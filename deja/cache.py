"""Semantic cache core: shared by the library wrapper and the proxy."""
import hashlib
import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

log = logging.getLogger("deja")
CACHEABLE_FINISH = {"stop", "tool_calls"}
SCHEMA_VERSION = 4  # bump when the entries table changes
STREAM_KEYS = ("stream", "stream_options")  # a streamed and a non-streamed request share an entry

# USD per 1M tokens (input, output), matched by longest model-name prefix. Unknown models count as $0.
# Check https://openai.com/api/pricing and override with Cache(prices=...).
PRICES = {
    "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00),
    "gpt-5-nano": (0.05, 0.40), "gpt-5-mini": (0.25, 2.00), "gpt-5": (1.25, 10.00),
}


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def cost(model, usage, prices=PRICES) -> float:
    key = max((k for k in prices if (model or "").startswith(k)), key=len, default=None)
    if not key or not usage:
        return 0.0
    p_in, p_out = prices[key]
    return (usage.get("prompt_tokens", 0) * p_in + usage.get("completion_tokens", 0) * p_out) / 1e6


def _estimated_usage(req: dict, resp: dict) -> dict:
    """Token counts from text length, for responses without usage (streams unless stream_options.include_usage)."""
    # ponytail: ~4 characters per token; good enough for savings stats, use tiktoken if exact numbers matter
    out = sum(len((c.get("message") or {}).get("content") or "") for c in resp.get("choices") or []) // 4
    inp = len(json.dumps(req.get("messages") or [])) // 4
    return {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}


def split_request(req: dict) -> tuple[str, str, str | None, list[dict]] | None:
    """Return (context_key, chat_key, query, history) or None if the request isn't cacheable.

    query = text of the last user message; context_key = hash of everything else.
    chat_key = the same minus the conversation history (model, settings, system prompt only), so a
    question asked mid-chat can find answers from other chats; history = those earlier turns.
    With no history, chat_key == context_key.
    query is None when the message carries an image, audio or file: exact match only,
    because the answer depends on the attachment, which the text can't capture.
    """
    if req.get("n", 1) != 1:
        return None
    msgs = req.get("messages") or []
    if not msgs or msgs[-1].get("role") != "user":
        return None
    content = msgs[-1].get("content")
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        content = "\n".join(texts) if len(texts) == len(content) else None
    elif not isinstance(content, str):
        content = None
    params = {k: v for k, v in req.items() if k not in ("messages", "user", *STREAM_KEYS)}
    prior = msgs[:-1]
    system = [m for m in prior if m.get("role") in ("system", "developer")]
    history = [m for m in prior if m.get("role") not in ("system", "developer")]
    return _hash({**params, "messages": prior}), _hash({**params, "messages": system}), content, history


def to_chunks(resp: dict) -> list[dict]:
    """Replay a cached completion as stream chunks: the whole message as one delta, then finish."""
    base = {"id": resp.get("id"), "object": "chat.completion.chunk", "created": resp.get("created"), "model": resp.get("model")}
    first, last = [], []
    for c in resp["choices"]:
        delta = {k: v for k, v in c["message"].items() if v is not None}
        if delta.get("tool_calls"):
            delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(delta["tool_calls"])]
        first.append({"index": c["index"], "delta": delta, "finish_reason": None})
        last.append({"index": c["index"], "delta": {}, "finish_reason": c["finish_reason"]})
    return [{**base, "choices": first}, {**base, "choices": last, "usage": resp.get("usage")}]


def from_chunks(chunks: list[dict]) -> dict | None:
    """Rebuild a completion from stream chunks, or None if it can't be rebuilt."""
    choices, usage = {}, None
    for ch in chunks:
        usage = ch.get("usage") or usage
        for c in ch.get("choices") or []:
            d = c.get("delta") or {}
            if d.get("tool_calls"):  # ponytail: tool-call deltas aren't reassembled, so those streams go uncached
                return None
            acc = choices.setdefault(c["index"], {"index": c["index"], "message": {"role": "assistant", "content": ""}, "finish_reason": None})
            acc["message"]["content"] += d.get("content") or ""
            acc["finish_reason"] = c.get("finish_reason") or acc["finish_reason"]
    if not choices:
        return None
    first = chunks[0]
    return {"id": first["id"], "object": "chat.completion", "created": first["created"], "model": first["model"],
            "choices": [choices[i] for i in sorted(choices)], "usage": usage}


def openai_embedder(model: str = "text-embedding-3-small", client=None):
    from openai import OpenAI

    client = client or OpenAI()

    def embed(texts: list[str]) -> np.ndarray:
        data = client.embeddings.create(model=model, input=texts).data
        return np.array([d.embedding for d in data], dtype=np.float32)

    return embed


VERIFY_PROMPT = (
    "You guard a response cache. You get two user requests sent in the same conversation context. "
    "Reply 'yes' only if a single correct answer fully answers both: same entities, numbers, units, "
    "direction, polarity, language and format. Otherwise reply 'no'. Reply with one word."
)


CHAT_VERIFY_PROMPT = (
    "You guard a response cache. You see a conversation, the user's NEW MESSAGE in it, and a CACHED ANSWER "
    "that was written for an EARLIER QUESTION in a different conversation. Reply 'yes' only if sending the "
    "cached answer as the reply to the new message would be fully correct and appropriate: it answers what the "
    "new message means in this conversation (resolve words like 'it', 'that' or 'what about' from the "
    "conversation), with the same entities, numbers and direction; it follows every instruction and preference "
    "set earlier in the conversation (language, format, length, level of detail, and constraints such as diet, "
    "budget, location, age or tools); and it doesn't refer to anything the user never said. "
    "Otherwise reply 'no'. Reply with one word."
)


def _yes_no(client, model, system, user, tracker, lock) -> bool:
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=1,
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    with lock:  # the proxy calls verifiers from a threadpool
        tracker.calls += 1
        tracker.dollars += cost(r.model, r.usage.model_dump() if r.usage else None)
    return (r.choices[0].message.content or "").strip().lower().startswith("y")


def _conversation_text(history: list[dict]) -> str:
    """Every user message (where instructions and constraints live) plus the last two assistant replies."""
    # ponytail: keeps the last 20 kept messages at 400 chars each; summarise if long chats lose early constraints
    last_assistant = [i for i, m in enumerate(history) if m.get("role") == "assistant"][-2:]
    kept = [m for i, m in enumerate(history) if m.get("role") != "assistant" or i in last_assistant][-20:]
    text = lambda c: c if isinstance(c, str) else json.dumps(c)
    return "\n".join(f"{m.get('role')}: {text(m.get('content'))[:400]}" for m in kept) or "(none)"


def openai_chat_verifier(model: str = "gpt-4o-mini", client=None):
    """Check for reuse across conversations: is the cached answer a correct reply to the new message,
    read in its own conversation? verify(history, new, cached_question, cached_answer) -> bool.
    Tracks spend like openai_verifier."""
    from openai import OpenAI

    client = client or OpenAI()

    def verify(history: list[dict], new: str, cached: str, answer: str) -> bool:
        t = time.perf_counter()
        user = (f"CONVERSATION SO FAR:\n{_conversation_text(history)}\n\nNEW MESSAGE:\n{new}\n\n"
                f"EARLIER QUESTION:\n{cached}\n\nCACHED ANSWER:\n{answer[:1500]}")
        ok = _yes_no(client, model, CHAT_VERIFY_PROMPT, user, verify, lock)
        with lock:
            verify.checks += 1
            verify.seconds += time.perf_counter() - t
        return ok

    lock = threading.Lock()
    verify.calls, verify.dollars, verify.checks, verify.seconds = 0, 0.0, 0, 0.0
    return verify


def openai_verifier(model: str = "gpt-4o-mini", client=None, both_orders=True):
    """Second check on a semantic hit: does the cached answer really answer the new request?

    both_orders asks twice in parallel, with A/B swapped, and needs two yeses: on the eval set it
    cut false hits 5/541 -> 2/541 at the cost of reuse 75% -> 65% (README). Same latency, 2x cost.
    The returned function tracks its own spend in `.calls`, `.dollars`, and per check in `.checks`, `.seconds`.
    """
    from openai import OpenAI

    client = client or OpenAI()

    def ask(first: str, second: str) -> bool:
        return _yes_no(client, model, VERIFY_PROMPT, f"Request A:\n{first}\n\nRequest B:\n{second}", verify, lock)

    def verify(new: str, cached: str) -> bool:
        t = time.perf_counter()
        if both_orders:
            swapped = pool.submit(ask, new, cached)
            ok = ask(cached, new) & swapped.result()  # `&`, not `and`: always collect the second call
        else:
            ok = ask(cached, new)
        with lock:
            verify.checks += 1
            verify.seconds += time.perf_counter() - t
        return ok

    pool = ThreadPoolExecutor(8)
    lock = threading.Lock()
    verify.calls, verify.dollars, verify.checks, verify.seconds = 0, 0.0, 0, 0.0
    return verify


class Cache:
    def __init__(self, path=":memory:", embed=None, threshold=0.65, ttl=None, namespace="default",
                 max_entries=None, prices=PRICES, verify=True, log_candidates=False, cost_aware=True,
                 verify_chat=True):
        # threshold is per embedding model: pick it with benchmark/threshold.py, don't guess.
        # With verify on, the threshold only picks a candidate; verify(new, cached) -> bool decides.
        # verify=True uses openai_verifier(); False turns it off (unsafe: see README results).
        # log_candidates records each near-match (sim >= threshold - 0.1) with its verdict, so real
        # traffic can be labelled later (benchmark/export_candidates.py). It stores prompt text.
        # cost_aware skips the check when calling the model is both cheaper and faster than verifying,
        # judged from the candidate's stored cost/latency vs. the verifier's running average.
        # verify_chat lets a message reuse an answer from a different conversation (or from mid-chat):
        # verify_chat(history, new, cached_question, cached_answer) -> bool sees the conversation and the
        # answer. True uses openai_chat_verifier("gpt-4.1-mini"); False keeps reuse within identical contexts.
        self.embed = embed or openai_embedder()
        self.verify = openai_verifier() if verify is True else verify or None
        self.verify_chat = openai_chat_verifier("gpt-4.1-mini") if verify_chat is True else verify_chat or None
        self.threshold = threshold
        self.ttl = ttl
        self.namespace = namespace
        self.max_entries = max_entries
        self.prices = prices
        self.log_candidates = log_candidates
        self.cost_aware = cost_aware
        self.stats = {"exact_hits": 0, "semantic_hits": 0, "misses": 0, "bypassed": 0, "cross_chat_hits": 0, "verify_rejects": 0, "not_worth_verifying": 0, "errors": 0,
                      "tokens_saved": 0, "dollars_saved": 0.0, "seconds_saved": 0.0}
        # ponytail: one global lock around sqlite, fine for a single process; per-bucket locks if contention shows up
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        # It's a cache: on a schema change, drop the old entries rather than migrate them.
        if self._db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            self._db.executescript(f"DROP TABLE IF EXISTS entries; DROP TABLE IF EXISTS candidates; PRAGMA user_version = {SCHEMA_VERSION};")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                namespace TEXT, ctx TEXT, chat TEXT, full_hash TEXT, query TEXT, vec BLOB,
                response TEXT, latency REAL, cost REAL, tokens INTEGER, created REAL, used REAL);
            CREATE INDEX IF NOT EXISTS ix_bucket ON entries(namespace, chat);
            CREATE INDEX IF NOT EXISTS ix_full ON entries(namespace, full_hash);
            CREATE INDEX IF NOT EXISTS ix_used ON entries(used);
            CREATE TABLE IF NOT EXISTS candidates (
                namespace TEXT, new_query TEXT, cached_query TEXT, sim REAL, verdict INTEGER, created REAL,
                history TEXT, answer TEXT);
        """)

    def _alive(self) -> float:
        return time.time() - self.ttl if self.ttl else 0.0

    def get(self, req: dict, namespace=None, info=None) -> dict | None:
        """Cached response or None. Pass a dict as `info` to learn why: kind (exact/semantic/None),
        closest (nearest cached query), similarity, verdict (True/False, None = not checked),
        cross_chat (the match came from a different conversation), query, and on a hit saved_dollars/saved_seconds."""
        ns = namespace or self.namespace
        info = {} if info is None else info
        info.update(kind=None, closest=None, similarity=None, verdict=None, skipped=False, cross_chat=False,
                    query=None, saved_dollars=0.0, saved_seconds=0.0)
        split = split_request(req)
        if split is None:
            self.stats["bypassed"] += 1
            return None
        ctx, chat, query, history = split
        info["query"] = query
        with self._lock:
            row = self._db.execute(
                "SELECT rowid, response, latency, cost, tokens FROM entries WHERE namespace=? AND full_hash=? AND created>=? LIMIT 1",
                (ns, _hash(self._exact(req)), self._alive())).fetchone()
            if row:
                info.update(kind="exact", closest=query, similarity=1.0, verdict=True)
                return self._hit(row, "exact_hits", info)
            rows = self._db.execute(
                "SELECT vec, query, rowid, response, latency, cost, tokens, ctx FROM entries WHERE namespace=? AND chat=? AND created>=?",
                (ns, chat, self._alive())).fetchall()
        # attachment entries are exact-match only; other conversations only with a chat verifier
        rows = [r for r in rows if r[0] is not None and (r[7] == ctx or self.verify_chat)]
        if query is None or not rows:
            self.stats["misses"] += 1
            return None
        try:
            # ponytail: brute-force cosine over one bucket; FAISS/sqlite-vec if a bucket passes ~100k entries
            q = self._vec(query)
            sims = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows]) @ q
            best = int(sims.argmax())
            # ponytail: only the top candidate is verified; check the next few if rejected-but-matchable shows up in evals
            cached_q, response, cross = rows[best][1], rows[best][3], rows[best][7] != ctx
            answer = ((json.loads(response).get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            verifier = self.verify_chat if cross else self.verify
            verdict = None
            if sims[best] >= self.threshold:
                if verifier and not self._worth_verifying(*rows[best][4:6], verifier):
                    self.stats["not_worth_verifying"] += 1
                    info["skipped"] = True
                elif cross:  # an answer from another chat must be text the checker can read
                    verdict = bool(answer) and verifier(history, query, cached_q, answer)
                else:
                    verdict = not verifier or verifier(query, cached_q)
            info.update(closest=cached_q, similarity=float(sims[best]), verdict=verdict, cross_chat=cross)
            if self.log_candidates and sims[best] >= self.threshold - 0.1:
                with self._lock:
                    self._db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?)",
                                     (ns, query, cached_q, float(sims[best]), verdict, time.time(),
                                      json.dumps(history) if history else None, answer[:1500] if cross else None))
                    self._db.commit()
            if verdict:
                info["kind"] = "semantic"
                self.stats["cross_chat_hits"] += cross
                with self._lock:
                    return self._hit(rows[best][2:7], "semantic_hits", info)
            if verdict is False:
                self.stats["verify_rejects"] += 1
        except Exception:  # embedding/verifier outage (rate limit, network): degrade to a miss, never fail the request
            log.warning("deja lookup failed, treating as miss", exc_info=True)
            self.stats["errors"] += 1
        self.stats["misses"] += 1
        return None

    def put(self, req: dict, resp: dict, latency: float = 0.0, namespace=None) -> bool:
        split = split_request(req)
        choices = resp.get("choices") or []
        if split is None or not choices or any(c.get("finish_reason") not in CACHEABLE_FINISH for c in choices):
            return False
        ctx, chat, query, _ = split
        try:
            vec = None if query is None else self._vec(query).tobytes()
        except Exception:
            log.warning("deja store failed, response not cached", exc_info=True)
            self.stats["errors"] += 1
            return False
        usage = resp.get("usage") or _estimated_usage(req, resp)
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO entries (namespace, ctx, chat, full_hash, query, vec, response, latency, cost, tokens, created, used) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (namespace or self.namespace, ctx, chat, _hash(self._exact(req)), query, vec, json.dumps(resp),
                 latency, cost(resp.get("model") or req.get("model"), usage, self.prices), usage["total_tokens"], now, now))
            if self.max_entries:  # evict least recently used
                self._db.execute("DELETE FROM entries WHERE rowid NOT IN (SELECT rowid FROM entries ORDER BY used DESC LIMIT ?)",
                                 (self.max_entries,))
            self._db.commit()
        return True

    def clear(self):
        with self._lock:
            self._db.execute("DELETE FROM entries")
            self._db.commit()

    def purge_expired(self):
        with self._lock:
            self._db.execute("DELETE FROM entries WHERE created<?", (self._alive(),))
            self._db.commit()

    def _worth_verifying(self, latency, dollars, verifier) -> bool:
        checks = getattr(verifier, "checks", 0)
        if not self.cost_aware or not checks:  # custom verifier, or no measurements yet
            return True
        saves_money = not dollars or dollars > verifier.dollars / checks  # $0 = unknown price: assume it does
        saves_time = (latency or 0.0) > verifier.seconds / checks
        return saves_money or saves_time

    @staticmethod
    def _exact(req: dict) -> dict:
        return {k: v for k, v in req.items() if k not in STREAM_KEYS}

    def _vec(self, text: str) -> np.ndarray:
        v = np.asarray(self.embed([text])[0], dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def _hit(self, row, kind, info) -> dict:  # caller holds the lock
        rowid, response, latency, dollars, tokens = row
        self._db.execute("UPDATE entries SET used=? WHERE rowid=?", (time.time(), rowid))
        self._db.commit()
        resp = json.loads(response)
        self.stats[kind] += 1
        self.stats["tokens_saved"] += tokens or 0
        self.stats["dollars_saved"] += dollars or 0.0
        self.stats["seconds_saved"] += latency or 0.0
        info.update(saved_dollars=dollars or 0.0, saved_seconds=latency or 0.0)
        return resp

    def wrap(self, client):
        """deja.wrap(OpenAI()) -> same client, chat.completions.create now cached.

        With stream=True you get a plain iterator of chunks (no `with` / .close() helpers).
        """
        from openai.types.chat import ChatCompletion, ChatCompletionChunk

        original = client.chat.completions.create

        def create(**kwargs):
            hit = self.get(kwargs)
            if hit is not None:
                if kwargs.get("stream"):
                    return iter([ChatCompletionChunk.model_validate(c) for c in to_chunks(hit)])
                return ChatCompletion.model_validate(hit)
            t = time.perf_counter()
            resp = original(**kwargs)
            if kwargs.get("stream"):
                return self._tee(kwargs, resp, t)
            self.put(kwargs, resp.model_dump(), time.perf_counter() - t)
            return resp

        client.chat.completions.create = create
        return client

    def _tee(self, req, stream, t):
        """Pass chunks through; cache the rebuilt completion once the stream finishes."""
        chunks = []
        for chunk in stream:
            chunks.append(chunk.model_dump())
            yield chunk
        resp = from_chunks(chunks)
        if resp:
            self.put(req, resp, time.perf_counter() - t)

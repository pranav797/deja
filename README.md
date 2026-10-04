# Déjà

A semantic cache for LLM calls. If a new prompt *means the same thing* as one it has already answered, Déjà returns the stored answer instead of calling the model.

Only the last user message is matched by meaning. The model, system prompt, temperature and tools must match exactly. Earlier turns of the conversation don't have to: a question asked mid-chat can reuse an answer from another chat, but only after a second checker (gpt-4.1-mini) reads the conversation and the cached answer and agrees it's a correct reply there. That stops reuse for follow-ups ("what about France?"), and when earlier messages set constraints ("I'm vegan") or instructions ("reply in Spanish").

Embedding similarity only finds a *candidate*. Before a semantic hit is served, a small model (gpt-4o-mini by default) checks that one answer really fits both requests. Embeddings alone confuse "translate to French" with "translate from French".

## Setup

You need Python 3.10+, [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
git clone https://github.com/pranav797/deja.git
cd deja
uv sync --extra proxy
```

Put your key in a `.env` file in the project folder. It's listed in `.gitignore`, so it won't be committed:

```
OPENAI_API_KEY=sk-...
```

The commands below load it with `uv run --env-file .env`.

## Library

```python
import deja
from openai import OpenAI

client = deja.wrap(OpenAI(), path="deja.db")  # threshold=0.65, verifier on
client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Who wrote Pride and Prejudice?"}])
client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Who is the author of Pride and Prejudice?"}])  # hit: similarity 0.944, verifier says yes
```

## Proxy (no code change)

```bash
pip install -e ".[proxy]"
deja-proxy
```

Point your client at it: `OpenAI(base_url="http://127.0.0.1:8000/v1")`. Only chat completions are cached. Every other `/v1/...` request (model lists, embeddings, the Responses API, files, audio) is passed through to OpenAI unchanged, marked `x-deja-cache: bypass`, so apps that use more than chat still work. Responses carry `x-deja-cache: hit|miss`; metrics are at `GET /deja/stats`.
Settings: `DEJA_UPSTREAM`, `DEJA_DB`, `DEJA_THRESHOLD`, `DEJA_VERIFY_MODEL` (`off` disables), `DEJA_VERIFY_BOTH_ORDERS` (default `1`), `DEJA_VERIFY_CHAT_MODEL` (default `gpt-4.1-mini`, `off` = reuse only within an identical conversation), `DEJA_COST_AWARE` (default `1`), `DEJA_LOG_CANDIDATES`, `DEJA_TTL`, `DEJA_MAX_ENTRIES` (LRU eviction), `DEJA_HOST`, `DEJA_PORT`.
Streaming works: a streamed miss is cached once it finishes cleanly, and a hit is replayed as SSE. Streamed and non-streamed requests share entries. Per-request namespace: `X-Deja-Namespace` header.

## Playground

The proxy serves a small chat page for trying the cache by hand:

```bash
uv run --env-file .env --extra proxy deja-proxy
```

Then open http://127.0.0.1:8000. Each answer shows whether it was reused, which earlier question it matched, the similarity, and what the checker decided. The sidebar shows money saved against money spent on checks. Questions are sent one at a time, with no chat history, because a reworded question only matches when everything before it in the conversation is identical. To collect your own questions for labelling, start the proxy with `DEJA_LOG_CANDIDATES=1` (see *Measuring on real traffic*).

The page uses the proxy's `OPENAI_API_KEY` when a request arrives without its own key. The proxy only listens on 127.0.0.1 by default; don't expose it on a network with a key in its environment.

## Using it with Open WebUI (Windows)

[Open WebUI](https://github.com/open-webui/open-webui) is a ChatGPT-style app that runs on your computer. To route it through Déjà:

```bash
uv tool install --python 3.11 open-webui
powershell -ExecutionPolicy Bypass -File start-webui.ps1
```

Then open http://127.0.0.1:8080 and create your account; it's stored only on your computer. To watch what Déjà decides while you chat, open the live dashboard at http://127.0.0.1:8000/dashboard. It shows net savings, reuse rate and a feed of every chat request: reused or not, which earlier question it matched, the similarity, and the time taken. It updates every second. The script starts Déjà with near-match logging on, and Open WebUI pointed at it, both reachable only from this computer. It turns off Open WebUI's background title, tag and follow-up calls: their prompts share long templates, so Déjà would treat different chats as near-duplicates. Questions asked mid-chat can be reused too (see *Reuse inside conversations*); follow-ups that depend on the chat are not. Open WebUI keeps its data in `%USERPROFILE%\.open-webui`; Déjà's cache and log stay in `deja.db`.

## Picking the threshold

Measure it for your embedding model and verifier:

```bash
uv run --env-file .env python benchmark/threshold.py --verify gpt-4o-mini
```

`benchmark/pairs.jsonl` and `benchmark/hard_negatives.jsonl` hold paraphrases plus hard negatives (entity swaps, negation, numbers, direction). The script prints hit rate and false hits at several thresholds and picks the lowest threshold with a false-hit rate at or below `--target`.
With the verifier on, a lower threshold means more hits but more verifier calls (about 430 ms each) on requests that end up as misses.

## Exact match only

Messages with an image, audio or file attached are only reused for an identical request. Two "translate this photo" requests look the same as text but are about different images.

## Never cached

Streams with tool calls, partial or disconnected streams, `n > 1`, requests that don't end in a user message, truncated or filtered responses, and errors. If the embedding or verifier call fails (a rate limit, say), the lookup counts as a miss and the request goes to the model.

## When it pays off

Exact repeats are always free. A semantic hit costs an embedding plus two verifier calls (about $0.00003 with gpt-4o-mini), so it saves money only when the answer it replaces costs more than that. That's true for larger models and long answers, but not for short gpt-4o-mini answers: in one end-to-end test the verifier cost $0.000086 and saved $0.000063. Déjà handles this itself (`cost_aware`, on by default). Before checking a candidate, it compares the stored cost and latency of the answer it would reuse with the verifier's running average. If calling the model again is both cheaper and faster, it skips the check and treats the request as a miss (`not_worth_verifying` in stats). A short gpt-4o-mini answer that took 1.4 s is still checked, because the check (~0.4 s) saves time.

## Measuring on real traffic

Turn on `log_candidates=True` (or `DEJA_LOG_CANDIDATES=1`) to record each near-match with its similarity and verdict. Then export a random sample, label `same` by hand and evaluate:

```bash
python benchmark/export_candidates.py deja.db --n 200 > benchmark/data/real.jsonl
uv run --env-file .env python benchmark/threshold.py benchmark/data/real.jsonl --verify gpt-4o-mini
```

The log and the export contain real prompts, so keep them out of git (`benchmark/data/` is ignored).

## Results (text-embedding-3-small, Oct 2026)

**Embedding similarity alone is not safe.** Some pairs that need different answers score higher than every real rewording ("translate to French" / "from French" 0.971, "sort ascending" / "descending" 0.959, best real rewording 0.944). Without a verifier, no threshold gets any hits at ≤1% false hits, and 0.9 gives 2/15 false hits.

**With the verifier: 730 pairs** (`benchmark/threshold.py --verify gpt-4o-mini`, threshold 0.65, two runs):

| slice | pairs | result | 95% CI |
|---|---|---|---|
| hand-written trick pairs (entity, number, negation, direction, time, language, format, scope) | 390 | **2–3 false hits** (0.5–0.8%) | 0.1%–2.2% |
| Quora pairs that need different answers | 151 | 2 false hits (1.3%) | 0.4%–4.7% |
| **all "different" pairs** | 541 | **4–5 false hits (0.7–0.9%)** | 0.3%–2.1% |
| all "same" pairs: reused | 189 | 75% | 69%–81% |

Fetch the Quora sample with `uv run --with pyarrow python benchmark/fetch_qqp.py`. It isn't committed, because Quora's terms may not allow redistribution.

**Quora labels.** Quora labels pairs as *duplicate questions*, which isn't the same as *same answer*. `benchmark/qqp_relabel.json` relabels 33 of the 300 sampled pairs by the rule "would the cached answer to `a` fully satisfy someone asking `b`?". The file stores only Quora row IDs, never question text. The relabelling was done by Claude after seeing some v6 errors, so spot-check it.

**What still gets through:**
- "Flight time New York → Tokyo" vs "Tokyo → New York": a real direction miss.
- Two "translate this photo from Arabic" requests about different images: a real miss. Requests that depend on an attachment can't be judged from the text alone.
- "Who is the lady blindfolded in courts?" vs "Who blindfolded Lady Justice?": a real miss.
- "list index out of range on line 12" vs the same error without a line number: arguably mislabelled, since the specific answer also answers the general question.
- "Is it safe to eat raw eggs?" vs "Why is it not safe…": appears in one run of two. The verifier isn't fully deterministic on borderline pairs.

**Does it meet "≤1% false hits"?** The point estimate does (0.7–0.9%), but the 95% upper bound is about 2%. Bringing the upper bound under 1% needs either about 2,000 trick pairs at this error rate or fixing the failure modes above.

**Threshold:** the sweep's lowest threshold at ≤1% false hits is 0.646. False hits are identical from 0.65 to 0.75 and reuse rises from 69% to 75%, so the default is now 0.65. The cost is more verifier calls (~420 ms each) on candidates that end up rejected.

**v8: verifier asks in both orders** (`--both-orders`, one run, threshold 0.65):

| verifier | false hits | reused |
|---|---|---|
| one call, cached request first | 5/541 (0.9%) | 141/189 (75%) |
| one call, new request first | 6/541 (1.1%) | 137/189 (72%) |
| **both orders must say yes (default)** | **2/541 (0.4%, 95% CI 0.1%–1.3%)** | 123/189 (65%) |

The two-yes rule removes the raw-eggs and Arabic-photo misses. The two left are the New York/Tokyo flight time and Lady Justice. Both calls run in parallel, so latency is unchanged and cost doubles. Turn it off with `openai_verifier(both_orders=False)` or `DEJA_VERIFY_BOTH_ORDERS=0` if reuse matters more than the extra false hits.

**v9: stronger verifier models** (both orders, threshold 0.65):

| verifier | false hits | reused | cost per check |
|---|---|---|---|
| **gpt-4o-mini (default)** | 2/541 (0.4%, 95% CI 0.1%–1.3%) | 123/189 (65%) | 1× |
| gpt-4.1-mini | 1/541 (0.2%, 95% CI 0.0%–1.0%) | 100/189 (53%) | ~2.7× |
| gpt-4o | not measured: hit the account's 30k tokens/min limit, which would also throttle a busy proxy | | ~17× |

gpt-4.1-mini is the only configuration whose upper bound reaches 1%. But 1 vs 2 false hits is within noise, while the drop in reuse is not, so it's offered as the stricter option rather than the default: `Cache(verify=openai_verifier("gpt-4.1-mini"))` or `DEJA_VERIFY_MODEL=gpt-4.1-mini`. Its one false hit is new: "leap year function in Python" vs the same "without using the calendar module".

**Reuse inside conversations** (`benchmark/chat_eval.py`, 93 hand-written cases, threshold 0.65):

| chat checker | reused when right | reused when wrong |
|---|---|---|
| gpt-4o-mini | 7/37 (19%) | 2/56 (3.6%) |
| **gpt-4.1-mini (default)** | **32/37 (86%, 95% CI 72%–94%)** | **2/56 (3.6%, 95% CI 1.0%–12.1%)** |

By category with gpt-4.1-mini: standalone questions asked mid-chat 30/30 reused; follow-ups like "what about France?" or "make it shorter" 0/20; earlier constraints (diet, budget, country, tools) 0/12 ignored; earlier instructions 1/10 ignored ("only give me code" served an answer with an explanation); answers that refer to their own chat 1/8 ("in your stylesheet from before…"); different subjects 0/6. Follow-ups that would be fine to reuse, like "when was it founded?" after asking about Audi, are below the threshold on their own words (0.44–0.48), so only 2/7 are reused. Matching those would need the question rewritten from the conversation first.

gpt-4o-mini is too cautious here: it refused plain repeats like "what's the capital of Australia?". Making the two missed rules more explicit in the prompt changed nothing, so the prompt stays as is. The wrong-reuse rate is higher than for first messages (0.4%) on a set built to be hard; real chats are mostly ordinary follow-ups, which the checker reliably refuses. To keep the stricter old behavior, set `DEJA_VERIFY_CHAT_MODEL=off`.

Streamed answers usually carry no token counts, so Déjà estimates them from text length (about 4 characters per token) for its savings stats.

**Benchmark** (`benchmark/run.py` on the earlier 27-pair set, before the defaults changed: gpt-4o-mini answers, threshold 0.75, gpt-4o-mini verifier):

| metric | no cache | déjà |
|---|---|---|
| p50 latency | 1401 ms | 1183 ms |
| p95 latency | 10802 ms | 8097 ms |
| cost (incl. verifier) | $0.00394 | $0.00243 |
| hit rate on paraphrases | – | 92% (11/12) |
| false hits on hard negatives | – | 1/15 |

Latency savings are modest here because a semantic hit still pays for an embedding and a verifier call; exact repeats skip both.

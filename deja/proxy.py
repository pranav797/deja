"""OpenAI-compatible proxy. Point your client's base_url at http://localhost:8000/v1.

Env: DEJA_UPSTREAM (default https://api.openai.com/v1), DEJA_DB (default deja.db),
DEJA_THRESHOLD (default 0.65), DEJA_TTL seconds, DEJA_MAX_ENTRIES (default unlimited),
DEJA_VERIFY_MODEL (default gpt-4o-mini, "off" disables verification),
DEJA_VERIFY_BOTH_ORDERS (default 1; 0 = one verifier call, more reuse, more false hits),
DEJA_VERIFY_CHAT_MODEL (default gpt-4.1-mini; checks reuse across conversations, "off" = same context only),
DEJA_LOG_CANDIDATES (1 = record near-matches for labelling; stores prompt text),
DEJA_COST_AWARE (default 1; skip verifying when calling the model is cheaper and faster),
OPENAI_API_KEY for embeddings, and for upstream calls that arrive without an Authorization header.
Namespace per request via the `X-Deja-Namespace` header.
Only POST /v1/chat/completions is cached; every other /v1/... request is passed through unchanged.
GET / serves a playground page for trying the cache by hand; GET /dashboard shows live totals and decisions.
"""
import asyncio
import itertools
import json
import os
import pathlib
import time
from collections import deque
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from .cache import Cache, from_chunks, openai_chat_verifier, openai_verifier, to_chunks

UPSTREAM = os.environ.get("DEJA_UPSTREAM", "https://api.openai.com/v1").rstrip("/")
cache = Cache(
    path=os.environ.get("DEJA_DB", "deja.db"),
    threshold=float(os.environ.get("DEJA_THRESHOLD", "0.65")),
    verify=False if (VERIFY_MODEL := os.environ.get("DEJA_VERIFY_MODEL", "gpt-4o-mini")) == "off"
    else openai_verifier(VERIFY_MODEL, both_orders=os.environ.get("DEJA_VERIFY_BOTH_ORDERS", "1") == "1"),
    verify_chat=False if (CHAT_MODEL := os.environ.get("DEJA_VERIFY_CHAT_MODEL", "gpt-4.1-mini")) == "off"
    else openai_chat_verifier(CHAT_MODEL),
    log_candidates=os.environ.get("DEJA_LOG_CANDIDATES") == "1",
    cost_aware=os.environ.get("DEJA_COST_AWARE", "1") == "1",
    ttl=float(os.environ["DEJA_TTL"]) if os.environ.get("DEJA_TTL") else None,
    max_entries=int(os.environ["DEJA_MAX_ENTRIES"]) if os.environ.get("DEJA_MAX_ENTRIES") else None,
)
http = httpx.AsyncClient(timeout=600)
app = FastAPI(title="deja")
EVENTS, _ids = deque(maxlen=200), itertools.count(1)  # recent decisions for the dashboard, in memory only
BOOT = str(time.time())  # lets the dashboard notice a restart, when ids start over


def _event(body: dict, info: dict) -> dict:
    """One dashboard row: what Déjà decided for this chat request and why."""
    if info["kind"] == "exact":
        result = "exact"
    elif info["kind"]:
        result = "other-chat" if info["cross_chat"] else "reworded"
    elif info["query"] is None:
        result = "not-cacheable"
    else:
        result = "blocked" if info["verdict"] is False else "skipped" if info["skipped"] else "new"
    return {"question": info["query"], "model": body.get("model"), "result": result, "closest": info["closest"],
            "similarity": info["similarity"], "saved_dollars": info["saved_dollars"], "saved_seconds": info["saved_seconds"]}


def _record(event: dict, t: float):
    EVENTS.append({**event, "id": next(_ids), "time": time.time(), "ms": round((time.perf_counter() - t) * 1000)})


FWD_HEADERS = ("authorization", "content-type", "accept", "user-agent", "idempotency-key")
# Not forwarded: hop-by-hop and length headers (httpx sets its own), and anything a browser adds (cookies, origin).
DROP_RESPONSE = {"content-encoding", "content-length", "transfer-encoding", "connection"}  # aiter_bytes() decodes


def _fwd_headers(request: Request) -> dict:
    h = {k: v for k, v in request.headers.items() if k.lower() in FWD_HEADERS or k.lower().startswith("openai-")}
    if not any(k.lower() == "authorization" for k in h) and os.environ.get("OPENAI_API_KEY"):
        # lets the playground (and local tools) call without holding a key; the proxy binds to 127.0.0.1 by default
        h["authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    return h


def _deja_headers(info: dict) -> dict:
    """Why the request hit or missed, for the playground and for debugging."""
    h = {"x-deja-cache": "hit" if info["kind"] else "miss", "x-deja-kind": info["kind"] or "none",
         "x-deja-cross-chat": "1" if info["cross_chat"] else "0"}
    if info["closest"] is not None:
        h["x-deja-closest"] = quote(info["closest"])  # headers must be ASCII
        h["x-deja-similarity"] = f"{info['similarity']:.3f}"
        h["x-deja-verdict"] = "skipped" if info["skipped"] else {True: "yes", False: "no", None: "below-threshold"}[info["verdict"]]
    return h


_background = set()  # strong refs so store tasks aren't garbage-collected mid-flight


def _store_stream(raw: bytes, body: dict, ns, latency: float):
    """Cache a relayed stream from a background task: clients (Open WebUI, the OpenAI SDK) disconnect the
    moment they read [DONE], and Starlette then cancels the response, which would cut off an inline store.
    A stream that was cut off mid-answer has no finish_reason, so cache.put rejects it."""
    chunks = []
    for line in raw.decode(errors="replace").splitlines():
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            try:
                chunks.append(json.loads(line[5:]))
            except json.JSONDecodeError:  # half-received last line of a cut-off stream
                return
    resp = from_chunks(chunks)
    if resp:
        task = asyncio.create_task(run_in_threadpool(cache.put, body, resp, latency, ns))
        _background.add(task)
        task.add_done_callback(_background.discard)


async def _relay(upstream: httpx.Response, body: dict, ns, t: float, event: dict):
    """Pass the SSE stream through; cache the rebuilt completion if it finished cleanly."""
    raw = bytearray()
    try:
        async for b in upstream.aiter_bytes():
            raw += b
            yield b
    finally:  # runs on normal end and on cancellation; no await before the store is handed off
        _record(event, t)
        if upstream.status_code == 200:
            _store_stream(bytes(raw), body, ns, time.perf_counter() - t)
        await upstream.aclose()


@app.post("/v1/chat/completions")
async def chat(request: Request):
    t = time.perf_counter()
    body = await request.json()
    ns = request.headers.get("x-deja-namespace")
    info = {}
    hit = await run_in_threadpool(cache.get, body, ns, info)  # embedding call is blocking
    event = _event(body, info)
    if hit is not None:
        _record(event, t)
        if body.get("stream"):
            sse = [f"data: {json.dumps(c)}\n\n" for c in to_chunks(hit)] + ["data: [DONE]\n\n"]
            return StreamingResponse(iter(sse), media_type="text/event-stream", headers=_deja_headers(info))
        return JSONResponse(hit, headers=_deja_headers(info))

    url = f"{UPSTREAM}/chat/completions"
    if body.get("stream"):
        upstream = await http.send(http.build_request("POST", url, json=body, headers=_fwd_headers(request)), stream=True)
        return StreamingResponse(_relay(upstream, body, ns, t, event), status_code=upstream.status_code,
                                 media_type=upstream.headers.get("content-type"), headers=_deja_headers(info))

    r = await http.post(url, json=body, headers=_fwd_headers(request))
    if r.status_code == 200:
        await run_in_threadpool(cache.put, body, r.json(), time.perf_counter() - t, ns)
    _record(event, t)
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"),
                    headers=_deja_headers(info))


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def passthrough(path: str, request: Request):
    """Everything Déjà doesn't cache (models, embeddings, responses, files, audio, ...) goes upstream untouched."""
    # ponytail: request body is read into memory; stream it with request.stream() if large file uploads matter
    req = http.build_request(request.method, f"{UPSTREAM}/{path}", params=request.query_params,
                             content=await request.body(), headers=_fwd_headers(request))
    upstream = await http.send(req, stream=True)
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in DROP_RESPONSE}
    return StreamingResponse(upstream.aiter_bytes(), status_code=upstream.status_code,
                             headers={**headers, "x-deja-cache": "bypass"}, background=BackgroundTask(upstream.aclose))


@app.get("/deja/stats")
def stats():
    s = cache.stats
    hits = s["exact_hits"] + s["semantic_hits"]
    return {**s, "hit_rate": hits / ((hits + s["misses"]) or 1),
            "verify_dollars": getattr(cache.verify, "dollars", 0.0) + getattr(cache.verify_chat, "dollars", 0.0)}


@app.get("/deja/events")
def events(after: int = 0):
    return {"boot": BOOT, "events": [e for e in EVENTS if e["id"] > after]}


@app.get("/dashboard")
def dashboard():
    return FileResponse(pathlib.Path(__file__).with_name("dashboard.html"))


@app.post("/deja/clear")
def clear():
    cache.clear()
    return {"cleared": True}


@app.get("/")
def playground():
    return FileResponse(pathlib.Path(__file__).with_name("playground.html"))


def main():
    import uvicorn

    uvicorn.run(app, host=os.environ.get("DEJA_HOST", "127.0.0.1"), port=int(os.environ.get("DEJA_PORT", "8000")))

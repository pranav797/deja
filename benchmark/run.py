"""Before/after benchmark on a stated workload: benchmark/pairs.jsonl.

Baseline: send every `b` prompt straight to the model.
Déjà:     warm the cache with every `a`, then send every `b` (only the `b` pass is measured).
A hit on a `same: false` pair is a false hit: the user got the answer to a different question.
Déjà's cost includes verifier calls; embeddings (text-embedding-3-small, $0.02 / 1M tokens) are left out as negligible.

    uv run --env-file .env python benchmark/run.py [--threshold 0.65] [--verify gpt-4o-mini|off]
"""
import argparse
import json
import pathlib
import time

import numpy as np
from openai import OpenAI

from deja import Cache, openai_verifier
from deja.cache import cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="?", default=pathlib.Path(__file__).with_name("pairs.jsonl"))
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--verify", default="gpt-4o-mini", help='verifier model, or "off"')
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]

    def ask(client, q):
        t = time.perf_counter()
        r = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": q}])
        return r, time.perf_counter() - t

    raw = OpenAI()
    base = [ask(raw, r["b"]) for r in rows]

    verify = False if args.verify == "off" else openai_verifier(args.verify)
    cache = Cache(threshold=args.threshold, verify=verify)
    cached = cache.wrap(OpenAI())
    for a in dict.fromkeys(r["a"] for r in rows):
        ask(cached, a)
    runs, hits = [], []
    for r in rows:
        before = cache.stats["exact_hits"] + cache.stats["semantic_hits"]
        runs.append(ask(cached, r["b"]))
        hits.append(cache.stats["exact_hits"] + cache.stats["semantic_hits"] > before)

    hits, same = np.array(hits), np.array([r["same"] for r in rows])
    b_lat, c_lat = np.array([s for _, s in base]), np.array([s for _, s in runs])
    b_cost = sum(cost(resp.model, resp.usage.model_dump()) for resp, _ in base)
    c_cost = sum(cost(resp.model, resp.usage.model_dump()) for (resp, _), h in zip(runs, hits) if not h)
    c_cost += getattr(verify, "dollars", 0.0)

    print(f"model={args.model}  threshold={args.threshold}  verify={args.verify}  requests={len(rows)} "
          f"({same.sum()} paraphrases, {(~same).sum()} hard negatives)\n")
    print("| metric | no cache | déjà |\n|---|---|---|")
    print(f"| p50 latency | {np.percentile(b_lat, 50)*1000:.0f} ms | {np.percentile(c_lat, 50)*1000:.0f} ms |")
    print(f"| p95 latency | {np.percentile(b_lat, 95)*1000:.0f} ms | {np.percentile(c_lat, 95)*1000:.0f} ms |")
    print(f"| model cost | ${b_cost:.5f} | ${c_cost:.5f} |")
    print(f"| hit rate on paraphrases | – | {hits[same].mean():.0%} |")
    print(f"| false hits on hard negatives | – | {hits[~same].sum()}/{(~same).sum()} |")
    for r, h in zip(rows, hits):
        if h and not r["same"]:
            print(f"  FALSE HIT: {r['b']!r} served the answer to {r['a']!r}-like prompt")


if __name__ == "__main__":
    main()

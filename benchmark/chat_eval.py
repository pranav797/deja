"""Measure reuse inside conversations: can a cached answer from one chat answer a message in another?

    uv run --env-file .env python benchmark/chat_eval.py [--verify gpt-4o-mini]

Each row in chat_pairs.jsonl has a conversation `history`, the new message `b`, an earlier question `a` from a
different chat, and `answer`, the cached answer to `a`. `same` = sending that answer as the reply to `b` would be
correct. Categories cover standalone mid-chat questions (reuse is right), follow-ups that resolve to the cached
question (right), and follow-ups, earlier constraints, earlier instructions, answers that point at their own chat,
and different subjects (all wrong). Hand-written by Claude: spot-check before trusting.
"""
import argparse
import json
import pathlib
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from threshold import wilson

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="?", default=HERE / "chat_pairs.jsonl")
    ap.add_argument("--verify", default="gpt-4o-mini")
    ap.add_argument("--threshold", type=float, default=0.65)
    args = ap.parse_args()

    from openai import OpenAI
    from deja import openai_embedder
    from deja.cache import openai_chat_verifier

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    if any(r["same"] is None for r in rows):
        raise SystemExit('some pairs have "same": null; label them first')
    vecs = openai_embedder()([r["a"] for r in rows] + [r["b"] for r in rows])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = (vecs[: len(rows)] * vecs[len(rows):]).sum(axis=1)
    same = np.array([r["same"] for r in rows])
    cats = np.array([r["category"] for r in rows])

    verify = openai_chat_verifier(args.verify, client=OpenAI(max_retries=10))

    def check(r):
        t = time.perf_counter()
        return verify(r["history"], r["b"], r["a"], r["answer"]), time.perf_counter() - t

    with ThreadPoolExecutor(4) as pool:
        out = list(pool.map(check, rows))
    verdict = np.array([v for v, _ in out])
    hit = (sims >= args.threshold) & verdict

    n_pos, n_neg = same.sum(), (~same).sum()
    k_pos, k_neg = hit[same].sum(), hit[~same].sum()
    print(f"verifier={args.verify}  threshold={args.threshold}  pairs={len(rows)} ({n_pos} reuse-is-right, {n_neg} reuse-is-wrong)")
    print(f"verifier p50 {np.median([s for _, s in out]) * 1000:.0f} ms, ${verify.dollars:.4f} for {verify.calls} calls\n")
    lo, hi = wilson(k_pos, n_pos)
    flo, fhi = wilson(k_neg, n_neg)
    print(f"reused when right: {k_pos}/{n_pos} ({k_pos / n_pos:.0%}, 95% CI {lo:.0%}–{hi:.0%})")
    print(f"reused when wrong: {k_neg}/{n_neg} ({k_neg / n_neg:.1%}, 95% CI {flo:.1%}–{fhi:.1%})\n")
    print("| category | reuse right? | reused |\n|---|---|---|")
    for c in dict.fromkeys(cats):
        m = cats == c
        print(f"| {c} | {'yes' if same[m][0] else 'no'} | {hit[m].sum()}/{m.sum()} |")

    errors = defaultdict(list)
    for r, s, v, h in zip(rows, sims, verdict, hit):
        if h != r["same"]:
            why = "WRONG REUSE" if h else ("checker said no" if s >= args.threshold else "below threshold")
            errors[why].append(f"  {s:.3f} [{r['category']}] new={r['b']!r} cached={r['a']!r}")
    for why, lines in errors.items():
        print(f"\n{why} ({len(lines)}):\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

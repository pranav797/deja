"""Pick the similarity threshold from labelled pairs instead of guessing.

    uv run --env-file .env python benchmark/threshold.py [files.jsonl ...] [--verify gpt-4o-mini] [--threshold 0.65]

Defaults to benchmark/pairs.jsonl + hard_negatives.jsonl, plus data/qqp.jsonl if fetch_qqp.py has been run.
A pair counts as a hit if its similarity clears the threshold (and, with --verify, the verifier says yes).
Prints hit rate and false hits per threshold with 95% Wilson intervals, a per-category breakdown at
--threshold, and every error at --threshold so the labels can be checked by hand.
Pairwise eval approximates the cache, which only verifies its single closest entry.
Re-run whenever the embedding model, verifier model or verifier prompt changes.
"""
import argparse
import json
import math
import pathlib
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a rate of k out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre, spread = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - spread) / (1 + z * z / n)), min(1.0, (centre + spread) / (1 + z * z / n))


def sweep(sims: np.ndarray, same: np.ndarray, target: float):
    """Return (threshold, recall, false_hit_rate) for the lowest threshold meeting target."""
    for t in np.unique(np.round(sims, 4)):
        hit = sims >= t
        fhr = (hit & ~same).sum() / max((~same).sum(), 1)
        if fhr <= target:
            return float(t), hit[same].mean(), fhr
    return 1.0, 0.0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*")
    ap.add_argument("--target", type=float, default=0.01, help="max acceptable false-hit rate")
    ap.add_argument("--threshold", type=float, default=0.65, help="threshold for the breakdown and error list")
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--verify", metavar="MODEL", help="also run openai_verifier(MODEL) on every pair")
    ap.add_argument("--both-orders", action="store_true", help="also ask with A/B swapped; a hit needs both to say yes")
    args = ap.parse_args()

    from deja import openai_embedder, openai_verifier

    files = args.pairs or [p for p in (HERE / "pairs.jsonl", HERE / "hard_negatives.jsonl", HERE / "data" / "qqp.jsonl") if p.exists()]
    rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]
    unlabelled = sum(r["same"] is None for r in rows)
    if unlabelled:
        raise SystemExit(f"{unlabelled} pairs have \"same\": null; label them first")
    embed = openai_embedder(args.model)
    texts = [r["a"] for r in rows] + [r["b"] for r in rows]
    vecs = np.concatenate([embed(texts[i:i + 1000]) for i in range(0, len(texts), 1000)])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = (vecs[: len(rows)] * vecs[len(rows):]).sum(axis=1)
    same = np.array([r["same"] for r in rows])
    cats = np.array([r.get("category", "?") for r in rows])

    verdict = np.ones(len(rows), dtype=bool)
    if args.verify:
        from openai import OpenAI

        verify = openai_verifier(args.verify, client=OpenAI(max_retries=10), both_orders=False)  # ride out 429s; errors must not become "no"

        def check(r):
            t0 = time.perf_counter()
            v = verify(r["b"], r["a"])
            lat = time.perf_counter() - t0
            return v, verify(r["a"], r["b"]) if args.both_orders else v, lat

        with ThreadPoolExecutor(4) as pool:
            out = list(pool.map(check, rows))
        verdict = np.array([v for v, _, _ in out])
        lat = [s for _, _, s in out]
        if args.both_orders:
            swapped = np.array([v for _, v, _ in out])
            print(f"at threshold {args.threshold}:\n| order | false hits | reused |\n|---|---|---|")
            for name, v in (("cached first (default)", verdict), ("new first", swapped), ("both must say yes", verdict & swapped)):
                hit = (sims >= args.threshold) & v
                print(f"| {name} | {hit[~same].sum()}/{(~same).sum()} | {hit[same].sum()}/{same.sum()} |")
            print()
            verdict = verdict & swapped

    gated = np.where(verdict, sims, -1.0)  # verifier "no" = never a hit
    n_pos, n_neg = same.sum(), (~same).sum()
    print(f"files={[str(f) for f in files]}\nembedding={args.model}  verify={args.verify}  "
          f"pairs={len(rows)} ({n_pos} same, {n_neg} different)\n")
    if args.verify:
        print(f"verifier: {(verdict != same).sum()}/{len(rows)} verdicts disagree with the label, "
              f"p50 {np.median(lat) * 1000:.0f} ms, ${verify.dollars:.4f} for {verify.calls} calls\n")

    print("| threshold | hit rate on same (95% CI) | false hits | false-hit rate (95% CI) |\n|---|---|---|---|")
    for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        hit = gated >= t
        k_pos, k_neg = hit[same].sum(), hit[~same].sum()
        lo, hi = wilson(k_pos, n_pos)
        flo, fhi = wilson(k_neg, n_neg)
        print(f"| {t:.2f} | {k_pos / n_pos:.0%} ({lo:.0%}–{hi:.0%}) | {k_neg}/{n_neg} | "
              f"{k_neg / n_neg:.1%} ({flo:.1%}–{fhi:.1%}) |")
    t, recall, fhr = sweep(gated, same, args.target)
    print(f"\nlowest threshold with false-hit rate <= {args.target:.1%}: {t:.3f} "
          f"(hit rate {recall:.0%}, false-hit rate {fhr:.1%})")

    hit = gated >= args.threshold
    print(f"\nby category at threshold {args.threshold}:\n| category | same: hits | different: false hits |\n|---|---|---|")
    for c in sorted(set(cats)):
        m = cats == c
        pos, neg = m & same, m & ~same
        print(f"| {c} | {hit[pos].sum()}/{pos.sum()} | {hit[neg].sum()}/{neg.sum()} |")

    errors = defaultdict(list)
    for r, s, v, h in zip(rows, sims, verdict, hit):
        if h != r["same"]:
            why = "FALSE HIT" if h else ("verifier said no" if s >= args.threshold else "below threshold")
            errors[why].append(f"  {s:.3f} [{r.get('category', '?')}] {r['a']!r} / {r['b']!r}")
    for why, lines in errors.items():
        print(f"\n{why} ({len(lines)}):")
        print("\n".join(lines))


if __name__ == "__main__":
    main()

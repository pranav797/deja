"""Export near-matches logged from real traffic for labelling (Cache(log_candidates=True) / DEJA_LOG_CANDIDATES=1).

    python benchmark/export_candidates.py deja.db [--n 200] > benchmark/data/real.jsonl

Each line is a pair in the eval format with "same": null plus what Déjà did (sim, verdict).
Set "same" to true/false by hand (would the cached answer fully satisfy the new request?), then:

    uv run --env-file .env python benchmark/threshold.py benchmark/data/real.jsonl --verify gpt-4o-mini

Near-matches from other conversations come out with their `history` and cached `answer`; put those in a
separate file and evaluate them with benchmark/chat_eval.py instead.

Pairs are sampled at random, not just the hits, so missed reuse gets measured too.
The output contains real user prompts: keep it out of git (benchmark/data/ is ignored).
"""
import argparse
import json
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    rows = sqlite3.connect(args.db).execute(
        "SELECT DISTINCT cached_query, new_query, sim, verdict, history, answer FROM candidates ORDER BY random() LIMIT ?",
        (args.n,))
    for a, b, sim, verdict, history, answer in rows:
        row = {"a": a, "b": b, "same": None, "category": "real", "sim": round(sim, 3),
               "verdict": None if verdict is None else bool(verdict)}
        if answer is not None:  # matched across conversations: chat_eval.py format
            row.update(category="real-chat", history=json.loads(history) if history else [], answer=answer)
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

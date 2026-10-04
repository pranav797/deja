"""Sample real user-question pairs from Quora Question Pairs (GLUE validation split).

    uv run --with pyarrow python benchmark/fetch_qqp.py [--n 300]

Writes benchmark/data/qqp.jsonl (gitignored: Quora's terms may not allow redistribution).
QQP labels mean "duplicate question", which is close to but not the same as "one answer fits both",
and they carry some known label noise. For the default sample, qqp_relabel.json overrides 33 labels
with "would the cached answer to a fully satisfy someone asking b?" (category qqp-relabelled).
"""
import argparse
import json
import pathlib
import random
import urllib.request

URL = "https://huggingface.co/datasets/nyu-mll/glue/resolve/main/qqp/validation-00000-of-00001.parquet"
DATA = pathlib.Path(__file__).with_name("data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="pairs to sample, half duplicates and half not")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    DATA.mkdir(exist_ok=True)
    src = DATA / "qqp_validation.parquet"
    if not src.exists():
        urllib.request.urlretrieve(URL, src)
    rows = pq.read_table(src).to_pylist()
    rng = random.Random(args.seed)
    picked = []
    for label in (1, 0):
        picked += rng.sample([r for r in rows if r["label"] == label], args.n // 2)
    # Relabelled "same answer?" (see qqp_relabel.json); only valid for the default sample.
    relabel = json.loads((DATA.parent / "qqp_relabel.json").read_text())["changed"] if (args.n, args.seed) == (300, 0) else {}
    with open(DATA / "qqp.jsonl", "w", encoding="utf-8") as f:
        for r in picked:
            label = relabel.get(str(r["idx"]))
            f.write(json.dumps({"a": r["question1"], "b": r["question2"], "same": bool(r["label"]) if label is None else label,
                                "category": "qqp" if label is None else "qqp-relabelled", "idx": r["idx"],
                                "qqp_label": bool(r["label"])}) + "\n")
    print(f"wrote {len(picked)} pairs to {DATA / 'qqp.jsonl'} ({len(relabel)} relabelled)")


if __name__ == "__main__":
    main()

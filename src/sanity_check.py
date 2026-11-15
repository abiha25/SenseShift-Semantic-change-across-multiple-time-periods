"""
Sanity check for src/scoring.py using "dumb" systems whose scores we can predict.
If these numbers look as expected, the scoring setup is trustworthy.

Expected on Swedish dev with the official scorer of 8 Nov 2026 (commit 0e69238):
    one_cluster   B3 P 0.646, R 0.981, F1 0.756, 1b rho 0.000
    gold_labels   1.000 everywhere
    Subtask 2     all_1 0.236, all_0 0.361 (macro F1 over both classes per word)
(Before 8 Nov 2026: all_1 was 0.472 and the one-cluster P/R were shown swapped.)

Run from the repo root:
    python src/sanity_check.py            (uses data/SV)
    python src/sanity_check.py --lang SV --seed 42

Writes the dumb prediction files to results/sanity/ and a summary to
results/sanity/summary.json.
"""

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import score_1a, score_1b, score_2  # noqa: E402

warnings.filterwarnings("ignore")  # sklearn warns on words with no positive gold labels


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="SV")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    gold_dir = Path("data") / a.lang
    out_dir = Path("results") / "sanity"
    g1, g2 = gold_dir / "subtask1.jsonl", gold_dir / "subtask2.jsonl"
    usages = [(r["word"], r["sentence_id"]) for r in read_jsonl(g1)]
    gold1 = read_jsonl(g1)

    systems_1 = {
        "one_cluster": lambda i: [1],                       # everything together
        "each_own_cluster": lambda i: [i],                  # every usage alone
        "random_5_clusters": lambda i: [rng.randint(1, 5)],
        "gold_labels": lambda i: gold1[i]["label"],         # upper bound: must score 1.0
    }
    systems_2 = {
        "all_1": lambda i: 1,
        "all_0": lambda i: 0,
        "random": lambda i: rng.randint(0, 1),
    }

    summary = {"subtask1": {}, "subtask2": {}}

    print(f"Subtask 1 ({a.lang}):")
    print(f"  {'system':<20}{'B3 P':>8}{'B3 R':>8}{'B3 F1':>8}{'1b rho':>9}")
    for name, fn in systems_1.items():
        path = out_dir / f"{a.lang}_subtask1_{name}.jsonl"
        write_jsonl(path, [{"word": w, "sentence_id": s, "label": fn(i)}
                           for i, (w, s) in enumerate(usages)])
        r1a = score_1a(path, g1)["macro"]
        r1b = score_1b(path, g1)["rho"]
        summary["subtask1"][name] = {"bcubed": r1a, "spearman": r1b}
        print(f"  {name:<20}{r1a['precision']:>8.3f}{r1a['recall']:>8.3f}"
              f"{r1a['fcubed']:>8.3f}{r1b:>9.3f}")

    print(f"\nSubtask 2 ({a.lang}):")
    for name, fn in systems_2.items():
        path = out_dir / f"{a.lang}_subtask2_{name}.jsonl"
        write_jsonl(path, [{"word": w, "sentence_id": s, "label": fn(i)}
                           for i, (w, s) in enumerate(usages)])
        r2 = score_2(path, g2)["macro"]
        summary["subtask2"][name] = r2
        print(f"  {name:<20} macro F1 = {r2:.3f}")

    # Gold labels for Subtask 2 must also score a perfect 1.0 (on words with positives).
    gold2_path = out_dir / f"{a.lang}_subtask2_gold_labels.jsonl"
    write_jsonl(gold2_path, read_jsonl(g2))
    per_word = score_2(gold2_path, g2)["per_word"]
    summary["subtask2"]["gold_labels_per_word"] = per_word
    print(f"  {'gold_labels':<20} per word = {per_word}")

    # NaN (e.g. Spearman for the one-cluster baseline) is not valid JSON,
    # so write it as null.
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, float) and x != x:  # NaN check
            return None
        return x

    (out_dir / "summary.json").write_text(
        json.dumps(clean(summary), indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    print(f"\nSaved to {out_dir}/summary.json")


if __name__ == "__main__":
    main()
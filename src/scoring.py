"""
Shared scoring module for the SenseShift team (SemEval-2027 Task 3).

Everyone uses THIS file to score predictions. Do not write your own metric code.
It reuses the official organiser code (evaluation.py / evaluation_utils.py in the
repo root) and only adds three things the official script does not give us:

  1. Scoring a single JSONL file (the official script only accepts a ZIP).
  2. Per-word scores (the official script only returns the average).
  3. Scoring a subset of words (needed for leave-one-word-out validation, and
     for the organisers' example submission, which only covers 2 words).

Usage from Python:
    from src.scoring import score_1a, score_1b, score_2, validate_submission
    res = score_1a("results/my_preds_subtask1.jsonl", "data/SV/subtask1.jsonl")
    print(res["macro"]["fcubed"])

Usage from the command line (run from the repo root):
    python src/scoring.py 1  results/my_preds_subtask1.jsonl data/SV/subtask1.jsonl
    python src/scoring.py 2  results/my_preds_subtask2.jsonl data/SV/subtask2.jsonl
    python src/scoring.py zip my_submission.zip data
    python src/scoring.py make-zip my_submission.zip SV results/SV_subtask1.jsonl results/SV_subtask2.jsonl

Add  --only-predicted-words  to score only the words that appear in the
prediction file (instead of requiring every gold word).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

# Make the official scripts (in the repo root) importable from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import (  # noqa: E402  (official code, never edit it)
    compute_bcubed,
    compute_f1_score,
    compute_spearman,
    filter_jsd_by_min_instances,
)
from evaluation_utils import (  # noqa: E402
    calculate_JSD_distances,
    load_subtask1_labels,
    load_subtask2_labels,
)

ALLOWED_LANGUAGES = {"EN", "SV", "RU", "NL", "IT", "ES"}
MIN_INSTANCES_1B = 10  # same value the official script uses


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _words_in(path: Path) -> set[str]:
    return {row["word"] for row in _read_jsonl(path)}


def _prepare_gold(gold_file, pred_file, words, only_predicted_words, tmpdir):
    """
    Returns a path to the gold file to use.
    If a subset of words is requested, writes a filtered copy of gold to tmpdir,
    so the official loaders only expect those words.
    """
    if only_predicted_words:
        words = _words_in(pred_file) if words is None else set(words) & _words_in(pred_file)
    if words is None:
        return Path(gold_file)

    words = set(words)
    rows = [r for r in _read_jsonl(gold_file) if r["word"] in words]
    if not rows:
        raise ValueError(f"None of the requested words are in the gold file: {sorted(words)}")
    out = Path(tmpdir) / "gold_filtered.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def _split_by_word(labels: dict[str, object]) -> dict[str, dict[str, object]]:
    """Official labels are keyed 'word;sentence_id'. Group them per word."""
    by_word: dict[str, dict[str, object]] = {}
    for key, value in labels.items():
        word = key.split(";", 1)[0]
        by_word.setdefault(word, {})[key] = value
    return by_word


# --------------------------------------------------------------------------
# Subtask 1a: BCubed F1
# --------------------------------------------------------------------------
def score_1a(pred_file, gold_file, words=None, only_predicted_words=False) -> dict:
    """
    BCubed precision / recall / F1 per word, plus the macro-average
    (exactly what the official script reports for 1a).
    """
    with tempfile.TemporaryDirectory() as tmp:
        gold_path = _prepare_gold(gold_file, pred_file, words, only_predicted_words, tmp)
        _, _, pred, gold = load_subtask1_labels(Path(pred_file), gold_path)

    gold_by_word = _split_by_word(gold)
    per_word = {}
    for word, gold_w in sorted(gold_by_word.items()):
        pred_w = {k: pred[k] for k in gold_w}
        # Official function on a single word = that word's score.
        per_word[word] = _fix_names(compute_bcubed(gold_w, pred_w, str(pred_file)))

    macro = _fix_names(compute_bcubed(gold, pred, str(pred_file)))  # official macro-average
    return {"per_word": per_word, "macro": macro}


def _fix_names(scores: dict) -> dict:
    """
    The official code calls bcubed.precision(gold, pred), but the bcubed library
    expects (pred, gold). So the official "precision" is really BCubed RECALL
    and vice versa. F1 is symmetric, so F1 (the ranked metric) is unaffected.
    We return the correctly named values; Codabench will show them swapped.
    """
    return {
        "precision": scores["recall"],
        "recall": scores["precision"],
        "fcubed": scores["fcubed"],
    }


# --------------------------------------------------------------------------
# Subtask 1b: JSD between consecutive periods, Spearman over all pairs
# --------------------------------------------------------------------------
def score_1b(pred_file, gold_file, words=None, only_predicted_words=False) -> dict:
    """
    Spearman rho between predicted and gold JSD over all (word, period-pair)
    items. Pairs where either period has < 10 gold usages are dropped,
    like the official script. Also returns the JSD values so you can inspect them.
    """
    with tempfile.TemporaryDirectory() as tmp:
        gold_path = _prepare_gold(gold_file, pred_file, words, only_predicted_words, tmp)
        pred_counts, gold_counts, _, _ = load_subtask1_labels(Path(pred_file), gold_path)

    gold_jsd = calculate_JSD_distances(gold_counts)
    pred_jsd = calculate_JSD_distances(pred_counts)
    gold_jsd = filter_jsd_by_min_instances(gold_jsd, gold_counts, min_instances=MIN_INSTANCES_1B)

    result = compute_spearman(gold_jsd, pred_jsd, str(pred_file))
    rho = result["rho"]
    note = None
    if math.isnan(rho):
        note = ("Spearman is NaN: the predicted (or gold) JSD values are all identical, "
                "e.g. every usage in one cluster. Treat this as 'no signal' (≈0). "
                "Note: the official script would fail on this when writing JSON.")
    return {
        "rho": rho,
        "pvalue": result["pvalue"],
        "n_pairs": len(gold_jsd),
        "gold_jsd": {k: float(gold_jsd[k]) for k in gold_jsd},
        "pred_jsd": {k: float(pred_jsd[k]) for k in gold_jsd},
        "note": note,
    }


# --------------------------------------------------------------------------
# Subtask 2: F1 per word, macro-averaged
# --------------------------------------------------------------------------
def score_2(pred_file, gold_file, words=None, only_predicted_words=False) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        gold_path = _prepare_gold(gold_file, pred_file, words, only_predicted_words, tmp)
        pred, gold = load_subtask2_labels(Path(pred_file), gold_path)

    per_word = {}
    for word, gold_w in sorted(_split_by_word(gold).items()):
        pred_w = {k: pred[k] for k in gold_w}
        per_word[word] = compute_f1_score(gold_w, pred_w, str(pred_file))

    macro = compute_f1_score(gold, pred, str(pred_file))  # official macro-average
    return {"per_word": per_word, "macro": macro}


# --------------------------------------------------------------------------
# Submission ZIP helpers
# --------------------------------------------------------------------------
def make_submission_zip(zip_path, language, subtask1_file=None, subtask2_file=None) -> Path:
    """Package prediction files with the exact names Codabench expects."""
    language = language.upper()
    if language not in ALLOWED_LANGUAGES:
        raise ValueError(f"Unknown language {language}; expected one of {sorted(ALLOWED_LANGUAGES)}")
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        if subtask1_file:
            z.write(subtask1_file, arcname=f"{language}_subtask1.jsonl")
        if subtask2_file:
            z.write(subtask2_file, arcname=f"{language}_subtask2.jsonl")
    return zip_path


def validate_submission(zip_path, gold_folder) -> list[str]:
    """
    Checks a submission ZIP BEFORE uploading it. Returns a list of problems
    (empty list = valid). Uses the official loaders, so anything they would
    reject is reported here.
    """
    problems: list[str] = []
    zip_path = Path(zip_path)
    if not zipfile.is_zipfile(zip_path):
        return [f"{zip_path} is not a ZIP file."]

    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as tmp:
        names = [n for n in z.namelist() if not n.endswith("/")]
        for name in names:
            if "/" in name:
                problems.append(f"'{name}' is inside a folder. Files must be at the top level of the ZIP.")
        good = [n for n in names if n.endswith(("_subtask1.jsonl", "_subtask2.jsonl"))]
        if not good:
            problems.append("No files named like 'SV_subtask1.jsonl' or 'SV_subtask2.jsonl' found.")
        z.extractall(tmp)

        for name in good:
            base = Path(name).name
            lang, subtask = base.split("_", 1)
            if lang not in ALLOWED_LANGUAGES:
                problems.append(f"{base}: unknown language code '{lang}' (it would be skipped).")
                continue
            gold = Path(gold_folder) / lang / subtask
            if not gold.exists():
                problems.append(f"{base}: no gold file at {gold} to check against.")
                continue
            loader = load_subtask1_labels if subtask == "subtask1.jsonl" else load_subtask2_labels
            try:
                loader(Path(tmp) / name, gold)
            except Exception as exc:  # official loader error = invalid submission
                problems.append(f"{base}: {exc}")
    return problems


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def _clean_nan(x):
    """NaN is not valid JSON; show it as null."""
    if isinstance(x, dict):
        return {k: _clean_nan(v) for k, v in x.items()}
    if isinstance(x, float) and x != x:
        return None
    return x


def _print(obj):
    print(json.dumps(_clean_nan(obj), indent=2, ensure_ascii=False, allow_nan=False, default=str))


def main():
    p = argparse.ArgumentParser(description="SenseShift scoring (wraps the official scorer).")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("1", "2"):
        s = sub.add_parser(name, help=f"Score Subtask {name} predictions")
        s.add_argument("pred_file")
        s.add_argument("gold_file")
        s.add_argument("--only-predicted-words", action="store_true")
        s.add_argument("--show-jsd", action="store_true", help="(Subtask 1) print JSD values too")

    z = sub.add_parser("zip", help="Validate a submission ZIP")
    z.add_argument("zip_path")
    z.add_argument("gold_folder")

    m = sub.add_parser("make-zip", help="Create a submission ZIP")
    m.add_argument("zip_path")
    m.add_argument("language")
    m.add_argument("subtask1_file", nargs="?")
    m.add_argument("subtask2_file", nargs="?")

    a = p.parse_args()

    if a.cmd == "1":
        r1a = score_1a(a.pred_file, a.gold_file, only_predicted_words=a.only_predicted_words)
        r1b = score_1b(a.pred_file, a.gold_file, only_predicted_words=a.only_predicted_words)
        if not a.show_jsd:
            r1b = {k: v for k, v in r1b.items() if k not in ("gold_jsd", "pred_jsd")}
        _print({"subtask1a": r1a, "subtask1b": r1b})
    elif a.cmd == "2":
        _print({"subtask2": score_2(a.pred_file, a.gold_file, only_predicted_words=a.only_predicted_words)})
    elif a.cmd == "zip":
        problems = validate_submission(a.zip_path, a.gold_folder)
        print("VALID ✔" if not problems else "INVALID ✘")
        for pr in problems:
            print(" -", pr)
        sys.exit(1 if problems else 0)
    elif a.cmd == "make-zip":
        print("Wrote", make_submission_zip(a.zip_path, a.language, a.subtask1_file, a.subtask2_file))


if __name__ == "__main__":
    main()
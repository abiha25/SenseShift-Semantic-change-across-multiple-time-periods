"""
Data splits for SemEval-2027 Task 3 (team SenseShift).   Owner: Person 3.

WHY THIS FILE EXISTS
    The task releases no training set: the only labelled data are the 10
    Swedish development words (1,983 usages). The test words (released
    10 Jan 2027) come without labels. So we build the splits ourselves, and
    we split BY TARGET WORD, never by usage:

      * The fine-tuned encoder learns from same-sense / different-sense
        usage pairs. If it saw usages of the word it is tested on, the test
        score would be inflated (leakage).
      * Repeated sentences (5 pairs) always belong to the same word, so a
        word-level split also keeps every copy on the same side.

STRATEGY: nested leave-one-word-out (LOWO), one fold per word
    fold i:  test       = word i                        (scored once)
             validation = the next word in sorted order  (early stopping and
                                                          checkpoint selection)
             train      = the remaining 8 words          (fine-tuning, tuning)
    Every word is the test word exactly once and the validation word exactly
    once. The rotation is deterministic, so no randomness is involved, and
    the folds are saved to a JSON file that everyone loads.

    The 10 test-word predictions are written into ONE file and scored once,
    so Subtask 1b uses all 60 scored period pairs (as in Assignment 1).

Usage
    python -m src.splits                         # writes the file from the config
    python -m src.splits --config configs/default.yaml

    from src.splits import load_splits, fold_frames
    splits = load_splits('data/splits/SV_splits.json', df)   # also verifies the data
    for fold in splits['folds']:
        train, val, test = fold_frames(df, fold)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STRATEGY = 'nested_lowo_rotating_validation'
DATA_FILES = ['usages.jsonl', 'subtask1.jsonl', 'subtask2.jsonl', 'definitions.jsonl']


# ---------------------------------------------------------------------------
# Building the folds
# ---------------------------------------------------------------------------

def make_nested_lowo(words, validation_offset=1):
    """
    One fold per word. With words sorted, fold i tests word i and validates
    on word (i + validation_offset) mod n. Needs at least 3 words.
    """
    words = sorted(set(words))
    n = len(words)
    if n < 3:
        raise ValueError(f'Need at least 3 words for train/validation/test, got {n}')
    if validation_offset % n == 0:
        raise ValueError('validation_offset must not make the validation word the test word')
    folds = []
    for i, test_word in enumerate(words):
        val_word = words[(i + validation_offset) % n]
        train = [w for w in words if w not in (test_word, val_word)]
        folds.append({'fold': i, 'test': [test_word], 'validation': [val_word], 'train': train})
    return folds


def check_no_leakage(splits):
    """
    Raise if any word appears in two parts of one fold, if a fold does not
    cover all words, or if a word is not tested exactly once.
    Returns True so it can be used in an assert.
    """
    words = set(splits['words'])
    tested = []
    for f in splits['folds']:
        tr, va, te = set(f['train']), set(f['validation']), set(f['test'])
        if tr & va or tr & te or va & te:
            raise ValueError(f"Fold {f['fold']}: a word is in two parts: "
                             f"{sorted((tr & va) | (tr & te) | (va & te))}")
        if tr | va | te != words:
            raise ValueError(f"Fold {f['fold']}: parts do not cover all words")
        tested += f['test']
    if sorted(tested) != sorted(words):
        raise ValueError('Every word must be the test word exactly once')
    return True


# ---------------------------------------------------------------------------
# Data fingerprint: proves the split was made on exactly these files
# ---------------------------------------------------------------------------

def data_fingerprint(data_dir, language='SV'):
    """SHA-256 over the raw data files (in a fixed order). If anyone's copy
    of the data differs, load_splits() will refuse to use the split."""
    h = hashlib.sha256()
    lang_dir = Path(data_dir) / language
    for name in DATA_FILES:
        path = lang_dir / name
        if path.exists():
            h.update(name.encode())
            h.update(path.read_bytes().replace(b'\r\n', b'\n'))  # same hash on Windows and Colab
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------

def build_splits(df, data_dir, language='SV', seed=42, validation_offset=1):
    """Folds + everything needed to describe them in the report (Section 7)."""
    words = sorted(df['word'].unique())
    counts = df.groupby('word').size().to_dict()
    folds = make_nested_lowo(words, validation_offset)
    for f in folds:
        f['n_usages'] = {part: int(sum(counts[w] for w in f[part]))
                         for part in ('train', 'validation', 'test')}
    splits = {
        'strategy': STRATEGY,
        'language': language,
        'seed': seed,  # recorded for completeness; the rotation itself is not random
        'validation_offset': validation_offset,
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'data_fingerprint': data_fingerprint(data_dir, language),
        'n_usages_total': int(len(df)),
        'words': words,
        'usages_per_word': {w: int(counts[w]) for w in words},
        'folds': folds,
    }
    check_no_leakage(splits)
    return splits


def save_splits(splits, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding='utf-8')
    return path


def load_splits(path, df=None, data_dir=None):
    """
    Load a saved split and verify it:
      * no word in two parts of a fold (check_no_leakage)
      * if df is given: the split's words are exactly the words in df
      * if data_dir is given: the data files are byte-identical to the ones
        the split was made from (fingerprint)
    """
    splits = json.loads(Path(path).read_text(encoding='utf-8'))
    check_no_leakage(splits)
    if df is not None:
        have = sorted(df['word'].unique())
        if have != splits['words']:
            raise ValueError(f'Split words {splits["words"]} != data words {have}')
    if data_dir is not None:
        fp = data_fingerprint(data_dir, splits['language'])
        if fp != splits['data_fingerprint']:
            raise ValueError('The data files differ from the ones this split was made from. '
                             'Re-download the official data or tell the team before re-splitting.')
    return splits


def fold_frames(df, fold):
    """(train_df, validation_df, test_df) for one fold."""
    return (df[df['word'].isin(fold['train'])],
            df[df['word'].isin(fold['validation'])],
            df[df['word'].isin(fold['test'])])


def describe(splits):
    """Short text summary (used for the report and the command line)."""
    lines = [f"{splits['strategy']}: {len(splits['folds'])} folds, "
             f"{len(splits['words'])} words, {splits['n_usages_total']} usages"]
    for f in splits['folds']:
        n = f['n_usages']
        lines.append(f"  fold {f['fold']}: test={f['test'][0]:<10} ({n['test']:>3})  "
                     f"val={f['validation'][0]:<10} ({n['validation']:>3})  "
                     f"train=8 words ({n['train']})")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    from data_loader import load_language
    from src.config import load_config

    ap = argparse.ArgumentParser(description='Write the shared word-level split file.')
    ap.add_argument('--config', default=str(REPO_ROOT / 'configs' / 'default.yaml'))
    ap.add_argument('--force', action='store_true', help='overwrite an existing split file')
    a = ap.parse_args()

    cfg = load_config(a.config)
    data_dir, lang = cfg['data']['data_dir'], cfg['data']['language']
    out = Path(cfg['split']['file'])
    if out.exists() and not a.force:
        splits = load_splits(out, load_language(data_dir, lang), data_dir)
        print(f'{out} already exists and matches the data (use --force to rebuild).')
    else:
        df = load_language(data_dir, lang)
        splits = build_splits(df, data_dir, lang, seed=cfg['seed'],
                              validation_offset=cfg['split']['validation_offset'])
        save_splits(splits, out)
        print(f'Wrote {out}')
    print(describe(splits))


if __name__ == '__main__':
    main()

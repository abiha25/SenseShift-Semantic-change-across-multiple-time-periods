"""
Shared data loader for SemEval-2027 Task 3 (team SenseShift).

Everyone imports this file so that all
baselines read the data in exactly the same way.

Usage
-----
    from data_loader import load_language, check_integrity, leave_one_word_out

    df = load_language('/content/drive/MyDrive/SemEval2027/dev_data', 'SV')
    report = check_integrity(df)
    for train_words, test_word in leave_one_word_out(df):
        ...

One row of the returned DataFrame = one usage (one sentence containing
the target word). The key that identifies a usage is (word, sentence_id),
the same key the official evaluation scripts use.
"""

import json
import os
import re

import pandas as pd

KEY = ['word', 'sentence_id']


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path):
    """Read a .jsonl file (one JSON object per line) into a DataFrame."""
    rows = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def period_start_year(period_label):
    """'1880-1883' -> 1880. Same ordering rule as the official scorer."""
    match = re.search(r'\d{4}', str(period_label))
    return int(match.group()) if match else float('inf')


def period_end_year(period_label):
    """'1880-1883' -> 1883 (last four-digit year in the label)."""
    years = re.findall(r'\d{4}', str(period_label))
    return int(years[-1]) if years else float('inf')


def repair_offsets(df):
    """
    Some gold offsets point at the wrong token (e.g. skär usages where
    text[start:end] is 'hon' or 'trä'). For each usage this adds:

        start_fixed, end_fixed, target_fixed   offsets/span to use in models
        offset_status
            'ok'             span already starts like the target word
            'repaired'       span replaced by the closest token that starts
                             with the word's stem (first 4 letters)
            'kept_original'  no such token found; the span is probably an
                             irregular form (e.g. suga -> sög) and is kept

    The original start/end/target columns are never changed.
    """
    starts, ends, targets, status = [], [], [], []
    for text, word, s, e, tgt in zip(df['text'], df['word'], df['start'], df['end'], df['target']):
        word_l = word.lower()
        if tgt.lower().startswith(word_l[:3]):
            starts.append(s); ends.append(e); targets.append(tgt); status.append('ok')
            continue
        stem = re.escape(word_l[:4])
        matches = [m for m in re.finditer(r'\b' + stem + r'\w*', text, flags=re.IGNORECASE)]
        if matches:
            best = min(matches, key=lambda m: abs(m.start() - s))
            starts.append(best.start()); ends.append(best.end())
            targets.append(best.group()); status.append('repaired')
        else:
            starts.append(s); ends.append(e); targets.append(tgt); status.append('kept_original')
    df['start_fixed'] = starts
    df['end_fixed'] = ends
    df['target_fixed'] = targets
    df['offset_status'] = status
    return df


def ordered_periods(df):
    """All period labels in the data, sorted by start year."""
    return sorted(df['period_label'].unique(), key=period_start_year)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_language(data_dir, language='SV'):
    """
    Load all files for one language and merge them into one table.

    Works with and without gold labels: the evaluation data released in
    January will have no subtask1/subtask2 label files, and then the
    label columns are simply left out.

    Returns a DataFrame with one row per usage and these columns:
        word, sentence_id, text, pos, period_label, start, end, year
            straight from usages.jsonl
        target          the exact characters text[start:end]
        period_index    0 for the earliest period, 1 for the next, ...
        definition      the Subtask 2 sense definition for this word
        senses          (gold only) list of sense ids from subtask1.jsonl
        n_senses        (gold only) how many senses this usage has
        target_sense    (gold only) 1/0 from subtask2.jsonl
    """
    lang_dir = os.path.join(data_dir, language)

    usages = _read_jsonl(os.path.join(lang_dir, 'usages.jsonl'))

    # Some numeric fields can arrive as text (e.g. "1883" instead of 1883).
    # Convert them to numbers and remember how many needed fixing, so the
    # Dataset section can report it. Values that cannot be read become NaN.
    type_fixes = {}
    for col in ['start', 'end', 'year']:
        raw = usages[col]
        not_int = raw.apply(lambda v: not (isinstance(v, int) and not isinstance(v, bool)))
        type_fixes[col] = int(not_int.sum())
        usages[col] = pd.to_numeric(raw, errors='coerce')
        if usages[col].notna().all():
            usages[col] = usages[col].astype(int)

    usages['target'] = [
        text[int(s):int(e)] if pd.notna(s) and pd.notna(e) else ''
        for text, s, e in zip(usages['text'], usages['start'], usages['end'])
    ]

    usages = repair_offsets(usages)

    periods = sorted(usages['period_label'].unique(), key=period_start_year)
    usages['period_index'] = usages['period_label'].map(
        {p: i for i, p in enumerate(periods)}
    )

    # Subtask 2 definitions (one per word).
    def_path = os.path.join(lang_dir, 'definitions.jsonl')
    if os.path.exists(def_path):
        definitions = _read_jsonl(def_path)
        usages = usages.merge(definitions, on='word', how='left')

    # Gold labels for Subtask 1 (sense lists).
    s1_path = os.path.join(lang_dir, 'subtask1.jsonl')
    if os.path.exists(s1_path):
        s1 = _read_jsonl(s1_path).rename(
            columns={'label': 'senses', 'period_label': 'period_label_s1'}
        )
        usages = usages.merge(s1, on=KEY, how='left')
        usages['n_senses'] = usages['senses'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

    # Gold labels for Subtask 2 (0/1).
    s2_path = os.path.join(lang_dir, 'subtask2.jsonl')
    if os.path.exists(s2_path):
        s2 = _read_jsonl(s2_path).rename(columns={'label': 'target_sense'})
        usages = usages.merge(s2, on=KEY, how='left')

    usages['language'] = language
    usages = usages.sort_values(['word', 'period_index', 'sentence_id']).reset_index(drop=True)
    usages.attrs['type_fixes'] = type_fixes
    return usages


# ---------------------------------------------------------------------------
# Integrity checks (feeds the Dataset section of the report)
# ---------------------------------------------------------------------------

def check_integrity(df, verbose=True):
    """
    Run all sanity checks and return a dict of problem counts.
    Each value is the number of usages with that problem (0 = clean).
    The matching rows are stored under the same name + '_rows'.
    """
    report = {}

    def record(name, mask):
        report[name] = int(mask.sum())
        report[name + '_rows'] = df[mask]

    # 1. The (word, sentence_id) key must be unique.
    record('duplicate_keys', df.duplicated(KEY, keep=False))

    # 2. Same sentence text appearing twice for the same word.
    record('duplicate_texts', df.duplicated(['word', 'text'], keep=False))

    # 3. Empty or whitespace-only sentences.
    record('empty_text', df['text'].fillna('').str.strip() == '')

    # 4. Offsets must lie inside the sentence and be non-empty.
    text_len = df['text'].fillna('').str.len()
    record('offset_out_of_range',
           (df['start'] < 0) | (df['end'] > text_len) | (df['start'] >= df['end']))

    # 5. sentence_id ends in _start_end; does that agree with start/end?
    id_offsets = df['sentence_id'].str.extract(r'_(\d+)_(\d+)$').astype(float)
    record('id_offset_mismatch',
           (id_offsets[0] != df['start']) | (id_offsets[1] != df['end']))

    # 6. Does the marked span look like the target word?
    #    Swedish inflects words (kvinna -> kvinnor, kvinnans), so we only
    #    flag spans that do not even share the first 3 letters.
    record('target_not_like_word',
           ~df.apply(lambda r: r['target'].lower().startswith(r['word'].lower()[:3]), axis=1))

    # 7. Is the year inside its period label?
    start_y = df['period_label'].map(period_start_year)
    end_y = df['period_label'].map(period_end_year)
    record('year_outside_period', (df['year'] < start_y) | (df['year'] > end_y))

    # 8. Gold-file consistency (only if gold labels were loaded).
    if 'senses' in df.columns:
        record('missing_subtask1_label', df['senses'].isna())
        record('period_mismatch_subtask1',
               df['period_label_s1'].notna() & (df['period_label_s1'] != df['period_label']))
    if 'target_sense' in df.columns:
        record('missing_subtask2_label', df['target_sense'].isna())

    # 8b. Duplicate sentences whose copies got different gold labels.
    if 'senses' in df.columns or 'target_sense' in df.columns:
        dup = df[df.duplicated(['word', 'text'], keep=False)].copy()
        label_cols = [c for c in ['senses', 'target_sense'] if c in dup.columns]
        dup['_labels'] = dup[label_cols].astype(str).agg('|'.join, axis=1)
        conflict_groups = dup.groupby(['word', 'text'])['_labels'].nunique()
        conflict_groups = conflict_groups[conflict_groups > 1].index
        mask = df.set_index(['word', 'text']).index.isin(conflict_groups)
        record('duplicate_label_conflict', pd.Series(mask, index=df.index))

    # 9. Numeric fields that could not be read as numbers at all.
    record('unreadable_numbers', df[['start', 'end', 'year']].isna().any(axis=1))

    if verbose:
        print(f'{len(df)} usages, {df["word"].nunique()} words, '
              f'{df["period_label"].nunique()} periods')
        fixes = df.attrs.get('type_fixes', {})
        if any(fixes.values()):
            print(f'  note: values stored as text and converted to numbers: {fixes}')
        if 'offset_status' in df.columns:
            print(f'  note: offset repair: {df["offset_status"].value_counts().to_dict()}')
        for name, value in report.items():
            if not name.endswith('_rows'):
                flag = 'OK ' if value == 0 else '!! '
                print(f'  {flag}{name}: {value}')
    return report


# ---------------------------------------------------------------------------
# Train / dev handling
# ---------------------------------------------------------------------------

def leave_one_word_out(df):
    """
    All labelled data is the 10 dev words, so there is no separate
    training set. Use this to tune on 9 words and test on the 10th.

        for train_words, test_word in leave_one_word_out(df):
            train = df[df.word.isin(train_words)]
            test  = df[df.word == test_word]
    """
    words = sorted(df['word'].unique())
    for test_word in words:
        yield [w for w in words if w != test_word], test_word


# ---------------------------------------------------------------------------
# Writing predictions in the official submission format
# ---------------------------------------------------------------------------

def write_subtask1(df, pred_column, out_path):
    """
    Write {"word", "sentence_id", "label": [int, ...]} per line.
    pred_column holds an int or a list of ints per usage.
    The file must be named <LANG>_subtask1.jsonl inside the submission zip.
    """
    with open(out_path, 'w', encoding='utf-8') as fh:
        for _, r in df.iterrows():
            label = r[pred_column]
            label = [int(x) for x in label] if isinstance(label, (list, tuple)) else [int(label)]
            fh.write(json.dumps({'word': r['word'], 'sentence_id': r['sentence_id'],
                                 'label': label}, ensure_ascii=False) + '\n')


def write_subtask2(df, pred_column, out_path):
    """Write {"word", "sentence_id", "label": 0 or 1} per line (<LANG>_subtask2.jsonl)."""
    with open(out_path, 'w', encoding='utf-8') as fh:
        for _, r in df.iterrows():
            fh.write(json.dumps({'word': r['word'], 'sentence_id': r['sentence_id'],
                                 'label': int(r[pred_column])}, ensure_ascii=False) + '\n')


# ===========================================================================
# ASSIGNMENT 2 ADDITIONS (data pipeline)
#
# Everything above this line is unchanged from Assignment 1, so Baseline A
# and Baseline B load the data exactly as before. The functions below add:
#
#   load_and_prepare()      one call: load + model inputs + duplicate flags
#   add_model_inputs()      config-driven preprocessing that keeps the target
#                           offsets valid (window, spelling, lower-casing)
#   flag_duplicates()       marks repeated sentences and conflicting labels
#   cap_usages_per_period() optional, seeded sampling cap (training only)
#   group_usages()          {word: {period: usages}} in time order
#
# Splits live in src/splits.py, settings in configs/default.yaml.
# ===========================================================================

# Default preprocessing. The real values come from configs/default.yaml
# (section "preprocessing"); these defaults mean "change nothing".
DEFAULT_PREPROCESSING = {
    'spelling_normalisation': False,   # pre-1906 Swedish spelling -> modern (EDA F4)
    'lowercase': False,                # separate switch (A1 mixed it with spelling)
    'context_window': None,            # k words each side of the target; None = whole sentence
}

# The same five pre-1906 rules as src/features.py (EDA F4), but matched
# case-insensitively and with the original capitalisation kept, so that
# spelling normalisation no longer forces lower-casing. On lower-cased text
# they give exactly the same result as src/features.normalise().
SWEDISH_SPELLING_RULES = [
    (re.compile(r'\bhv', re.IGNORECASE), 'v'),     # hvad  -> vad
    (re.compile(r'fv', re.IGNORECASE), 'v'),       # öfver -> över
    (re.compile(r'dt\b', re.IGNORECASE), 'tt'),    # godt  -> gott
    (re.compile(r'\baf\b', re.IGNORECASE), 'av'),  # af    -> av
    (re.compile(r'\bqv', re.IGNORECASE), 'kv'),    # qvinna -> kvinna
]
SPELLING_RULES_BY_LANGUAGE = {'SV': SWEDISH_SPELLING_RULES}

# A token "counts" for the context window if it contains a letter or digit,
# so punctuation such as "," or "—" does not use up the window (as in A1).
_TOKEN_RE = re.compile(r'\S+')
_WORDLIKE_RE = re.compile(r'\w')


def _match_case(original, replacement):
    """'Hv' -> 'V', 'HV' -> 'V', 'hv' -> 'v': keep the capitalisation."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _move(pos, m_start, m_end, new_len):
    """Where does character position `pos` end up after text[m_start:m_end]
    is replaced by a string of length new_len?"""
    if pos <= m_start:
        return pos
    if pos >= m_end:
        return pos + new_len - (m_end - m_start)
    return m_start + min(pos - m_start, new_len)


def normalise_spelling(text, start, end, language='SV'):
    """
    Rewrite old spellings and return (new_text, new_start, new_end).

    Rules are applied to the WHOLE sentence (not to left/target/right chunks
    separately, as A1 did), so a word boundary is never invented at the edge
    of the target. Every replacement shifts the offsets by its change in
    length, so new_text[new_start:new_end] is still the target.
    Languages without rules are returned unchanged.
    """
    for pattern, repl in SPELLING_RULES_BY_LANGUAGE.get(language, []):
        pieces, last = [], 0
        shift_start = shift_end = 0          # how far the offsets have moved so far
        for m in pattern.finditer(text):     # positions refer to the text BEFORE this rule
            new = _match_case(m.group(), repl)
            pieces += [text[last:m.start()], new]
            last = m.end()
            shift_start += _move(start, m.start(), m.end(), len(new)) - start
            shift_end += _move(end, m.start(), m.end(), len(new)) - end
        pieces.append(text[last:])
        text, start, end = ''.join(pieces), start + shift_start, end + shift_end
    return text, start, end


def lowercase_keep_offsets(text, start, end):
    """
    Lower-case without moving any character. str.lower() can change the
    length for a few characters (e.g. 'İ' -> 'i̇'); those are left as they are
    so that the offsets stay exact in every language.
    """
    out = ''.join(c.lower() if len(c.lower()) == 1 else c for c in text)
    return out, start, end


def window_around_target(text, start, end, k):
    """
    Keep at most k word-like tokens on each side of the target and return
    (cut_text, new_start, new_end). k=None keeps the whole sentence.

    Why (EDA F3): historical sentences are ~40% longer than modern ones, so
    a whole-sentence input gives the two eras different amounts of context.
    """
    if k is None:
        return text, start, end
    left = [m for m in _TOKEN_RE.finditer(text, 0, start) if _WORDLIKE_RE.search(m.group())]
    right = [m for m in _TOKEN_RE.finditer(text, end) if _WORDLIKE_RE.search(m.group())]
    cut_from = left[-k].start() if k > 0 and len(left) > k else (start if k == 0 else 0)
    if k == 0:
        cut_to = end
    else:
        cut_to = right[k - 1].end() if len(right) > k else len(text)
    return text[cut_from:cut_to], start - cut_from, end - cut_from


def prepare_usage(text, start, end, language='SV', spelling_normalisation=False,
                  lowercase=False, context_window=None):
    """
    The single place where one usage is turned into model input.
    Order: window (on the original tokens) -> spelling -> lower-case.
    Returns (text, start, end) with text[start:end] = the (normalised) target.
    """
    text, start, end = window_around_target(text, int(start), int(end), context_window)
    if spelling_normalisation:
        text, start, end = normalise_spelling(text, start, end, language)
    if lowercase:
        text, start, end = lowercase_keep_offsets(text, start, end)
    return text, start, end


def add_model_inputs(df, preprocessing=None):
    """
    Add the columns every model should read:
        model_text, model_start, model_end, model_target
    built from the REPAIRED offsets (start_fixed / end_fixed) with the
    preprocessing settings from the config. The original columns are kept.
    """
    settings = dict(DEFAULT_PREPROCESSING)
    settings.update({k: v for k, v in (preprocessing or {}).items() if k in DEFAULT_PREPROCESSING})
    unknown = set(preprocessing or {}) - set(DEFAULT_PREPROCESSING) - {'max_usages_per_period'}
    if unknown:
        raise ValueError(f'Unknown preprocessing setting(s): {sorted(unknown)}')
    df = df.copy()
    rows = [prepare_usage(t, s, e, lang, **settings)
            for t, s, e, lang in zip(df['text'], df['start_fixed'], df['end_fixed'], df['language'])]
    df['model_text'] = [r[0] for r in rows]
    df['model_start'] = [r[1] for r in rows]
    df['model_end'] = [r[2] for r in rows]
    df['model_target'] = [t[s:e] for t, s, e in rows]
    df.attrs['preprocessing'] = settings
    return df


def flag_duplicates(df):
    """
    Add three columns (nothing is removed, because the scorer expects every
    sentence_id):
        is_duplicate_text   same word + same sentence occurs more than once
        dup_group           id shared by the copies (-1 = not duplicated)
        dup_label_conflict  the copies received different gold labels
    Use dup_label_conflict to keep contradictory pairs out of fine-tuning.
    """
    df = df.copy()
    dup = df.duplicated(['word', 'text'], keep=False)
    df['is_duplicate_text'] = dup
    df['dup_group'] = -1
    df.loc[dup, 'dup_group'] = df[dup].groupby(['word', 'text'], sort=True).ngroup()
    label_cols = [c for c in ['senses', 'target_sense'] if c in df.columns]
    df['dup_label_conflict'] = False
    if label_cols and dup.any():
        labels = df.loc[dup, label_cols].astype(str).agg('|'.join, axis=1)
        n_distinct = labels.groupby(df.loc[dup, 'dup_group']).transform('nunique')
        df.loc[dup, 'dup_label_conflict'] = (n_distinct > 1).values
    return df


def cap_usages_per_period(df, max_n, seed=42):
    """
    Keep at most max_n usages per (word, period), sampled with a fixed seed.

    For TRAINING / embedding budgets only. Never cap the usages you write
    predictions for: the official scorer rejects a file with missing ids.
    (The organisers already capped the dev data at 30, so on dev this is
    a no-op for max_n >= 30.)
    """
    if max_n is None:
        return df
    parts = [g if len(g) <= max_n else g.sample(n=max_n, random_state=seed)
             for _, g in df.groupby(['word', 'period_label'], sort=True)]
    return pd.concat(parts).sort_index()


def group_usages(df):
    """
    The structure the work plan asks for: usages grouped by word and period.

        groups = group_usages(df)
        groups['herre']['1880-1883']   -> DataFrame of those usages

    Words are sorted alphabetically, periods by start year (the same order
    the official 1b scorer uses), so iterating gives consecutive periods.
    Row indices are kept, so results can be written back into df.
    """
    out = {}
    for word, g in df.groupby('word', sort=True):
        periods = sorted(g['period_label'].unique(), key=period_start_year)
        out[word] = {p: g[g['period_label'] == p] for p in periods}
    return out


def load_and_prepare(data_dir, language='SV', preprocessing=None, check=False, verbose=False):
    """
    The one call experiments should use:

        df = load_and_prepare('data', 'SV', cfg['preprocessing'])

    = load_language() (unchanged A1 loader) + optional integrity report
      + model input columns + duplicate flags.
    Works on test data without gold labels as well.
    """
    df = load_language(data_dir, language)
    if check:
        check_integrity(df, verbose=verbose)
    df = add_model_inputs(df, preprocessing)
    df = flag_duplicates(df)
    return df

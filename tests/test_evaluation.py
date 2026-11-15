"""
Tests for the shared evaluation script. Run from the repo root:

    python -m pytest tests -q

They pin the scores of "dumb" systems whose values we know, so any change to
the official scorer or to src/scoring.py / src/evaluate.py that moves a
number is caught immediately.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import (check_graded, check_predictions, compare,  # noqa: E402
                          evaluate_predictions, read_jsonl, write_jsonl, _gold_tables)

GOLD1 = ROOT / 'data' / 'SV' / 'subtask1.jsonl'
GOLD2 = ROOT / 'data' / 'SV' / 'subtask2.jsonl'


def _constant(tmp_path, gold, label, name):
    rows = [{'word': r['word'], 'sentence_id': r['sentence_id'], 'label': label}
            for r in read_jsonl(gold)]
    return write_jsonl(tmp_path / name, rows)


@pytest.fixture(scope='module')
def trivial(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('trivial')
    s1 = _constant(tmp, GOLD1, [1], 'one_cluster.jsonl')
    s2 = _constant(tmp, GOLD2, 1, 'all_1.jsonl')
    return evaluate_predictions(s1, s2, name='trivial', out_dir=tmp, verbose=False,
                                shuffle=False, bootstrap=False)


def test_one_cluster_reference(trivial):
    m = trivial['subtask1']['subtask1a']['macro']
    assert round(m['fcubed'], 3) == 0.756
    # One cluster is impure (low precision) but keeps senses together (high
    # recall). If this fails, precision and recall are swapped again.
    assert m['precision'] < m['recall']
    assert round(m['precision'], 3) == 0.646 and round(m['recall'], 3) == 0.981


def test_constant_change_gives_rho_zero(trivial):
    assert trivial['subtask1']['subtask1b']['rho'] == 0
    assert trivial['subtask1']['subtask1b']['n_pairs'] == 60


def test_subtask2_uses_new_macro_f1(trivial, tmp_path):
    assert round(trivial['subtask2']['macro_f1'], 3) == 0.236           # all 1
    assert round(trivial['subtask2']['word_mean']['positive_f1'], 3) == 0.472  # A1 metric
    s2 = _constant(tmp_path, GOLD2, 0, 'all_0.jsonl')
    r = evaluate_predictions(s2=s2, out_dir=tmp_path, verbose=False, bootstrap=False)
    assert round(r['subtask2']['macro_f1'], 3) == 0.361


def test_gold_labels_score_perfectly(tmp_path):
    r = evaluate_predictions(GOLD1, GOLD2, out_dir=tmp_path, verbose=False,
                             shuffle=False, bootstrap=False)
    assert r['scores']['1a_bcubed_f1'] == pytest.approx(1.0)
    assert r['scores']['1b_spearman'] == pytest.approx(1.0)
    assert r['scores']['2_macro_f1'] == pytest.approx(1.0)


def test_format_errors_are_caught(tmp_path):
    rows = read_jsonl(GOLD1)
    bad = [dict(r) for r in rows if r['word'] != 'dumpa']
    bad[0]['label'] = 3              # int instead of list
    bad.append(dict(bad[1]))         # duplicate
    chk = check_predictions(write_jsonl(tmp_path / 'bad.jsonl', bad), GOLD1, '1')
    text = ' '.join(chk['errors'] + chk['warnings'])
    assert 'LIST' in text and 'duplicate' in text and 'MISSING WORDS' in text
    assert not chk['complete']


def test_missing_and_extra_words_warn_but_score(tmp_path):
    rows = [r for r in read_jsonl(GOLD2) if r['word'] != 'skär']
    rows.append({'word': 'banan', 'sentence_id': 'x', 'label': 1})
    r = evaluate_predictions(s2=write_jsonl(tmp_path / 'p.jsonl', rows), out_dir=tmp_path,
                             verbose=False, bootstrap=False)
    warns = ' '.join(r['checks']['subtask2']['warnings'])
    assert 'MISSING WORDS' in warns and 'EXTRA WORDS' in warns
    assert 'skär' not in r['words_scored'] and len(r['words_scored']) == 9


def test_graded_file_needs_every_scored_pair(tmp_path):
    gold = _gold_tables(ROOT / 'data', 'SV')
    rows = [{'word': k.rsplit('_', 2)[0], 'period1': k.rsplit('_', 2)[1],
             'period2': k.rsplit('_', 2)[2], 'score': v} for k, v in gold['gold_jsd'].items()]
    path = write_jsonl(tmp_path / 'g.jsonl', rows[:-1])
    assert check_graded(path, gold['gold_jsd'])['errors']
    path = write_jsonl(tmp_path / 'g.jsonl', rows)
    r = evaluate_predictions(graded=path, out_dir=tmp_path, verbose=False, bootstrap=False)
    assert r['graded']['rho'] == pytest.approx(1.0)


def test_compare_identical_systems_is_not_significant(trivial):
    r = compare(trivial, trivial, '1a', n=200)
    assert r['difference_a_minus_b'] == 0 and r['p_value'] == 1.0
    assert r['verdict'].startswith('not significant')

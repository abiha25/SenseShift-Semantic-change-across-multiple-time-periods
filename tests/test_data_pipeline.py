"""
Tests for the shared data pipeline. Run from the repo root:

    python -m pytest tests -q

Each test protects one promise the rest of the team relies on.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_loader import (add_model_inputs, cap_usages_per_period, group_usages,  # noqa: E402
                         load_and_prepare, load_language, normalise_spelling,
                         period_start_year, prepare_usage, write_subtask1)
from src import splits as sp  # noqa: E402
from src.config import apply_overrides, load_config, save_run_config  # noqa: E402
from src.features import normalise as a1_normalise  # noqa: E402
from src.scoring import score_1a  # noqa: E402

DATA = ROOT / 'data'


@pytest.fixture(scope='module')
def df():
    return load_and_prepare(DATA, 'SV', {'spelling_normalisation': True, 'lowercase': True})


# --- loader ----------------------------------------------------------------

def test_counts_match_report(df):
    assert len(df) == 1983 and df['word'].nunique() == 10 and df['period_label'].nunique() == 10


def test_a1_columns_unchanged(df):
    """The A1 loader output is a subset of the new output, value for value."""
    old = load_language(DATA, 'SV')
    for col in old.columns:
        assert old[col].astype(str).equals(df[col].astype(str)), col


def test_works_without_gold_labels(tmp_path):
    """Test data (January) has no subtask1/subtask2 files."""
    lang = tmp_path / 'SV'
    lang.mkdir()
    for name in ['usages.jsonl', 'definitions.jsonl']:
        (lang / name).write_bytes((DATA / 'SV' / name).read_bytes())
    d = load_and_prepare(tmp_path, 'SV')
    assert len(d) == 1983 and 'senses' not in d.columns and 'model_text' in d.columns


# --- preprocessing ---------------------------------------------------------

def test_spelling_matches_assignment1_on_every_sentence(df):
    for t in df['text']:
        assert normalise_spelling(t.lower(), 0, 0)[0] == a1_normalise(t)


def test_target_offsets_survive_preprocessing(df):
    expected = [a1_normalise(t[s:e]) for t, s, e in zip(df.text, df.start_fixed, df.end_fixed)]
    assert list(df['model_target']) == expected


def test_offsets_shift_after_shorter_replacement():
    # 'hvad' -> 'vad' removes one character BEFORE the target
    text = 'Hvad sade qvinnan i går'
    s = text.index('qvinnan')
    t, s2, e2 = normalise_spelling(text, s, s + len('qvinnan'))
    assert t == 'Vad sade kvinnan i går' and t[s2:e2] == 'kvinnan'


def test_spelling_keeps_case_when_lowercase_is_off():
    t, _, _ = prepare_usage('Hvad skall QVINNAN göra', 0, 4, spelling_normalisation=True)
    assert t == 'Vad skall KVINNAN göra'


def test_spelling_only_for_swedish():
    t, _, _ = prepare_usage('hvad af', 0, 4, language='EN', spelling_normalisation=True)
    assert t == 'hvad af'


def test_context_window(df):
    d = add_model_inputs(df, {'context_window': 3})
    assert (d['model_target'] == d['target_fixed']).all()
    t, s, e = prepare_usage('a b c d TARGET e f g h', 8, 14, context_window=2)
    assert (t, t[s:e]) == ('c d TARGET e f', 'TARGET')


def test_unknown_preprocessing_key_is_refused(df):
    with pytest.raises(ValueError):
        add_model_inputs(df, {'lowercasing': True})


def test_duplicate_flags(df):
    assert int(df['is_duplicate_text'].sum()) == 10          # 5 sentences x 2 copies
    conflicts = df[df['dup_label_conflict']]
    assert len(conflicts) == 4 and set(conflicts['word']) == {'suga'}


def test_cap_is_seeded(df):
    a = cap_usages_per_period(df, 10, seed=1)
    b = cap_usages_per_period(df, 10, seed=1)
    assert a.index.equals(b.index) and a.groupby(['word', 'period_label']).size().max() == 10


def test_group_usages_order(df):
    g = group_usages(df)
    assert list(g) == sorted(g)
    for periods in g.values():
        assert list(periods) == sorted(periods, key=period_start_year)
    assert sum(len(x) for p in g.values() for x in p.values()) == len(df)


def test_new_dataframe_still_scores_like_a1(df, tmp_path):
    """One cluster per word must still give the reported 0.756."""
    path = tmp_path / 'SV_subtask1.jsonl'
    write_subtask1(df.assign(_p=0), '_p', path)
    assert round(score_1a(path, DATA / 'SV' / 'subtask1.jsonl')['macro']['fcubed'], 3) == 0.756


# --- splits ----------------------------------------------------------------

def test_split_has_no_leakage(df):
    s = sp.build_splits(df, DATA, 'SV')
    assert sp.check_no_leakage(s) and len(s['folds']) == 10
    for f in s['folds']:
        train, val, test = sp.fold_frames(df, f)
        assert len(train) + len(val) + len(test) == len(df)
        assert not set(train.word) & set(test.word) and not set(val.word) & set(test.word)
    assert sorted(f['validation'][0] for f in s['folds']) == s['words']


def test_leakage_check_catches_overlap():
    bad = {'words': ['a', 'b', 'c'],
           'folds': [{'fold': 0, 'train': ['a', 'b'], 'validation': ['b'], 'test': ['c']}]}
    with pytest.raises(ValueError):
        sp.check_no_leakage(bad)


def test_saved_split_matches_data(df):
    s = sp.load_splits(ROOT / 'data' / 'splits' / 'SV_splits.json', df, DATA)
    assert s['n_usages_total'] == 1983


def test_fingerprint_detects_changed_data(tmp_path):
    lang = tmp_path / 'SV'
    lang.mkdir()
    for name in sp.DATA_FILES:
        (lang / name).write_bytes((DATA / 'SV' / name).read_bytes())
    before = sp.data_fingerprint(tmp_path)
    with open(lang / 'usages.jsonl', 'a', encoding='utf-8') as fh:
        fh.write('\n{"x": 1}\n')
    assert sp.data_fingerprint(tmp_path) != before


# --- config ----------------------------------------------------------------

def test_config_overrides_and_typos():
    cfg = load_config(overrides=['training.epochs=1', 'split.folds=[0]', 'seed=13'])
    assert cfg['training']['epochs'] == 1 and cfg['split']['folds'] == [0] and cfg['seed'] == 13
    with pytest.raises(KeyError):
        apply_overrides(cfg, ['trainig.epochs=1'])


def test_run_record(tmp_path):
    cfg = load_config()
    save_run_config(cfg, tmp_path)
    info = json.loads((tmp_path / 'run_info.json').read_text(encoding='utf-8'))
    assert (tmp_path / 'config.yaml').exists() and info['data_fingerprint']
    assert load_config(tmp_path / 'config.yaml')['seed'] == cfg['seed']


# --- seeds and checkpoints (need PyTorch) ----------------------------------
# importorskip is inside each test, so a computer without PyTorch skips only
# these three tests, not the whole file.

def test_same_seed_same_numbers():
    torch = pytest.importorskip('torch')
    from src.reproducibility import set_seed
    set_seed(7)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(7)
    b = (np.random.rand(3), torch.rand(3))
    assert np.allclose(a[0], b[0]) and torch.equal(a[1], b[1])


def test_checkpoint_round_trip_and_resume(tmp_path):
    torch = pytest.importorskip('torch')
    from src.reproducibility import load_checkpoint, save_checkpoint, set_seed
    set_seed(0)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.randn(5, 4)).sum().backward()
    opt.step()
    save_checkpoint(tmp_path, model, opt, epoch=3, metrics={'val': 0.5}, config={'seed': 0})
    next_draw = torch.rand(1)

    model2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.1)
    meta = load_checkpoint(tmp_path, model2, opt2, restore_rng=True)
    assert meta['epoch'] == 3 and meta['metrics']['val'] == 0.5
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), model2.parameters()))
    assert opt2.state_dict()['state'][0]['step'] == opt.state_dict()['state'][0]['step']
    assert torch.equal(torch.rand(1), next_draw)          # random stream resumes exactly


def test_keeper_selects_best_and_stops(tmp_path):
    torch = pytest.importorskip('torch')
    from src.reproducibility import CheckpointKeeper
    model = torch.nn.Linear(2, 1)
    keeper = CheckpointKeeper(tmp_path, metric='val_f1', mode='max', patience=2)
    for epoch, score in enumerate([0.50, 0.62, 0.60, 0.58]):
        with torch.no_grad():
            model.weight.fill_(epoch)                      # mark the weights with the epoch
        keeper.update(score, model, epoch)
        if keeper.should_stop:
            break
    assert keeper.best_epoch == 1 and epoch == 3
    fresh = torch.nn.Linear(2, 1)
    assert keeper.load_best(fresh)['epoch'] == 1 and fresh.weight[0, 0].item() == 1.0

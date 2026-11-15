"""
Shared configuration handling (team SenseShift).

    from src.config import load_config, make_run_dir, save_run_config
    cfg = load_config('configs/default.yaml', overrides=['training.epochs=1'])
    run_dir = make_run_dir(cfg, 'prelim_xlmr')
    save_run_config(cfg, run_dir)        # config.yaml + run_info.json in the run folder

WHY
    * One file holds every setting, so a result can always be traced back to
      the exact settings that produced it (instructor brief, Section 7).
    * Overrides from the command line mean nobody edits the shared file for
      a one-off experiment.
    * save_run_config() writes the config AND the environment (Python and
      library versions, GPU, git commit, data fingerprint) next to the
      results: this is the record Section 10 (Reproducibility) reports.

The config is a plain nested dict (cfg['training']['learning_rate']), so it
needs no special classes and is easy to print, save and compare.
"""

from __future__ import annotations

import copy
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'default.yaml'

# Settings every run needs. load_config() fails early if one is missing.
REQUIRED_KEYS = [
    'seed',
    'data.data_dir', 'data.language',
    'split.file', 'split.validation_offset',
    'preprocessing.spelling_normalisation', 'preprocessing.lowercase',
    'preprocessing.context_window',
    'output.results_dir', 'output.checkpoints_dir',
]

# Path settings that are resolved against the repo root when relative, so
# the same config works from any working directory (local or Colab).
PATH_KEYS = ['data.data_dir', 'split.file', 'output.results_dir', 'output.checkpoints_dir']


# ---------------------------------------------------------------------------
# Dotted-key helpers: 'training.learning_rate' <-> cfg['training']['learning_rate']
# ---------------------------------------------------------------------------

def get(cfg, dotted, default=KeyError):
    node = cfg
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            if default is KeyError:
                raise KeyError(f'Missing config key: {dotted}')
            return default
        node = node[part]
    return node


def set_value(cfg, dotted, value):
    parts = dotted.split('.')
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_overrides(cfg, overrides):
    """
    overrides: list like ['training.epochs=1', 'seed=13', 'split.folds=[0]'].
    The value is parsed as YAML, so numbers, booleans, null and lists work.
    A key that does not exist yet is refused, which catches typos such as
    'trainig.epochs=1' that would otherwise be ignored silently.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, raw = item.split('=', 1)
        key = key.strip()
        get(cfg, key)  # raises KeyError for unknown keys
        set_value(cfg, key, yaml.safe_load(raw))
    return cfg


def _resolve_paths(cfg):
    for key in PATH_KEYS:
        value = get(cfg, key, None)
        if value is not None and not Path(value).is_absolute():
            set_value(cfg, key, str((REPO_ROOT / value).resolve()))
    return cfg


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(path=DEFAULT_CONFIG, overrides=None, resolve_paths=True):
    """Read the YAML file, apply overrides, check required keys, resolve paths."""
    with open(path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    cfg = apply_overrides(cfg, overrides)
    missing = [k for k in REQUIRED_KEYS if get(cfg, k, None) is None and k not in (
        'preprocessing.context_window',)]  # null is a valid window (= whole sentence)
    if missing:
        raise KeyError(f'Config {path} is missing required keys: {missing}')
    cfg['_config_file'] = str(Path(path).resolve())
    return _resolve_paths(cfg) if resolve_paths else cfg


def add_cli_arguments(parser):
    """Adds --config and repeatable --set key=value to any script's argparse."""
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='KEY=VALUE', help='override a config value, e.g. --set seed=13')
    return parser


# ---------------------------------------------------------------------------
# Recording a run
# ---------------------------------------------------------------------------

def make_run_dir(cfg, name):
    """results/runs/<YYYYmmdd-HHMMSS>_<name>_seed<seed>/ (created)."""
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = Path(cfg['output']['results_dir']) / f"{stamp}_{name}_seed{cfg['seed']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _git_commit():
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return {'commit': sha or None, 'uncommitted_changes': bool(dirty)} if sha else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info():
    """Software and hardware facts for the reproducibility section."""
    info = {'python': platform.python_version(), 'platform': platform.platform(),
            'executable': sys.executable, 'packages': {}}
    for name in ['numpy', 'pandas', 'scipy', 'sklearn', 'torch', 'transformers', 'bcubed', 'yaml']:
        try:
            module = __import__(name)
            info['packages'][name] = getattr(module, '__version__', 'unknown')
        except ImportError:
            info['packages'][name] = None
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info['cuda_version'] = torch.version.cuda
    except ImportError:
        info['cuda_available'], info['gpu'] = False, None
    return info


def save_run_config(cfg, run_dir, extra=None):
    """
    Write <run_dir>/config.yaml (exact settings used) and
    <run_dir>/run_info.json (time, environment, git commit, data fingerprint).
    """
    from src.splits import data_fingerprint

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith('_')}
    with open(run_dir / 'config.yaml', 'w', encoding='utf-8') as fh:
        yaml.safe_dump(clean, fh, sort_keys=False, allow_unicode=True)
    info = {
        'started': datetime.now().isoformat(timespec='seconds'),
        'config_file': cfg.get('_config_file'),
        'command': ' '.join(sys.argv),
        'git': _git_commit(),
        'data_fingerprint': data_fingerprint(cfg['data']['data_dir'], cfg['data']['language']),
        'environment': environment_info(),
        **(extra or {}),
    }
    (run_dir / 'run_info.json').write_text(json.dumps(info, indent=2, default=str), encoding='utf-8')
    return run_dir

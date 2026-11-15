"""
Seeds and checkpoints (team SenseShift).

    from src.reproducibility import set_seed, seed_worker, torch_generator, CheckpointKeeper

    set_seed(cfg['seed'], deterministic=cfg['training']['deterministic'])
    loader = DataLoader(ds, shuffle=True, worker_init_fn=seed_worker,
                        generator=torch_generator(cfg['seed']))

    keeper = CheckpointKeeper(run_dir / 'checkpoints', metric='val_bcubed_f1', mode='max',
                              patience=cfg['training']['early_stopping_patience'])
    for epoch in range(cfg['training']['epochs']):
        ... train ...
        score = evaluate_on_validation_word(...)
        keeper.update(score, model=model, optimizer=opt, scheduler=sch, epoch=epoch, config=cfg)
        if keeper.should_stop:
            break
    keeper.load_best(model)      # the checkpoint with the best VALIDATION score

WHY
    * Seeds: fine-tuning, K-means, shuffling and sampling all use random
      numbers. Fixing every generator means the same config gives the same
      result, and differences between seeds (Section 10: 3-5 seeds,
      mean +- sd) measure real run-to-run variation.
    * Checkpoints: the model is selected by its best score on the
      VALIDATION word, never the test word (leakage). Saving the optimizer,
      scheduler and random-number states as well lets a stopped Colab run
      continue exactly where it stopped.

PyTorch is imported only inside the functions that need it, so the rest of
the pipeline (loader, splits, config, scoring) works without it.
"""

from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = True) -> int:
    """
    Fix every random-number generator we use: Python, NumPy, PyTorch (CPU
    and all GPUs). With deterministic=True, PyTorch is also told to use
    repeatable GPU algorithms (a little slower).

    Note: Python's own hash randomisation (PYTHONHASHSEED) can only be fixed
    before the interpreter starts; we set it here so that any SUBPROCESS
    inherits it. Our code never depends on set/dict-of-string ordering.
    """
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Needed by cuBLAS for deterministic matrix products; must be set
        # before the first CUDA call, so call set_seed() at the very start.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: an operation without a deterministic version prints a
        # warning instead of crashing; such warnings are reported in Section 9.
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def seed_worker(worker_id: int) -> None:
    """worker_init_fn for torch DataLoader: each worker gets a derived, fixed seed."""
    import torch
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int):
    """A seeded torch.Generator for DataLoader(shuffle=True, generator=...)."""
    import torch
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def rng_state() -> dict:
    """Snapshot of all random states (stored inside checkpoints)."""
    state = {'python': random.getstate(), 'numpy': np.random.get_state()}
    try:
        import torch
        state['torch'] = torch.get_rng_state()
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if 'torch' in state:
        import torch
        torch.set_rng_state(state['torch'])
        if 'torch_cuda' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['torch_cuda'])


# ---------------------------------------------------------------------------
# Checkpoints
#
# Layout of one checkpoint folder:
#     model_state.pt      model weights (state_dict)
#     training_state.pt   optimizer, scheduler, random states
#     meta.json           human-readable: epoch, metrics, time, config
# ---------------------------------------------------------------------------

def save_checkpoint(ckpt_dir, model, optimizer=None, scheduler=None, epoch=None, step=None,
                    metrics=None, config=None, extra=None) -> Path:
    """
    Save everything needed to (a) reuse the model and (b) resume training.
    Works for any torch.nn.Module, including Hugging Face models: the model
    is re-created from config['model'] and then load_checkpoint() fills in
    the weights.
    """
    import torch

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / 'model_state.pt')
    torch.save({
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'rng_state': rng_state(),
    }, ckpt_dir / 'training_state.pt')
    meta = {
        'saved': datetime.now().isoformat(timespec='seconds'),
        'epoch': epoch,
        'step': step,
        'metrics': metrics or {},
        'model_class': type(model).__name__,
        'n_parameters': int(sum(p.numel() for p in model.parameters())),
        'config': {k: v for k, v in (config or {}).items() if not str(k).startswith('_')},
        **(extra or {}),
    }
    (ckpt_dir / 'meta.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    return ckpt_dir


def load_checkpoint(ckpt_dir, model=None, optimizer=None, scheduler=None,
                    map_location='cpu', restore_rng=False) -> dict:
    """
    Load the weights into `model` (built beforehand with the same settings)
    and, if given, the optimizer/scheduler state. restore_rng=True continues
    the random-number streams exactly (for resuming an interrupted run).
    Returns meta.json as a dict.
    """
    import torch

    ckpt_dir = Path(ckpt_dir)
    if not (ckpt_dir / 'meta.json').exists():
        raise FileNotFoundError(f'No checkpoint in {ckpt_dir}')
    if model is not None:
        state = torch.load(ckpt_dir / 'model_state.pt', map_location=map_location, weights_only=True)
        model.load_state_dict(state)
    if optimizer is not None or scheduler is not None or restore_rng:
        # weights_only=False: the file holds optimizer/RNG objects that WE wrote.
        # Never load a training_state.pt from an untrusted source.
        ts = torch.load(ckpt_dir / 'training_state.pt', map_location=map_location, weights_only=False)
        if optimizer is not None and ts['optimizer'] is not None:
            optimizer.load_state_dict(ts['optimizer'])
        if scheduler is not None and ts['scheduler'] is not None:
            scheduler.load_state_dict(ts['scheduler'])
        if restore_rng:
            restore_rng_state(ts['rng_state'])
    return json.loads((ckpt_dir / 'meta.json').read_text(encoding='utf-8'))


class CheckpointKeeper:
    """
    Keeps two checkpoints per training run:
        <dir>/best   the epoch with the best VALIDATION score (model selection)
        <dir>/last   the most recent epoch (to resume after a Colab disconnect)
    and counts epochs without improvement for early stopping.
    """

    def __init__(self, ckpt_dir, metric='val_score', mode='max', patience=None, min_delta=0.0):
        if mode not in ('max', 'min'):
            raise ValueError("mode must be 'max' or 'min'")
        self.dir = Path(ckpt_dir)
        self.metric, self.mode = metric, mode
        self.patience, self.min_delta = patience, min_delta
        self.best_score = -math.inf if mode == 'max' else math.inf
        self.best_epoch = None
        self.bad_epochs = 0
        self.history = []

    def _improved(self, score):
        if score is None or (isinstance(score, float) and math.isnan(score)):
            return False
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def update(self, score, model, epoch, **save_kwargs) -> bool:
        """Save 'last' always and 'best' when the score improved. Returns True if improved."""
        metrics = {self.metric: score, **save_kwargs.pop('metrics', {})}
        improved = self._improved(score)
        save_checkpoint(self.dir / 'last', model, epoch=epoch, metrics=metrics, **save_kwargs)
        if improved:
            self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
            save_checkpoint(self.dir / 'best', model, epoch=epoch, metrics=metrics, **save_kwargs)
        else:
            self.bad_epochs += 1
        self.history.append({'epoch': epoch, self.metric: score, 'improved': improved})
        (self.dir / 'history.json').write_text(json.dumps(self.history, indent=2, default=str),
                                               encoding='utf-8')
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience is not None and self.bad_epochs >= self.patience

    def load_best(self, model, map_location='cpu') -> dict:
        return load_checkpoint(self.dir / 'best', model, map_location=map_location)

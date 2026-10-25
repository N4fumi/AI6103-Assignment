"""Configuration, provenance, deterministic state, and atomic local I/O."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ('lr_05', 'lr_005', 'lr_0001', 'constant', 'cosine', 'wd_5e4', 'wd_1e4', 'mixup')
METRICS = ('epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy',
           'lr', 'next_lr', 'epoch_seconds', 'elapsed_seconds')


def path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def read_json(filename):
    with Path(filename).open(encoding='utf-8') as f:
        return json.load(f)


def atomic_bytes(filename, content):
    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=target.name + '.', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(filename, value):
    atomic_bytes(filename, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(filename):
    h = hashlib.sha256()
    with Path(filename).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_config(experiment=None):
    import yaml
    with (ROOT / 'configs/base.yaml').open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if experiment:
        if experiment not in EXPERIMENTS:
            raise ValueError(f'Unknown experiment: {experiment}')
        with (ROOT / f'configs/{experiment}.yaml').open(encoding='utf-8') as f:
            override = yaml.safe_load(f) or {}
        unknown = set(override) - set(cfg)
        if unknown:
            raise ValueError(f'Unknown configuration keys: {unknown}')
        cfg.update(override)
        cfg['experiment'] = experiment
    validate_config(cfg)
    return cfg


def validate_config(c):
    fixed = {'model': 'efficientnet_v2_s', 'pretrained': False, 'optimizer': 'SGD',
             'momentum': 0.9, 'batch_size': 128, 'image_size': 100, 'padding': 12,
             'flip_probability': 0.5, 'num_classes': 120, 'split_counts': [7155, 1533, 1534]}
    for k, v in fixed.items():
        if c[k] != v:
            raise ValueError(f'Assignment requires {k}={v}; got {c[k]}')
    if c['precision'] not in ('fp32', 'amp_fp16', 'amp_bf16'):
        raise ValueError('precision must be fp32, amp_fp16, or amp_bf16')
    if c['device'] not in ('cpu', 'cuda') or c['scheduler'] not in ('constant', 'cosine'):
        raise ValueError('Invalid device or scheduler')
    if not isinstance(c['workers'], int) or c['workers'] < 0 or not 0 < c['memory_fraction'] <= 1:
        raise ValueError('Invalid worker count or memory fraction')
    if c['epochs'] not in (15, 300) or c['weight_decay'] not in (0.0, 0.0001, 0.0005):
        raise ValueError('Invalid assignment epochs or weight decay')
    if c['lr'] is not None and c['lr'] not in (0.5, 0.05, 0.001):
        raise ValueError('Learning rate must be 0.5, 0.05, or 0.001')
    if c['mixup_alpha'] not in (0.0, 0.2):
        raise ValueError('Mixup alpha must be 0 or 0.2')
    if c['content_duplicate_policy'] not in ('error', 'preserve_official'):
        raise ValueError('content_duplicate_policy must be error or preserve_official')
    exp = c.get('experiment')
    if exp in EXPERIMENTS[:3]:
        required_lr = dict(zip(EXPERIMENTS[:3], (0.5, 0.05, 0.001)))[exp]
        if (c['lr'], c['epochs'], c['scheduler'], c['weight_decay'], c['mixup_alpha']) != (required_lr, 15, 'constant', 0.0, 0.0):
            raise ValueError(f'{exp}: LR study requires its specified LR, 15 epochs, constant schedule, no decay/Mixup')
    elif exp in EXPERIMENTS[3:]:
        if c['epochs'] != 300 or c['mixup_alpha'] != (0.2 if exp == 'mixup' else 0.0):
            raise ValueError(f'{exp}: requires 300 epochs and its specified Mixup setting')
        if exp != 'mixup':
            expected = {'constant': ('constant', 0.0), 'cosine': ('cosine', 0.0),
                        'wd_5e4': ('cosine', 0.0005), 'wd_1e4': ('cosine', 0.0001)}[exp]
            if (c['scheduler'], c['weight_decay']) != expected:
                raise ValueError(f'{exp}: scheduler/decay must be {expected}')


def seed_all(seed, deterministic=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(filename, value):
    import io
    import torch
    buf = io.BytesIO()
    torch.save(value, buf)
    atomic_bytes(filename, buf.getvalue())


def load_checkpoint(filename):
    import torch
    # Local checkpoints contain Python/NumPy RNG state. Never load untrusted files.
    value = torch.load(filename, map_location='cpu', weights_only=False)
    required = {'model', 'optimizer', 'scheduler', 'scaler', 'epoch', 'config',
                'rng', 'history', 'data_signature', 'best_epoch', 'best_key', 'best_model'}
    if not required.issubset(value):
        raise ValueError(f'Invalid checkpoint: missing {required - set(value)}')
    if value['epoch'] != len(value['history']):
        raise ValueError('Checkpoint epoch/history mismatch')
    return value


def write_history(filename, history):
    import io
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=METRICS)
    writer.writeheader()
    writer.writerows(history)
    atomic_bytes(filename, buf.getvalue().encode())


def read_history(filename):
    with Path(filename).open(newline='', encoding='utf-8') as f:
        return [{k: int(v) if k == 'epoch' else float(v) for k, v in row.items()}
                for row in csv.DictReader(f)]

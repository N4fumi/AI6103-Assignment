"""Explicitly invoked training only. No top-level execution or auto-pipeline."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import time

import torch
import torch.nn.functional as F

from .dataset import load_data_metadata, make_loader
from .metrics import Metrics, validation_key, weighted_correct
from .mixup import mixup_batch, mixup_loss
from .model import build_model
from .utils import (ROOT, digest, file_hash, load_checkpoint, path, restore_rng, rng_state,
                    save_checkpoint, seed_all, write_history, write_json)


def source_signature():
    return digest({p.name: file_hash(p) for p in sorted((ROOT / 'src').glob('*.py'))})


def environment_signature():
    return digest({'python': platform.python_version(), 'torch': torch.__version__,
                   'torchvision': importlib.metadata.version('torchvision'), 'numpy': importlib.metadata.version('numpy'),
                   'pillow': importlib.metadata.version('Pillow'), 'cuda': torch.version.cuda})


def setup_device(cfg):
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable. Run scripts/check_environment.py; CPU requires --device cpu.')
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
        # This catches wheels which list CUDA but cannot actually execute on this GPU.
        x = torch.ones((2, 2), device=device)
        torch.mm(x, x)
        torch.cuda.synchronize()
        if cfg['precision'] == 'amp_bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unsupported. Use --precision fp32 for every configuration in the comparison.')
    elif cfg['precision'] != 'fp32':
        raise ValueError('CPU execution requires explicit --precision fp32')
    return device


def autocast_context(cfg, device):
    if cfg['precision'] == 'fp32':
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16 if cfg['precision'] == 'amp_fp16' else torch.bfloat16)


def make_scheduler(optimizer, cfg):
    if cfg['scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=0.0)
    return None


@torch.inference_mode()
def evaluate_loader(model, loader, device):
    """FP32 deterministic validation/test CE and ordinary top-1 accuracy."""
    model.eval()
    meter = Metrics()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        meter.update(loss.item(), weighted_correct(logits, y), len(y))
    return meter.values()


def train_epoch(model, loader, optimizer, scaler, cfg, device):
    model.train()
    meter = Metrics()
    skipped = 0
    for batch, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if cfg['channels_last']:
            x = x.contiguous(memory_format=torch.channels_last)
        x, a, b, lam = mixup_batch(x, y, cfg['mixup_alpha'])
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(cfg, device):
            logits = model(x)
            loss = mixup_loss(logits, a, b, lam)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training CE. Stop and rerun this experiment in FP32 with --overwrite; keep comparisons consistent.')
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < before)
        else:
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite gradient. Last complete epoch remains resumable; consider a fresh FP32 experiment.')
            optimizer.step()
        meter.update(loss.item(), weighted_correct(logits.detach(), a, b, lam), len(y))
        if batch == 1 or batch % 20 == 0 or batch == len(loader):
            print(f'  batch {batch}/{len(loader)} CE={meter.values()[0]:.5f}', flush=True)
    if skipped:
        raise FloatingPointError(f'FP16 GradScaler skipped {skipped} updates. Epoch not committed. Use a fresh FP32 run to preserve methodology.')
    return meter.values()


def best_artifact(state):
    row = state['history'][state['best_epoch'] - 1]
    return {'kind': 'inference_best_validation', 'model': state['best_model'], 'epoch': state['best_epoch'],
            'config': state['config'], 'data_signature': state['data_signature'], 'classes': state['classes'],
            'statistics': state['statistics'], 'source_signature': state['source_signature'],
            'validation': row, 'criterion': 'highest validation accuracy, lowest CE, earliest epoch'}


def run(cfg, resume=False, overwrite=False):
    """Train exactly one experiment, only after an explicit CLI request."""
    if resume and overwrite:
        raise ValueError('--resume and --overwrite are mutually exclusive')
    exp = cfg['experiment']
    results, checkpoints = ROOT / 'results' / exp, ROOT / 'checkpoints' / exp
    lock = ROOT / 'results' / f'.{exp}.lock'
    results.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as e:
        raise RuntimeError(f'{lock} exists. Ensure no process is running; remove only a stale lock manually.') from e
    os.close(fd)
    try:
        _run_locked(cfg, results, checkpoints, resume, overwrite)
    finally:
        lock.unlink(missing_ok=True)


def _run_locked(cfg, results, checkpoints, resume, overwrite):
    exp = cfg['experiment']
    manifest, stats = load_data_metadata(cfg, verify_hashes=True)
    data_signature = digest({'manifest': manifest['signature'], 'statistics': stats})
    seed_all(cfg['seed'], cfg['deterministic'])
    device = setup_device(cfg)
    existing = results.exists() or checkpoints.exists()
    if existing and not (resume or overwrite):
        raise FileExistsError(f'{exp} already has outputs. Use --resume or explicit --overwrite.')
    if resume and not (checkpoints / 'last.pt').is_file():
        raise FileNotFoundError(f'No resumable checkpoint: {checkpoints / "last.pt"}')
    if overwrite:
        # Preserve previous work in-place under a timestamped archive instead of deleting.
        stamp = str(time.time_ns())
        for directory in (results, checkpoints, ROOT / 'figures' / exp):
            if directory.exists():
                shutil.move(str(directory), str(directory.with_name(directory.name + '.archived.' + stamp)))
    results.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg['num_classes']).to(device)
    if cfg['channels_last']:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = make_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['precision'] == 'amp_fp16' and device.type == 'cuda')
    history, best_key, best_epoch, best_model, elapsed = [], None, 0, None, 0.0
    code_signature = source_signature()
    if resume:
        state = load_checkpoint(checkpoints / 'last.pt')
        if (state['config'] != cfg or state['data_signature'] != data_signature or
                state['source_signature'] != code_signature or state.get('environment_signature') != environment_signature()):
            raise ValueError('Resume refused: configuration, source code, software environment, or data changed since checkpoint')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler'])
            if scheduler.last_epoch != state['epoch'] or scheduler.get_last_lr() != [g['lr'] for g in optimizer.param_groups]:
                raise ValueError('Checkpoint scheduler progress/LR does not match completed epochs and optimizer')
        elif state['scheduler'] is not None:
            raise ValueError('Unexpected scheduler state in a constant-LR checkpoint')
        scaler.load_state_dict(state['scaler'])
        history, best_key, best_epoch, best_model = state['history'], state['best_key'], state['best_epoch'], state['best_model']
        elapsed = history[-1]['elapsed_seconds'] if history else 0.0
        restore_rng(state['rng'])
        # Canonical history is the atomic last.pt; repair CSV/best after a crash.
        write_history(results / 'metrics.csv', history)
        save_checkpoint(checkpoints / 'best.pt', best_artifact(state))
        if len(history) == cfg['epochs']:
            finish(cfg, results, checkpoints, state)
            print(f'{exp} was already complete; recovered derived outputs.')
            return
    metadata = {'python': platform.python_version(), 'platform': platform.platform(),
                'packages': {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'numpy', 'Pillow', 'PyYAML', 'matplotlib')},
                'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                'architecture': str(model), 'parameters': sum(p.numel() for p in model.parameters()),
                'weights': None, 'data_signature': data_signature, 'source_signature': code_signature,
                'dataset_integrity': manifest.get('integrity'),
                'training_metrics': 'online pre-update CE and top-1 on augmented samples; Mixup uses lambda-weighted CE and correctness; excludes weight penalty',
                'validation_metrics': 'FP32 eval-mode CE and top-1 on deterministic unmixed validation images'}
    write_json(results / 'config.json', cfg)
    write_json(results / 'metadata.json', metadata)
    write_json(results / 'status.json', {'status': 'running', 'completed_epochs': len(history)})
    train_loader = make_loader(cfg, manifest, stats, 'train')
    val_loader = make_loader(cfg, manifest, stats, 'val')
    try:
        for epoch in range(len(history) + 1, cfg['epochs'] + 1):
            start = time.perf_counter()
            train_loader.dataset.epoch[0] = epoch
            train_loader.generator.manual_seed(cfg['seed'] + epoch)
            # Dedicated val generator isolates validation from dropout/Mixup RNG.
            val_loader.generator.manual_seed(cfg['seed'] + 100000 + epoch)
            used_lr = optimizer.param_groups[0]['lr']
            print(f'{exp}: epoch {epoch}/{cfg["epochs"]}, lr={used_lr:.8g}', flush=True)
            train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, scaler, cfg, device)
            val_loss, val_accuracy = evaluate_loader(model, val_loader, device)
            # Exactly once after the epoch's optimizer updates. Last next_lr is zero.
            if scheduler:
                scheduler.step()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            elapsed += seconds
            row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                       val_loss=val_loss, val_accuracy=val_accuracy, lr=used_lr,
                       next_lr=optimizer.param_groups[0]['lr'], epoch_seconds=seconds, elapsed_seconds=elapsed)
            history.append(row)
            key = validation_key(row)
            if best_key is None or key < best_key:
                best_key, best_epoch = key, epoch
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict() if scheduler else None, 'scaler': scaler.state_dict(),
                     'epoch': epoch, 'seed': cfg['seed'], 'rng': rng_state(), 'config': cfg,
                     'history': history, 'best_key': best_key, 'best_epoch': best_epoch, 'best_model': best_model,
                     'data_signature': data_signature, 'classes': manifest['classes'], 'statistics': stats,
                     'source_signature': code_signature, 'environment_signature': environment_signature()}
            save_checkpoint(checkpoints / 'last.pt', state)
            save_checkpoint(checkpoints / 'best.pt', best_artifact(state))
            write_history(results / 'metrics.csv', history)
            write_json(results / 'history.json', history)
            write_json(results / 'status.json', {'status': 'running', 'completed_epochs': epoch})
            print(f'  train CE={train_loss:.5f}, acc={train_accuracy:.2%}; val CE={val_loss:.5f}, acc={val_accuracy:.2%}; {seconds:.1f}s', flush=True)
        finish(cfg, results, checkpoints, state)
    except (KeyboardInterrupt, Exception) as error:
        write_json(results / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                           'error': str(error), 'resume': f'python -m src.experiments --experiment {exp} --resume'})
        print('Stopped. Resume from last.pt to replay any unfinished epoch; no partial epoch was committed.', flush=True)
        raise


def finish(cfg, results, checkpoints, state):
    save_checkpoint(checkpoints / 'final.pt', state)
    save_checkpoint(checkpoints / 'best.pt', best_artifact(state))
    history = state['history']
    write_history(results / 'metrics.csv', history)
    write_json(results / 'history.json', history)
    summary = {'status': 'complete', 'config': cfg, 'config_signature': digest(cfg),
               'data_signature': state['data_signature'], 'source_signature': state['source_signature'],
               'history_signature': digest(history), 'final': history[-1],
               'best_validation': history[state['best_epoch'] - 1],
               'checkpoint_sha256': {n: file_hash(checkpoints / n) for n in ('best.pt', 'final.pt')}}
    write_json(results / 'summary.json', summary)
    write_json(results / 'status.json', {'status': 'complete', 'completed_epochs': len(history)})

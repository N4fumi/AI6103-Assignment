"""Audited execution-only optimizations; the frozen experiment recipe stays unchanged.

This adapter deliberately keeps the original src/ and configs/ byte-for-byte intact.
Its own SHA256 and equivalence evidence are recorded separately in every new artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader


class CompatiblePersistentLoader(DataLoader):
    """Reuse workers while matching a newly-created iterator's sampler RNG draws."""

    def __iter__(self):
        if self.persistent_workers and self.num_workers and self._iterator is not None:
            # _BaseDataLoaderIter.__init__ consumes this draw before RandomSampler.
            # A persistent iterator's _reset does not. Preserve the original order.
            torch.empty((), dtype=torch.int64).random_(generator=self.generator).item()
        return super().__iter__()


def make_optimized_loader(cfg, manifest, stats, split):
    from src.dataset import DogDataset, make_transform, seed_worker
    ds = DogDataset(manifest, split, make_transform(cfg, stats, split == 'train'), cfg['seed'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    kwargs = dict(batch_size=cfg['batch_size'], shuffle=split == 'train',
                  num_workers=cfg['workers'], pin_memory=cfg['pin_memory'] and cfg['device'] == 'cuda',
                  drop_last=False, generator=generator, worker_init_fn=seed_worker)
    if cfg['workers']:
        kwargs.update(persistent_workers=True, prefetch_factor=cfg['prefetch_factor'])
    return CompatiblePersistentLoader(ds, **kwargs)


def gradients_finite(parameters):
    """Same per-element finite checks, one CPU decision instead of one per tensor."""
    checks = [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
    return not checks or bool(torch.stack(checks).all().item())


def optimized_train_epoch(model, loader, optimizer, scaler, cfg, device):
    from src.train import autocast_context
    from src.metrics import Metrics, weighted_correct
    from src.mixup import mixup_batch, mixup_loss
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
            raise FloatingPointError('Nonfinite training CE; optimization does not permit automatic retries.')
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < before)
        else:
            if not gradients_finite(model.parameters()):
                raise FloatingPointError('Nonfinite gradient; last completed epoch remains recoverable.')
            optimizer.step()
        meter.update(loss.item(), weighted_correct(logits.detach(), a, b, lam), len(y))
        if batch == 1 or batch % 20 == 0 or batch == len(loader):
            print(f'  batch {batch}/{len(loader)} CE={meter.values()[0]:.5f}', flush=True)
    if skipped:
        raise FloatingPointError(f'FP16 GradScaler skipped {skipped} updates; epoch not committed.')
    return meter.values()


def shutdown_loader(loader):
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()


def install(cfg, start_epoch, evidence):
    """Patch this process only; never modify the frozen source/config files."""
    from src import train
    from src.utils import ROOT, file_hash
    adapter = Path(__file__).resolve()
    report = ROOT / 'results/performance_optimization/equivalence.json'
    if evidence.get('status') != 'passed' or evidence.get('adapter_sha256') != file_hash(adapter):
        raise RuntimeError('Optimization requires passed equivalence evidence for this exact adapter.')
    runtime = dict(version=1, adapter=str(adapter.relative_to(ROOT)), adapter_sha256=file_hash(adapter),
                   evidence=str(report.relative_to(ROOT)), evidence_sha256=file_hash(report),
                   effective_from_epoch=start_epoch, persistent_workers=bool(cfg['workers']),
                   sampler_rng_compatible=True, gradient_checks='same finite predicate; one host synchronization',
                   best_checkpoint_policy='write on improvement; last.pt still written every epoch',
                   frozen_recipe_unchanged=True, timing_comparability='execution changed; recipe and numerical trajectory verified')
    original_save, original_write = train.save_checkpoint, train.write_json
    original_train, original_eval = train.train_epoch, train.evaluate_loader
    original_make = train.make_loader
    profile_path = ROOT / 'results' / cfg['experiment'] / 'performance.jsonl'
    profile = {'train_seconds': 0.0, 'validation_seconds': 0.0, 'checkpoint_seconds': 0.0,
               'best_checkpoint_skipped': False}
    best_saved_epoch = None

    def timed_train(*args, **kwargs):
        torch.cuda.synchronize() if args[-1].type == 'cuda' else None
        started = time.perf_counter()
        result = optimized_train_epoch(*args, **kwargs)
        torch.cuda.synchronize() if args[-1].type == 'cuda' else None
        profile['train_seconds'] = time.perf_counter() - started
        return result

    def timed_eval(*args, **kwargs):
        started = time.perf_counter()
        result = original_eval(*args, **kwargs)
        torch.cuda.synchronize() if args[-1].type == 'cuda' else None
        profile['validation_seconds'] = time.perf_counter() - started
        return result

    def save(filename, value):
        nonlocal best_saved_epoch
        filename = Path(filename)
        if filename.name == 'best.pt' and best_saved_epoch == value['epoch'] and filename.exists():
            profile['best_checkpoint_skipped'] = True
            return
        started = time.perf_counter()
        original_save(filename, dict(value, runtime_optimization=runtime))
        profile['checkpoint_seconds'] += time.perf_counter() - started
        if filename.name == 'best.pt':
            best_saved_epoch = value['epoch']

    def write(filename, value):
        filename = Path(filename)
        if filename.name in ('metadata.json', 'summary.json'):
            value = dict(value, runtime_optimization=runtime)
        original_write(filename, value)
        if filename.name == 'status.json' and value.get('status') == 'running':
            epoch = value.get('completed_epochs', 0)
            if epoch >= start_epoch:
                entry = dict(profile, epoch=epoch, runtime=runtime['adapter_sha256'])
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                with profile_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(entry, allow_nan=False) + '\n')
                print('  performance: ' + json.dumps(entry), flush=True)
            profile.update(train_seconds=0.0, validation_seconds=0.0, checkpoint_seconds=0.0,
                           best_checkpoint_skipped=False)

    train.make_loader = make_optimized_loader
    train.train_epoch, train.evaluate_loader = timed_train, timed_eval
    train.save_checkpoint, train.write_json = save, write
    return runtime

"""Explicit, isolated equivalence/throughput checks; never commits experiment epochs."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from optimized_runtime import (gradients_finite, make_optimized_loader,
                               optimized_train_epoch, shutdown_loader)
from src.dataset import load_data_metadata, make_loader
from src.model import build_model
from src.train import evaluate_loader, make_scheduler, train_epoch
from src.utils import file_hash, load_checkpoint, load_config, restore_rng, rng_state, seed_all, write_json


def fingerprint(value):
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            h.update(str((str(item.dtype), tuple(item.shape))).encode())
            h.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(str((str(item.dtype), item.shape)).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                h.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


def loader_equivalence(cfg, manifest, stats):
    # Real image transforms; enough samples to cross multiple minibatches.
    small = dict(manifest, records=[r for split in ('train', 'val')
                                   for r in [r for r in manifest['records'] if r['split'] == split][:263]])
    evidence = []
    for split in ('train', 'val'):
        old = make_loader(cfg, small, stats, split)
        new = make_optimized_loader(cfg, small, stats, split)
        try:
            for epoch in (132, 133, 134):
                seed = cfg['seed'] + (100000 if split == 'val' else 0) + epoch
                for loader in (old, new):
                    loader.dataset.epoch[0] = epoch
                    loader.generator.manual_seed(seed)
                old_batches = [fingerprint((x, y)) for x, y in old]
                new_batches = [fingerprint((x, y)) for x, y in new]
                assert old_batches == new_batches, f'{split} batch order/augmentation changed at epoch {epoch}'
                assert torch.equal(old.generator.get_state(), new.generator.get_state()), 'Loader RNG differs'
                evidence.append(dict(split=split, epoch=epoch, batches=len(old_batches),
                                     identical=True, batch_hashes=old_batches))
        finally:
            shutdown_loader(new)
    p = torch.nn.Parameter(torch.zeros(4))
    for data, expected in (([0., 1., -2., 4.], True), ([0., float('nan'), 1., 2.], False),
                           ([float('inf'), 0., 0., 0.], False), ([-float('inf'), 0., 0., 0.], False)):
        p.grad = torch.tensor(data)
        assert gradients_finite([p]) == expected
    assert gradients_finite([])
    return evidence


def branch(state, cfg, manifest, stats, optimized, epochs=2, limited_batches=None):
    device = torch.device(cfg['device'])
    seed_all(cfg['seed'], cfg['deterministic'])
    model = build_model(cfg['num_classes']).to(device)
    model.load_state_dict(state['model'])
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    if cfg['weight_decay'] != state['config']['weight_decay']:
        for group in optimizer.param_groups:
            group['weight_decay'] = cfg['weight_decay']
    scheduler = make_scheduler(optimizer, cfg)
    if scheduler:
        scheduler.load_state_dict(state['scheduler'])
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    factory = make_optimized_loader if optimized else make_loader
    loaders = {split: factory(cfg, manifest, stats, split) for split in ('train', 'val')}
    restore_rng(state['rng'])
    rows = []
    try:
        for epoch in range(state['epoch'] + 1, state['epoch'] + 1 + epochs):
            loaders['train'].dataset.epoch[0] = epoch
            loaders['train'].generator.manual_seed(cfg['seed'] + epoch)
            loaders['val'].generator.manual_seed(cfg['seed'] + 100000 + epoch)
            torch.cuda.synchronize()
            started = time.perf_counter()
            metrics = (optimized_train_epoch if optimized else train_epoch)(model, loaders['train'], optimizer, scaler, cfg, device)
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - started
            val_started = time.perf_counter()
            val_metrics = evaluate_loader(model, loaders['val'], device)
            if scheduler:
                scheduler.step()
            torch.cuda.synchronize()
            row = dict(epoch=epoch, train_seconds=train_seconds,
                       validation_seconds=time.perf_counter()-val_started,
                       epoch_seconds=time.perf_counter()-started,
                       train=list(metrics), validation=list(val_metrics),
                       model_sha256=fingerprint(model.state_dict()),
                       optimizer_sha256=fingerprint(optimizer.state_dict()),
                       scheduler_sha256=fingerprint(scheduler.state_dict() if scheduler else None),
                       rng_sha256=fingerprint(rng_state()),
                       loader_rng_sha256=fingerprint({k: v.generator.get_state() for k, v in loaders.items()}))
            print(json.dumps(dict(optimized=optimized, **row)), flush=True)
            rows.append(row)
    finally:
        for loader in loaders.values():
            shutdown_loader(loader)
        del model, optimizer, scheduler, scaler, loaders
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def compare(old, new):
    keys = ('epoch', 'train', 'validation', 'model_sha256', 'optimizer_sha256',
            'scheduler_sha256', 'rng_sha256', 'loader_rng_sha256')
    for a, b in zip(old, new, strict=True):
        for key in keys:
            assert a[key] == b[key], f'Numerical equivalence failed at epoch {a["epoch"]}: {key}'
    return dict(identical_fields=list(keys), epochs=len(old), bitwise_identical=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loader-only', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    output = ROOT / 'results/performance_optimization'
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config('cosine')
    manifest, stats = load_data_metadata(cfg, verify_hashes=True)
    if args.loader_only:
        cfg['device'], cfg['precision'] = 'cpu', 'fp32'
        proof = loader_equivalence(cfg, manifest, stats)
        write_json(output / 'loader_equivalence.json', dict(status='passed', evidence=proof))
        print('Loader equivalence passed for three epochs of train and validation.', flush=True)
        return
    if args.checkpoint is None:
        parser.error('--checkpoint is required for GPU equivalence checks')
    state = load_checkpoint(args.checkpoint)
    cfg = state['config']
    report = dict(status='running', checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint),
                  adapter_sha256=file_hash(Path(__file__).with_name('optimized_runtime.py')),
                  source_signature=state['source_signature'], original_config=cfg,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    write_json(output / 'equivalence.json', report)
    try:
        old = branch(state, cfg, manifest, stats, False)
        new = branch(state, cfg, manifest, stats, True)
        report.update(baseline=old, optimized=new, comparison=compare(old, new))
        # Exercise Mixup RNG and a remaining weight-decay setting without accessing test data.
        mixed_cfg = dict(cfg, mixup_alpha=0.2, weight_decay=0.0005)
        small = dict(manifest, records=[r for split, count in (('train', 243), ('val', 128))
                                       for r in [r for r in manifest['records'] if r['split'] == split][:count]])
        mixed_old = branch(state, mixed_cfg, small, stats, False, epochs=2)
        mixed_new = branch(state, mixed_cfg, small, stats, True, epochs=2)
        report.update(mixup_baseline=mixed_old, mixup_optimized=mixed_new,
                      mixup_comparison=compare(mixed_old, mixed_new), status='passed',
                      test_data_used=False, experiment_results_written=False)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'equivalence.json', report)
    print('PASSED: metrics, model/optimizer/scheduler, RNG and loader RNG are identical.', flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

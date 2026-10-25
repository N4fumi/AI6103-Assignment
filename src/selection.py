"""Selection uses genuine completed histories and never launches an experiment."""
from __future__ import annotations

import math

from .metrics import validation_key
from .utils import EXPERIMENTS, ROOT, digest, file_hash, read_history, read_json, validate_config, write_json

LR_IDS = EXPERIMENTS[:3]
BASELINES = ('constant', 'cosine', 'wd_5e4', 'wd_1e4')


def completed(experiment, root=ROOT):
    directory = root / 'results' / experiment
    summary = read_json(directory / 'summary.json')
    history = read_history(directory / 'metrics.csv')
    cfg = read_json(directory / 'config.json')
    validate_config(cfg)
    if summary.get('status') != 'complete' or len(history) != cfg['epochs']:
        raise ValueError(f'{experiment} is incomplete')
    if cfg['experiment'] != experiment or summary['config'] != cfg or summary['config_signature'] != digest(cfg):
        raise ValueError(f'{experiment}: inconsistent configuration metadata')
    if [r['epoch'] for r in history] != list(range(1, cfg['epochs'] + 1)):
        raise ValueError(f'{experiment}: missing/repeated epochs')
    if any(not math.isfinite(v) for r in history for v in r.values()):
        raise ValueError(f'{experiment}: nonfinite history')
    if summary['history_signature'] != digest(history) or summary['final'] != history[-1]:
        raise ValueError(f'{experiment}: summary/history mismatch')
    best = min(history, key=validation_key)
    if best != summary['best_validation']:
        raise ValueError(f'{experiment}: invalid best-validation record')
    for name in ('best.pt', 'final.pt'):
        p = root / 'checkpoints' / experiment / name
        if not p.is_file() or file_hash(p) != summary['checkpoint_sha256'][name]:
            raise ValueError(f'{experiment}: {name} is missing or changed')
    return summary


def require(experiments, root=ROOT):
    results, errors = {}, []
    for exp in experiments:
        try:
            results[exp] = completed(exp, root)
        except (FileNotFoundError, ValueError, KeyError) as e:
            errors.append(f'{exp}: {e}\n  python -m src.experiments --experiment {exp}  # add --resume if interrupted')
    if errors:
        raise ValueError('Missing/incomplete/invalid prerequisites; no selection made:\n' + '\n'.join(errors))
    return results


def comparable(results, varying):
    items = list(results.values())
    reference = items[0]
    def signature(s):
        return {k: v for k, v in s['config'].items() if k not in set(varying) | {'experiment', 'dependencies'}}
    for item in items[1:]:
        if signature(item) != signature(reference) or item['data_signature'] != reference['data_signature'] or item['source_signature'] != reference['source_signature']:
            raise ValueError('Experiments are not comparable: data, precision, seed, runtime or fixed settings differ.')


def selection_record(winner, summaries, root):
    summary = summaries[winner]
    return {'experiment': winner, 'config': summary['config'], 'data_signature': summary['data_signature'],
            'checkpoint': f'checkpoints/{winner}/best.pt',
            'checkpoint_sha256': summary['checkpoint_sha256']['best.pt'],
            'best_validation': summary['best_validation'],
            'source_summaries': {e: file_hash(root / 'results' / e / 'summary.json') for e in summaries}}


def select_lr(root=ROOT, persist=True):
    summaries = require(LR_IDS, root)
    comparable(summaries, {'lr'})
    winner = min(LR_IDS, key=lambda e: (summaries[e]['final']['train_loss'], LR_IDS.index(e)))
    result = selection_record(winner, summaries, root)
    result.update(criterion='lowest final training cross-entropy only; exact ties use listed experiment order',
                  lr=summaries[winner]['config']['lr'],
                  metric_winners={key: (min if 'loss' in key else max)(LR_IDS, key=lambda e: summaries[e]['final'][key])
                                  for key in ('train_loss', 'train_accuracy', 'val_loss', 'val_accuracy')})
    if persist:
        write_json(root / 'results/selected_lr.json', result)
    return result


def select_baseline(root=ROOT, persist=True):
    lr = select_lr(root, persist)
    summaries = require(BASELINES, root)
    comparable(summaries, {'scheduler', 'weight_decay'})
    if any(s['config']['lr'] != lr['lr'] or s['data_signature'] != lr['data_signature'] for s in summaries.values()):
        raise ValueError('Baseline experiments do not use the current selected LR/data')
    # Cross-stage runtime/seed/precision must also agree with the LR search.
    comparable({**summaries, 'lr_search': require([lr['experiment']], root)[lr['experiment']]},
               {'epochs', 'scheduler', 'weight_decay'})
    winner = min(BASELINES, key=lambda e: (validation_key(summaries[e]['best_validation']), e))
    result = selection_record(winner, summaries, root)
    result['criterion'] = 'highest validation accuracy over epochs; lowest validation CE, earliest epoch, experiment ID break ties'
    if persist:
        write_json(root / 'results/selected_baseline.json', result)
    return result


def select_final(root=ROOT, persist=True):
    baseline = select_baseline(root, persist)
    summaries = require([baseline['experiment'], 'mixup'], root)
    comparable(summaries, {'mixup_alpha'})
    if summaries['mixup']['config']['dependencies']['baseline'] != baseline['experiment']:
        raise ValueError('Mixup was trained with a stale baseline selection')
    winner = min(summaries, key=lambda e: (validation_key(summaries[e]['best_validation']), e))
    result = selection_record(winner, summaries, root)
    result['criterion'] = baseline['criterion']
    result['baseline'] = baseline['experiment']
    if persist:
        write_json(root / 'results/final_selection.json', result)
    return result


def resolve_config(cfg, root=ROOT):
    import copy
    cfg = copy.deepcopy(cfg)
    exp = cfg['experiment']
    if exp in LR_IDS:
        return cfg
    lr = select_lr(root)
    cfg['lr'] = lr['lr']
    cfg['dependencies'] = {'lr_experiment': lr['experiment'], 'lr_sources': lr['source_summaries']}
    if exp == 'mixup':
        base = select_baseline(root)
        cfg['scheduler'] = base['config']['scheduler']
        cfg['weight_decay'] = base['config']['weight_decay']
        cfg['dependencies'].update(baseline=base['experiment'], baseline_sources=base['source_summaries'])
    return cfg

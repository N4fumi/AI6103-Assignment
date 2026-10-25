"""Verify and export existing final measurements. Never trains or evaluates a model."""
import _bootstrap
import argparse
import csv
from datetime import datetime, timezone
import io
import math
from pathlib import Path

from src.selection import completed, select_final
from src.utils import (EXPERIMENTS, ROOT, atomic_bytes, digest, file_hash,
                       read_history, read_json, write_json)


def finalize():
    summaries = {experiment: completed(experiment) for experiment in EXPERIMENTS}
    selected = read_json(ROOT / 'results/final_selection.json')
    if selected != select_final(persist=False):
        raise ValueError('Saved final selection differs from validation-only recomputation')
    test_path = ROOT / 'results/final_test.json'
    test = read_json(test_path)
    expected = dict(experiment=selected['experiment'], epoch=selected['best_validation']['epoch'],
                    checkpoint_sha256=selected['checkpoint_sha256'], data_signature=selected['data_signature'],
                    selection_signature=digest(selected), count=selected['config']['split_counts'][2],
                    precision='fp32')
    if any(test.get(key) != value for key, value in expected.items()):
        raise ValueError('Final test provenance/count does not match the frozen selected model')
    if file_hash(ROOT / selected['checkpoint']) != selected['checkpoint_sha256']:
        raise ValueError('Final selected checkpoint hash changed')
    if not (math.isfinite(test['test_accuracy']) and math.isfinite(test['test_loss']) and
            0 <= test['test_accuracy'] <= 1 and test['test_loss'] >= 0):
        raise ValueError('Invalid final test metrics')
    control = read_json(ROOT / 'results/performance_optimization/control.json')
    backup = Path(control['backup'])
    previous = read_json(backup / 'completed_files.json')
    for entry in previous:
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError(f'Previously completed output changed: {entry["path"]}')
    old_history = read_history(backup / 'cosine_results/metrics.csv')
    if read_history(ROOT / 'results/cosine/metrics.csv')[:len(old_history)] != old_history:
        raise ValueError('Pre-optimization cosine history changed')
    for folder in ('src', 'configs'):
        for item in (backup / folder).rglob('*'):
            if item.is_file() and '__pycache__' not in item.parts:
                if file_hash(item) != file_hash(ROOT / folder / item.relative_to(backup / folder)):
                    raise ValueError(f'Frozen recipe changed: {item}')
    integrity = read_json(ROOT / 'data/manifest.json')['integrity']
    audit = {key: integrity[key] for key in ('duplicate_content_groups', 'cross_split_content_groups',
                                             'conflicting_label_groups', 'path_overlap', 'content_duplicate_policy')}
    rows = []
    for experiment, summary in summaries.items():
        row = dict(experiment=experiment, epochs=summary['final']['epoch'],
                   best_validation_epoch=summary['best_validation']['epoch'],
                   best_validation_accuracy=summary['best_validation']['val_accuracy'],
                   best_validation_loss=summary['best_validation']['val_loss'],
                   final_training_loss=summary['final']['train_loss'],
                   final_training_accuracy=summary['final']['train_accuracy'],
                   selected_for_test=experiment == selected['experiment'],
                   test_accuracy=test['test_accuracy'] if experiment == selected['experiment'] else '',
                   test_loss=test['test_loss'] if experiment == selected['experiment'] else '')
        rows.append(row)
    finished_at = datetime.now(timezone.utc).isoformat()
    result = dict(status='complete', verified_at=finished_at, accuracy_units='fraction',
                  experiments=rows, final_selection=selected, final_test=test,
                  final_test_sha256=file_hash(test_path), dataset_audit=audit,
                  preservation=dict(previously_completed_files_unchanged=len(previous),
                                    pre_optimization_cosine_epochs_unchanged=len(old_history),
                                    src_and_configs_unchanged=True),
                  figures=dict(png_count=len(list((ROOT / 'figures').rglob('*.png'))),
                               pdf_count=len(list((ROOT / 'figures').rglob('*.pdf')))),
                  limitations=['Single seed; no confidence interval from repeated runs.',
                               'Official split membership and labels retained despite duplicate content and conflicting labels.',
                               'Pre/post optimization timing reflects different execution implementations; numerical equivalence is separately audited.'])
    write_json(ROOT / 'results/final_summary.json', result)
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(ROOT / 'results/final_summary.csv', buf.getvalue().encode())
    execution_path = ROOT / 'results/final_test_execution.json'
    execution = read_json(execution_path)
    execution.update(status='complete', finished_at=finished_at,
                     result_sha256=result['final_test_sha256'], checkpoint_sha256=test['checkpoint_sha256'])
    write_json(execution_path, execution)
    state_path = ROOT / 'results/continuation.json'
    state = read_json(state_path)
    state.update(status='complete', updated_at=finished_at, current_experiment=None, child_pid=None,
                 final_test=str(test_path), final_summary=str(ROOT / 'results/final_summary.json'),
                 final_test_accuracy=test['test_accuracy'], final_experiment=test['experiment'],
                 note='All eight experiments, validation-only final selection, one authorized hold-out evaluation, numeric summaries and plots completed; no report prose generated.')
    write_json(state_path, state)
    control.update(status='complete', completed_at=finished_at)
    write_json(ROOT / 'results/performance_optimization/control.json', control)
    print(f"Complete: {test['experiment']} epoch {test['epoch']}; validation={selected['best_validation']['val_accuracy']:.4%}; test={test['test_accuracy']:.4%} ({test['count']} images).")
    print(f'Preserved {len(previous)} completed files and {len(old_history)} earlier cosine epochs; frozen recipe unchanged.')
    return result


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    finalize()

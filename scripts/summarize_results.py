"""Export only actual saved measurements; missing results stay missing."""
import _bootstrap
import argparse
import csv
import io
import json

from src.selection import completed, select_lr
from src.utils import EXPERIMENTS, ROOT, atomic_bytes, read_history, read_json, write_json


def summarize(root=ROOT):
    rows = []
    for exp in EXPERIMENTS:
        row = {'experiment': exp, 'status': 'missing', 'epochs_recorded': 0}
        directory = root / 'results' / exp
        if directory.exists():
            row['status'] = 'incomplete'
        if (directory / 'status.json').is_file():
            row['run_status'] = read_json(directory / 'status.json').get('status', 'unknown')
        if (directory / 'metrics.csv').is_file():
            history = read_history(directory / 'metrics.csv')
            row.update(status='incomplete', epochs_recorded=len(history))
            if history:
                last = history[-1]
                row.update({k: last[k] for k in ('train_loss', 'val_loss', 'train_accuracy', 'val_accuracy')})
                row['accuracy_gap'] = last['train_accuracy'] - last['val_accuracy']
        if (directory / 'config.json').is_file():
            cfg = read_json(directory / 'config.json')
            row.update({k: cfg[k] for k in ('lr', 'scheduler', 'weight_decay', 'mixup_alpha', 'precision', 'seed')})
        try:
            summary = completed(exp, root)
            row.update(status='complete', best_validation_accuracy=summary['best_validation']['val_accuracy'],
                       best_validation_epoch=summary['best_validation']['epoch'])
        except (FileNotFoundError, ValueError, KeyError) as error:
            if row['status'] != 'missing':
                row['detail'] = str(error)
        rows.append(row)
    result = {'experiments': rows, 'accuracy_units': 'fraction',
              'gap_definition': 'online augmented training accuracy minus deterministic validation accuracy; not a clean-set generalization estimate'}
    try:
        result['lr_selection'] = select_lr(root)
    except ValueError as error:
        result['lr_selection_pending'] = str(error)
    write_json(root / 'results/comparison.json', result)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(root / 'results/comparison.csv', buf.getvalue().encode())
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = summarize()
    for r in result['experiments']:
        values = ', '.join(f'{k}={r[k]:.5f}' for k in ('train_loss', 'val_loss', 'train_accuracy', 'val_accuracy') if k in r)
        print(f'{r["experiment"]:9s} {r["status"]:10s} epochs={r["epochs_recorded"]} {values}')
    if 'lr_selection_pending' in result:
        print(result['lr_selection_pending'])
    else:
        print('LR selected by final training loss:', result['lr_selection']['lr'])
    from generate_report_tables import generate_tables
    generate_tables(result)


if __name__ == '__main__':
    main()

"""Audit preserved outputs, then verify the actual resumed epochs against the probe."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1] / 'outputs/AI6103_Assignment'
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
from src.utils import file_hash, load_checkpoint, read_history, read_json, write_json
from verify_optimization import fingerprint

parser = argparse.ArgumentParser()
parser.add_argument('--live', action='store_true')
args = parser.parse_args()
folder = PROJECT / 'results/performance_optimization'
control = read_json(folder / 'control.json')
backup = Path(control['backup'])
evidence = read_json(folder / 'equivalence.json')
assert evidence['status'] == 'passed'
assert evidence['adapter_sha256'] == file_hash(PROJECT / 'scripts/optimized_runtime.py')
old_history = read_history(backup / 'cosine_results/metrics.csv')
entries = read_json(backup / 'completed_files.json')
for item in entries:
    assert file_hash(item['path']) == item['sha256'], f'Completed result changed: {item["path"]}'
for name in ('src', 'configs'):
    for item in (backup / name).rglob('*'):
        if item.is_file() and '__pycache__' not in item.parts:
            assert file_hash(item) == file_hash(PROJECT / name / item.relative_to(backup / name))
active_history = read_history(PROJECT / 'results/cosine/metrics.csv')
assert active_history[:len(old_history)] == old_history
preservation = dict(status='passed', backup=str(backup), preserved_active_epochs=len(old_history),
                    completed_experiments=['lr_05', 'lr_005', 'lr_0001', 'constant'],
                    immutable_files_verified=len(entries), src_and_configs_unchanged=True,
                    active_history_prefix_identical=True,
                    verified_at=datetime.now(timezone.utc).isoformat())
if not args.live:
    assert file_hash(PROJECT / 'checkpoints/cosine/last.pt') == evidence['checkpoint_sha256']
    control.update(status='migrating', evidence=str(folder / 'equivalence.json'))
    write_json(folder / 'control.json', control)
else:
    expected = evidence['optimized']
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        status = read_json(PROJECT / 'results/cosine/status.json')
        if status['status'] in ('failed', 'interrupted'):
            raise RuntimeError(status)
        if status.get('completed_epochs', 0) >= expected[-1]['epoch']:
            break
        time.sleep(2)
    else:
        raise RuntimeError('The resumed run did not complete its first two epochs in time.')
    active_history = read_history(PROJECT / 'results/cosine/metrics.csv')
    assert active_history[:len(old_history)] == old_history
    for row in expected:
        actual = active_history[row['epoch'] - 1]
        assert [actual['train_loss'], actual['train_accuracy']] == row['train']
        assert [actual['val_loss'], actual['val_accuracy']] == row['validation']
    checkpoint = load_checkpoint(PROJECT / 'checkpoints/cosine/last.pt')
    preservation['resumed_metrics_match_probe'] = True
    preservation['actual_checkpoint_epoch'] = checkpoint['epoch']
    if checkpoint['epoch'] == expected[-1]['epoch']:
        comparisons = {'model': 'model_sha256', 'optimizer': 'optimizer_sha256',
                       'scheduler': 'scheduler_sha256', 'rng': 'rng_sha256'}
        for field, key in comparisons.items():
            assert fingerprint(checkpoint[field]) == expected[-1][key], f'Live {field} differs from equivalence probe'
        preservation['actual_state_bitwise_matches_probe'] = True
    control.update(status='active_optimized', resumed_from_epoch=len(old_history),
                   runtime='scripts/train_optimized.py', verified_at=datetime.now(timezone.utc).isoformat())
    write_json(folder / 'control.json', control)
write_json(folder / 'preservation.json', preservation)
print(preservation, flush=True)

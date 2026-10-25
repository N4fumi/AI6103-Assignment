"""Finish an intentional optimization handoff after the old exec job has exited."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1] / 'outputs/AI6103_Assignment'
sys.path.insert(0, str(PROJECT))
from src.utils import file_hash, load_checkpoint, read_history, read_json, write_json

state_path = PROJECT / 'results/continuation.json'
state = read_json(state_path)
status = read_json(PROJECT / 'results/cosine/status.json')
checkpoint = load_checkpoint(PROJECT / 'checkpoints/cosine/last.pt')
history = read_history(PROJECT / 'results/cosine/metrics.csv')
assert checkpoint['epoch'] == status['completed_epochs'] == len(history)
assert checkpoint['history'] == history
assert state['current_experiment'] == 'cosine'
assert read_json(PROJECT / 'results/performance_optimization/loader_equivalence.json')['status'] == 'passed'
backup = PROJECT / 'results/performance_optimization' / ('backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir()
for source, name in (('results/cosine', 'cosine_results'), ('checkpoints/cosine', 'cosine_checkpoints'),
                     ('src', 'src'), ('configs', 'configs')):
    shutil.copytree(PROJECT / source, backup / name)
shutil.copy2(state_path, backup / 'continuation.before.json')
immutable = []
for experiment in state['completed_experiments']:
    for folder in ('results', 'checkpoints'):
        for item in sorted((PROJECT / folder / experiment).rglob('*')):
            if item.is_file():
                immutable.append(dict(path=str(item), sha256=file_hash(item)))
write_json(backup / 'completed_files.json', immutable)
for name in ('.continuation.lock', '.cosine.lock'):
    item = PROJECT / 'results' / name
    if item.exists():
        shutil.copy2(item, backup / name)
        item.unlink()
handoff = dict(status='benchmarking', backup=str(backup), completed_epoch=checkpoint['epoch'],
               stopped_training_pid=18868, stopped_watcher_pid=30892,
               process_exit='Old exec job ended with its watcher; last fully committed epoch verified.',
               checkpoint=str(backup / 'cosine_checkpoints/last.pt'),
               updated_at=datetime.now(timezone.utc).isoformat())
write_json(PROJECT / 'results/performance_optimization/control.json', handoff)
state.update(status='optimization_handoff', completed_epochs=checkpoint['epoch'], child_pid=None,
             watcher_pid=None, updated_at=handoff['updated_at'])
write_json(state_path, state)
print(handoff, flush=True)

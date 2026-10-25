"""Save the actual post-resume measurements and audit summary."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1] / 'outputs/AI6103_Assignment'
sys.path.insert(0, str(PROJECT))
from src.utils import read_history, read_json, write_json

folder = PROJECT / 'results/performance_optimization'
probe = read_json(folder / 'equivalence.json')
preservation = read_json(folder / 'preservation.json')
history = read_history(PROJECT / 'results/cosine/metrics.csv')
sample = [row for row in history if 143 <= row['epoch'] <= 148]
assert len(sample) == 6 and probe['status'] == preservation['status'] == 'passed'
profiles = [json.loads(line) for line in (PROJECT / 'results/cosine/performance.jsonl').read_text().splitlines()]
profiles = [row for row in profiles if 143 <= row['epoch'] <= 148]
timings = [row['epoch_seconds'] for row in sample]
saving = statistics.mean(row['checkpoint_seconds'] for row in profiles)
before = statistics.mean(row['epoch_seconds'] for row in history if 118 <= row['epoch'] <= 127)
summary = dict(status='active_optimized', recorded_at=datetime.now(timezone.utc).isoformat(),
               previous_epoch_seconds=before, sample_epochs=[row['epoch'] for row in sample],
               epoch_seconds_mean=statistics.mean(timings), epoch_seconds_min=min(timings),
               epoch_seconds_max=max(timings), checkpoint_seconds_mean=saving,
               approx_epoch_wall_seconds=statistics.mean(timings)+saving,
               training_validation_speedup=before/statistics.mean(timings),
               same_checkpoint_probe_baseline=probe['baseline'][-1]['epoch_seconds'],
               same_checkpoint_probe_optimized=probe['optimized'][-1]['epoch_seconds'],
               exact_equivalence_passed=True, preserved_old_epochs=141,
               completed_experiment_files_unchanged=preservation['immutable_files_verified'],
               actual_resumed_state_bitwise_matches_probe=preservation['actual_state_bitwise_matches_probe'],
               note='Epoch timer includes training and FP32 validation; checkpoint saving is timed separately. First epoch after worker startup is excluded from the steady-state sample.')
write_json(folder / 'summary.json', summary)
print(json.dumps(summary, indent=2), flush=True)

"""User-authorized sequential continuation; never evaluates the hold-out test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE / 'outputs/AI6103_Assignment'
sys.path.insert(0, str(PROJECT))

from src.selection import completed
from src.utils import EXPERIMENTS, digest, file_hash, read_json, write_json

STATE = PROJECT / 'results/continuation.json'
LOGS = PROJECT / 'results/console'
LOCK = PROJECT / 'results/.continuation.lock'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def inputs_signature():
    return digest({p.relative_to(PROJECT).as_posix(): file_hash(p)
                   for folder, pattern in [('src', '*.py'), ('configs', '*.yaml')]
                   for p in sorted((PROJECT / folder).glob(pattern))})


class Continuation:
    def __init__(self, optimized=False, resume_continuation=False):
        self.optimized = optimized
        self.authorized_resume = None
        self.signature = inputs_signature()
        self.state = {'status': 'starting', 'watcher_pid': os.getpid(), 'python': sys.executable,
                      'project': str(PROJECT), 'source_config_signature': self.signature,
                      'created_at': utc_now(), 'updated_at': utc_now(),
                      'completed_experiments': [], 'current_experiment': None,
                      'test_evaluation_authorized': True,
                      'test_evaluation_policy': 'A follow-up in this chat may evaluate the frozen final model after training if the user has not returned or changed instructions. This watcher does not itself evaluate test data.'}
        if resume_continuation:
            previous = read_json(STATE)
            control = read_json(PROJECT / 'results/performance_optimization/control.json')
            evidence = read_json(PROJECT / 'results/performance_optimization/equivalence.json')
            if (not optimized or previous['status'] != 'optimization_handoff' or
                    previous['source_config_signature'] != self.signature or evidence['status'] != 'passed' or
                    control['status'] != 'migrating'):
                raise RuntimeError('A verified, user-authorized optimization handoff is required.')
            self.authorized_resume = previous['current_experiment']
            current = PROJECT / 'checkpoints' / self.authorized_resume / 'last.pt'
            if file_hash(current) != evidence['checkpoint_sha256']:
                raise RuntimeError('Current checkpoint differs from the backed-up, verified checkpoint.')
            self.state.update(created_at=previous['created_at'], previous_watcher_pid=control['stopped_watcher_pid'],
                              optimization_evidence='results/performance_optimization/equivalence.json',
                              execution_adapter='scripts/train_optimized.py',
                              test_evaluation_authorized=previous['test_evaluation_authorized'])

    def update(self, **changes):
        self.state.update(changes, updated_at=utc_now())
        write_json(STATE, self.state)
        print(utc_now(), json.dumps(changes, ensure_ascii=True), flush=True)

    def check_inputs(self):
        if inputs_signature() != self.signature:
            raise RuntimeError('Source/configuration changed during the authorized sequence; stopped without launching another run.')

    def wait_existing(self, experiment):
        lock = PROJECT / 'results' / f'.{experiment}.lock'
        status_file = PROJECT / 'results' / experiment / 'status.json'
        observed_epoch = None
        last_progress = time.monotonic()
        while lock.exists():
            status = read_json(status_file) if status_file.exists() else {}
            epoch = status.get('completed_epochs', 0)
            if epoch != observed_epoch:
                observed_epoch = epoch
                last_progress = time.monotonic()
                self.update(status='waiting_for_existing_run', current_experiment=experiment,
                            completed_epochs=epoch, existing_status=status.get('status', 'unknown'))
            if status.get('status') in ('failed', 'interrupted'):
                raise RuntimeError(f'{experiment} stopped: {status}. No automatic restart or overwrite was attempted.')
            if time.monotonic() - last_progress > 1800:
                raise RuntimeError(f'{experiment}: no completed-epoch progress for 30 minutes. The existing process was left untouched.')
            time.sleep(5)
        # A lock disappearing alone is not evidence of successful completion.
        completed(experiment)

    def launch_one(self, experiment, resume=False):
        self.check_inputs()
        other_locks = [p.name for p in (PROJECT / 'results').glob('.*.lock') if p != LOCK]
        if other_locks:
            raise RuntimeError(f'Another experiment is active: {other_locks}. Refusing concurrent GPU training.')
        stdout = LOGS / f'{experiment}.stdout.log'
        stderr = LOGS / f'{experiment}.stderr.log'
        if not resume and (stdout.exists() or stderr.exists()):
            raise FileExistsError(f'Console logs already exist for {experiment}; refusing an ambiguous restart.')
        command = ([sys.executable, '-u', 'scripts/train_optimized.py'] if self.optimized else
                   [sys.executable, '-u', '-m', 'src.experiments']) + ['--experiment', experiment]
        if resume:
            command.append('--resume')
        with stdout.open('a' if resume else 'x', encoding='utf-8') as out, stderr.open('a' if resume else 'x', encoding='utf-8') as err:
            if resume:
                out.write('\n=== User-authorized verified optimization resume; existing history preserved ===\n')
                out.flush()
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUNBUFFERED'] = '1'
            process = subprocess.Popen(command, cwd=PROJECT, stdout=out, stderr=err, env=env,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.update(status='training', current_experiment=experiment, child_pid=process.pid,
                        command=command, stdout=str(stdout), stderr=str(stderr), completed_epochs=0)
            last_epoch = None
            while process.poll() is None:
                status_file = PROJECT / 'results' / experiment / 'status.json'
                if status_file.exists():
                    status = read_json(status_file)
                    epoch = status.get('completed_epochs', 0)
                    if epoch != last_epoch:
                        last_epoch = epoch
                        self.update(completed_epochs=epoch)
                time.sleep(5)
            if process.returncode != 0:
                raise RuntimeError(f'{experiment} exited with {process.returncode}. Inspect {stderr}; no subsequent experiment launched.')
        completed(experiment)

    def utility(self, arguments):
        self.check_inputs()
        self.update(status='selection_or_export', command=[sys.executable, *arguments], child_pid=None)
        result = subprocess.run([sys.executable, *arguments], cwd=PROJECT, check=False)
        if result.returncode:
            raise RuntimeError(f'Utility failed ({result.returncode}): {arguments}')

    def run(self):
        for experiment in EXPERIMENTS:
            self.check_inputs()
            summary_file = PROJECT / 'results' / experiment / 'summary.json'
            experiment_lock = PROJECT / 'results' / f'.{experiment}.lock'
            if experiment_lock.exists():
                self.wait_existing(experiment)
            elif summary_file.exists():
                completed(experiment)
            elif (PROJECT / 'results' / experiment).exists() or (PROJECT / 'checkpoints' / experiment).exists():
                if experiment != self.authorized_resume:
                    raise RuntimeError(f'{experiment} has partial outputs without an active lock. No overwrite or automatic restart is authorized.')
                self.authorized_resume = None
                self.launch_one(experiment, resume=True)
            else:
                self.launch_one(experiment)
            self.state['completed_experiments'].append(experiment)
            self.update(status='experiment_complete', current_experiment=experiment, child_pid=None)
            if experiment == 'lr_0001':
                self.utility(['-m', 'src.experiments', '--select-lr'])
            elif experiment == 'wd_1e4':
                self.utility(['-m', 'src.experiments', '--select-baseline'])
        self.utility(['-m', 'src.experiments', '--select-final'])
        self.utility(['scripts/summarize_results.py'])
        self.utility(['scripts/generate_figures.py'])
        self.utility(['scripts/generate_report_tables.py'])
        self.update(status='training_complete_pending_conditional_test', current_experiment=None, child_pid=None,
                    final_selection=str(PROJECT / 'results/final_selection.json'),
                    note='Training/validation selection and numeric/figure exports complete. A heartbeat must check latest human instructions before the conditionally authorized hold-out evaluation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Print the sequence without launching anything')
    parser.add_argument('--optimized', action='store_true', help='Use the separately verified execution adapter')
    parser.add_argument('--resume-continuation', action='store_true', help='Only the explicitly authorized optimization handoff')
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({'experiments': EXPERIMENTS, 'test_evaluation': False, 'project': str(PROJECT)}, indent=2))
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    runner = Continuation(args.optimized, args.resume_continuation)
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as stream:
        stream.write(str(os.getpid()))
    try:
        if STATE.exists() and not args.resume_continuation:
            raise FileExistsError('A continuation state already exists; inspect it before starting another watcher.')
        runner.update(status='starting')
        runner.run()
    except (Exception, KeyboardInterrupt) as error:
        runner.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

"""Explicit single-experiment entry point for the verified execution adapter."""
import argparse
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.utils import EXPERIMENTS, load_checkpoint, load_config, read_json
    from src.selection import resolve_config
    from src.train import run
    from optimized_runtime import install
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = resolve_config(load_config(args.experiment))
    last = ROOT / 'checkpoints' / args.experiment / 'last.pt'
    start = load_checkpoint(last)['epoch'] + 1 if args.resume else 1
    evidence = read_json(ROOT / 'results/performance_optimization/equivalence.json')
    runtime = install(cfg, start, evidence)
    print('Verified execution adapter:', runtime, flush=True)
    run(cfg, resume=args.resume)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

"""Manual, single-experiment entry point with explicit prerequisites."""
import argparse
import json

from .selection import resolve_config, select_baseline, select_final, select_lr
from .utils import EXPERIMENTS, load_config, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--experiment', choices=EXPERIMENTS)
    actions.add_argument('--validate-configs', action='store_true')
    actions.add_argument('--select-lr', action='store_true')
    actions.add_argument('--select-baseline', action='store_true')
    actions.add_argument('--select-final', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--resume', action='store_true')
    modes.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Resolve prerequisites/config only. Never train.')
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--precision', choices=('fp32', 'amp_fp16', 'amp_bf16'))
    parser.add_argument('--workers', type=int)
    parser.add_argument('--data-root')
    parser.add_argument('--memory-fraction', type=float)
    args = parser.parse_args()
    try:
        if args.validate_configs:
            for exp in EXPERIMENTS:
                cfg = load_config(exp)
                print(f'{exp}: valid; lr={cfg["lr"] if cfg["lr"] else "requires completed LR selection"}, epochs={cfg["epochs"]}')
            return
        for option, function in ((args.select_lr, select_lr), (args.select_baseline, select_baseline), (args.select_final, select_final)):
            if option:
                print(json.dumps(function(), indent=2))
                return
        cfg = load_config(args.experiment)
        for key in ('device', 'precision', 'workers', 'data_root', 'memory_fraction'):
            value = getattr(args, key)
            if value is not None:
                cfg[key] = value
        cfg = resolve_config(cfg)
        validate_config(cfg)
        if args.dry_run:
            print(json.dumps(cfg, indent=2))
            return
        from .train import run
        run(cfg, resume=args.resume, overwrite=args.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as e:
        parser.exit(2, f'ERROR: {e}\n')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

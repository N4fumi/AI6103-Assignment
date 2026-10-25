"""Compute channel population moments on resized, unaugmented training images."""
import _bootstrap
import argparse
import json

from src.dataset import compute_statistics
from src.utils import load_config, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root')
    args = parser.parse_args()
    cfg = load_config()
    if args.data_root:
        cfg['data_root'] = args.data_root
    if path(cfg['statistics']).exists():
        parser.exit(2, 'Statistics already exist; reuse them. Archive the file explicitly before recomputing.\n')
    try:
        print(json.dumps(compute_statistics(cfg), indent=2))
    except (ValueError, FileNotFoundError) as e:
        parser.exit(2, f'ERROR: {e}\n')


if __name__ == '__main__':
    main()

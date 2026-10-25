"""Inspect layout first. Explicit preparation preserves official splits."""
import _bootstrap
import argparse
import json

from src.dataset import (create_manifest, inventory, load_data_metadata, make_loader, validate_manifest)
from src.utils import load_config, path, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root')
    parser.add_argument('--prepare', action='store_true', help='Create a manifest from official splits')
    parser.add_argument('--create-splits', action='store_true', help='Explicitly authorize deterministic splitting only when no official split exists')
    parser.add_argument('--official-csv', help='Explicit CSV adapter: path,label,split')
    parser.add_argument('--verify-loaders', action='store_true', help='Read one train and validation minibatch; never evaluate test')
    parser.add_argument('--verify-hashes', action='store_true')
    args = parser.parse_args()
    cfg = load_config()
    if args.data_root:
        cfg['data_root'] = args.data_root
    try:
        info = inventory(path(cfg['data_root']))
        if args.prepare or args.create_splits or args.official_csv:
            _, info['manifest'] = create_manifest(cfg, args.create_splits, args.official_csv)
        elif path(cfg['manifest']).is_file():
            manifest = read_json(path(cfg['manifest']))
            info['manifest'] = validate_manifest(manifest, cfg['split_counts'], cfg['num_classes'], True, args.verify_hashes)
        if args.verify_loaders:
            manifest, stats = load_data_metadata(cfg, args.verify_hashes)
            cfg['workers'] = 0
            info['loader_shapes'] = {}
            for split in ('train', 'val'):
                x, y = next(iter(make_loader(cfg, manifest, stats, split)))
                assert tuple(x.shape[1:]) == (3, 100, 100)
                assert y.min() >= 0 and y.max() < cfg['num_classes']
                info['loader_shapes'][split] = {'images': list(x.shape), 'labels': list(y.shape)}
        write_json(path('data/inspection.json'), info)
        print(json.dumps(info, indent=2))
        if not info['image_count']:
            parser.exit(2, 'No images found. Copy the extracted dataset into data/raw; do not train yet.\n')
    except (FileNotFoundError, ValueError, FileExistsError) as e:
        parser.exit(2, f'ERROR: {e}\n')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

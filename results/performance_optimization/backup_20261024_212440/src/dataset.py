"""Manifest-based RGB input pipeline. No dataset is downloaded on import."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path
import random

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .utils import digest, file_hash, path, read_json, write_json

SPLITS = ('train', 'val', 'test')
ALIASES = {'training': 'train', 'valid': 'val', 'validation': 'val', 'testing': 'test'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def normalize_split(s):
    s = s.lower().strip()
    s = ALIASES.get(s, s)
    if s not in SPLITS:
        raise ValueError(f'Unknown split {s!r}')
    return s


def image_files(root):
    return sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in IMAGE_EXT)


def inventory(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory absent: {root}. See data/README.md.')
    images = image_files(root)
    return {'root': str(root), 'image_count': len(images),
            'top_level': sorted(p.name for p in root.iterdir()),
            'metadata_files': [str(p.relative_to(root)) for p in root.rglob('*')
                               if p.suffix.lower() in ('.csv', '.json', '.txt', '.npy', '.npz', '.pkl', '.pt')],
            'example_paths': [str(p.relative_to(root)) for p in images[:12]]}


def stratified_split(records, counts, seed):
    """Exact counts with largest-remainder allocations within each class."""
    if sum(counts) != len(records) or min(counts) <= 0:
        raise ValueError(f'Expected {counts} (total {sum(counts)}), found {len(records)} labeled images')
    groups = defaultdict(list)
    for row in sorted(records, key=lambda r: r['path']):
        groups[row['label']].append(dict(row))
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    for split, requested in zip(SPLITS[:-1], counts[:-1]):
        remaining = sum(map(len, groups.values()))
        quotas = {k: requested * len(v) / remaining for k, v in groups.items()}
        alloc = {k: int(v) for k, v in quotas.items()}
        order = sorted(groups, key=lambda k: (-(quotas[k] - alloc[k]), k))
        for k in order[:requested - sum(alloc.values())]:
            alloc[k] += 1
        for k in sorted(groups):
            n = alloc[k]
            result.extend(dict(r, split=split) for r in groups[k][:n])
            groups[k] = groups[k][n:]
    result.extend(dict(r, split='test') for k in sorted(groups) for r in groups[k])
    return sorted(result, key=lambda r: (SPLITS.index(r['split']), r['path']))


def csv_records(filename, root, split=None):
    """Accept path,label,split or Kaggle id,breed with image ID resolution."""
    files = image_files(root)
    by_stem = defaultdict(list)
    for p in files:
        by_stem[p.stem].append(p)
    rows = []
    with Path(filename).open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            label = row.get('label') or row.get('breed') or row.get('class')
            ref = row.get('path') or row.get('filename') or row.get('id')
            if label is None or ref is None:
                raise ValueError(f'{filename}: requires path,label[,split] or id,breed columns')
            p = (root / ref).resolve()
            if not p.is_file():
                matches = by_stem[Path(ref).stem]
                if len(matches) != 1:
                    raise ValueError(f'Cannot uniquely resolve image {ref!r}: {len(matches)} matches')
                p = matches[0].resolve()
            try:
                rel = p.relative_to(root.resolve()).as_posix()
            except ValueError as e:
                raise ValueError(f'Image path escapes dataset root: {ref}') from e
            item = {'path': rel, 'label': str(label).strip()}
            s = row.get('split') or split
            if s:
                item['split'] = normalize_split(s)
            rows.append(item)
    return rows


def discover_records(root, official_csv=None):
    """Preserve explicit CSV or split/class/image layout. Never infer a new split here."""
    root = Path(root).resolve()
    info = inventory(root)
    if not info['image_count']:
        raise FileNotFoundError(f'No images under {root}. Extract the dataset first.')
    if official_csv:
        return csv_records(Path(official_csv).resolve(), root), 'provided_csv'
    split_csv = root / 'splits.csv'
    if split_csv.is_file():
        return csv_records(split_csv, root), 'official_splits_csv'
    split_files = {}
    for s in SPLITS:
        names = [s] + [k for k, v in ALIASES.items() if v == s]
        matches = [root / filename for n in names for filename in (f'{n}.csv', f'{n}_labels.csv')
                   if (root / filename).is_file()]
        if len(matches) > 1:
            raise ValueError(f'Ambiguous split CSV files for {s}: {matches}')
        if matches:
            split_files[s] = matches[0]
    if split_files:
        if set(split_files) != set(SPLITS):
            raise ValueError('Partial split CSVs found. Supply all three labeled split CSVs or --official-csv.')
        return [r for s in SPLITS for r in csv_records(split_files[s], root, s)], 'official_split_csvs'
    # Detect supported split folders even when the archive adds one enclosing folder.
    split_dirs = {}
    for p in [root] + sorted(p for p in root.iterdir() if p.is_dir()):
        candidates = {}
        for d in p.iterdir():
            if d.is_dir() and (d.name.lower() in SPLITS or d.name.lower() in ALIASES):
                s = normalize_split(d.name)
                if s in candidates:
                    raise ValueError(f'Multiple folders map to {s}')
                candidates[s] = d
        if set(candidates) == set(SPLITS):
            if split_dirs:
                raise ValueError('Multiple official split folder sets detected; point data_root to the intended one')
            split_dirs = candidates
    if split_dirs:
        records = []
        for s, directory in split_dirs.items():
            for p in image_files(directory):
                rel = p.relative_to(directory)
                if len(rel.parts) < 2:
                    raise ValueError('Split folders contain flat images; provide a labeled official CSV.')
                records.append({'path': p.relative_to(root).as_posix(), 'label': rel.parts[0], 'split': s})
        return records, 'official_split_folders'
    # Unknown split indices are a reason to stop, never to overwrite the official protocol.
    suspicious = [s for s in info['metadata_files'] if
                  any(k in Path(s).stem.lower() for k in ('split', 'valid', 'test', 'index', 'indices'))
                  and 'sample_submission' not in s.lower()]
    if suspicious:
        raise ValueError(f'Unrecognized possible split metadata: {suspicious}. Convert to path,label,split CSV and use --official-csv.')
    labels = list(root.rglob('labels.csv'))
    if len(labels) == 1:
        return csv_records(labels[0], root), 'unsplit_labels_csv'
    files = image_files(root)
    parent = root
    while True:
        children = [p for p in parent.iterdir() if p.is_dir() and image_files(p)]
        if len(children) == 1 and all(p.is_relative_to(children[0]) for p in files):
            parent = children[0]
        else:
            break
    if all(len(p.relative_to(parent).parts) == 2 for p in files):
        return [{'path': p.relative_to(root).as_posix(), 'label': p.parent.name} for p in files], 'unsplit_class_folders'
    raise ValueError('Unsupported or ambiguous layout. See data/README.md for the explicit CSV adapter.')


def validate_manifest(manifest, expected_counts=None, expected_classes=None, verify_files=False, verify_hashes=False):
    rows = manifest['records']
    classes = manifest['classes']
    if expected_classes is not None and len(classes) != expected_classes:
        raise ValueError(f'Expected {expected_classes} classes, found {len(classes)}')
    if classes != sorted(set(classes)):
        raise ValueError('Class vocabulary must be sorted and unique')
    counts = Counter(r['split'] for r in rows)
    if expected_counts is not None and [counts[s] for s in SPLITS] != list(expected_counts):
        raise ValueError(f'Split counts {dict(counts)} do not match {expected_counts}')
    seen_paths, content_groups = set(), defaultdict(list)
    root = Path(manifest['root']).resolve()
    for r in rows:
        normalize_split(r['split'])
        if r['label'] not in classes or r['path'] in seen_paths:
            raise ValueError(f'Invalid label or overlapping image path: {r}')
        seen_paths.add(r['path'])
        p = (root / r['path']).resolve()
        if not p.is_relative_to(root):
            raise ValueError('Manifest path escapes dataset root')
        if verify_files and not p.is_file():
            raise FileNotFoundError(p)
        h = r.get('sha256')
        if h:
            content_groups[h].append(r)
            if verify_hashes and file_hash(p) != h:
                raise ValueError(f'Image changed after manifest creation: {p}')
    duplicates = [group for group in content_groups.values() if len(group) > 1]
    conflicting = [group for group in duplicates if len({r['label'] for r in group}) > 1]
    cross_split = [group for group in duplicates if len({r['split'] for r in group}) > 1]
    official = manifest.get('source', '').startswith(('official_', 'provided_csv')) and 'stratified' not in manifest.get('source', '')
    if (cross_split or conflicting) and not (official and manifest.get('content_duplicate_policy') == 'preserve_official'):
        raise ValueError(f'{len(cross_split)} cross-split duplicate groups; {len(conflicting)} conflicting-label groups. Only an explicitly audited official dataset can preserve these defects.')
    for s in SPLITS:
        if {r['label'] for r in rows if r['split'] == s} != set(classes):
            raise ValueError(f'{s} does not contain every class')
    return {'counts': {s: counts[s] for s in SPLITS}, 'num_classes': len(classes),
            'path_overlap': 0, 'duplicate_content_groups': len(duplicates), 'cross_split_content_groups': len(cross_split),
            'conflicting_label_groups': len(conflicting),
            'content_duplicate_policy': manifest.get('content_duplicate_policy', 'error'),
            'duplicate_groups': [[{'path': r['path'], 'split': r['split'], 'label': r['label']} for r in group] for group in duplicates],
            'class_distribution': {s: dict(sorted(Counter(r['label'] for r in rows if r['split'] == s).items())) for s in SPLITS}}


def create_manifest(cfg, allow_new_split=False, official_csv=None):
    target = path(cfg['manifest'])
    if target.exists():
        raise FileExistsError(f'{target} already exists. Reuse it; archive it explicitly to replace it.')
    root = path(cfg['data_root'])
    rows, source = discover_records(root, official_csv)
    has_split = ['split' in r for r in rows]
    if any(has_split) and not all(has_split):
        raise ValueError('Only some records have a split')
    if not all(has_split):
        if not allow_new_split:
            raise ValueError('No official split found. Review inspection, then explicitly pass --create-splits.')
        rows = stratified_split(rows, cfg['split_counts'], cfg['seed'])
        source += '_stratified_largest_remainder'
    for r in rows:
        r['sha256'] = file_hash(root / r['path'])
    manifest = {'version': 1, 'root': str(root), 'source': source, 'seed': cfg['seed'],
                'content_duplicate_policy': cfg['content_duplicate_policy'],
                'classes': sorted({r['label'] for r in rows}), 'records': rows}
    report = validate_manifest(manifest, cfg['split_counts'], cfg['num_classes'], verify_files=True)
    manifest['signature'] = digest({'classes': manifest['classes'], 'records': rows})
    manifest['integrity'] = report
    write_json(target, manifest)
    if report['cross_split_content_groups']:
        print(f'WARNING: preserving official data with {report["cross_split_content_groups"]} cross-split duplicate-content groups and {report["conflicting_label_groups"]} conflicting-label groups. See manifest integrity audit.')
    return manifest, report


def load_data_metadata(cfg, verify_hashes=False):
    manifest = read_json(path(cfg['manifest']))
    if path(cfg['data_root']) != Path(manifest['root']).resolve():
        raise ValueError('data_root differs from manifest root. Inspect or relocate the manifest explicitly.')
    report = validate_manifest(manifest, cfg['split_counts'], cfg['num_classes'], verify_files=True, verify_hashes=verify_hashes)
    if manifest.get('content_duplicate_policy', 'error') != cfg['content_duplicate_policy']:
        raise ValueError('Manifest duplicate policy differs from configuration')
    if report['cross_split_content_groups']:
        print(f'WARNING: official data have {report["cross_split_content_groups"]} cross-split duplicate groups and {report["conflicting_label_groups"]} conflicting-label groups; original membership/labels are preserved and audited.')
    signature = digest({'classes': manifest['classes'], 'records': manifest['records']})
    if signature != manifest['signature']:
        raise ValueError('Manifest signature mismatch')
    stats = read_json(path(cfg['statistics']))
    if stats['manifest_signature'] != signature or stats['image_size'] != cfg['image_size'] or stats['split'] != 'train':
        raise ValueError('Statistics do not match the current training manifest/preprocessing')
    if stats['count'] != cfg['split_counts'][0] or len(stats['mean']) != 3 or len(stats['std']) != 3:
        raise ValueError('Invalid training statistics')
    if not all(np.isfinite(v) for v in stats['mean'] + stats['std']) or min(stats['std']) <= 0:
        raise ValueError('Nonfinite or zero RGB standard deviation')
    return manifest, stats


def make_transform(cfg, stats, training):
    operations = [T.Resize((cfg['image_size'], cfg['image_size']), interpolation=T.InterpolationMode.BILINEAR, antialias=True)]
    if training:
        operations += [T.Pad(cfg['padding'], fill=0, padding_mode='constant'),
                       T.RandomCrop(cfg['image_size']), T.RandomHorizontalFlip(cfg['flip_probability'])]
    return T.Compose(operations + [T.ToTensor(), T.Normalize(stats['mean'], stats['std'])])


class DogDataset(Dataset):
    def __init__(self, manifest, split, transform, seed=6103):
        self.rows = [r for r in manifest['records'] if r['split'] == split]
        self.root = Path(manifest['root'])
        self.classes = manifest['classes']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform, self.seed = transform, seed
        # Shared epoch supports stateless augmentations even with persistent workers.
        self.epoch = torch.zeros(1, dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as im:
            im = im.convert('RGB')
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed((self.seed + int(self.epoch[0]) * 1000003 + index) % (2**63 - 1))
                x = self.transform(im)
        return x, self.class_to_idx[row['label']]


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(cfg, manifest, stats, split):
    ds = DogDataset(manifest, split, make_transform(cfg, stats, split == 'train'), cfg['seed'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    kwargs = {'batch_size': cfg['batch_size'], 'shuffle': split == 'train',
              'num_workers': cfg['workers'], 'pin_memory': cfg['pin_memory'] and cfg['device'] == 'cuda',
              'drop_last': False, 'generator': generator, 'worker_init_fn': seed_worker}
    if cfg['workers']:
        kwargs.update(persistent_workers=cfg['persistent_workers'], prefetch_factor=cfg['prefetch_factor'])
    return DataLoader(ds, **kwargs)


def compute_statistics(cfg):
    manifest = read_json(path(cfg['manifest']))
    if path(cfg['data_root']) != Path(manifest['root']).resolve():
        raise ValueError('data_root differs from manifest root; statistics must use the frozen training manifest')
    validate_manifest(manifest, cfg['split_counts'], cfg['num_classes'], verify_files=True, verify_hashes=True)
    if digest({'classes': manifest['classes'], 'records': manifest['records']}) != manifest['signature']:
        raise ValueError('Manifest signature mismatch')
    resize = T.Resize((cfg['image_size'], cfg['image_size']), interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    pixels = count = 0
    for row in manifest['records']:
        if row['split'] != 'train':
            continue
        with Image.open(Path(manifest['root']) / row['path']) as im:
            x = np.asarray(resize(im.convert('RGB')), dtype=np.float64).reshape(-1, 3) / 255.0
        total += x.sum(axis=0)
        squares += np.square(x).sum(axis=0)
        pixels += len(x)
        count += 1
    mean = total / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean**2, 0))
    if min(std) <= 0:
        raise ValueError('Zero channel standard deviation')
    result = {'mean': mean.tolist(), 'std': std.tolist(), 'count': count, 'pixels_per_channel': pixels,
              'split': 'train', 'image_size': cfg['image_size'], 'variance_definition': 'population; pixels scaled to [0,1]',
              'interpolation': 'PIL bilinear, antialias', 'manifest_signature': manifest['signature']}
    write_json(path(cfg['statistics']), result)
    return result

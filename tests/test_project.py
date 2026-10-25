"""Non-training tests. Any optimizer step during the suite is a hard failure."""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src import dataset, selection
from src.dataset import DogDataset, discover_records, make_loader, make_transform, stratified_split, validate_manifest
from src.metrics import Metrics, validation_key, weighted_correct
from src.mixup import mixup_batch, mixup_loss
from src.model import build_model
from src.train import make_scheduler
from src.utils import (EXPERIMENTS, digest, file_hash, load_checkpoint, load_config,
                       restore_rng, rng_state, save_checkpoint, seed_all, write_history, write_json)


def setUpModule():
    global STEP_GUARD
    STEP_GUARD = patch.object(torch.optim.SGD, 'step', side_effect=AssertionError('Optimizer steps are prohibited in these tests'))
    STEP_GUARD.start()


def tearDownModule():
    STEP_GUARD.stop()


class DataTests(unittest.TestCase):
    def test_deterministic_exact_stratified_split(self):
        rows = [{'path': f'{c}/{i}.jpg', 'label': c} for c in ('a', 'b', 'c') for i in range(20)]
        first = stratified_split(rows, [42, 9, 9], 6103)
        self.assertEqual(first, stratified_split(list(reversed(rows)), [42, 9, 9], 6103))
        self.assertNotEqual(first, stratified_split(rows, [42, 9, 9], 6104))
        m = {'root': '.', 'classes': ['a', 'b', 'c'], 'records': first}
        self.assertEqual(validate_manifest(m, [42, 9, 9], 3)['counts'], {'train': 42, 'val': 9, 'test': 9})
        m['records'].append(dict(first[0], split='test'))
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_required_large_split_counts(self):
        rows = [{'path': f'{i}.jpg', 'label': str(i % 120)} for i in range(10222)]
        result = stratified_split(rows, [7155, 1533, 1534], 6103)
        info = validate_manifest({'root': '.', 'classes': sorted({r['label'] for r in rows}), 'records': result}, [7155, 1533, 1534], 120)
        self.assertEqual(info['num_classes'], 120)

    def test_official_duplicate_content_is_audited_but_never_silent(self):
        m = {'root': '.', 'classes': ['a'], 'source': 'official_split_csvs',
             'content_duplicate_policy': 'preserve_official', 'records': [
                 {'path': 'train/a.jpg', 'label': 'a', 'split': 'train', 'sha256': 'same'},
                 {'path': 'val/a.jpg', 'label': 'a', 'split': 'val', 'sha256': 'same'},
                 {'path': 'test/a.jpg', 'label': 'a', 'split': 'test', 'sha256': 'different'}]}
        report = validate_manifest(m, [1, 1, 1], 1)
        self.assertEqual(report['path_overlap'], 0)
        self.assertEqual(report['cross_split_content_groups'], 1)
        self.assertEqual(len(report['duplicate_groups']), 1)
        with self.assertRaises(ValueError):
            validate_manifest(dict(m, content_duplicate_policy='error'))
        with self.assertRaises(ValueError):
            validate_manifest(dict(m, source='unsplit_labels_csv_stratified_largest_remainder'))
        m['classes'] = ['a', 'b']
        m['records'] += [{'path': f'{s}/b.jpg', 'label': 'b', 'split': s, 'sha256': 'same' if s == 'train' else s + 'b'} for s in ('train', 'val', 'test')]
        report = validate_manifest(m, [2, 2, 2], 2)
        self.assertEqual(report['conflicting_label_groups'], 1)
        self.assertEqual(m['records'][3]['label'], 'b')

    def test_official_folder_split_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for s in ('train', 'validation', 'test'):
                for i, c in enumerate(('a', 'b')):
                    p = root / s / c / f'{s}_{c}.png'
                    p.parent.mkdir(parents=True)
                    Image.new('RGB', (12, 17), (i * 20, 30, 40)).save(p)
            rows, source = discover_records(root)
            self.assertEqual(source, 'official_split_folders')
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r['split'] == 'val' for r in rows if r['path'].startswith('validation/')))

    def test_explicit_csv_and_unknown_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (12, 17)).save(root / 'a.png')
            (root / 'train_indices.npy').write_bytes(b'placeholder for unknown official index format')
            with self.assertRaisesRegex(ValueError, 'possible split metadata'):
                discover_records(root)
            (root / 'adapter.csv').write_text('path,label,split\na.png,dog,validation\n', encoding='utf-8')
            rows, _ = discover_records(root, root / 'adapter.csv')
            self.assertEqual(rows[0]['split'], 'val')

    def test_actual_assignment_labels_csv_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for s in ('train', 'val', 'test'):
                folder = root / s / 'images'
                folder.mkdir(parents=True)
                Image.new('RGB', (12, 17)).save(folder / f'{s}_image.jpg')
                (root / f'{s}_labels.csv').write_text(f'id,breed\n{s}_image,breed_a\n', encoding='utf-8')
            rows, source = discover_records(root)
            self.assertEqual(source, 'official_split_csvs')
            self.assertEqual({r['split'] for r in rows}, {'train', 'val', 'test'})
            self.assertTrue(all(r['path'] == f'{r["split"]}/images/{r["split"]}_image.jpg' and r['label'] == 'breed_a' for r in rows))

    def test_preprocessing_order_shapes_and_determinism(self):
        cfg = load_config()
        stats = {'mean': [.2, .3, .4], 'std': [.1, .2, .3]}
        image = Image.fromarray(np.arange(145 * 110 * 3, dtype=np.uint8).reshape(145, 110, 3))
        train = make_transform(cfg, stats, True)
        self.assertEqual([type(t).__name__ for t in train.transforms], ['Resize', 'Pad', 'RandomCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize'])
        val = make_transform(cfg, stats, False)
        self.assertEqual(tuple(train(image).shape), (3, 100, 100))
        self.assertTrue(torch.equal(val(image), val(image)))
        self.assertEqual(train.transforms[1].fill, 0)

    def test_stateless_augmentation_ignores_other_rng_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(np.random.default_rng(7).integers(0, 256, (130, 140, 3), dtype=np.uint8)).save(root / 'a.png')
            m = {'root': str(root), 'classes': ['a'], 'records': [{'path': 'a.png', 'label': 'a', 'split': 'train'}]}
            ds = DogDataset(m, 'train', make_transform(load_config(), {'mean': [0]*3, 'std': [1]*3}, True))
            a = ds[0][0]
            torch.rand(39)
            self.assertTrue(torch.equal(a, ds[0][0]))
            ds.epoch[0] = 2
            self.assertFalse(torch.equal(a, ds[0][0]))

    def test_train_only_statistics_population_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i, (s, color) in enumerate([('train', (0, 20, 40)), ('train', (100, 120, 140)), ('val', (250, 250, 250)), ('test', (255, 255, 255))]):
                name = f'{i}.png'
                Image.new('RGB', (6, 8), color).save(root / name)
                rows.append({'path': name, 'label': 'a', 'split': s, 'sha256': file_hash(root / name)})
            m = {'root': str(root), 'classes': ['a'], 'records': rows}
            m['signature'] = digest({'classes': m['classes'], 'records': rows})
            write_json(root / 'manifest.json', m)
            cfg = dict(load_config(), data_root=str(root), manifest=str(root / 'manifest.json'), statistics=str(root / 'stats.json'), split_counts=[2, 1, 1], num_classes=1)
            result = dataset.compute_statistics(cfg)
            np.testing.assert_allclose(result['mean'], np.array([50, 70, 90]) / 255, atol=1e-10)
            np.testing.assert_allclose(result['std'], np.array([50, 50, 50]) / 255, atol=1e-10)
            self.assertEqual(result['count'], 2)
            with self.assertRaisesRegex(ValueError, 'data_root differs'):
                dataset.compute_statistics(dict(cfg, data_root=str(root / 'wrong')))

    def test_windows_dataloader_workers_and_shared_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(4):
                Image.fromarray(np.random.default_rng(i).integers(0, 256, (130, 140, 3), dtype=np.uint8)).save(root / f'{i}.png')
                rows.append({'path': f'{i}.png', 'label': 'a', 'split': 'train'})
            m = {'root': str(root), 'classes': ['a'], 'records': rows}
            cfg = dict(load_config(), device='cpu', precision='fp32', workers=2, persistent_workers=True)
            stats = {'mean': [0]*3, 'std': [1]*3}
            loader = make_loader(cfg, m, stats, 'train')
            try:
                loader.dataset.epoch[0] = 1
                loader.generator.manual_seed(7)
                first, labels = next(iter(loader))
                self.assertEqual(tuple(first.shape), (4, 3, 100, 100))
                self.assertTrue(torch.equal(labels, torch.zeros(4, dtype=torch.int64)))
                loader.dataset.epoch[0] = 2
                loader.generator.manual_seed(7)
                second, _ = next(iter(loader))
                self.assertFalse(torch.equal(first, second))
            finally:
                if loader._iterator is not None:
                    loader._iterator._shutdown_workers()


class MathAndModelTests(unittest.TestCase):
    def test_model_output_on_meta_device_no_numeric_forward(self):
        # Shape propagation only. The environment script provides the optional real GPU forward.
        with torch.device('meta'):
            model = build_model(120).eval()
            output = model(torch.empty(2, 3, 100, 100))
        self.assertEqual(tuple(output.shape), (2, 120))
        self.assertEqual(model.classifier[-1].out_features, 120)

    def test_mixup_pairing_and_interpolated_loss(self):
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        y = torch.arange(4)
        with patch('numpy.random.beta', return_value=.25), patch('torch.randperm', return_value=torch.tensor([3, 2, 1, 0])):
            mixed, a, b, lam = mixup_batch(x, y, .2)
        self.assertEqual(lam, .25)
        self.assertTrue(torch.equal(mixed, .25*x + .75*x.flip(0)))
        logits = torch.tensor([[2., 0, 0, 0]]).repeat(4, 1)
        expected = .25*torch.nn.functional.cross_entropy(logits, a) + .75*torch.nn.functional.cross_entropy(logits, b)
        self.assertAlmostEqual(mixup_loss(logits, a, b, lam).item(), expected.item())
        self.assertTrue(torch.equal(mixup_batch(x, y, 0)[0], x))

    def test_metric_weighting_and_mixup_correctness(self):
        meter = Metrics()
        meter.update(1., 2, 2)
        meter.update(4., 0, 1)
        self.assertEqual(meter.values(), (2., 2/3))
        logits = torch.tensor([[3., 0.], [0., 3.]])
        self.assertAlmostEqual(weighted_correct(logits, torch.tensor([0, 0]), torch.tensor([1, 1]), .2), 1.)
        self.assertLess(validation_key({'val_accuracy': .8, 'val_loss': 2., 'epoch': 2}), validation_key({'val_accuracy': .7, 'val_loss': .1, 'epoch': 1}))
        with self.assertRaises(FloatingPointError):
            meter.update(float('nan'), 0, 1)

    def test_scheduler_configuration_state_and_formula(self):
        cfg = load_config('cosine')
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.05, momentum=.9)
        scheduler = make_scheduler(optimizer, cfg)
        self.assertEqual(scheduler.T_max, 300)
        self.assertEqual(scheduler.eta_min, 0.)
        self.assertEqual(scheduler.get_last_lr(), [.05])
        self.assertIsNone(make_scheduler(optimizer, dict(cfg, scheduler='constant')))
        saved = scheduler.state_dict()
        restored = make_scheduler(optimizer, cfg)
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict(), saved)
        for t, expected in ((0, .05), (150, .025), (300, 0)):
            self.assertAlmostEqual(.05 * (1 + math.cos(math.pi * t / 300)) / 2, expected)

    def test_checkpoint_and_rng_roundtrip_without_optimization(self):
        seed_all(6103)
        random_state = rng_state()
        expected = torch.rand(5), np.random.rand(5)
        restore_rng(random_state)
        self.assertTrue(torch.equal(expected[0], torch.rand(5)))
        np.testing.assert_equal(expected[1], np.random.rand(5))
        state = {'model': {'weight': torch.ones(2)}, 'optimizer': {}, 'scheduler': None,
                 'scaler': {}, 'epoch': 1, 'history': [{'epoch': 1}], 'config': load_config('lr_05'),
                 'rng': random_state, 'data_signature': 'example', 'best_epoch': 1, 'best_key': (0, 0, 1), 'best_model': {}}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'last.pt'
            save_checkpoint(p, state)
            loaded = load_checkpoint(p)
            self.assertTrue(torch.equal(loaded['model']['weight'], state['model']['weight']))
            self.assertEqual(loaded['epoch'], 1)
            state['epoch'] = 2
            save_checkpoint(p, state)
            with self.assertRaises(ValueError):
                load_checkpoint(p)

    def test_scheduler_restores_midpoint_without_any_updates(self):
        cfg = load_config('cosine')
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.05, momentum=.9)
        scheduler = make_scheduler(optimizer, cfg)
        # Handcrafted midpoint state is serialization test data, not an executed run.
        state = scheduler.state_dict()
        state.update(last_epoch=150, _last_lr=[.025], _step_count=151)
        opt_state = optimizer.state_dict()
        opt_state['param_groups'][0]['lr'] = .025
        other_opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.05, momentum=.9)
        other_schedule = make_scheduler(other_opt, cfg)
        other_opt.load_state_dict(opt_state)
        other_schedule.load_state_dict(state)
        self.assertEqual(other_schedule.last_epoch, 150)
        self.assertEqual(other_schedule.get_last_lr(), [.025])
        self.assertEqual(other_opt.param_groups[0]['lr'], .025)


def synthetic_completed(root, exp, cfg, train_loss, accuracy, val_loss=1.0):
    """Synthetic fixtures stay in TemporaryDirectory and are never project results."""
    cfg = copy.deepcopy(cfg)
    cfg['experiment'] = exp
    rows = [dict(epoch=i, train_loss=float(train_loss), train_accuracy=.4, val_loss=float(val_loss),
                 val_accuracy=float(accuracy), lr=cfg['lr'], next_lr=cfg['lr'], epoch_seconds=1., elapsed_seconds=float(i))
            for i in range(1, cfg['epochs']+1)]
    rd, cd = root / 'results' / exp, root / 'checkpoints' / exp
    cd.mkdir(parents=True)
    for n in ('best.pt', 'final.pt'):
        (cd / n).write_bytes(b'UNIT-TEST FIXTURE; NOT A REAL CHECKPOINT')
    write_json(rd / 'config.json', cfg)
    write_history(rd / 'metrics.csv', rows)
    summary = {'status': 'complete', 'config': cfg, 'config_signature': digest(cfg), 'data_signature': 'same-fixture-data',
               'source_signature': 'same-fixture-source', 'history_signature': digest(rows), 'final': rows[-1],
               'best_validation': rows[0], 'checkpoint_sha256': {n: file_hash(cd / n) for n in ('best.pt', 'final.pt')}}
    write_json(rd / 'summary.json', summary)
    return summary


class DependencyTests(unittest.TestCase):
    def test_all_configurations_and_corrected_lr(self):
        for exp in EXPERIMENTS:
            load_config(exp)
        self.assertEqual(load_config('lr_005')['lr'], .05)
        self.assertEqual(load_config('mixup')['mixup_alpha'], .2)

    def test_missing_lr_prerequisite_fails_with_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'python -m src.experiments --experiment lr_05'):
                selection.resolve_config(load_config('cosine'), Path(tmp))

    def test_lr_uses_only_final_training_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for exp, loss, acc in [('lr_05', 3., .99), ('lr_005', 1., .2), ('lr_0001', 2., .5)]:
                synthetic_completed(root, exp, load_config(exp), loss, acc)
            result = selection.select_lr(root)
            self.assertEqual(result['experiment'], 'lr_005')
            self.assertEqual(result['lr'], .05)
            self.assertEqual(result['metric_winners']['val_accuracy'], 'lr_05')
            # Tampering with history blocks selection.
            p = root / 'results/lr_005/metrics.csv'
            p.write_text(p.read_text().replace('1.0', '9.0', 1))
            with self.assertRaises(ValueError):
                selection.select_lr(root)

    def test_full_dependency_graph_and_mixup_need_not_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for exp in EXPERIMENTS[:3]:
                synthetic_completed(root, exp, load_config(exp), 1 if exp == 'lr_005' else 2, .3)
            with self.assertRaises(ValueError):
                selection.resolve_config(load_config('mixup'), root)
            for exp, acc in [('constant', .5), ('cosine', .6), ('wd_5e4', .7), ('wd_1e4', .65)]:
                cfg = selection.resolve_config(load_config(exp), root)
                synthetic_completed(root, exp, cfg, 1., acc)
            cfg = selection.resolve_config(load_config('mixup'), root)
            self.assertEqual(cfg['weight_decay'], .0005)
            self.assertEqual(cfg['scheduler'], 'cosine')
            self.assertEqual(cfg['lr'], .05)
            synthetic_completed(root, 'mixup', cfg, 1.2, .68)
            self.assertEqual(selection.select_final(root)['experiment'], 'wd_5e4')

    def test_final_test_cannot_run_without_selection(self):
        from src import evaluate
        with tempfile.TemporaryDirectory() as tmp, patch.object(evaluate, 'ROOT', Path(tmp)), patch('sys.argv', ['evaluate', '--test']):
            with self.assertRaises(SystemExit) as raised:
                evaluate.main()
            self.assertEqual(raised.exception.code, 2)

    def test_readonly_final_selection_does_not_rewrite_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for exp in EXPERIMENTS[:3]:
                synthetic_completed(root, exp, load_config(exp), 1. if exp == 'lr_005' else 2., .3)
            for exp in selection.BASELINES:
                synthetic_completed(root, exp, selection.resolve_config(load_config(exp), root), 1., .6)
            cfg = selection.resolve_config(load_config('mixup'), root)
            synthetic_completed(root, 'mixup', cfg, 1., .65)
            selected = selection.select_final(root, persist=False)
            self.assertEqual(selected['experiment'], 'mixup')
            self.assertFalse((root / 'results/final_selection.json').exists())


if __name__ == '__main__':
    unittest.main()

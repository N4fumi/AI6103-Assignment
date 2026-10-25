"""Explicit hold-out evaluation, gated on a verified final model selection."""
import argparse
import json

import torch

from .dataset import load_data_metadata, make_loader
from .model import build_model
from .selection import select_final
from .train import evaluate_loader, setup_device
from .utils import ROOT, digest, file_hash, read_json, seed_all, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true', required=True, help='Explicitly authorize a single hold-out evaluation')
    parser.add_argument('--workers', type=int, help='Optional inference-only worker count')
    parser.add_argument('--device', choices=('cpu', 'cuda'), help='Optional inference-only device override')
    args = parser.parse_args()
    try:
        selection_file = ROOT / 'results/final_selection.json'
        if not selection_file.is_file():
            raise ValueError('Run python -m src.experiments --select-final after all experiments complete.')
        saved = read_json(selection_file)
        # Recompute only selection from saved validation metrics, never training.
        current = select_final(persist=False)
        if saved != current:
            raise ValueError('Final selection changed. Review it and rerun --select-final explicitly before test evaluation.')
        target = ROOT / 'results/final_test.json'
        if target.exists():
            raise FileExistsError('Hold-out evaluation already exists at results/final_test.json. Repeated tuning on test results is disallowed.')
        checkpoint = ROOT / saved['checkpoint']
        if file_hash(checkpoint) != saved['checkpoint_sha256']:
            raise ValueError('Selected checkpoint integrity check failed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state.get('kind') != 'inference_best_validation' or state['config'] != saved['config'] or state['epoch'] != saved['best_validation']['epoch']:
            raise ValueError('Selected checkpoint metadata mismatch')
        cfg = dict(saved['config'])
        manifest, stats = load_data_metadata(cfg, verify_hashes=True)
        if digest({'manifest': manifest['signature'], 'statistics': stats}) != saved['data_signature'] or stats != state['statistics'] or manifest['classes'] != state['classes']:
            raise ValueError('Test preprocessing, classes or data changed after model selection')
        if args.workers is not None:
            if args.workers < 0:
                raise ValueError('workers must be nonnegative')
            cfg['workers'] = args.workers
        if args.device:
            cfg['device'] = args.device
        cfg['precision'] = 'fp32'
        seed_all(cfg['seed'], cfg['deterministic'])
        device = setup_device(cfg)
        model = build_model(cfg['num_classes']).to(device)
        model.load_state_dict(state['model'])
        loader = make_loader(cfg, manifest, stats, 'test')
        loss, accuracy = evaluate_loader(model, loader, device)
        result = {'test_loss': loss, 'test_accuracy': accuracy, 'count': len(loader.dataset),
                  'experiment': saved['experiment'], 'epoch': state['epoch'],
                  'checkpoint_sha256': saved['checkpoint_sha256'], 'data_signature': saved['data_signature'],
                  'selection_signature': digest(saved), 'precision': 'fp32', 'device': str(device)}
        write_json(target, result)
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as e:
        parser.exit(2, f'ERROR: {e}\n')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

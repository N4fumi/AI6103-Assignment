"""Read-only environment checks; optional single inference pass, never training."""
import _bootstrap
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys

from src.utils import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward-check', action='store_true', help='One eval-mode EfficientNetV2-S inference, no backward/optimizer')
    args = parser.parse_args()
    result = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'training_executed': False}
    result['packages'] = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'PyYAML', 'matplotlib'):
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = 'MISSING'
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=15)
        result['nvidia_smi'] = gpu.stdout.strip() or gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        result['nvidia_smi'] = str(error)
    try:
        import torch
        import torchvision
        result.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda_runtime=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(), compiled_architectures=torch.cuda.get_arch_list())
        if torch.cuda.is_available():
            result['device'] = torch.cuda.get_device_name(0)
            result['compute_capability'] = list(torch.cuda.get_device_capability(0))
            result['bf16_supported'] = torch.cuda.is_bf16_supported()
            x = torch.ones((2, 2), device='cuda')
            result['cuda_tensor_check'] = bool(torch.all(x @ x == 2).item())
            torch.cuda.synchronize()
        if args.forward_check:
            if not torch.cuda.is_available():
                raise RuntimeError('GPU inference check skipped: CUDA unavailable')
            from src.model import build_model
            with torch.inference_mode():
                model = build_model(120).eval().cuda()
                output = model(torch.zeros(1, 3, 100, 100, device='cuda'))
                result['forward_shape'] = list(output.shape)
                result['forward_finite'] = bool(torch.isfinite(output).all().item())
                assert list(output.shape) == [1, 120] and result['forward_finite']
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    write_json(ROOT / 'environment_check.json', result)
    print(json.dumps(result, indent=2))
    if 'error' in result or any(v == 'MISSING' for v in result['packages'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

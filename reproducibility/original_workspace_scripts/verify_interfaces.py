"""Read-only CLI verification. Every experiment command here has --dry-run."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1] / 'outputs' / 'AI6103_Assignment'
cases = [
    (['-m', 'src.experiments', '--validate-configs'], 0),
    (['-m', 'src.experiments', '--experiment', 'lr_005', '--dry-run'], 0),
    (['-m', 'src.experiments', '--experiment', 'cosine', '--dry-run'], 2),
    (['-m', 'src.experiments', '--select-final'], 2),
    (['-m', 'src.evaluate', '--help'], 0),
    (['scripts/check_environment.py', '--help'], 0),
    (['scripts/inspect_dataset.py', '--help'], 0),
    (['scripts/compute_statistics.py', '--help'], 0),
    (['scripts/generate_figures.py', '--help'], 0),
    (['scripts/summarize_results.py', '--help'], 0),
    (['scripts/generate_report_tables.py', '--help'], 0),
]


def check(case):
    args, expected = case
    result = subprocess.run([sys.executable] + args, cwd=root, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=120)
    return {'arguments': args, 'expected_exit': expected, 'actual_exit': result.returncode,
            'passed': result.returncode == expected, 'output': (result.stdout + result.stderr)[:3000]}


with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(check, cases))
(root / 'verification/interface_checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
for r in results:
    print(('PASS' if r['passed'] else 'FAIL') + ' ' + ' '.join(r['arguments']))
raise SystemExit(0 if all(r['passed'] for r in results) else 1)

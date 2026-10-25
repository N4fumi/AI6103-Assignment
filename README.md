# AI6103: Hyperparameter Analysis of EfficientNetV2-S

Author: **Zhang Mufan** (GitHub: [N4fumi](https://github.com/N4fumi)).

Completed experiments for dog breed classification with EfficientNetV2-S trained from scratch.
The repository contains the source, configurations, all eight recorded histories, figures,
validation selections, final test result, and report source. The selected model is a
[Release attachment](https://github.com/N4fumi/AI6103-Assignment/releases/tag/v1.0-results).

## Results

| Experiment | Epochs | Final train CE | Final validation CE | Final validation accuracy (%) | Best validation accuracy (%) |
|---|---:|---:|---:|---:|---:|
| lr_05 | 15 | 4.6273 | 4.6938 | 1.37 | 2.54 |
| lr_005 | 15 | 4.1588 | 4.2930 | 4.83 | 4.96 |
| lr_0001 | 15 | 4.7874 | 4.7906 | 1.37 | 1.44 |
| constant | 300 | 0.0088 | 8.9741 | 18.13 | 19.70 |
| cosine | 300 | 0.0022 | 8.4297 | 19.05 | 19.90 |
| wd_5e4 | 300 | 0.0199 | 3.2910 | 32.88 | 34.77 |
| wd_1e4 | 300 | 0.0036 | 5.5271 | 26.61 | 27.33 |
| mixup | 300 | 0.8900 | 2.7401 | 38.29 | 38.62 |

The selected configuration uses initial learning rate **0.05**, cosine annealing,
weight decay **0.0005**, and Mixup alpha **0.2**. Its checkpoint at epoch **273**
achieves **38.62% validation accuracy** and **37.55% test accuracy**
(test cross-entropy **2.7505**, 1,534 test images).
Machine-readable results are in `results/final_summary.json`, `results/final_summary.csv`,
`results/final_selection.json`, and `results/final_test.json`. CSV accuracy fields are fractions.

## Repository contents

- `src/`: training, evaluation, dataset processing, selection, and visualization.
- `configs/`: all eight experiment settings and common settings.
- `scripts/`: dataset inspection, statistics, figures, summaries, and verified runtime adapter.
- `tests/`, `verification/`: non-training verification code and recorded checks.
- `results/`: per-epoch CSV/JSON records, configuration/provenance, console logs, selections,
  performance-equivalence evidence, and the recorded final hold-out result.
- `figures/`: all comparison and individual-run PNG/PDF plots.
- `data/`: recorded split manifest, training statistics, duplicate audit, and dataset instructions.
- `report/`: the LaTeX report, bibliography, supplied AAAI style files, and figure assets.
- `reproducibility/original_workspace_scripts/`: the actual local continuation, optimization,
  and report-plot scripts. These preserve their original workspace-layout assumptions;
  the normal project entry points are in `src/` and `scripts/`.
- `artifacts/`: the selected model's location, byte size, and SHA-256 checksum.

The other 23 training checkpoints and optimization backup weights are not included in
this compact publication. Raw dataset images and label CSV files are supplied separately
through the assignment's dataset link in [data/README.md](data/README.md).

## Selected checkpoint

Download `mixup-best-epoch273.pt` from the [results release](https://github.com/N4fumi/AI6103-Assignment/releases/tag/v1.0-results)
and place it at `checkpoints/mixup/best.pt` when working with the saved final selection.
The expected SHA-256 is:

```text
0609a1441414064c03bc50baad4e50d953aaf8dd0a3af364fb948996c4d4b973
```

## Environment and reproduction

The recorded runs used seed 6103, SGD momentum 0.9, batch size 128, 100 x 100 inputs,
BF16 training and FP32 validation/test, with deterministic algorithms enabled.
The measured environment used Python 3.10.18, PyTorch 2.9.0.dev20250905+cu129,
torchvision 0.24.0.dev20250901+cu129 and an RTX 5070 Ti Laptop GPU.
See `environment_check.json` for the actual inspection record. `requirements.txt`
documents an alternative stable dependency set from the initial setup, rather than
the exact nightly environment used for the recorded results.

Use the supplied dataset splits: 7,155 training, 1,533 validation, and 1,534 test images
across 120 classes. The frozen `data/manifest.json` records the original absolute data
location. On another computer, update only its `root` field to the absolute local
`data/` directory; keep its records, class list, signature, and statistics unchanged.
The loader verifies the files and their hashes before use.

To inspect data and regenerate summaries/figures from existing records:

```text
python scripts/inspect_dataset.py --verify-loaders --verify-hashes
python scripts/summarize_results.py
python scripts/generate_figures.py
python scripts/generate_report_tables.py
```

Full experiment commands and dependency order are in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)
and `RUN_COMMANDS.md`. Existing experiment records are protected against accidental overwrite.
The hold-out evaluator has already produced `results/final_test.json`; keep that result
when reproducing plots or reviewing the saved model.

## Dataset audit

The official data contain 15 byte-identical image groups, including 10 spanning splits;
six duplicate-content groups have conflicting labels. The official membership and
labels were retained. See `data/duplicate_audit.json` and the recorded inspection.

## Report

[Download the compiled PDF](report/AI6103_Report_ZhangMufan_G2505269G.pdf).

Open `report/main.tex` in a LaTeX editor, or upload the `report/` directory to Overleaf.
The report uses the supplied AAAI 2024 template. Repository and model-release links are provided in this README.

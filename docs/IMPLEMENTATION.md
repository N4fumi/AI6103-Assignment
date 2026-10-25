# AI6103 Assignment: reproducible EfficientNetV2-S experiments

This project implements the provided Fall 2026 assignment and its corrections.
**Nothing trains on import, during setup, or in the test suite. You launch each
experiment yourself.** There are no fabricated empirical results. The hold-out
test has a separate, explicit entry point.

Deadline supplied in the request: **26 October 2026, 23:59 Singapore time**.

The completed results are published in this repository. The report source is in `report/`. This document retains the original implementation and command reference.

## 1. Environment

The environment inspected on this computer is:

- Python: `D:\WORKSPACE\Anaconda\envs\ntuai\python.exe`, Python 3.10.18.
- PyTorch `2.9.0.dev20250905+cu129`; torchvision `0.24.0.dev20250901+cu129`.
- NVIDIA RTX 5070 Ti Laptop GPU; 12,227 MiB; driver 621.01; compute capability 12.0.
- CUDA available, `sm_120` present, BF16 supported, CUDA tensor check passed.
- One FP32 inference-only check returned `[1,120]` with finite outputs. No backward
  pass or optimizer update was executed. Batch-128 training memory has **not**
  been benchmarked.
- NumPy 2.2.6, Pillow 11.3.0, PyYAML 6.0.3, matplotlib 3.10.6.
- The Windows `python` alias on PATH is unusable. Activate the Conda environment
  or use its explicit executable.

Use the existing environment without altering it:

```powershell
Set-Location "C:\Users\ROG\Documents\Codex\2026-10-24\files-pasted-by-the-user-ai6103\outputs\AI6103_Assignment"
& "D:\WORKSPACE\Anaconda\shell\condabin\conda-hook.ps1"
conda activate ntuai
python scripts/check_environment.py
```

If Conda activation is unavailable in this shell, use the executable directly:

```powershell
& "D:\WORKSPACE\Anaconda\envs\ntuai\python.exe" scripts/check_environment.py
```

In VS Code, select the same interpreter. Save the actual environment before any
real experiments, and retain it for all configurations:

```powershell
python -m pip freeze > environment-lock.txt
```

For an isolated stable environment, use an installed Python 3.10--3.12 interpreter
and a **new** directory; do not replace an existing `.venv`:

```powershell
& "D:\WORKSPACE\Anaconda\envs\ntuai\python.exe" -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

The stable pair is pinned for reproducibility, not advertised as the latest
release. The inspected nightly environment has different versions; the stable
pair has not been installed or tested in this session. Official installation
commands: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).
Blackwell-capable CUDA wheels are required for this GPU; a driver reporting CUDA
13.5 does not mean the installed torch wheel uses that runtime.
See [PyTorch's Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/).
Do not install CPU-only or older GPU wheels accidentally.

## 2. Dataset and preprocessing

In this workspace, the official manifest, training statistics, and loader
verification are **already complete**. Reuse `data/manifest.json` and
`data/statistics.json`; run inspection without `--prepare` to check them.
The preparation commands below are for a fresh project copy. The code-only ZIP
excludes dataset images/labels and local absolute-root manifests.

The provided dataset is in `data/`: `train/images`, `val/images`, `test/images`,
with `train_labels.csv`, `val_labels.csv`, and `test_labels.csv`. The configured
root is `data`; preserve its official membership and metadata.
Content hashing found 15 duplicate-image groups, including 10 crossing splits.
Six of these groups have conflicting official labels.
The explicit `preserve_official` policy retains the mandated split/counts,
records the audit, and warns at runtime. Path membership has no overlap;
byte-identical content does. We do not silently remove data or claim that the
official hold-out is free of this contamination. See `data/README.md`.
Read [data/README.md](data/README.md) for supported layouts and CSV adapters.

```powershell
python scripts/inspect_dataset.py
python scripts/inspect_dataset.py --prepare
```

If inspection establishes that no official split exists, use the following
**instead of** `--prepare`:

```powershell
python scripts/inspect_dataset.py --create-splits
```

Then:

```powershell
python scripts/compute_statistics.py
python scripts/inspect_dataset.py --verify-loaders --verify-hashes
```

Required sizes: 7,155 train, 1,533 validation, 1,534 test; 120 classes. The split
manifest is fixed, with image-content hashes. Statistics use only resized RGB
training images, no random augmentation, and population variance over pixels.

Training: bilinear resize to 100 x 100 -> zero pad 12 on every side -> random
100 x 100 crop -> random horizontal flip (p=0.5) -> tensor -> normalize.
Validation/test: resize -> tensor -> normalize. No ImageNet constants are used.

The loader uses Windows-safe top-level dataset/worker definitions and a guarded
CLI. CPU-generated sample augmentation is stateless in `(seed, epoch, index)`,
and its epoch counter is shared with workers. This makes resumed augmentation
independent of worker scheduling, including optional persistent workers.
Shuffling uses a dedicated epoch-seeded generator. No data are silently dropped.

## 3. Non-training verification

```powershell
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python -m src.experiments --validate-configs
python -m src.experiments --experiment lr_005 --dry-run
```

Tests cover exact and deterministic splitting, official split preservation,
unknown metadata rejection, preprocessing order/shape, stateless augmentation,
training-only statistics, model shape propagation on the meta device, Mixup,
metric weighting, scheduler configuration/state, RNG/checkpoint round trips,
all dependency stages, integrity checks, and test gating. An optimizer-step
guard causes the suite to fail if any SGD update is attempted. Synthetic test
fixtures live only in temporary directories and are never project results.

An optional real **single inference**, without gradients, is available:

```powershell
python scripts/check_environment.py --forward-check
```

The project author already performed this check once; it is not a training or
batch-size memory benchmark. Do not use training smoke tests as setup checks.

## 4. Eight manual experiments, in order

Every experiment uses a fresh randomly initialized torchvision EfficientNetV2-S,
SGD momentum 0.9, batch size 128, input 100 x 100, and seed 6103.

| ID | Initial LR | Epochs | Schedule | Weight decay | Mixup alpha |
|---|---:|---:|---|---:|---:|
| lr_05 | 0.5 | 15 | constant | 0 | 0 |
| lr_005 | **0.05** | 15 | constant | 0 | 0 |
| lr_0001 | 0.001 | 15 | constant | 0 | 0 |
| constant | selected by training CE | 300 | constant | 0 | 0 |
| cosine | same selected LR | 300 | cosine | 0 | 0 |
| wd_5e4 | same selected LR | 300 | cosine | 0.0005 | 0 |
| wd_1e4 | same selected LR | 300 | cosine | 0.0001 | 0 |
| mixup | same selected LR | 300 | selected baseline | selected baseline | 0.2 |

First, manually run the three LR experiments:

```powershell
python -m src.experiments --experiment lr_05
python -m src.experiments --experiment lr_005
python -m src.experiments --experiment lr_0001
python -m src.experiments --select-lr
```

LR selection requires all three genuine 15-epoch histories and intact
checkpoints. It minimizes **final training CE only**. Exact ties use the listed
experiment order. Four metric winners are reported separately. The selected
value and evidence are written to `results/selected_lr.json`.

Next:

```powershell
python -m src.experiments --experiment constant
python -m src.experiments --experiment cosine
python -m src.experiments --experiment wd_5e4
python -m src.experiments --experiment wd_1e4
python -m src.experiments --select-baseline
```

Baseline selection requires **all four** 300-epoch non-Mixup runs. This implements
“best setup so far” by considering the constant schedule and the cosine/decay
options, including zero decay. Criterion: highest validation accuracy over
epochs; lower validation CE, earlier epoch, then lexicographic experiment ID
break ties. All completed epochs are eligible; there is no early stopping.
The baseline is written to `results/selected_baseline.json`.

Finally:

```powershell
python -m src.experiments --experiment mixup
python -m src.experiments --select-final
```

Mixup inherits the selected baseline's scheduler and decay. It is a new
300-epoch training run, not fine-tuning. Final selection compares its best
validation checkpoint with the selected non-Mixup baseline using the same
criterion. Mixup is not presumed to win. `results/final_selection.json` records
the checkpoint, its hash, configuration, epoch, criterion, and source evidence.

No command launches multiple experiments. Dependencies are rechecked before
each dependent run. Missing, incomplete, altered, or incomparable results
produce actionable errors and manual commands. No default LR/winner is invented.
Selection commands inspect saved validation/training results only.

## 5. Runtime settings and metrics

`configs/base.yaml` centralizes fixed and runtime settings. Each experiment has
a small override file. Default precision is CUDA BF16 autocast (no scaling);
FP16 autocast uses a GradScaler. Validation/test inference is FP32. Pinned
memory and nonblocking transfers are enabled; workers=4, prefetch=2, persistent
workers disabled by default. Deterministic algorithms are required, cuDNN
benchmark and TF32 are disabled. Results across different devices/software
versions are not guaranteed bit-identical.

Runtime-only CLI options include:

```powershell
python -m src.experiments --experiment lr_05 --precision fp32 --workers 0
python -m src.experiments --experiment lr_05 --device cpu --precision fp32 --workers 0
```

These commands **train if executed**. Choose settings before the first run and
keep the same settings for the comparison. The dependency checker rejects
inconsistent precision/seed/runtime settings. Use base.yaml to avoid having to
repeat flags. `--memory-fraction 0.9` is available; it limits the allocator and
may cause OOM earlier rather than making batch 128 fit. `channels_last` and
`persistent_workers` are optional base settings, recorded with the run. We do
not change the model, batch size, use gradient accumulation, or add warmup.

The LR schedule is `eta(t)=eta0*(1+cos(pi*t/300))/2`, with `t` equal to completed
epochs. The scheduler steps once **after** each epoch's optimizer updates.
Epoch 1 uses eta0; epoch 300 uses eta(299); the stored next LR after the final
epoch is zero. CSV columns `lr` and `next_lr` make this convention explicit.

Every epoch records `epoch`, online augmented `train_loss`, `train_accuracy`,
deterministic `val_loss`, `val_accuracy`, `lr`, `next_lr`, `epoch_seconds`,
and accumulated `elapsed_seconds`. Times cover epoch data loading, computation,
validation, and device synchronization; they exclude checkpoint serialization,
startup/hash checking, and time spent stopped. Accuracy is stored as a fraction.
Loss means are sample-weighted, including the short final minibatch, and always
exclude the optimizer's regularization penalty. PyTorch SGD applies the same
coupled decay to all trainable parameters; this is not decoupled AdamW.

Mixup draws one `Beta(0.2,0.2)` coefficient per batch, pairs images with a random
permutation, and interpolates the two CE terms. Training accuracy is
`lambda*I(pred=label_a)+(1-lambda)*I(pred=label_b)`, averaged over examples.
Its training metrics differ from hard-label metrics, so use ordinary unmixed
validation metrics for the cross-method comparison. Online training/validation
gaps are diagnostic and not clean-train generalization gaps.

Nonfinite CE/gradients or skipped FP16 updates stop the run instead of silently
continuing with altered settings. For numerical instability use FP32 in a
**fresh** run, explicitly archive prior outputs with `--overwrite`, and keep all
compared experiments consistent. Do not resume a run with changed precision.

## 6. Checkpoints, stopping, and resumption

Stop with Ctrl+C. The latest **completed epoch** is atomically committed to
`checkpoints/EXPERIMENT/last.pt`. An unfinished epoch is replayed on resume;
it is not appended as a completed epoch. Sudden termination can leave a stale
`results/.EXPERIMENT.lock`; verify the old process is gone before removing it.

```powershell
python -m src.experiments --experiment cosine --resume
```

Repeat the same runtime flags if the original command used them. Resume checks
configuration, software versions, source signature, preprocessing/data hashes,
and history integrity. `last.pt` and `final.pt` contain model/optimizer/scheduler/
scaler, completed epoch, RNG states, complete history, metadata, and best model
weights. Scheduler position is restored, not restarted. `best.pt` is an explicitly
labeled inference-only artifact for the best validation epoch; resume always
uses `last.pt`. It contains model, config, classes, statistics, provenance, and
validation record rather than an unrelated optimizer state.

After a crash, `last.pt` is the canonical history. Resume repairs CSV/JSON and
best artifacts from it. Starting an existing experiment without a flag fails.
`--overwrite` explicitly archives old result/checkpoint/figure folders with a
timestamp; it never silently deletes completed work. Downstream selections
must be regenerated if their evidence changes. Checkpoints use torch's pickle
format for RNG state: load only this project's trusted local checkpoints.

## 7. Outputs, figures, and numeric table exports

```powershell
python scripts/summarize_results.py
python scripts/generate_figures.py
python scripts/generate_report_tables.py
```

`summarize_results.py` exports CSV, JSON, and LaTeX tables, final metrics,
validation best epochs, configurations, and diagnostic accuracy gaps. Missing
and incomplete runs are labeled; they are not assigned numerical results in
the table. Figures use saved unsmoothed trajectories, 350-DPI PNG and
vector PDF output, consistent legends, labeled axes, and [0,1] accuracy axes.
Comparison plots cover all four metrics for LRs, schedules, decay (including
the zero-decay cosine baseline), and Mixup. A separate decay gap plot is made.
Partial histories may be plotted and are identified as partial in console output.

The mathematical Beta and normalized schedule diagrams can be generated before
training with `python scripts/generate_figures.py --theory-only`. They are
explicitly theoretical, not invented measurements. Beta's infinite endpoint
densities are excluded by plotting the interval [0.001,0.999].

Expected locations:

```text
data/manifest.json, statistics.json, inspection.json
environment_check.json, environment-lock.txt
results/lr_05/config.json, metadata.json, status.json
results/lr_05/metrics.csv, history.json, summary.json
checkpoints/lr_05/last.pt, best.pt, final.pt
results/comparison.csv, comparison.json
results/selected_lr.json, selected_baseline.json, final_selection.json
results/final_test.json             # only after the explicit test command
figures/lr_05/loss.{png,pdf}, accuracy.{png,pdf}, learning_rate.{png,pdf}
figures/comparisons/{learning_rates,schedules,weight_decay,mixup,accuracy_gap}.{png,pdf}
figures/theory/{beta,schedule}.{png,pdf}
results/tables/results.tex, statistics.tex, preprocessing_excerpt.py
```

The same per-run paths apply to the other seven experiment IDs.

`generate_report_tables.py` retains its originally requested command name but
only exports numeric LaTeX tables and the actual preprocessing code excerpt to
`results/tables/`. It does not generate a report or write scientific conclusions.

## 8. Explicit final hold-out evaluation

Only after all experiments and final validation-based selection:

```powershell
python -m src.evaluate --test
python scripts/generate_report_tables.py
```

The evaluator requires an existing final selection, revalidates it without
changing it, checks the selected checkpoint's hash and frozen preprocessing,
and reports test CE/top-1 in `results/final_test.json`. It refuses to overwrite
an existing final test result. No training, plotting, table-export, or selection
command calls test evaluation. Do not use the test result to change parameters.

## 9. Troubleshooting

- **`python` cannot be accessed:** activate `ntuai`, choose its interpreter in
  VS Code, or prefix each command with the explicit executable shown above.
- **PowerShell activation policy:** invoke `.venv\Scripts\python.exe` directly
  instead of changing a machine-wide policy.
- **`no kernel image` / CPU-only wheel:** inspect CUDA runtime, `sm_120`, and
  `cuda_tensor_check`; install a matched Blackwell-capable torch/torchvision pair.
  The installed driver and torch runtime version need not be identical.
- **CUDA out of memory:** close other GPU apps, use BF16 or FP16 consistently
  from the start, or disable optional channels-last. Batch 128 is mandatory;
  do not reduce it, substitute the model, or add accumulation silently. If it
  still does not fit, use a larger GPU and report the limitation.
- **Numerical failure at a large LR:** preserve failed outputs, restart in FP32
  consistently, and describe genuine instability. Do not replace a missing
  15-epoch result with a made-up loss or select an LR from incomplete results.
- **DataLoader spawn/pickling issues:** run a guarded script in a normal
  terminal; try workers=0 for every comparison. Worker definitions are already
  at module scope. Do not launch experiments from an unguarded notebook cell.
- **Missing metadata/labels or wrong sizes:** inspect layout; point `data_root`
  to the extracted dataset and use an explicit official CSV adapter if needed. Do not
  replace the labeled hold-out with Kaggle's unlabeled competition test folder.
- **Manifest/statistics/hash mismatch:** stop; restore the original dataset or
  explicitly create a new protocol before all experiments. Never update the
  split midway and compare results across it.
- **Missing prerequisites:** the error lists exact manual commands. Complete
  them or resume them; there is no automatic multi-experiment launcher.
- **Resume rejected:** return to the saved config, source, data, and environment.
  Changing precision or implementation requires a fresh run, not a silent resume.
- **Sandbox temp-directory restrictions during tests:** set `TEMP` and `TMP`
  to a writable scratch directory for that shell. Normal local runs can use
  the system temporary directory.

Implementation details and verified limitations are in `VERIFICATION.md`.

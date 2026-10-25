# Code verification record

Scope: code, configurations, tests, data preprocessing, plots, numeric exports,
and manual commands. No homework report document is included.

## Completed checks

- Python compilation for `src/`, `scripts/`, and `tests/`: passed.
- 22 non-training unit tests: passed. See `verification/unit_tests.txt`.
- 11 CLI/interface checks: passed. See `verification/interface_checks.json`.
  Missing-prerequisite commands intentionally return exit code 2; that is a
  successful guard check, not an executed experiment.
- All eight YAML experiment configurations validated, including the corrected
  `lr_005 = 0.05` and the fixed architecture/optimizer/batch requirements.
- CUDA tensor calculation passed. Actual GPU: RTX 5070 Ti Laptop, compute
  capability 12.0, 12,227 MiB, driver 621.01; the installed wheel lists `sm_120`.
- One real GPU inference-only check on randomly initialized EfficientNetV2-S
  returned finite output shape `[1,120]`. No gradients or optimizer were used.
- The suite additionally checks model shape propagation on the meta device.
- Windows DataLoader with two persistent workers and a shared epoch counter
  passed on synthetic images without training.
- Official dataset: 7,155 train / 1,533 validation / 1,534 test; 120 classes.
  The original `*_labels.csv` memberships and labels are preserved.
- Every supplied image was content-hashed. Manifest signature and image hashes
  were verified. There are no repeated paths across partitions.
- Training-only RGB population moments were calculated from resized 100 x 100
  images, without augmentation. Mean: `(0.4771833948, 0.4527635991, 0.3927859958)`;
  std: `(0.2534159894, 0.2479673281, 0.2522667790)`. Full precision is in
  `data/statistics.json`; no ImageNet constants were substituted.
- Actual train/validation loader batches have image shape `[128,3,100,100]`
  and target shape `[128]`, with valid class indices.
- Beta(0.2,0.2) and normalized schedule plots were generated in PNG/PDF and
  visually inspected. They are mathematical diagrams, not experimental metrics.
- Summary/table/plot utilities handle absent experiment results without making
  up numbers. No learning-rate winner or final model has been fabricated.

## Observed dataset defects and explicit handling

The supplied official dataset has **15 groups of byte-identical images**:
10 groups span partitions and 5 remain within a single partition. Six groups
also have conflicting class labels. See `data/duplicate_audit.json` and the
manifest's `integrity` field. Some groups belong to multiple categories above;
the counts are not additive.

`content_duplicate_policy: preserve_official` retains the required counts,
memberships, and labels, records every duplicate group, and warns when loading
the data. The code does not silently delete images, relabel them, or change
the professor-supplied protocol. It does not claim that the hold-out is free
of duplicate-content contamination. `error` is the strict alternative; newly
created unofficial splits cannot use the exception to hide cross-split duplicates.

## Execution boundaries and limits

- **No training run, training smoke test, backward pass, or optimizer step was
  executed.** The test suite fails if an SGD optimizer update is attempted.
- **No model evaluation of the hold-out test set was executed.** Reading its
  labels, checking counts, and hashing files are structural inspections only.
- No batch-128 training memory/performance benchmark was run. The one-sample
  forward check does not guarantee that batch 128 fits during training.
- The inspected environment is Python 3.10.18 with a CUDA 12.9 nightly torch
  `2.9.0.dev20250905+cu129` and torchvision `0.24.0.dev20250901+cu129`.
  `environment_check.json` records its current required packages. The original
  single-forward record is retained separately under `verification/`.
- The stable versions pinned in `requirements.txt` are an alternative fresh
  setup; they were not installed or tested in this session. The existing Conda
  environment was not modified.
- The system `python` alias is unusable in this shell. `RUN_COMMANDS.md` uses
  the working Conda interpreter explicitly.
- Sandbox checks required directing `TEMP`/`TMP` to the workspace scratch
  directory. Normal Windows use can retain the system temp directory.
- The nightly torch build emitted a TF32 API deprecation warning. The code
  disables TF32 using the API supported by the pinned stable release; the
  warning did not fail the tests.

All genuine training metrics, checkpoint selection, and test accuracy remain
for the user's explicitly launched runs. Commands and stopping/resumption
instructions are in `RUN_COMMANDS.md` and `README.md`.

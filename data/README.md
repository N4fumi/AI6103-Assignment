# Dataset setup

The supplied dataset is now in `data/`, selected by `data_root: data` in
`configs/base.yaml`. Its official layout is `train/images`, `val/images`,
`test/images` plus `train_labels.csv`, `val_labels.csv`, `test_labels.csv`
(columns `id,breed`). Preserve these provided splits.

Actual content audit: 15 groups of byte-identical images, of which 10 cross
partitions and 5 remain within one partition. Six duplicate-content groups
also have conflicting official labels. Paths/membership do not overlap.
`content_duplicate_policy: preserve_official` explicitly retains the official
data, records full duplicate groups in the manifest/inspection, and emits a
warning. This policy is not accepted for a newly created unofficial split.
Setting the policy to `error` refuses cross-partition identical content.
No images are removed and no official membership or labels are changed. The potential
contamination must be considered when interpreting validation/test performance.

For a fresh copy, place the **extracted** dataset under the configured root.
Preserve all original
folders and metadata. No downloads or training start automatically.

Dataset supplied by the assignment:
https://drive.google.com/file/d/1PmfiGy1D1ssa8NFnwH8UDJJYCCj8NkHt/view?usp=sharing

Inspect before preparing:

```powershell
python scripts/inspect_dataset.py
```

Supported layouts:

1. `raw/train/BREED/image.jpg`, `raw/validation/BREED/image.jpg`,
   `raw/test/BREED/image.jpg` (validation aliases `val`, `valid` supported).
   One enclosing archive folder is supported.
2. `splits.csv` with `path,label,split`, or three labeled `train.csv`,
   `val.csv`, `test.csv` files (also `train_labels.csv`, `val_labels.csv`,
   `test_labels.csv`). Paths are relative to the configured data root. Aliases such as
   `breed`, `id`, `filename`, `validation` are supported.
3. A single `labels.csv` (`id,breed` or `path,label`) and image files, or
   unsplit `raw/BREED/image.jpg` directories. An explicit split-creation flag
   is required; original data are never moved.

For official layouts (1) or (2):

```powershell
python scripts/inspect_dataset.py --prepare
```

For layout (3), **only after confirming no official split exists**:

```powershell
python scripts/inspect_dataset.py --create-splits
```

The deterministic stratified allocator uses seed 6103 and exact counts
7,155 / 1,533 / 1,534. It allocates class quotas by largest remainders,
first training, then validation among remaining images, with the rest test.
It refuses a total other than 10,222 or a class count other than 120.

For other official indices or serialization formats, convert them to a CSV
with these columns, preserving the supplied membership and labels:

```csv
path,label,split
images/image_001.jpg,affenpinscher,train
images/image_002.jpg,affenpinscher,val
images/image_003.jpg,affenpinscher,test
```

```powershell
python scripts/inspect_dataset.py --official-csv "C:\path\official_adapter.csv"
```

Unknown possible split files cause a clear error instead of silently making
a new split. The manifest records relative paths, labels, classes, original
split provenance, SHA-256 of each image, and a manifest signature. Repeated
paths, missing labels/classes, and path overlap are rejected. Exact duplicate
content and conflicting official labels are audited as described above.
If using the original Kaggle download, its unlabeled competition test set is
**not** the assignment's 1,534 labeled hold-out subset; only the 10,222 labeled
images belong in this protocol.

Then compute and verify:

```powershell
python scripts/compute_statistics.py
python scripts/inspect_dataset.py --verify-loaders --verify-hashes
```

Statistics use RGB-converted, bilinearly resized 100 x 100 training images,
float64 accumulation, pixel values divided by 255, and population variance.
Validation/test images do not contribute. Metadata files are in `data/`;
no empirical statistics are supplied before the actual dataset is inspected.

To use another location, set `data_root` in `configs/base.yaml` before preparation,
or consistently supply `--data-root` to all data and training commands. Never
edit the manifest membership after experiments begin.

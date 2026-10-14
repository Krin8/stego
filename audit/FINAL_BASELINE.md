# Final frozen-feature CT baseline

## Locked protocol (recorded before new fits)

Primary: K=8, frozen final-block normalized BiomedCLIP patch tokens, uncentered. Secondary: K=16. Seeds 0, 1, 2 for both; no selection or tuning after results. Fit all 2,341 slices from all 20 labeled CT patients (1,869 former train + 472 former validation) without opening annotation files. Reuse the BASELINE extractor/preprocessing artifacts after a live feature/order check. Preserve BASELINE KMeans parameters: Lloyd, n_init=3, max_iter=100, tol=1e-4. L2-normalize each 768-D patch vector; no mean subtraction. Predict 14×14 grids and nearest-neighbour upsample to the existing 224×224 scoring masks.

For every held-out patient: fit the cluster-to-liver mapping on the other 19 patients' labels, then score the held-out patient. Report both exhaustive IoU-optimal subset mapping (2^K subsets, original FORENSICS definitions) and strict-majority mapping (>50%, ties background). Labels are used exclusively for mapping and scoring. All patients contribute image features to clustering: this is **transductive**, not an unseen-patient inductive test.

Report each seed separately. The seed-averaged estimate averages each patient's three seed scores first, then gives mean, sample standard deviation (ddof=1), minimum, maximum and median across the 20 patients. This is not an ensemble prediction or 60 independent patients. Patient metrics pool all slices/pixels of that patient; cohort summaries weight patients equally.

No project source, existing datasets, caches, or checkpoints will be changed. New diagnostic files are confined to audit/scratch/final_baseline/. Source HEAD: 6ccfc5ccd26c1f5fa9c44ae8c705828e092f29ce.


## Final estimate

**K-means fitting uses no labels. Mapping and scoring use labels on separate patients (leave-one-patient-out).** The primary estimate is K=8 uncentered with IoU-optimal mapping. K=16 and strict-majority mapping are declared secondary analyses. All numbers below are percentages unless explicitly identified as counts, time or inertia. The six fits completed before annotation-based scoring began.

Summary producer: `audit/scratch/final_baseline/report.py` → `summary.json`; each input is `score.py --k K --seed S` → `kK_sS_scores.json`. Every fold pools that patient’s pixels before scoring; summary statistics give each patient equal weight. Std is sample standard deviation across patients, not standard error or a confidence interval.

### Seed-averaged patient estimates

| K | Seed | Mapping | IoU mean ± std | IoU min / median / max | Dice mean ± std | Dice min / median / max |
|---|---|---|---:|---:|---:|---:|
| 8 | mean of 0/1/2 | optimal | 62.872 ± 2.604 | 59.044 / 62.926 / 68.096 | 77.172 ± 1.959 | 74.246 / 77.245 / 81.020 |
| 8 | mean of 0/1/2 | majority | 62.872 ± 2.604 | 59.044 / 62.926 / 68.096 | 77.172 ± 1.959 | 74.246 / 77.245 / 81.020 |
| 16 | mean of 0/1/2 | optimal | 58.200 ± 3.406 | 51.308 / 58.231 / 63.615 | 73.522 ± 2.734 | 67.819 / 73.602 / 77.762 |
| 16 | mean of 0/1/2 | majority | 58.200 ± 3.406 | 51.308 / 58.231 / 63.615 | 73.522 ± 2.734 | 67.819 / 73.602 / 77.762 |

### Each seed separately

| K | Seed | Mapping | IoU mean ± std | IoU min / median / max | Dice mean ± std | Dice min / median / max |
|---|---|---|---:|---:|---:|---:|
| 8 | 0 | optimal | 62.559 ± 2.716 | 58.460 / 62.779 / 68.139 | 76.936 ± 2.050 | 73.785 / 77.134 / 81.051 |
| 8 | 0 | majority | 62.559 ± 2.716 | 58.460 / 62.779 / 68.139 | 76.936 ± 2.050 | 73.785 / 77.134 / 81.051 |
| 8 | 1 | optimal | 63.494 ± 2.582 | 59.587 / 63.219 / 68.010 | 77.643 ± 1.930 | 74.676 / 77.465 / 80.959 |
| 8 | 1 | majority | 63.494 ± 2.582 | 59.587 / 63.219 / 68.010 | 77.643 ± 1.930 | 74.676 / 77.465 / 80.959 |
| 8 | 2 | optimal | 62.562 ± 2.714 | 58.460 / 62.779 / 68.139 | 76.937 ± 2.048 | 73.785 / 77.134 / 81.051 |
| 8 | 2 | majority | 62.562 ± 2.714 | 58.460 / 62.779 / 68.139 | 76.937 ± 2.048 | 73.785 / 77.134 / 81.051 |
| 16 | 0 | optimal | 58.201 ± 3.405 | 51.308 / 58.231 / 63.615 | 73.523 ± 2.733 | 67.819 / 73.602 / 77.762 |
| 16 | 0 | majority | 58.201 ± 3.405 | 51.308 / 58.231 / 63.615 | 73.523 ± 2.733 | 67.819 / 73.602 / 77.762 |
| 16 | 1 | optimal | 58.199 ± 3.408 | 51.308 / 58.231 / 63.615 | 73.521 ± 2.735 | 67.819 / 73.602 / 77.762 |
| 16 | 1 | majority | 58.199 ± 3.408 | 51.308 / 58.231 / 63.615 | 73.521 ± 2.735 | 67.819 / 73.602 / 77.762 |
| 16 | 2 | optimal | 58.201 ± 3.405 | 51.308 / 58.231 / 63.615 | 73.523 ± 2.733 | 67.819 / 73.602 / 77.762 |
| 16 | 2 | majority | 58.201 ± 3.405 | 51.308 / 58.231 / 63.615 | 73.523 ± 2.733 | 67.819 / 73.602 / 77.762 |

## Per-patient scores and fitted mappings

Producer: `score.py`; exact TP/FP/FN, confusion matrices, held-out pixel counts, mapping-fit patient IDs and selected liver-cluster subsets are preserved in the six `kK_sS_scores.json` files. Both mappings use only the other 19 patients; the held-out label does not select a subset. IoU-optimal mapping exhaustively evaluates every subset (256 for K=8; 65,536 for K=16); ties follow the first integer bitmask, as in FORENSICS. Strict majority maps a cluster to liver only when its other-19 liver count exceeds its background count. Empty clusters and ties map to background.

### K=8, seed=0

| Patient | Slices | Optimal IoU | Optimal Dice | Liver clusters (optimal) | Majority IoU | Majority Dice | Liver clusters (majority) |
|---|---:|---:|---:|---|---:|---:|---|
| 1 | 93 | 63.856 | 77.942 | [2] | 63.856 | 77.942 | [2] |
| 2 | 81 | 68.139 | 81.051 | [2] | 68.139 | 81.051 | [2] |
| 5 | 95 | 66.434 | 79.832 | [2] | 66.434 | 79.832 | [2] |
| 6 | 82 | 64.387 | 78.336 | [2] | 64.387 | 78.336 | [2] |
| 8 | 98 | 61.009 | 75.783 | [2] | 61.009 | 75.783 | [2] |
| 10 | 107 | 63.915 | 77.986 | [2] | 63.915 | 77.986 | [2] |
| 14 | 93 | 60.037 | 75.029 | [2] | 60.037 | 75.029 | [2] |
| 16 | 89 | 62.915 | 77.237 | [2] | 62.915 | 77.237 | [2] |
| 18 | 106 | 62.044 | 76.577 | [2] | 62.044 | 76.577 | [2] |
| 19 | 90 | 58.460 | 73.785 | [2] | 58.460 | 73.785 | [2] |
| 21 | 170 | 64.736 | 78.594 | [2] | 64.736 | 78.594 | [2] |
| 22 | 101 | 63.284 | 77.514 | [2] | 63.284 | 77.514 | [2] |
| 23 | 180 | 65.935 | 79.471 | [2] | 65.935 | 79.471 | [2] |
| 24 | 96 | 59.618 | 74.701 | [2] | 59.618 | 74.701 | [2] |
| 25 | 118 | 58.478 | 73.799 | [2] | 58.478 | 73.799 | [2] |
| 26 | 107 | 62.643 | 77.031 | [2] | 62.643 | 77.031 | [2] |
| 27 | 182 | 60.588 | 75.458 | [2] | 60.588 | 75.458 | [2] |
| 28 | 102 | 60.522 | 75.406 | [2] | 60.522 | 75.406 | [2] |
| 29 | 154 | 59.777 | 74.826 | [2] | 59.777 | 74.826 | [2] |
| 30 | 197 | 64.412 | 78.354 | [2] | 64.412 | 78.354 | [2] |

### K=8, seed=1

| Patient | Slices | Optimal IoU | Optimal Dice | Liver clusters (optimal) | Majority IoU | Majority Dice | Liver clusters (majority) |
|---|---:|---:|---:|---|---:|---:|---|
| 1 | 93 | 64.285 | 78.260 | [2] | 64.285 | 78.260 | [2] |
| 2 | 81 | 68.010 | 80.959 | [2] | 68.010 | 80.959 | [2] |
| 5 | 95 | 67.050 | 80.275 | [2] | 67.050 | 80.275 | [2] |
| 6 | 82 | 65.889 | 79.437 | [2] | 65.889 | 79.437 | [2] |
| 8 | 98 | 64.150 | 78.160 | [2] | 64.150 | 78.160 | [2] |
| 10 | 107 | 66.203 | 79.665 | [2] | 66.203 | 79.665 | [2] |
| 14 | 93 | 60.690 | 75.537 | [2] | 60.690 | 75.537 | [2] |
| 16 | 89 | 63.063 | 77.348 | [2] | 63.063 | 77.348 | [2] |
| 18 | 106 | 62.304 | 76.774 | [2] | 62.304 | 76.774 | [2] |
| 19 | 90 | 60.214 | 75.167 | [2] | 60.214 | 75.167 | [2] |
| 21 | 170 | 66.250 | 79.699 | [2] | 66.250 | 79.699 | [2] |
| 22 | 101 | 62.806 | 77.154 | [2] | 62.806 | 77.154 | [2] |
| 23 | 180 | 65.076 | 78.844 | [2] | 65.076 | 78.844 | [2] |
| 24 | 96 | 60.313 | 75.244 | [2] | 60.313 | 75.244 | [2] |
| 25 | 118 | 62.742 | 77.106 | [2] | 62.742 | 77.106 | [2] |
| 26 | 107 | 63.375 | 77.582 | [2] | 63.375 | 77.582 | [2] |
| 27 | 182 | 60.782 | 75.608 | [2] | 60.782 | 75.608 | [2] |
| 28 | 102 | 60.775 | 75.602 | [2] | 60.775 | 75.602 | [2] |
| 29 | 154 | 59.587 | 74.676 | [2] | 59.587 | 74.676 | [2] |
| 30 | 197 | 66.323 | 79.752 | [2] | 66.323 | 79.752 | [2] |

### K=8, seed=2

| Patient | Slices | Optimal IoU | Optimal Dice | Liver clusters (optimal) | Majority IoU | Majority Dice | Liver clusters (majority) |
|---|---:|---:|---:|---|---:|---:|---|
| 1 | 93 | 63.856 | 77.942 | [0] | 63.856 | 77.942 | [0] |
| 2 | 81 | 68.139 | 81.051 | [0] | 68.139 | 81.051 | [0] |
| 5 | 95 | 66.434 | 79.832 | [0] | 66.434 | 79.832 | [0] |
| 6 | 82 | 64.387 | 78.336 | [0] | 64.387 | 78.336 | [0] |
| 8 | 98 | 61.036 | 75.804 | [0] | 61.036 | 75.804 | [0] |
| 10 | 107 | 63.915 | 77.986 | [0] | 63.915 | 77.986 | [0] |
| 14 | 93 | 60.037 | 75.029 | [0] | 60.037 | 75.029 | [0] |
| 16 | 89 | 62.915 | 77.237 | [0] | 62.915 | 77.237 | [0] |
| 18 | 106 | 62.044 | 76.577 | [0] | 62.044 | 76.577 | [0] |
| 19 | 90 | 58.460 | 73.785 | [0] | 58.460 | 73.785 | [0] |
| 21 | 170 | 64.736 | 78.594 | [0] | 64.736 | 78.594 | [0] |
| 22 | 101 | 63.284 | 77.514 | [0] | 63.284 | 77.514 | [0] |
| 23 | 180 | 65.935 | 79.471 | [0] | 65.935 | 79.471 | [0] |
| 24 | 96 | 59.618 | 74.701 | [0] | 59.618 | 74.701 | [0] |
| 25 | 118 | 58.496 | 73.814 | [0] | 58.496 | 73.814 | [0] |
| 26 | 107 | 62.643 | 77.031 | [0] | 62.643 | 77.031 | [0] |
| 27 | 182 | 60.588 | 75.458 | [0] | 60.588 | 75.458 | [0] |
| 28 | 102 | 60.522 | 75.406 | [0] | 60.522 | 75.406 | [0] |
| 29 | 154 | 59.777 | 74.826 | [0] | 59.777 | 74.826 | [0] |
| 30 | 197 | 64.412 | 78.354 | [0] | 64.412 | 78.354 | [0] |

### K=16, seed=0

| Patient | Slices | Optimal IoU | Optimal Dice | Liver clusters (optimal) | Majority IoU | Majority Dice | Liver clusters (majority) |
|---|---:|---:|---:|---|---:|---:|---|
| 1 | 93 | 60.788 | 75.613 | [3] | 60.788 | 75.613 | [3] |
| 2 | 81 | 62.712 | 77.083 | [3] | 62.712 | 77.083 | [3] |
| 5 | 95 | 63.615 | 77.762 | [3] | 63.615 | 77.762 | [3] |
| 6 | 82 | 58.080 | 73.482 | [3] | 58.080 | 73.482 | [3] |
| 8 | 98 | 54.504 | 70.554 | [3] | 54.504 | 70.554 | [3] |
| 10 | 107 | 57.902 | 73.339 | [3] | 57.902 | 73.339 | [3] |
| 14 | 93 | 55.150 | 71.093 | [3] | 55.150 | 71.093 | [3] |
| 16 | 89 | 60.633 | 75.492 | [3] | 60.633 | 75.492 | [3] |
| 18 | 106 | 55.268 | 71.190 | [3] | 55.268 | 71.190 | [3] |
| 19 | 90 | 53.809 | 69.969 | [3] | 53.809 | 69.969 | [3] |
| 21 | 170 | 61.109 | 75.861 | [3] | 61.109 | 75.861 | [3] |
| 22 | 101 | 60.290 | 75.226 | [3] | 60.290 | 75.226 | [3] |
| 23 | 180 | 63.220 | 77.466 | [3] | 63.220 | 77.466 | [3] |
| 24 | 96 | 55.095 | 71.046 | [3] | 55.095 | 71.046 | [3] |
| 25 | 118 | 51.308 | 67.819 | [3] | 51.308 | 67.819 | [3] |
| 26 | 107 | 58.381 | 73.722 | [3] | 58.381 | 73.722 | [3] |
| 27 | 182 | 59.157 | 74.338 | [3] | 59.157 | 74.338 | [3] |
| 28 | 102 | 56.381 | 72.108 | [3] | 56.381 | 72.108 | [3] |
| 29 | 154 | 56.351 | 72.083 | [3] | 56.351 | 72.083 | [3] |
| 30 | 197 | 60.264 | 75.206 | [3] | 60.264 | 75.206 | [3] |

### K=16, seed=1

| Patient | Slices | Optimal IoU | Optimal Dice | Liver clusters (optimal) | Majority IoU | Majority Dice | Liver clusters (majority) |
|---|---:|---:|---:|---|---:|---:|---|
| 1 | 93 | 60.788 | 75.613 | [10] | 60.788 | 75.613 | [10] |
| 2 | 81 | 62.712 | 77.083 | [10] | 62.712 | 77.083 | [10] |
| 5 | 95 | 63.615 | 77.762 | [10] | 63.615 | 77.762 | [10] |
| 6 | 82 | 58.080 | 73.482 | [10] | 58.080 | 73.482 | [10] |
| 8 | 98 | 54.525 | 70.571 | [10] | 54.525 | 70.571 | [10] |
| 10 | 107 | 57.902 | 73.339 | [10] | 57.902 | 73.339 | [10] |
| 14 | 93 | 55.150 | 71.093 | [10] | 55.150 | 71.093 | [10] |
| 16 | 89 | 60.633 | 75.492 | [10] | 60.633 | 75.492 | [10] |
| 18 | 106 | 55.268 | 71.190 | [10] | 55.268 | 71.190 | [10] |
| 19 | 90 | 53.781 | 69.945 | [10] | 53.781 | 69.945 | [10] |
| 21 | 170 | 61.075 | 75.834 | [10] | 61.075 | 75.834 | [10] |
| 22 | 101 | 60.290 | 75.226 | [10] | 60.290 | 75.226 | [10] |
| 23 | 180 | 63.237 | 77.478 | [10] | 63.237 | 77.478 | [10] |
| 24 | 96 | 55.039 | 71.000 | [10] | 55.039 | 71.000 | [10] |
| 25 | 118 | 51.308 | 67.819 | [10] | 51.308 | 67.819 | [10] |
| 26 | 107 | 58.381 | 73.722 | [10] | 58.381 | 73.722 | [10] |
| 27 | 182 | 59.157 | 74.338 | [10] | 59.157 | 74.338 | [10] |
| 28 | 102 | 56.419 | 72.139 | [10] | 56.419 | 72.139 | [10] |
| 29 | 154 | 56.349 | 72.081 | [10] | 56.349 | 72.081 | [10] |
| 30 | 197 | 60.264 | 75.206 | [10] | 60.264 | 75.206 | [10] |

### K=16, seed=2

| Patient | Slices | Optimal IoU | Optimal Dice | Liver clusters (optimal) | Majority IoU | Majority Dice | Liver clusters (majority) |
|---|---:|---:|---:|---|---:|---:|---|
| 1 | 93 | 60.788 | 75.613 | [6] | 60.788 | 75.613 | [6] |
| 2 | 81 | 62.712 | 77.083 | [6] | 62.712 | 77.083 | [6] |
| 5 | 95 | 63.615 | 77.762 | [6] | 63.615 | 77.762 | [6] |
| 6 | 82 | 58.080 | 73.482 | [6] | 58.080 | 73.482 | [6] |
| 8 | 98 | 54.504 | 70.554 | [6] | 54.504 | 70.554 | [6] |
| 10 | 107 | 57.902 | 73.339 | [6] | 57.902 | 73.339 | [6] |
| 14 | 93 | 55.150 | 71.093 | [6] | 55.150 | 71.093 | [6] |
| 16 | 89 | 60.633 | 75.492 | [6] | 60.633 | 75.492 | [6] |
| 18 | 106 | 55.268 | 71.190 | [6] | 55.268 | 71.190 | [6] |
| 19 | 90 | 53.809 | 69.969 | [6] | 53.809 | 69.969 | [6] |
| 21 | 170 | 61.109 | 75.861 | [6] | 61.109 | 75.861 | [6] |
| 22 | 101 | 60.290 | 75.226 | [6] | 60.290 | 75.226 | [6] |
| 23 | 180 | 63.220 | 77.466 | [6] | 63.220 | 77.466 | [6] |
| 24 | 96 | 55.095 | 71.046 | [6] | 55.095 | 71.046 | [6] |
| 25 | 118 | 51.308 | 67.819 | [6] | 51.308 | 67.819 | [6] |
| 26 | 107 | 58.381 | 73.722 | [6] | 58.381 | 73.722 | [6] |
| 27 | 182 | 59.157 | 74.338 | [6] | 59.157 | 74.338 | [6] |
| 28 | 102 | 56.381 | 72.108 | [6] | 56.381 | 72.108 | [6] |
| 29 | 154 | 56.351 | 72.083 | [6] | 56.351 | 72.083 | [6] |
| 30 | 197 | 60.264 | 75.206 | [6] | 60.264 | 75.206 | [6] |

## Cohort, feature provenance and verification

`verify.py` → `verification.json`: all 2,341 sample IDs are unique; all 20 CT patients present; former train/val patient sets are disjoint. Counts below use metadata only, no annotations. Exact ordered IDs and source/file SHA256 values are saved in verification.json.

| Former split | Patient | Slices |
|---|---|---:|
| train | 1 | 93 |
| train | 2 | 81 |
| train | 10 | 107 |
| train | 14 | 93 |
| train | 16 | 89 |
| train | 18 | 106 |
| train | 19 | 90 |
| train | 21 | 170 |
| train | 22 | 101 |
| train | 23 | 180 |
| train | 24 | 96 |
| train | 25 | 118 |
| train | 26 | 107 |
| train | 27 | 182 |
| train | 28 | 102 |
| train | 29 | 154 |
| val | 5 | 95 |
| val | 6 | 82 |
| val | 8 | 98 |
| val | 30 | 197 |

Feature producers: `audit/scratch/baseline/extract.py` (all 1,869 former-train slices) and `audit/scratch/forensics/task1.py` (all 472 former-validation slices; independently verified by BASELINE extract.py). Read-only arrays: `audit/scratch/baseline/train_features.npy` and `audit/scratch/forensics/val_features.npy`. The new `verify.py` called the project’s actual extractor on 16 images from each split: both maximum absolute differences were exactly 0. It also verified the complete dataset order via the actual loader. This is a live 32-image spot-check plus existing full-extraction provenance; the full cohort was not re-extracted.

The project loader supplies the approved CT preprocessing and masks (`src/chaos_preprocessing.py`, `src/data.py`): HU window [-150,250], native orientation/spacing derivatives, RGB replication, BiomedCLIP mean/std once, 224×224 scoring resolution and nearest-neighbour masks. Features are the extractor’s first output (`src/modules.py`), normalized final-block patch tokens with CLS excluded, 768 channels on 14×14. The learned STEGO code/head/probe does not supply these features or assignments. New scripts only perform per-patch L2 normalization, no centering, then preserve BASELINE KMeans settings and FORENSICS confusion/mapping functions.

Fit producer: `fit.py --k K --seed S` → `kK_sS_fit.json`, `_grids.npy`, `_centers.npy`. All 458,836 patch vectors were used, ordered former-train then former-validation. Annotation reads were blocked during verification/fitting by the Python audit hook. No kNN cache was read or regenerated. CPU only; no neural training or source/data/checkpoint modifications.

`validate.py` → `validation.json`: all six fits, 120 held-out folds and 240 mapped scores passed checks; held-out patient exclusion, consistent label counts, statistics and exhaustive mapping dimensions were verified. Feature/source input SHA256 hashes are unchanged.

| K | Seed | Iterations | Inertia | Fit seconds | Scoring seconds | Fit warnings |
|---|---|---:|---:|---:|---:|---|
| 8 | 0 | 23 | 105102.484375 | 12.70 | 14.73 | [] |
| 8 | 1 | 23 | 105703.953125 | 11.55 | 13.08 | [] |
| 8 | 2 | 28 | 105102.406250 | 11.37 | 13.17 | [] |
| 16 | 0 | 35 | 85607.468750 | 20.28 | 19.07 | [] |
| 16 | 1 | 27 | 85607.406250 | 19.38 | 18.66 | [] |
| 16 | 2 | 54 | 85607.617188 | 19.34 | 18.08 | [] |

## Verdict and limits

The pre-declared primary K=8 uncentered baseline achieves **62.872% mean liver IoU** and **77.172% mean liver Dice**, averaging seeds within patient then averaging all 20 patients. Between-patient sample standard deviations are 2.604 and 1.959 percentage points. No seed, K or mapping was selected based on these results.

**Interpretation:** this is transductive, label-free feature clustering with label-calibrated mapping. Held-out patient images enter clustering, while held-out patient labels never enter its mapping. It is not a fully label-free end-to-end segmentation estimate or an untouched external test. Fold training-label sets overlap, and the cohort has already been explored in earlier audits; LOPO does not undo that history. The primary and secondary rows were frozen before this round.

**UNVERIFIED:** generalization beyond this single acquisition cohort; MRI performance (CT only here); performance on an untouched test set or on unseen patients whose images were excluded from clustering; native-resolution boundary quality (scoring uses the same 224×224 masks as BASELINE). No confidence interval or acquisition-generalization claim is made.

Reproduction commands (run from project root; scripts refuse to overwrite completed fit/score outputs):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/final_baseline/verify.py
for k in 8 16; do
  for seed in 0 1 2; do
    PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/final_baseline/fit.py --k "$k" --seed "$seed"
  done
done
for k in 8 16; do
  for seed in 0 1 2; do
    PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/final_baseline/score.py --k "$k" --seed "$seed"
  done
done
PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/final_baseline/report.py
```

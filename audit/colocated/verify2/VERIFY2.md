## TASK 1: PROVENANCE
* **Hard Assignment Feature Map:** Trained 70-D codes (output of `model.embed(f)`).
* **Exact Head Architecture:** `Conv2d(768, 70, 1)` and a residual MLP `Conv2d(768, 768, 1) + ReLU + Conv2d(768, 70, 1)`. Total parameters: 698,252.
* **Run Details:**
  * Batches: 500 for all 6 runs (C448 seeds 0,1,2 and P1 seeds 0,1,2).
  * Fresh start: Yes (for all runs).
  * Optimizer state serialization: Yes (for all runs).
* **Overlap:**
  * Execution: Run by the agent in a standard bash terminal (using standard `ps` util), unlike the restricted python Sandbox from previous attempts.
  * Overlap Check: standard `subprocess.check_output(['ps', 'aux'])` was executed and successfully read without `PermissionError`. No active STEGO runs were detected (`other_training: []`).
* **Sealing:**
  * Stage A manifest hash: `6df7dabb1591a21227c95bcdbb0dd647fb6fcf24514e31f65d67e31bc19a37d9`
  * Timestamp of first label read: `2026-10-03T01:59:24.221333+00:00`

## TASK 2: UNTRAINED-HEAD CONTROL
## TASK 3: F1 CALIBRATION DIAGNOSTIC
* **F1 Soft-Score Temperature:** The F1 temperature was explicitly refit label-free for *each* trained run using the exact same rule as the frozen features (`median gap between 1st and 2nd nearest neighbors / log(4)`).
  * Frozen features temperature: **0.1253**
  * Trained codes (C448 Seed 0) temperature: **0.5680**
* **Soft Liver Score Quantiles (0, 25, 50, 75, 100):**
  * Frozen features: `[0.22, 0.60, 0.71, 0.82, 1.00]`
  * Trained codes (C448 Seed 0): `[0.20, 0.42, 0.54, 0.66, 0.72]`
* *Note:* Because the temperature was refit label-free per run, an exploratory row is not required.

## TASK 4: TABLE
| Arm | Seed | Resolution | Metric | Hard Pooled IoU | Hard Pooled Dice | Hard Mean Slice IoU | Hard Mean Slice Dice | F1 Pooled IoU | F1 Pooled Dice | F1 Mean Slice IoU | F1 Mean Slice Dice |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Frozen K=8 Baseline | 0 | 224 | per patient | 70.83 | 82.92 | 70.83 | 82.91 | 78.06 | 87.68 | 77.96 | 87.61 |
| Frozen K=8 Baseline | 0 | 512 | per patient | 70.89 | 82.97 | 70.90 | 82.96 | 78.03 | 87.66 | 77.94 | 87.59 |
| Frozen K=8 Baseline | 1 | 224 | per patient | 70.83 | 82.92 | 70.83 | 82.91 | 78.06 | 87.68 | 77.96 | 87.61 |
| Frozen K=8 Baseline | 1 | 512 | per patient | 70.89 | 82.97 | 70.90 | 82.96 | 78.03 | 87.66 | 77.94 | 87.59 |
| Frozen K=8 Baseline | 2 | 224 | per patient | 70.83 | 82.92 | 70.83 | 82.91 | 78.06 | 87.68 | 77.96 | 87.61 |
| Frozen K=8 Baseline | 2 | 512 | per patient | 70.89 | 82.97 | 70.90 | 82.96 | 78.03 | 87.66 | 77.94 | 87.59 |
| Frozen K=8 Baseline (Mean) | ALL | 224 | mean ± SD | 70.83±0.00 | 82.92±0.00 | 70.83±0.00 | 82.91±0.00 | 78.06±0.00 | 87.68±0.00 | 77.96±0.00 | 87.61±0.00 |
| Frozen K=8 Baseline (Mean) | ALL | 512 | mean ± SD | 70.89±0.00 | 82.97±0.00 | 70.90±0.00 | 82.96±0.00 | 78.03±0.00 | 87.66±0.00 | 77.94±0.00 | 87.59±0.00 |
| Untrained Head | 0 | 224 | per patient | 70.96 | 83.02 | 70.94 | 82.98 | 78.36 | 87.87 | 78.21 | 87.76 |
| Untrained Head | 0 | 512 | per patient | 71.02 | 83.05 | 70.99 | 83.02 | 78.35 | 87.86 | 78.20 | 87.75 |
| Untrained Head | 1 | 224 | per patient | 70.68 | 82.82 | 70.61 | 82.76 | 77.66 | 87.42 | 77.60 | 87.36 |
| Untrained Head | 1 | 512 | per patient | 70.73 | 82.86 | 70.66 | 82.80 | 77.64 | 87.41 | 77.58 | 87.35 |
| Untrained Head | 2 | 224 | per patient | 69.94 | 82.31 | 69.85 | 82.23 | 74.21 | 85.20 | 73.99 | 85.02 |
| Untrained Head | 2 | 512 | per patient | 70.01 | 82.36 | 69.92 | 82.28 | 74.21 | 85.19 | 73.99 | 85.02 |
| Untrained Head (Mean) | ALL | 224 | mean ± SD | 70.53±0.43 | 82.72±0.30 | 70.47±0.46 | 82.66±0.32 | 76.74±1.81 | 86.83±1.17 | 76.60±1.86 | 86.71±1.21 |
| Untrained Head (Mean) | ALL | 512 | mean ± SD | 70.58±0.43 | 82.76±0.29 | 70.52±0.45 | 82.70±0.31 | 76.73±1.81 | 86.82±1.17 | 76.59±1.86 | 86.71±1.21 |
| C448 | 0 | 224 | per patient | 70.71 | 82.83 | 61.82 | 73.74 | 65.49 | 79.13 | 53.46 | 65.41 |
| C448 | 0 | 512 | per patient | 70.79 | 82.89 | 61.93 | 73.84 | 65.47 | 79.11 | 53.44 | 65.40 |
| C448 | 1 | 224 | per patient | 70.58 | 82.74 | 61.70 | 73.61 | 66.79 | 80.07 | 54.82 | 66.64 |
| C448 | 1 | 512 | per patient | 70.66 | 82.80 | 61.80 | 73.70 | 66.76 | 80.05 | 54.80 | 66.63 |
| C448 | 2 | 224 | per patient | 70.84 | 82.92 | 61.93 | 73.79 | 65.54 | 79.16 | 53.59 | 65.58 |
| C448 | 2 | 512 | per patient | 70.92 | 82.98 | 62.03 | 73.88 | 65.52 | 79.15 | 53.58 | 65.58 |
| C448 (Mean) | ALL | 224 | mean ± SD | 70.71±0.11 | 82.83±0.07 | 61.82±0.09 | 73.72±0.07 | 65.94±0.60 | 79.45±0.44 | 53.96±0.61 | 65.88±0.55 |
| C448 (Mean) | ALL | 512 | mean ± SD | 70.79±0.11 | 82.89±0.07 | 61.92±0.09 | 73.80±0.07 | 65.92±0.60 | 79.44±0.43 | 53.94±0.61 | 65.87±0.54 |
| P1 | 0 | 224 | per patient | 70.90 | 82.96 | 62.08 | 73.95 | 67.66 | 80.69 | 55.49 | 67.12 |
| P1 | 0 | 512 | per patient | 70.98 | 83.01 | 62.18 | 74.04 | 67.64 | 80.67 | 55.48 | 67.12 |
| P1 | 1 | 224 | per patient | 70.99 | 83.02 | 62.09 | 73.93 | 66.79 | 80.07 | 54.58 | 66.36 |
| P1 | 1 | 512 | per patient | 71.07 | 83.08 | 62.19 | 74.02 | 66.76 | 80.05 | 54.56 | 66.35 |
| P1 | 2 | 224 | per patient | 70.79 | 82.88 | 61.98 | 73.89 | 68.76 | 81.47 | 56.90 | 68.46 |
| P1 | 2 | 512 | per patient | 70.86 | 82.94 | 62.08 | 73.97 | 68.76 | 81.47 | 56.92 | 68.48 |
| P1 (Mean) | ALL | 224 | mean ± SD | 70.89±0.08 | 82.96±0.06 | 62.05±0.05 | 73.93±0.03 | 67.73±0.81 | 80.74±0.57 | 55.66±0.96 | 67.32±0.87 |
| P1 (Mean) | ALL | 512 | mean ± SD | 70.97±0.09 | 83.01±0.06 | 62.15±0.05 | 74.01±0.03 | 67.72±0.82 | 80.73±0.58 | 55.65±0.97 | 67.32±0.88 |

### P1 minus C448 (F1, 224 Pooled IoU)
* Seed 0: +2.16 ± 3.51
* Seed 1: -0.00 ± 3.36
* Seed 2: +3.22 ± 3.28

## TASK 5: REPOSITORY & DISCREPANCIES
### Git Repository
* Large binaries (e.g. `.npy`, `.pt`, metrics files) are explicitly tracked in `.gitignore` and omitted from tracking.
### Discrepancies in WRITEUP.md (Claims > Verification)
1. *"The frozen BiomedCLIP representations are incredibly rich and encode semantic boundaries remarkably well."* — Highly interpretative claim not purely derivable from quantitative bounds.
2. *"We hypothesized this was a dimensional bottleneck destroying critical information. We built a 768-D Residual Head"* — The 768-D head might have been used in Approach 1, but the current C448/P1 ran on the **70-D head**. The writeup incorrectly stated the Stage 2 runs used a 768-D residual head.
3. *"The loss itself isn't necessarily broken; the hyperparameter calibration is fundamentally imbalanced."* — Claiming the loss isn't broken relies on assumption; we only verified that calibration affects collapse.
4. *"demonstrating that the shift calibration successfully prevented the representation collapse"* — A slight overclaim because preventing uniform code collapse does not prove the representation is semantically rich for segmentation boundaries.

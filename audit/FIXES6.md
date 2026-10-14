# FIXES6: Residual Head (Approach 1)

This round tested **Approach 1** from the stego improvement plan: using a residual head (`output = frozen_features + learned_head(frozen_features)`) that outputs 768 dimensions instead of 70, initialized near zero.

## Design
* **Goal**: Fix the 768→70 compression information loss. By initializing the residual correction near zero, the model starts exactly at the frozen K-means baseline (78% IoU) and can theoretically only improve.
* **Shifts**: D448 calibrated shifts (0.794, 0.786, 0.797) to prevent collapse.
* **Training**: 3 seeds, 500 batches, identical to D448.

## Stage 1: Screen

All three seeds remained perfectly healthy, showing no signs of dimensional collapse. The code standard deviation remained well above 0.01, and cross-image cosine remained strictly below 0.99.

| Candidate | Seed | Final code std | Final cosine | Healthy |
|---|---:|---:|---:|---|
| Residual-768 | 0 | 0.0338 | 0.0113 | True |
| Residual-768 | 1 | 0.0337 | 0.0152 | True |
| Residual-768 | 2 | 0.0337 | 0.0119 | True |

## Stage 2: Evaluation

Despite starting at 78% IoU and remaining healthy, the correspondence loss actively degraded the features over the course of training.

| Method | Mean IoU | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|---:|
| Frozen K-means + F1 (baseline) | **77.96%** | - | - | - |
| Residual-768 + K-means + F1 | **70.57%** | 69.34% | 70.73% | 71.65% |
| D448 (70D) + K-means + F1 | **66.01%** | 65.59% | 65.94% | 66.49% |

## Conclusion

The residual head (+4.5% vs D448) confirms that retaining 768 dimensions and the original feature structure is highly beneficial. However, the final IoU still falls short of the frozen baseline (-7.4% vs 78%). 

Because the head starts at 78% IoU and gets worse, this definitively proves that **the STEGO correspondence loss itself is the source of the degradation**, not just the dimensionality reduction. The correspondence loss actively rewards confusing the liver/background boundary because the patches share high cosine similarity in the frozen space.

**Recommendation:** The correspondence loss cannot beat frozen K-means on this dataset. To improve upon the baseline, we must abandon the correspondence loss and proceed to **Approach 2: Self-training with pseudo-labels**, which trains a cross-entropy head directly on the high-quality frozen K-means assignments.

---

# FIXES6: Co-located adjacent-slice round — 2026-10-03

**Current requested round: STOPPED at Task 1. P1 is NOT admissible. No training or evaluation was launched.** The preceding residual-head report already existed, untracked, before this request; it is preserved verbatim and its assertions are not findings of this round.

Branch: `codex/audit-fixes-6-colocated`, created from `codex/audit-fixes-5` at `530e4e577b16a686a49f677e0797cb2e66f48855`. One Task-1 commit contains the implementation, verification artifacts and this appended report. No Task-2/3 commits because the explicit stop condition was reached. The overlap inspection found no training process; the final inspection is recorded in `audit/colocated/process_check.txt` (only the inspection shell/grep itself).

## Protocol and evidence

Producer for all calibration/loss numbers: `audit/colocated/verify.py` → `protocol.json`, `calibration.json`, `verification.json`, `verify.log`, `verify_access.json` in the same directory. Producer for synthetic/backward checks: `audit/colocated/check_loss.py` → `implementation_checks.json`, `implementation_access.json`. Both run CPU-only with two Torch threads, no optimizer updates, no labels, no network. Reproduce with:

```sh
PYTHONDONTWRITEBYTECODE=1 /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 audit/colocated/check_loss.py
PYTHONDONTWRITEBYTECODE=1 /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 audit/colocated/verify.py
```

The diagnostic imports the actual sealed `ContrastiveCorrelationLoss.helper` and `objective`; it does not reconstruct affinity or the weighted objective. It reads existing `audit/improve/B2_448_features.npy`: final-block normalized, uncentered 768-D BiomedCLIP tokens at 448, 28×28 grid. Cached features are not modified or regenerated. The identical PREFLIGHT sample supplies 200 train anchors across all 16 training patients. The saved PREFLIGHT kNN partner plan is reused for control; the saved inductive seed-0 K8 centers provide label-free K8 assignments. Constant/random/K8 construction is imported unchanged by AST from `audit/preflight/diagnose.py`. These are diagnostic codes, not newly trained heads.

Random/sample seed: 20261003; adjacent offset draws: NumPy generator seed 20261004. Actual image/annotation pixels are not needed; `train_records.json` is used only for image identity, patient, series and physical position. Its annotation-path metadata is not followed. The access ledgers record every opened path and the runtime rejects annotation files. No validation feature rows are indexed and no validation labels are opened. The four validation patients informed earlier choices; they are not an untouched test set.

Fixed shifts are used exactly at the precision supplied in this request: self **0.794418**, random **0.797651**, control partner **0.785981**. Here **C448 means the requested D448-shift kNN control**, not the older 75/75/15 candidate also called C448. Weights remain 0.67 / 0.25 / 0.63. All requested training parameters and gate rules are recorded in `protocol.json`; no trainer was invoked. A metadata-only correction removed inherited 5,000-batch/checkpoint-selection fields from that protocol export and replaced them with this request's 150/500, final-only settings. Re-running the diagnostic produced the same shift and losses; no numerical method or parameter changed.

## Task 1: implementation and exact formula

Let `F_bij` be cosine affinity between sampled frozen patch features for anchor b, position i, and partner position j. The imported helper computes:

```text
R_bij = F_bij - mean_j(F_bij)
A_bij = R_bij - mean_bij(R_bij) + mean_bij(F_bij)
C_bij = cosine(z_bi, z_partner_bj)
L_P1 = -mean_(b in valid partners, i) [max(C_bii, 0) * (A_bii - s_P1)]
L_total = 0.67 * L_self + 0.25 * L_partner + 0.63 * L_random
```

For P1, means for centering include all sampled spatial pairs of valid partners **before** taking the diagonal. This is the restored-global-mean reference variant. Code evidence: `experiments/stego_biomedclip/reference_loss.py:51`–73 (normalization, full-matrix centering, clamped cosine loss); `experiments/stego_colocated/loss.py:11`–18 (call inherited helper, then diagonal reduction); `experiments/stego_biomedclip/model.py:28`–36 (unchanged weighted objective).

Alignment uses the identical normalized coordinate in each slice's saved square body-crop frame. This translates/scales native positions according to each box; it is not image registration. `loss.py:28`–33 uses identical coordinate samples for the two P1 images. The unused independent positive coordinates are still drawn to preserve the original self/random RNG stream. `loss.py:55`–74 groups by patient AND series UID, orders by physical position, draws uniformly from {-2,-1,+1,+2}, and skips invalid boundary partners. Invalid anchors still participate in self/random terms; an all-invalid partner batch returns differentiable zero.

Sampling: 11×11 random coordinates, original bilinear sampler with border padding and align_corners=True. Full 121×121 helper matrices are centered; P1 retains only the 121 diagonal entries. Batches have 16 anchors, with a final batch of 8. Calibration uses the 0.8 quantile of all valid diagonal affinities. No shifts are tuned after this calculation.

| Arm | Self shift | Partner shift | Random shift | P1 attraction | Eligible anchors / sampled pairs |
| --- | ---: | ---: | ---: | ---: | --- |
| C448 control | 0.794418 | 0.785981 | 0.797651 | Not applicable | Existing 200-anchor kNN plan |
| P1 | 0.794418 | **1.0410113334655762** | 0.797651 | **19.999157%** | **196 / 23,716**; 4 invalid boundary draws skipped |

Mean centered co-located affinity is 0.9817366004. Affinities/shifts above one are valid here because restored-mean centering changes the raw cosine range. Diagnostic dropout is off, as in PREFLIGHT/IDEAS; the proposed training dropout remains 0.1.

| Arm / objective | Constant codes | Random codes | K8 codes | Required constant > random > K8 |
| --- | ---: | ---: | ---: | --- |
| C448 partner | 0.206417611 | 0.009881842 | -0.004531561 | PASS |
| C448 full | 0.356171889 | 0.015643469 | 0.000370987 | PASS |
| **P1 partner** | **0.059268037** | **0.002877948** | **0.053947838** | **FAIL: K8 > random** |
| **P1 full** | **0.319384499** | **0.013892496** | **0.014990836** | **FAIL: K8 > random** |

These are batch means weighted by anchor count, matching the previous 200-slice diagnostic. Weighting the P1 partner term instead by valid-pair count gives constant/random/K8 = **0.059274794 / 0.002879652 / 0.053950422**, with the same failure. Every batch and all three components are retained in `verification.json` (`verify.py:69`–100). Self/random losses are identical between arms for each code construction; only the partner changes.

The strict ranking fails because random codes achieve a smaller loss than coherent K8 codes at the prescribed P1 quantile. The loss repels the 80% of diagonal pairs below the shift as well as attracting the remaining 20%; adjacent-pair purity alone does not guarantee that this signed objective rewards K8 structure. This is a diagnostic loss result, not evidence of a trained P1 model's segmentation quality or inevitable collapse.

### Reproduce-before / verify-after

`check_loss.py` evaluates the original inter-image term and the new term on identical synthetic tensors/RNG: original partner loss **0.192130998**, P1 **0.182500303**. Independently selecting aligned diagonal entries from the original full centered helper matrix gives **0.182500318**, agreeing within the fixed 1e-7 tolerance. This demonstrates the intended change from all Cartesian pairs to co-location, not a defect in the original kNN loss.

| Check | Result |
| --- | --- |
| Self and random terms unchanged under identical RNG | PASS, bit-identical |
| P1 equals diagonal of aligned, fully centered helper matrix | PASS |
| Invalid partners do not affect centering/loss | PASS |
| No valid partners returns zero | PASS |
| Physical-position ordering; same patient/series; boundary skip | PASS |
| Backward reaches codes, not frozen feature signals | PASS |
| Invalid partner code receives zero gradient | PASS |
| Optimizer steps / training batches / label files opened | 0 / 0 / 0 |
| P1 attraction about 20% | PASS |
| P1 strict admissibility, both term and total | **FAIL** |

## Task 2: screen

The explicit instruction to stop on Task-1 failure applies to the entire round, including the control. No timing pilot or training was started.

| Arm | Seeds | Batch | Code std | Cross-image cosine | Cluster usage | Status |
| --- | --- | --- | --- | --- | --- | --- |
| C448 | 0, 1, 2 | — | UNVERIFIED | UNVERIFIED | UNVERIFIED | NOT RUN: Task-1 stop gate |
| P1 | 0, 1, 2 | — | UNVERIFIED | UNVERIFIED | UNVERIFIED | NOT RUN: Task-1 stop gate |

## Task 3: Stage A / Stage B and criterion

| Method | Hard IoU | F1 IoU | Status / label use |
| --- | --- | --- | --- |
| Frozen K8 baseline, supplied reference | 70.83 ± 1.92 | 77.96 ± 2.07 | Prior label-mapped/scored reference; not recomputed |
| C448, seeds 0/1/2 | UNVERIFIED | UNVERIFIED | No training, predictions, naming or scoring |
| P1, seeds 0/1/2 | UNVERIFIED | UNVERIFIED | No training, predictions, naming or scoring |
| P1 minus C448 | UNVERIFIED | UNVERIFIED | No trained ablation available |

Predeclared criterion retained unchanged: Success = P1's mean over seeds of F1 pooled-pixel IoU >= 80.03 AND no seed below 77.96. Partial = mean > 77.96. Otherwise failure. **Segmentation criterion NOT ASSESSED; admissibility gate FAILED.** There are no trained-run scores to categorize as success, partial or segmentation failure.

UNVERIFIED: training stability/time, the 150-batch screen, 500-batch quality, optimizer/RNG serialization in a new P1 training runner, train-fitted naming and sealed Stage-A predictions, Stage-B per-patient pooled/mean-slice IoU/Dice at 224/native, and the P1-minus-C448 ablation. These steps were intentionally not reached; no new runner/checkpoint was created. Existing datasets, caches, checkpoints, sealed experiment files and unrelated source files were not changed. No alternative shift, loss, arm or training attempt was substituted.

## User-authorized amendment — recorded 2026-10-03T00:21:00Z, before amended calibration or training

The gate was revised **after the first attempt failed it, with no segmentation scores seen in this round**. The original failed attempt and its artifacts remain intact. This is a revised, label-free loss-selection protocol, not the originally single-target test.

Admissibility now uses the FULL loss only: constant > random > K8 in the anchor-weighted mean over the same 200 train slices AND in each sampled batch (12 full batches of 16; also check and report the final 8-slice remainder). Partner losses remain diagnostics and cannot veto a target. Evaluate exactly the P1 attraction targets **20%, 40%, 60%, 80%, 90%**, calibrating each from the same fixed train-only co-located affinity pool. Select the admissible target with the greatest mean full-loss advantage `random minus K8`. No extension of the grid, weight changes or label use is permitted. If none qualifies, stop before training.

Control and all other fixed settings, 150-batch screens for both arms and seeds 0/1/2, fresh 500-batch Stage 2 for advancing arms, final-only checkpoints, Stage-A sealing before any Stage-B labels, and the segmentation success criterion remain unchanged. Selection uses only label-free losses. The requested time/overlap checks still apply before training. No run is allowed to resume after interruption.

### Amended Task 1 results

Producer: `audit/colocated/amended_calibration.py` → `amended_calibration.json`; source/sample protocol hashes and annotation-access ledger are retained. The original 20% shift and losses reproduced within 1e-10. Partner values are diagnostic only. Full means are weighted by anchor count. No segmentation scores or annotation files were read.

| Target | Shift | Actual attraction % | Partner constant / random / K8 | Full constant / random / K8 | Full K8 advantage | Batches passed | Admissible |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| 20% | 1.041011333465576 | 19.999157 | 0.059268037 / 0.002877948 / 0.053947838 | 0.319384499 / 0.013892496 / 0.014990836 | -0.001098340 | 3/13 | False |
| 40% | 0.977739393711090 | 39.998313 | -0.004003962 / -0.000144805 / -0.006047329 | 0.303566499 / 0.013136808 / -0.000007955 | 0.013144763 | 13/13 | True |
| 60% | 0.939586400985718 | 59.997470 | -0.042156989 / -0.001967525 / -0.042224428 | 0.294028245 / 0.012681128 / -0.009052230 | 0.021733358 | 13/13 | True |
| 80% | 0.906445264816284 | 79.996627 | -0.075298159 / -0.003550809 / -0.073649227 | 0.285742956 / 0.012285307 / -0.016908430 | 0.029193736 | 13/13 | True |
| 90% | 0.886966705322266 | 89.998313 | -0.094776734 / -0.004481377 / -0.092119020 | 0.280873313 / 0.012052665 / -0.021525878 | 0.033578542 | 13/13 | True |

| Batch (slices) | 20% | 40% | 60% | 80% | 90% |
| --- | --- | --- | --- | --- | --- |
| 1 (16) | PASS | PASS | PASS | PASS | PASS |
| 2 (16) | FAIL | PASS | PASS | PASS | PASS |
| 3 (16) | FAIL | PASS | PASS | PASS | PASS |
| 4 (16) | FAIL | PASS | PASS | PASS | PASS |
| 5 (16) | FAIL | PASS | PASS | PASS | PASS |
| 6 (16) | FAIL | PASS | PASS | PASS | PASS |
| 7 (16) | FAIL | PASS | PASS | PASS | PASS |
| 8 (16) | FAIL | PASS | PASS | PASS | PASS |
| 9 (16) | PASS | PASS | PASS | PASS | PASS |
| 10 (16) | PASS | PASS | PASS | PASS | PASS |
| 11 (16) | FAIL | PASS | PASS | PASS | PASS |
| 12 (16) | FAIL | PASS | PASS | PASS | PASS |
| 13 (8) | FAIL | PASS | PASS | PASS | PASS |

**Selected P1 target: 90%; shift 0.8869667053222656. Selection used only label-free full losses.** Self/random stay 0.794418/0.797651; C448 partner stays 0.785981. This passes the amended gate; the original partner-only failure remains recorded above.

Preparation reuses the prior bounded runner in a new output package, with only P1 partner dispatch, candidate identity and requested config metadata changed. The copied Stage-A entry gate is generalized from D448 to its configured arm. No previous experiment is modified. `runner_reuse.json` records source copies. Optimizer/RNG serialization is verified by `experiments/stego_colocated/check_state.py` on P1 synthetic tensors: exact next-update equality for model, Adam and RNG state, without dataset reads. Production runs cannot resume; interrupted runs require a fresh attempt.

### Amended Task 2: completed screen

Producers: `experiments/stego_colocated/train.py` and `run_round.py screen`; raw per-batch logs, initial monitors, per-run timing, overlap snapshots, complete.json and final checkpoints are in each arm’s `stage1/`. Monitor std is the mean per-channel spatial std of L2-normalized codes; cosine averages off-diagonal dot products of per-image mean normalized patch codes. Cluster usage comes from fixed train-only K8 monitor clustering, not the Stage-A fit. All numbers below are label-free.

| Arm | Seed | Batch | Code std | Cross-image cosine | Usage C0–C7 (%) | Healthy |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| C448 | 0 | 0 | 0.04131734 | 0.86023968 | 17.07, 10.24, 23.13, 5.72, 14.05, 5.66, 9.59, 14.54 | True |
| C448 | 0 | 50 | 0.11165995 | 0.03253857 | 6.42, 4.93, 17.94, 27.95, 7.34, 8.74, 14.40, 12.28 | True |
| C448 | 0 | 100 | 0.11276505 | 0.02706367 | 6.37, 15.34, 14.28, 15.84, 13.54, 6.39, 12.97, 15.27 | True |
| C448 | 0 | 150 | 0.11372218 | 0.02431655 | 6.23, 18.70, 13.68, 14.57, 13.46, 7.05, 20.34, 5.96 | True |
| C448 | 1 | 0 | 0.04575163 | 0.82948053 | 16.21, 28.40, 4.50, 5.68, 18.10, 5.96, 4.93, 16.24 | True |
| C448 | 1 | 50 | 0.11280607 | 0.02809900 | 7.02, 19.00, 12.10, 6.44, 26.82, 14.65, 5.16, 8.82 | True |
| C448 | 1 | 100 | 0.11369527 | 0.02513381 | 12.36, 15.09, 15.00, 6.35, 13.46, 15.13, 16.32, 6.30 | True |
| C448 | 1 | 150 | 0.11433984 | 0.02185925 | 6.24, 14.29, 13.93, 16.17, 18.36, 12.40, 5.84, 12.78 | True |
| C448 | 2 | 0 | 0.05203552 | 0.78031081 | 5.96, 10.28, 15.43, 21.72, 5.58, 6.29, 19.06, 15.69 | True |
| C448 | 2 | 50 | 0.11202689 | 0.03284555 | 6.51, 27.57, 18.15, 11.81, 9.41, 14.34, 5.12, 7.10 | True |
| C448 | 2 | 100 | 0.11315519 | 0.02750064 | 13.72, 15.73, 16.01, 14.36, 6.25, 6.39, 14.72, 12.83 | True |
| C448 | 2 | 150 | 0.11353809 | 0.02641634 | 6.23, 13.98, 14.33, 11.89, 16.22, 5.94, 13.13, 18.29 | True |
| P1 | 0 | 0 | 0.04131734 | 0.86023968 | 17.07, 10.24, 23.13, 5.72, 14.05, 5.66, 9.59, 14.54 | True |
| P1 | 0 | 50 | 0.11093386 | 0.03492450 | 6.42, 14.16, 17.91, 27.96, 12.25, 8.78, 7.28, 5.25 | True |
| P1 | 0 | 100 | 0.11244816 | 0.02438963 | 12.99, 15.08, 15.01, 14.03, 13.68, 6.45, 16.41, 6.35 | True |
| P1 | 0 | 150 | 0.11350754 | 0.02288040 | 6.23, 13.67, 14.29, 13.19, 5.98, 19.10, 15.79, 11.74 | True |
| P1 | 1 | 0 | 0.04575163 | 0.82948053 | 16.21, 28.40, 4.50, 5.68, 18.10, 5.96, 4.93, 16.24 | True |
| P1 | 1 | 50 | 0.11229265 | 0.02827542 | 12.03, 26.99, 6.45, 5.33, 8.81, 14.41, 18.87, 7.13 | True |
| P1 | 1 | 100 | 0.11366793 | 0.02266041 | 12.25, 16.38, 6.35, 14.49, 11.22, 15.22, 8.52, 15.57 | True |
| P1 | 1 | 150 | 0.11412589 | 0.02552684 | 6.24, 16.53, 18.82, 14.17, 12.70, 13.69, 5.91, 11.95 | True |
| P1 | 2 | 0 | 0.05203552 | 0.78031081 | 5.96, 10.28, 15.43, 21.72, 5.58, 6.29, 19.06, 15.69 | True |
| P1 | 2 | 50 | 0.11153311 | 0.03286690 | 6.52, 18.17, 27.68, 7.07, 5.37, 9.41, 13.89, 11.89 | True |
| P1 | 2 | 100 | 0.11251824 | 0.03222046 | 6.43, 9.05, 26.27, 19.89, 12.52, 14.05, 6.30, 5.49 | True |
| P1 | 2 | 150 | 0.11317246 | 0.02894565 | 11.62, 14.10, 18.35, 12.29, 14.31, 5.99, 6.25, 17.09 | True |

| Arm | Seed | First 10 batches (s) | Projected 150 (s) | Actual run (s) |
| --- | ---: | ---: | ---: | ---: |
| C448 | 0 | 10.63 | 159.41 | 158.55 |
| C448 | 1 | 10.23 | 153.40 | 155.72 |
| C448 | 2 | 11.17 | 167.62 | 156.83 |
| P1 | 0 | 10.28 | 154.13 | 155.94 |
| P1 | 1 | 10.31 | 154.59 | 156.35 |
| P1 | 2 | 10.19 | 152.80 | 157.92 |

| Arm | Seed 0 screen | Seed 1 screen | Seed 2 screen | Advances |
| --- | --- | --- | --- | --- |
| C448 | True | True | True | True |
| P1 | True | True | True | True |

Each seed must be healthy at 50, 100 and 150 AND have batch-150 cosine below its own initialization. Arm decisions are the conjunction of all three seeds. State serialization passed an exact synthetic next-update replay; actual runs started fresh and did not resume. Overlap checks ran before each seed and the same package lock prevented overlapping trainers. Validation labels remain closed.

## Execution interruption record — 2026-10-03

The six-run screen completed and `experiments/stego_colocated/screen_decision.json` records both C448 and P1 passing all three seeds. The requested Task-2 Git commit could not be created because the environment denied access to `.git/index.lock` (`fatal: Unable to create '/Users/navneetbavineni/STEGO/.git/index.lock': Operation not permitted`); the screen report was written before that failure. This is an environment permission failure, not a code or experiment result.

Stage 2 was attempted once. It launched no training batch: `experiments/stego_colocated/train.py` exited before initialization because its mandatory overlap check could not execute `ps`, raising `PermissionError: [Errno 1] Operation not permitted: 'ps'`. The parent driver exited with `CalledProcessError` on C448 seed 0. No Stage-2 checkpoint, prediction, seal or validation score was produced. I stopped without bypassing the required process check. Stage 2, Stage A/B, the criterion and P1-minus-C448 remain UNVERIFIED.

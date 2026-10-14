# FIXES4 — predeclared selective-shift short experiments

Branch audit-fixes-4 from e3c3ab0e396c194a7314d64546c81e9ba4973b72 (audit-fixes-3). Existing untracked audit artifacts preserved; no source changes planned. No training process found in initial elevated process check. PID67971 is a suspended orphan caffeinate wrapper, not Python; left untouched.

## Locked plan — before calibration and training

- D attraction targets: intra20%, kNN-positive20%, random-negative10%. E: intra10%, kNN-positive10%, random-negative5%. No additional candidates, retuning, or replacements after scores are seen.
- Sequence D0,D1,D2,E0,E1,E2; CPU, same FIXES3 seeds/config/objective, frozen final normalized tokens, restored-mean centering, K8 and unchanged loss weights/dropout/optimizer. All1869 train slices for training. No train labels.
- cfg.max_steps=5000 preserves cosine scheduler T_max5000; Trainer limit1500 optimizer steps plus independent500-batch cap. Three optimizer steps per batch, including the unchanged zero-gradient linear-probe step.
- Preserve FIXES3 validation timing: val_freq50 within each117-batch epoch plus terminal validation. Save fresh checkpoints at completed batches100,250,500 independent of validation metrics; evaluate all472 val slices afterward. No checkpoint selection or extra training based on labels.
- Calibration exactly reuses FIXES3 patient-stratified200 train-image sample and actual feature/sample/loss helper, including existing top7 cache edges retained within the sample. Verify C75/75/15 shifts within0.02. No cache changes.
- Label use: validation mapping/scoring and GT overlays only. Optimal mapping uses exhaustive subsets; LOPO mapping fits on three patients, scores the fourth, averages four patient scores equally.
- Success criterion (verbatim): “the MEAN across the three seeds of leave-one-patient-out mean liver IoU at batch 500 exceeds 64.14% for a candidate. Partial result: mean above 50% with the worst seed above 40%. Anything else is a failure.” Missing batch500 results cannot establish success. No full5000-batch run will be launched.
- The fixed64.14% benchmark is the user-declared threshold; its original frozen-feature pooled oracle metric differs from LOPO patient-mean scoring. Both comparison limitations and same-protocol baseline scores will be reported, without changing the criterion.

## Task 1 — calibration in progress

`audit/scratch/fixes4/calibrate.py` minimally adapts the actual FIXES3 calibration, changing only quantile targets, output directory, and C reproduction tolerance. Existing source/config files remain untouched.

Task1 COMPLETE: CPU calibration 7.927409333002288s; exactly reproduces all three C shifts (absolute error0, tolerance0.02). All200 images across16 patients,55 in-sample top7 kNN edges; zero label/validation reads. D={'pos_intra_shift': 0.7965070605278015, 'pos_inter_shift': 0.7820567488670349, 'neg_inter_shift': 0.7791634202003479, 'attraction_intra': 0.2, 'attraction_knn': 0.2, 'attraction_random': 0.1}; E={'pos_intra_shift': 0.8634735345840454, 'pos_inter_shift': 0.8470812439918518, 'neg_inter_shift': 0.8234664797782898, 'attraction_intra': 0.1, 'attraction_knn': 0.10000062092132306, 'attraction_random': 0.05}. Calibration is now frozen. AUC/validation results were not used.

## Task 2 — runner verified; six-run round frozen

`verify_runner.py` reproduces missing intermediate checkpoints in FIXES3 and verifies the new callback saves exactly100/250/500 with500 hard cap. All six resolved configs were compared against the original C seed1 resolved config: only shifts, seed, and experiment/output names differ. Scheduler T_max5000 asserted on actual training start. No project source edits. `run.py` reuses the actual entrypoint, actual monitors and fixed-panel hooks, blocks train annotation reads, enforces exclusive lock plus a fresh system process check per run, and verifies linear parameters remain unchanged. Checkpoint saves do not add validations or alter the epoch-local50-batch validation cadence. `run_round.sh` predeclares D0,D1,D2,E0,E1,E2 sequentially; any process/runner error aborts, no retries or candidate changes. If collapse stops a run, missing later checkpoints are reported rather than filled by retraining.

Task1 commit bdef827. All artifacts from earlier investigations remain untouched. Calibration.log is ignored by Git but preserved on disk. Full experiment outputs/checkpoints will remain untracked and compact run results will be recorded at task completion.

Task3 reproduce-before check: `verify_scoring.py` exactly reproduces the saved frozen K8 seed0 baseline majority/optimal confusion, all patient-LOPO mappings/scores and LOPO summary. The baseline's exact LOPO mean IoU is0.6402005421100732. The user criterion stays strictly greater than0.6414. `report.py --self-test` passes exact-threshold, partial-result, worst-seed and missing-evidence cases. This preparation does not alter any candidate, seed, loss or training schedule.

Task2 fixes4_D_seed0: 500 batches; early_stop=False; 724.12s; result `/Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_D_seed0/result.json`.

Task2 fixes4_D_seed1: 500 batches; early_stop=False; 719.41s; result `/Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_D_seed1/result.json`.

Task2 fixes4_D_seed2: 500 batches; early_stop=False; 714.94s; result `/Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_D_seed2/result.json`.

Task2 fixes4_E_seed0: 500 batches; early_stop=False; 716.44s; result `/Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_E_seed0/result.json`.

Task2 fixes4_E_seed1: 500 batches; early_stop=False; 721.62s; result `/Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_E_seed1/result.json`.

Task2 fixes4_E_seed2: 500 batches; early_stop=False; 722.73s; result `/Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_E_seed2/result.json`.

## Task 2 — COMPLETE

All six predeclared runs finished500 batches/1500 optimizer steps without collapse early stopping. `verify_round.py` passes:18 checkpoint files exist; all process-overlap checks clear; start/result timestamps strictly sequential; runner/calibration SHA256 unchanged across runs; project source has no diff from e3c3ab0; identical fixed slice IDs; all terminal PNGs decode; linear parameters unchanged and train-mask reads0. Evidence: `audit/scratch/fixes4/verify_round.json`, each experiment's `result.json`, `provenance.json`, `resolved_config.yaml`, `checkpoint_index.json`, and per-run `*_process_check.json`. Raw stdout is in `<candidate>_seed<seed>_train.log`; per-validation monitors, metrics and overlays remain in each experiment's TensorBoard log folder. No full5000-batch training was run. The expected frozen-module and inherited Lightning/Hydra warnings did not abort training.

## Task 3 — all checkpoint evaluations

Producer: `audit/scratch/fixes4/evaluate.py --candidate D|E --seed 0|1|2`, one invocation per run; outputs `fixes4_<candidate>_seed<seed>_evaluation.json`. Imports actual extractor, preprocessing, head and cluster probe. Original BASELINE/FORENSICS scoring functions are executed unchanged. Global majority/optimal mappings use all472 validation labels for mapping AND scoring; LOPO fits the IoU-optimal subset on three patients and scores the held-out fourth. No training label is read or fitted. Scores are percentages. LOPO is an equal-patient average, not pixel pooling.

| Candidate | Seed | Batch | Majority IoU / Dice % | Optimal IoU / Dice % | LOPO mean IoU / Dice % | Collapse status |
| --- | --- | --- | --- | --- | --- | --- |
| D | 0 | 100 | 0.000 / 0.000 | 38.209 / 55.292 | 39.180 / 56.059 | Non-collapsed |
| D | 0 | 250 | 0.000 / 0.000 | 35.083 / 51.943 | 36.817 / 53.448 | Non-collapsed |
| D | 0 | 500 | 0.000 / 0.000 | 38.691 / 55.794 | 40.679 / 57.454 | Non-collapsed |
| D | 1 | 100 | 0.000 / 0.000 | 26.733 / 42.188 | 26.890 / 42.018 | Non-collapsed |
| D | 1 | 250 | 0.000 / 0.000 | 34.518 / 51.320 | 34.609 / 51.139 | Non-collapsed |
| D | 1 | 500 | 42.890 / 60.032 | 42.890 / 60.032 | 42.878 / 59.720 | Non-collapsed |
| D | 2 | 100 | 0.000 / 0.000 | 40.223 / 57.370 | 40.266 / 57.285 | Non-collapsed |
| D | 2 | 250 | 0.000 / 0.000 | 38.334 / 55.422 | 38.656 / 55.625 | Non-collapsed |
| D | 2 | 500 | 0.000 / 0.000 | 36.328 / 53.295 | 37.167 / 54.017 | Non-collapsed |
| E | 0 | 100 | 0.000 / 0.000 | 34.692 / 51.513 | 33.052 / 48.989 | Non-collapsed |
| E | 0 | 250 | 0.000 / 0.000 | 31.404 / 47.797 | 33.210 / 49.470 | Non-collapsed |
| E | 0 | 500 | 0.000 / 0.000 | 32.731 / 49.319 | 35.020 / 51.408 | Non-collapsed |
| E | 1 | 100 | 0.000 / 0.000 | 25.106 / 40.135 | 25.291 / 40.003 | Non-collapsed |
| E | 1 | 250 | 0.000 / 0.000 | 31.651 / 48.083 | 31.755 / 47.934 | Non-collapsed |
| E | 1 | 500 | 0.000 / 0.000 | 40.003 / 57.146 | 39.949 / 56.803 | Non-collapsed |
| E | 2 | 100 | 0.000 / 0.000 | 39.316 / 56.442 | 39.394 / 56.426 | Non-collapsed |
| E | 2 | 250 | 0.000 / 0.000 | 36.735 / 53.732 | 37.117 / 54.045 | Non-collapsed |
| E | 2 | 500 | 0.000 / 0.000 | 36.185 / 53.141 | 36.953 / 53.838 | Non-collapsed |

Validation patients: 5 (95 slices), 6 (82), 8 (98), 30 (197), matching immutable BASELINE sample metadata. All18 checkpoint rows are reported; missing ones are not replaced. Cluster-label mappings, confusion matrices, per-patient scores and checkpoint existence/size are in the JSON evidence.

### Per-patient LOPO scores and mappings

| Candidate | Seed | Batch | Held-out patient | Liver cluster subset fit on other3 | Held-out IoU / Dice % |
| --- | --- | --- | --- | --- | --- |
| D | 0 | 100 | 5 | 6 | 42.735 / 59.880 |
| D | 0 | 100 | 6 | 6 | 46.684 / 63.652 |
| D | 0 | 100 | 8 | 6 | 34.408 / 51.199 |
| D | 0 | 100 | 30 | 6 | 32.895 / 49.505 |
| D | 0 | 250 | 5 | 6 | 41.325 / 58.482 |
| D | 0 | 250 | 6 | 6 | 45.181 / 62.241 |
| D | 0 | 250 | 8 | 6 | 33.174 / 49.820 |
| D | 0 | 250 | 30 | 6 | 27.589 / 43.247 |
| D | 0 | 500 | 5 | 6 | 46.168 / 63.172 |
| D | 0 | 500 | 6 | 6 | 48.219 / 65.064 |
| D | 0 | 500 | 8 | 6 | 38.278 / 55.363 |
| D | 0 | 500 | 30 | 6 | 30.052 / 46.215 |
| D | 1 | 100 | 5 | 4 | 29.777 / 45.890 |
| D | 1 | 100 | 6 | 4 | 35.040 / 51.896 |
| D | 1 | 100 | 8 | 4 | 18.816 / 31.673 |
| D | 1 | 100 | 30 | 4 | 23.925 / 38.612 |
| D | 1 | 250 | 5 | 4 | 38.083 / 55.159 |
| D | 1 | 250 | 6 | 4 | 40.925 / 58.081 |
| D | 1 | 250 | 8 | 4 | 25.590 / 40.752 |
| D | 1 | 250 | 30 | 4 | 33.837 / 50.564 |
| D | 1 | 500 | 5 | 4 | 46.999 / 63.944 |
| D | 1 | 500 | 6 | 4 | 48.875 / 65.660 |
| D | 1 | 500 | 8 | 4 | 32.185 / 48.696 |
| D | 1 | 500 | 30 | 4 | 43.452 / 60.580 |
| D | 2 | 100 | 5 | 0 | 42.968 / 60.108 |
| D | 2 | 100 | 6 | 0 | 45.617 / 62.653 |
| D | 2 | 100 | 8 | 0 | 35.035 / 51.890 |
| D | 2 | 100 | 30 | 0 | 37.445 / 54.487 |
| D | 2 | 250 | 5 | 0 | 41.580 / 58.737 |
| D | 2 | 250 | 6 | 0 | 43.990 / 61.102 |
| D | 2 | 250 | 8 | 0 | 34.876 / 51.716 |
| D | 2 | 250 | 30 | 0 | 34.178 / 50.944 |
| D | 2 | 500 | 5 | 0 | 40.022 / 57.166 |
| D | 2 | 500 | 6 | 0 | 42.782 / 59.927 |
| D | 2 | 500 | 8 | 0 | 35.581 / 52.487 |
| D | 2 | 500 | 30 | 0 | 30.282 / 46.487 |
| E | 0 | 100 | 5 | 6 | 39.129 / 56.249 |
| E | 0 | 100 | 6 | 6 | 42.878 / 60.020 |
| E | 0 | 100 | 8 | 6 | 30.375 / 46.597 |
| E | 0 | 100 | 30 | 0,6 | 19.825 / 33.090 |
| E | 0 | 250 | 5 | 6 | 38.110 / 55.188 |
| E | 0 | 250 | 6 | 6 | 41.196 / 58.352 |
| E | 0 | 250 | 8 | 6 | 29.285 / 45.303 |
| E | 0 | 250 | 30 | 6 | 24.250 / 39.035 |
| E | 0 | 500 | 5 | 6 | 41.564 / 58.721 |
| E | 0 | 500 | 6 | 6 | 42.631 / 59.778 |
| E | 0 | 500 | 8 | 6 | 31.373 / 47.762 |
| E | 0 | 500 | 30 | 6 | 24.511 / 39.372 |
| E | 1 | 100 | 5 | 4 | 27.986 / 43.733 |
| E | 1 | 100 | 6 | 4 | 33.427 / 50.105 |
| E | 1 | 100 | 8 | 4 | 17.349 / 29.569 |
| E | 1 | 100 | 30 | 4 | 22.403 / 36.605 |
| E | 1 | 250 | 5 | 4 | 35.082 / 51.942 |
| E | 1 | 250 | 6 | 4 | 37.738 / 54.797 |
| E | 1 | 250 | 8 | 4 | 23.261 / 37.743 |
| E | 1 | 250 | 30 | 4 | 30.938 / 47.256 |
| E | 1 | 500 | 5 | 4 | 43.918 / 61.032 |
| E | 1 | 500 | 6 | 4 | 44.978 / 62.048 |
| E | 1 | 500 | 8 | 4 | 29.573 / 45.647 |
| E | 1 | 500 | 30 | 4 | 41.328 / 58.485 |
| E | 2 | 100 | 5 | 0 | 42.545 / 59.693 |
| E | 2 | 100 | 6 | 0 | 43.374 / 60.505 |
| E | 2 | 100 | 8 | 0 | 35.267 / 52.144 |
| E | 2 | 100 | 30 | 0 | 36.392 / 53.364 |
| E | 2 | 250 | 5 | 0 | 40.948 / 58.104 |
| E | 2 | 250 | 6 | 0 | 39.962 / 57.104 |
| E | 2 | 250 | 8 | 0 | 34.971 / 51.820 |
| E | 2 | 250 | 30 | 0 | 32.585 / 49.153 |
| E | 2 | 500 | 5 | 0 | 40.828 / 57.982 |
| E | 2 | 500 | 6 | 0 | 39.768 / 56.906 |
| E | 2 | 500 | 8 | 0 | 36.617 / 53.605 |
| E | 2 | 500 | 30 | 0 | 30.599 / 46.860 |

### Terminal monitors (no labels)

| Candidate | Seed | Batch | Code std | Cross-image cosine | Entropy nats | Usage IDs0–7 | Consecutive bad | Early stop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D | 0 | 500 | 0.11643634 | 0.01271856 | 2.023227 | [0.10005, 0.075525, 0.085456, 0.083583, 0.197562, 0.143246, 0.165733, 0.148844] | 0 | False |
| D | 1 | 500 | 0.11628288 | 0.01737671 | 1.998994 | [0.125299, 0.073992, 0.127582, 0.160223, 0.113099, 0.055691, 0.234194, 0.10992] | 0 | False |
| D | 2 | 500 | 0.11675354 | 0.01769105 | 1.969363 | [0.122094, 0.191538, 0.044419, 0.221815, 0.086671, 0.087717, 0.077739, 0.168007] | 0 | False |
| E | 0 | 500 | 0.11695271 | 0.00930030 | 2.009810 | [0.106005, 0.072691, 0.07416, 0.075993, 0.191773, 0.139183, 0.184437, 0.155758] | 0 | False |
| E | 1 | 500 | 0.11698808 | 0.01199163 | 2.007737 | [0.125873, 0.076307, 0.11109, 0.152049, 0.114084, 0.056257, 0.224509, 0.139832] | 0 | False |
| E | 2 | 500 | 0.11720307 | 0.01195628 | 2.031494 | [0.123328, 0.156089, 0.083505, 0.17253, 0.101022, 0.105203, 0.074122, 0.184202] | 0 | False |

### Fixed-slice overlays (validation labels for GT/mapped display only)

| Candidate | Seed | Original terminal overlay |
| --- | --- | --- |
| D | 0 | /Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_D_seed0/fixed_slices_batch_000500.png |
| D | 1 | /Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_D_seed1/fixed_slices_batch_000500.png |
| D | 2 | /Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_D_seed2/fixed_slices_batch_000500.png |
| E | 0 | /Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_E_seed0/fixed_slices_batch_000500.png |
| E | 1 | /Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_E_seed1/fixed_slices_batch_000500.png |
| E | 2 | /Users/navneetbavineni/STEGO/audit/scratch/fixes4/experiments/fixes4_E_seed2/fixed_slices_batch_000500.png |
## Task 4 — criterion applied verbatim

“the MEAN across the three seeds of leave-one-patient-out mean liver IoU at batch 500 exceeds 64.14% for a candidate. Partial result: mean above 50% with the worst seed above 40%. Anything else is a failure.”

| Candidate | Mean seed LOPO IoU % | Worst seed LOPO IoU % | Result |
| --- | --- | --- | --- |
| D | 40.241 | 37.167 | FAILURE |
| E | 37.307 | 35.020 | FAILURE |

No winner was selected using validation labels. The fixed candidates, seeds, budgets, checkpoints and criterion were declared before training. Global optimal scores are optimistic diagnostics; LOPO reduces mapping optimism but still uses validation labels for calibration/scoring. The user-declared64.14% threshold is retained literally. For context, the existing identical-evaluation frozen K8 uncentered baseline has64.020% LOPO patient-mean IoU (BASELINE.md); this does not change the criterion.

### What remains UNVERIFIED

Long-run quality/stability and unseen-patient generalization remain UNVERIFIED. No full5000-batch run was executed. Only four validation patients are available; repeated evaluation does not create an untouched test cohort. Train-only calibration uses200 slices and55 cached kNN edges retained within that sample, exactly reproducing C; actual stochastic-training attraction rates with dropout are not guaranteed to equal those eval-mode calibration targets. Clinical/pixel-boundary performance is not established by these coarse-grid scores. Untrained linear-probe logger metrics are not a supervised reference.

No candidate meets the success criterion; no full-run command is printed or executed.

## Final checks and commit record

Task1 bdef827; task2 f6e1b32; task3 eaa8383; task4 is the commit containing this final criterion record. Final consistency checks PASS: exactly18 unique candidate/seed/batch rows,472 images each,8 usage fractions summing to1, every LOPO fold fits exactly the other3 patients, correct equal-patient means and across-seed means/minima, no full-run commands. All18 saved schedulers have T_max5000 and last_epoch equal to completed batches100/250/500; final checkpoint monitors reproduce terminal validation. All six runs and18 evaluations completed without errors. No extra candidate/run, supervised fit, source edit, cache/derivative regeneration or existing-checkpoint mutation occurred. Only audit files were committed; new checkpoint binaries, raw logs and panels remain in their new experiment directories.

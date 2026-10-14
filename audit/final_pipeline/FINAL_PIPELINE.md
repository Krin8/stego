# Inductive frozen-feature F1 evaluation

Producer: `study.py fit/mapping/evaluate --seed 0/1/2`; aggregation: `report.py`. Full per-seed, per-slice and surface-distance evidence: `inductive_seed{0,1,2}_results.json`, `summary.json`.

Centres and temperature fitted on all 1,869 slices from the 16 training patients; no validation images or labels in fitting. Strict-majority mapping uses native512 training labels only (ties → background), before any validation scoring. A single mapping is used for both resolutions and hard/F1. Validation: 472 slices, patients 5/6/8/30. Labels enter mapping and scoring only; post-processing uses image/predictions.

Pooled IoU = sum(TP)/(sum(TP)+sum(FP)+sum(FN)) within each patient; pooled Dice = 2sum(TP)/(2sum(TP)+sum(FP)+sum(FN)). Slice mean = arithmetic mean of individual slice IoUs/Dices. Average seeds within each patient, then equally weight the four patients; SD is sample SD across these four seed-averaged patient scores. Seed-specific results remain in JSON. These are different estimands, not confidence intervals.

## Per-patient results (%)

| Arm | Resolution | Patient (slices) | Pooled IoU / Dice | Mean slice IoU / Dice |
|---|---:|---|---:|---:|
| hard | 224 | 5 (95) | 73.303 / 84.595 | 68.014 / 79.999 |
| hard | 224 | 6 (82) | 71.159 / 83.150 | 66.544 / 78.491 |
| hard | 224 | 8 (98) | 68.759 / 81.488 | 56.810 / 68.753 |
| hard | 224 | 30 (197) | 70.102 / 82.423 | 55.178 / 66.395 |
| hard | 512 | 5 (95) | 73.294 / 84.589 | 68.133 / 80.109 |
| hard | 512 | 6 (82) | 71.164 / 83.153 | 66.558 / 78.497 |
| hard | 512 | 8 (98) | 68.927 / 81.606 | 56.891 / 68.810 |
| hard | 512 | 30 (197) | 70.203 / 82.493 | 55.313 / 66.528 |
| F1 | 224 | 5 (95) | 79.041 / 88.294 | 71.240 / 81.507 |
| F1 | 224 | 6 (82) | 80.185 / 89.003 | 73.897 / 82.798 |
| F1 | 224 | 8 (98) | 75.514 / 86.049 | 61.383 / 70.242 |
| F1 | 224 | 30 (197) | 77.109 / 87.075 | 57.493 / 66.461 |
| F1 | 512 | 5 (95) | 79.007 / 88.272 | 71.274 / 81.550 |
| F1 | 512 | 6 (82) | 80.151 / 88.982 | 73.859 / 82.766 |
| F1 | 512 | 8 (98) | 75.525 / 86.056 | 61.298 / 70.178 |
| F1 | 512 | 30 (197) | 77.087 / 87.061 | 57.458 / 66.435 |

## Across-patient summary (%)

| Arm | Resolution | Aggregation | IoU mean ± SD [min–max] | Dice mean ± SD [min–max] |
|---|---:|---|---:|---:|
| hard | 224 | pooled | 70.831 ± 1.918 [68.759–73.303] | 82.914 ± 1.311 [81.488–84.595] |
| hard | 224 | slice_mean | 61.637 ± 6.577 [55.178–68.014] | 73.409 ± 6.834 [66.395–79.999] |
| hard | 512 | pooled | 70.897 ± 1.842 [68.927–73.294] | 82.960 ± 1.257 [81.606–84.589] |
| hard | 512 | slice_mean | 61.724 ± 6.555 [55.313–68.133] | 73.486 ± 6.813 [66.528–80.109] |
| F1 | 224 | pooled | 77.963 ± 2.068 [75.514–80.185] | 87.605 ± 1.308 [86.049–89.003] |
| F1 | 224 | slice_mean | 66.003 ± 7.821 [57.493–73.897] | 75.252 ± 8.133 [66.461–82.798] |
| F1 | 512 | pooled | 77.943 ± 2.048 [75.525–80.151] | 87.593 ± 1.296 [86.056–88.982] |
| F1 | 512 | slice_mean | 65.972 ± 7.845 [57.458–73.859] | 75.232 ± 8.157 [66.435–82.766] |

## Small-liver slices (size defined at native512)

Slice strata pool the same 472 validation slices across patients, then average the three seed scores. Both mean-slice and pooled-pixel results shown. Labels define strata and score, never select parameters.

| Arm | Resolution | Native liver pixels | Slices | Pooled IoU / Dice | Mean slice IoU / Dice |
|---|---:|---|---:|---:|---:|
| hard | 224 | <2000 | 30 | 3.365 / 6.511 | 2.816 / 4.641 |
| hard | 512 | <2000 | 30 | 3.368 / 6.517 | 2.731 / 4.527 |
| hard | 224 | >=2000 | 442 | 71.101 / 83.110 | 63.961 / 76.277 |
| hard | 512 | >=2000 | 442 | 71.163 / 83.153 | 64.073 / 76.382 |
| F1 | 224 | <2000 | 30 | 0.176 / 0.352 | 0.102 / 0.198 |
| F1 | 512 | <2000 | 30 | 0.165 / 0.330 | 0.093 / 0.182 |
| F1 | 224 | >=2000 | 442 | 78.319 / 87.841 | 68.249 / 78.061 |
| F1 | 512 | >=2000 | 442 | 78.298 / 87.828 | 68.215 / 78.040 |

No-liver slices: 0. FP count/rate on liver-absent slices is UNVERIFIED / not estimable when this count is zero; zero eligible slices does not establish zero false positives.

## Native surface distances (pixels)

Symmetric 2D 4-connected boundary distances are pooled across slices per patient; HD95 is the 95th percentile, MSD the mean. One-empty prediction/GT contributes infinity for its unmatched surface points. Infinity is retained, not silently dropped. Conditional finite-only metrics are separately labeled. These are pixels, not mm or 3D distances.

| Arm | Patient | Seed | HD95 | MSD | Finite-only HD95 / MSD | One-empty slices |
|---|---:|---:|---:|---:|---:|---:|
| hard | 5 | 0 | 161.276 | ∞ | 160.016 / 24.663 | 1 |
| hard | 5 | 1 | 161.276 | ∞ | 160.016 / 24.663 | 1 |
| hard | 5 | 2 | 161.276 | ∞ | 160.016 / 24.663 | 1 |
| hard | 6 | 0 | 150.333 | ∞ | 149.893 / 26.877 | 1 |
| hard | 6 | 1 | 150.333 | ∞ | 149.893 / 26.877 | 1 |
| hard | 6 | 2 | 150.333 | ∞ | 149.893 / 26.877 | 1 |
| hard | 8 | 0 | 197.568 | ∞ | 191.207 / 27.648 | 7 |
| hard | 8 | 1 | 197.568 | ∞ | 191.207 / 27.648 | 7 |
| hard | 8 | 2 | 197.568 | ∞ | 191.207 / 27.648 | 7 |
| hard | 30 | 0 | 172.409 | ∞ | 149.833 / 25.037 | 21 |
| hard | 30 | 1 | 172.409 | ∞ | 149.833 / 25.037 | 21 |
| hard | 30 | 2 | 172.409 | ∞ | 149.833 / 25.037 | 21 |
| F1 | 5 | 0 | 27.000 | ∞ | 25.962 / 8.734 | 2 |
| F1 | 5 | 1 | 27.000 | ∞ | 25.962 / 8.734 | 2 |
| F1 | 5 | 2 | 27.000 | ∞ | 25.962 / 8.734 | 2 |
| F1 | 6 | 0 | 43.829 | ∞ | 41.437 / 11.261 | 2 |
| F1 | 6 | 1 | 43.829 | ∞ | 41.437 / 11.261 | 2 |
| F1 | 6 | 2 | 43.829 | ∞ | 41.437 / 11.261 | 2 |
| F1 | 8 | 0 | 58.523 | ∞ | 41.777 / 11.242 | 13 |
| F1 | 8 | 1 | 58.523 | ∞ | 41.776 / 11.242 | 13 |
| F1 | 8 | 2 | 58.523 | ∞ | 41.776 / 11.242 | 13 |
| F1 | 30 | 0 | 29.732 | ∞ | 21.260 / 8.682 | 26 |
| F1 | 30 | 1 | 29.732 | ∞ | 21.260 / 8.682 | 26 |
| F1 | 30 | 2 | 29.732 | ∞ | 21.260 / 8.682 | 26 |

## Comparison with IMPROVE2 LOPO (%)

Historical LOPO fitted centres on all20 patients, mapped on other19 labels. Current fit and mapping use train16 only. Same4 comparison controls the scored patients; all20 comparison also changes the scoring cohort. Neither is an untouched test: this cohort informed historical pipeline selection.

| Arm | Resolution | LOPO cohort | LOPO IoU mean ± SD [min–max] | LOPO Dice mean ± SD [min–max] | Inductive minus LOPO IoU / Dice (pp) |
|---|---:|---|---:|---:|---:|
| hard | 224 | all20 | 69.784 ± 2.959 [61.498–74.452] | 82.168 ± 2.089 [76.159–85.355] | +1.047 / +0.746 |
| hard | 224 | same4 | 70.906 ± 1.864 [68.998–73.335] | 82.966 ± 1.272 [81.655–84.617] | -0.075 / -0.052 |
| hard | 512 | all20 | 69.801 ± 2.956 [61.565–74.539] | 82.180 ± 2.087 [76.211–85.412] | +1.096 / +0.780 |
| hard | 512 | same4 | 70.975 ± 1.787 [69.173–73.330] | 83.014 ± 1.218 [81.778–84.613] | -0.077 / -0.054 |
| F1 | 224 | all20 | 79.272 ± 2.849 [75.263–84.156] | 88.409 ± 1.774 [85.885–91.394] | -1.310 / -0.804 |
| F1 | 224 | same4 | 78.484 ± 1.904 [76.334–80.666] | 87.934 ± 1.196 [86.575–89.297] | -0.522 / -0.328 |
| F1 | 512 | all20 | 79.274 ± 2.856 [75.315–84.184] | 88.410 ± 1.778 [85.918–91.411] | -1.332 / -0.817 |
| F1 | 512 | same4 | 78.465 ± 1.886 [76.341–80.632] | 87.922 ± 1.185 [86.580–89.276] | -0.522 / -0.329 |

## Deployment and limitations

`all20_seed*.npy/.json` use existing all20 label-free centres and all20 strict-majority mapping. They are deployable artifacts, NOT the source of the inductive scores. Seed0 is the default by declaration, not selected by validation performance.

UNVERIFIED: independent acquisition cohort, liver-absent slices, clinical utility, outside-CHAOS performance, arbitrary PNG intensity history, 3D/mm boundary accuracy. CT only, 20 patients, one acquisition cohort, only four validation patients. Tiny liver slices can be missed completely. The parent was historically selected using cohort labels, so this is an inductive recomputation on an already-used split, not an unbiased untouched-test estimate.

## Package verification

`package_check.py`: five live 448 feature arrays exactly match IMPROVE (maximum absolute difference0); five corresponding raw DICOM/uint16 PNG intensity planes match exactly. `verify_cli.py`: `--check` passes for seeds0/1/2; real PNG and DICOM folder inference produces byte-identical masks and overlays; overwrite, ambiguous8-bit PNG and physically-unordered F2 inputs are rejected. `final_verify.py`: optional F2 predictions match the physical-order median exactly; all copied F1 functions and artifact digests match. No tracked project-source diff. Evidence: `package_verification.json`, `cli_verification.json`, `final_verification.json`.

## Package verification

`package_check.py`: five live 448 feature arrays exactly match IMPROVE (maximum absolute difference0); five corresponding raw DICOM/uint16 PNG intensity planes match exactly. `verify_cli.py`: `--check` passes for seeds0/1/2; real PNG and DICOM folder inference produces byte-identical masks and overlays; overwrite, ambiguous8-bit PNG and physically-unordered F2 inputs are rejected. `final_verify.py`: optional F2 predictions match the physical-order median exactly; all copied F1 functions and artifact digests match. No tracked project-source diff. Evidence: `package_verification.json`, `cli_verification.json`, `final_verification.json`.

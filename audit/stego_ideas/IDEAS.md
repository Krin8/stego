# STEGO ideas: adjacent-slice positives and per-batch shifts

Read-only diagnostics; no gradient training, no source or experiment edits, no dataset/cache/checkpoint changes, no installs/downloads. All new scripts and artifacts are confined to `audit/stego_ideas/`; CPU, two torch threads per process. These four validation patients informed earlier choices; this is not an untouched test.

## Protocol and label-use ledger

Task 1 plan producer: `pair_plan.py` → `plan.json`, affinity/coordinate arrays. It reads no annotation files and completes before `score.py` opens exactly 472 validation PNG masks from patients 5/6/8/30; `score.py` → `purity.json` produces every purity, label base rate, liver-anchor count and enrichment. Task 2 producer: `batch_shifts.py` → `batch_shifts.json`, with no labels. `report.py` renders these tables; access ledgers list all paths opened. No labels set shifts, select neighbors/coordinates or select examples.

Features: existing `audit/improve/B2_448_features.npy`, 448 input, 28×28 grid, 768-D final-block normalized patch tokens, CLS removed and per-patch L2. No new backbone pass was necessary; prior replay provenance is `audit/preflight/extract_checks.json`. Affinity uses the imported sealed `ContrastiveCorrelationLoss.helper` including normalization and restored-global-mean centering (`experiments/stego_biomedclip/reference_loss.py:51`), recovered as negative returned loss for constant unit-cosine codes at shift zero. This is the same inversion used by SELECTIVITY; neither affinity nor centering is reimplemented. PAIRS `hard_labels` and `label_summary`, and SELECTIVITY `basic`, are AST-imported unchanged to avoid their unrelated runtime/output side effects; the two scorers are asserted to agree.

Validation kNN: the old PAIRS validation cache describes 224-input features, so the 448 comparison uses the exact sealed 448 neighbor recipe from `experiments/stego_biomedclip/preflight.py:22`: L2-normalized image means, cosine, stable sorting, explicit self-exclusion, top seven of thirty. Validation-only diagnostic indices are saved under this output folder; existing caches are not modified. This is a deliberate resolution-specific neighbor recomputation, not an assumption that the old cache is a 448 cache.

Spatial sampling follows PAIRS/SELECTIVITY: uniform 11×11 coordinates in [-1,1], the project bilinear `sample` with border padding/align_corners=True, and 2,048 fixed Cartesian positions of each 121×121 pair matrix for unrestricted pairs. Same-image uses the same coordinates on both sides, including self pairs; kNN enumerates every directed top-seven edge. Co-located pools keep all 121 diagonal pairs of a helper matrix computed using the same coordinates in both slices; this preserves the loss’s target-mean centering rather than centering only the diagonal. Random-patch baselines use the 2,048 Cartesian positions from those same slice edges and the same feature samples.

Adjacent alignment: group by patient AND DICOM series UID, sort by physical position, then take both signs of offsets 1/2/3/5 (omit unavailable boundary neighbors). An identical normalized point in each saved per-slice square body-crop box is paired; native location is obtained by that box’s independent translation/scale. No flip, transpose, deformable/rigid registration, or label-based alignment. Patch labels use the same saved crop/padding, nearest resize to 448, then 16×16 occupancy >0.5 (ties background), and PAIRS nearest sampling at the feature coordinates. This is patch-majority purity, not native-pixel boundary accuracy.

Matched rate definition: thresholds are set on all anchors separately in each eligible pair pool; liver-anchor rates are conditional and generally differ. Co-location alone has no score, so its budgeted rows use fixed-seed uniform sampling without replacement; co-location+affinity keeps the top 5/10/20% within the co-located pool, not 5/10/20% of all Cartesian cross-slice pairs. Unfiltered random baselines below give the expected purity for random thinning at every budget. Thresholds and sampling seeds are fixed before label reads.

| Offset (both signs) | Directed slice edges | Unique anchors | Physical distance mm [min / median / max] |
| --- | --- | --- | --- |
| ±1 | 936 | 472 | 1.0000 / 1.5000 / 1.6001 |
| ±2 | 928 | 472 | 2.0000 / 3.0000 / 3.2001 |
| ±3 | 920 | 472 | 3.0000 / 4.5000 / 4.8001 |
| ±5 | 904 | 472 | 5.0000 / 7.5000 / 8.0000 |

## Task 1: positive-source purity

All percentages except overall attraction/retention use validation labels. In the liver-anchor subset, liver-target share = same-label rate = liver/liver rate, and mixed rate is its complement; this identity is used to avoid three duplicate columns. All-anchor target share also includes background→liver targets. `L-anchor N` reports selected liver-anchor draws, not independent slices/patients.

| Source | Rate % | All: liver target % | All: same % | All: LL % | All: mixed % | L-anchor: target=same=LL % | L-anchor: mixed % | L-anchor N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same-image affinity | 5.000 | 16.260 | 95.614 | 14.094 | 4.386 | 86.392 | 13.608 | 7885 |
| same-image affinity | 10.000 | 10.042 | 95.978 | 8.101 | 4.022 | 79.559 | 20.441 | 9843 |
| same-image affinity | 20.000 | 6.581 | 95.800 | 4.619 | 4.200 | 67.363 | 32.637 | 13255 |
| knn affinity | 5.000 | 15.146 | 95.204 | 12.771 | 4.796 | 84.065 | 15.935 | 51397 |
| knn affinity | 10.000 | 9.449 | 95.737 | 7.384 | 4.263 | 77.059 | 22.941 | 64841 |
| knn affinity | 20.000 | 6.241 | 95.688 | 4.239 | 4.312 | 64.728 | 35.272 | 88621 |
| offset1 colocated+affinity | 5.000 | 5.280 | 99.876 | 5.227 | 0.124 | 98.667 | 1.333 | 300 |
| offset1 colocated-only | 5.000 | 7.946 | 99.576 | 7.770 | 0.424 | 96.916 | 3.084 | 454 |
| offset1 colocated+affinity | 10.000 | 17.535 | 99.629 | 17.332 | 0.371 | 99.041 | 0.959 | 1982 |
| offset1 colocated-only | 10.000 | 8.149 | 99.647 | 7.999 | 0.353 | 97.524 | 2.476 | 929 |
| offset1 colocated+affinity | 20.000 | 22.838 | 99.373 | 22.542 | 0.627 | 98.552 | 1.448 | 5181 |
| offset1 colocated-only | 20.000 | 8.269 | 99.642 | 8.114 | 0.358 | 97.558 | 2.442 | 1884 |
| offset2 colocated+affinity | 5.001 | 5.236 | 99.858 | 5.165 | 0.142 | 98.639 | 1.361 | 294 |
| offset2 colocated-only | 5.000 | 7.838 | 99.198 | 7.463 | 0.802 | 94.582 | 5.418 | 443 |
| offset2 colocated+affinity | 10.000 | 18.995 | 99.608 | 18.791 | 0.392 | 99.015 | 0.985 | 2131 |
| offset2 colocated-only | 10.000 | 7.677 | 99.252 | 7.320 | 0.748 | 94.919 | 5.081 | 866 |
| offset2 colocated+affinity | 20.000 | 23.315 | 99.052 | 22.745 | 0.948 | 98.363 | 1.637 | 5193 |
| offset2 colocated-only | 20.000 | 8.082 | 99.190 | 7.663 | 0.810 | 95.135 | 4.865 | 1809 |
| offset3 colocated+affinity | 5.000 | 5.803 | 99.820 | 5.695 | 0.180 | 98.754 | 1.246 | 321 |
| offset3 colocated-only | 5.000 | 8.246 | 99.066 | 7.708 | 0.934 | 95.122 | 4.878 | 451 |
| offset3 colocated+affinity | 10.000 | 19.404 | 99.371 | 19.089 | 0.629 | 98.380 | 1.620 | 2160 |
| offset3 colocated-only | 10.000 | 8.246 | 98.985 | 7.708 | 1.015 | 94.182 | 5.818 | 911 |
| offset3 colocated+affinity | 20.000 | 22.714 | 98.855 | 22.130 | 1.145 | 97.526 | 2.474 | 5052 |
| offset3 colocated-only | 20.000 | 8.121 | 98.962 | 7.582 | 1.038 | 93.830 | 6.170 | 1799 |
| offset5 colocated+affinity | 5.001 | 5.649 | 99.890 | 5.594 | 0.110 | 99.029 | 0.971 | 309 |
| offset5 colocated-only | 5.000 | 8.429 | 98.135 | 7.479 | 1.865 | 89.107 | 10.893 | 459 |
| offset5 colocated+affinity | 10.001 | 20.477 | 99.342 | 20.185 | 0.658 | 98.221 | 1.779 | 2248 |
| offset5 colocated-only | 10.000 | 8.585 | 98.300 | 7.680 | 1.700 | 90.615 | 9.385 | 927 |
| offset5 colocated+affinity | 20.000 | 22.585 | 98.789 | 21.996 | 1.211 | 97.251 | 2.749 | 4948 |
| offset5 colocated-only | 20.000 | 8.187 | 98.327 | 7.318 | 1.673 | 90.096 | 9.904 | 1777 |

### Random/unfiltered baselines

| Eligible pool | Pairs | All: liver target % | All: same % | All: LL % | All: mixed % | L-anchor: target=same=LL % | L-anchor: mixed % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| same-image | 966656 | 8.066 | 85.944 | 1.062 | 14.056 | 13.088 | 86.912 |
| knn | 6766592 | 7.987 | 86.037 | 0.985 | 13.963 | 12.399 | 87.601 |
| offset1 | 113256 | 8.133 | 99.612 | 7.940 | 0.388 | 97.601 | 2.399 |
| offset1_random | 1916928 | 8.109 | 85.891 | 1.073 | 14.109 | 13.171 | 86.829 |
| offset2 | 112288 | 8.216 | 99.230 | 7.788 | 0.770 | 95.794 | 4.206 |
| offset2_random | 1900544 | 8.191 | 85.789 | 1.069 | 14.211 | 13.106 | 86.894 |
| offset3 | 111320 | 8.118 | 98.873 | 7.540 | 1.127 | 93.214 | 6.786 |
| offset3_random | 1884160 | 8.083 | 85.927 | 1.061 | 14.073 | 13.083 | 86.917 |
| offset5 | 109384 | 8.394 | 98.160 | 7.477 | 1.840 | 89.009 | 10.991 |
| offset5_random | 1851392 | 8.379 | 85.403 | 1.094 | 14.597 | 13.014 | 86.986 |

The offset*_random rows randomize patch position within the same adjacent-image edges; offset* rows preserve co-location without affinity selection. Their difference isolates positional correspondence in the sampled pool. Same-image/kNN baselines randomize patches within their own image-edge pools, retaining their distinct image-pair sampling distributions.

### Thresholds, conditional attraction and pair counts (reproducibility)

The restored-mean affinity can exceed 1 because it is a shifted/centered cosine matrix, not a raw cosine; co-located cutoffs above 1 are therefore expected from the imported helper.

| Source | Overall rate % | Affinity cutoff | Liver-anchor retained % | Selected pairs |
| --- | --- | --- | --- | --- |
| same-image affinity | 5.000 | 0.91373980 | 10.053 | 48333 |
| same-image affinity | 10.000 | 0.86602759 | 12.549 | 96665 |
| same-image affinity | 20.000 | 0.79438776 | 16.899 | 193330 |
| knn affinity | 5.000 | 0.90146458 | 9.559 | 338330 |
| knn affinity | 10.000 | 0.85591900 | 12.059 | 676659 |
| knn affinity | 20.000 | 0.78695291 | 16.482 | 1353318 |
| offset1 colocated+affinity | 5.000 | 1.16455555 | 3.256 | 5663 |
| offset1 colocated-only | 5.000 | uniform thinning | 4.927 | 5663 |
| offset1 colocated+affinity | 10.000 | 1.10507047 | 21.511 | 11326 |
| offset1 colocated-only | 10.000 | uniform thinning | 10.082 | 11326 |
| offset1 colocated+affinity | 20.000 | 1.04625130 | 56.230 | 22651 |
| offset1 colocated-only | 20.000 | uniform thinning | 20.447 | 22651 |
| offset2 colocated+affinity | 5.001 | 1.15008891 | 3.221 | 5615 |
| offset2 colocated-only | 5.000 | uniform thinning | 4.853 | 5614 |
| offset2 colocated+affinity | 10.000 | 1.09061468 | 23.343 | 11229 |
| offset2 colocated-only | 10.000 | uniform thinning | 9.486 | 11229 |
| offset2 colocated+affinity | 20.000 | 1.03233099 | 56.885 | 22458 |
| offset2 colocated-only | 20.000 | uniform thinning | 19.816 | 22458 |
| offset3 colocated+affinity | 5.000 | 1.14141476 | 3.565 | 5566 |
| offset3 colocated-only | 5.000 | uniform thinning | 5.009 | 5566 |
| offset3 colocated+affinity | 10.000 | 1.08204842 | 23.989 | 11132 |
| offset3 colocated-only | 10.000 | uniform thinning | 10.118 | 11132 |
| offset3 colocated+affinity | 20.000 | 1.02496839 | 56.108 | 22264 |
| offset3 colocated-only | 20.000 | uniform thinning | 19.980 | 22264 |
| offset5 colocated+affinity | 5.001 | 1.13092303 | 3.363 | 5470 |
| offset5 colocated-only | 5.000 | uniform thinning | 4.995 | 5469 |
| offset5 colocated+affinity | 10.001 | 1.07147610 | 24.464 | 10939 |
| offset5 colocated-only | 10.000 | uniform thinning | 10.088 | 10938 |
| offset5 colocated+affinity | 20.000 | 1.01389647 | 53.847 | 21877 |
| offset5 colocated-only | 20.000 | uniform thinning | 19.338 | 21877 |

## Task 2: self-calibrating shifts

The exact PREFLIGHT 200-train-slice order, seed 20261003, saved positive-image choices, code construction and coordinate/permutation RNG are reused. There are twelve full batches of 16 and one final batch of 8; keeping that last batch includes all 200 slices without repetition. Each batch independently takes affinity quantiles 0.8/0.8/0.9; negative quantiles pool its five actual non-self batch permutations. Constant/random/one-hot K8 codes and all three losses use the imported objective, pinned weights 0.67/0.25/0.63, dropout disabled, no optimization. K8 assignments use the saved train-only seed-0 frozen-feature centers; no label mapping is used. Mean/SD below weight each batch equally (sample SD, ddof=1).

| Term | Fixed D448 | Per-batch mean ± SD | Min–max |
| --- | --- | --- | --- |
| Intra | 0.79441762 | 0.79409725 ± 0.00611515 | 0.78290045–0.80517256 |
| kNN | 0.78598076 | 0.78582525 ± 0.00587632 | 0.77690756–0.79741770 |
| Random | 0.79765058 | 0.79794113 ± 0.00986634 | 0.78492260–0.81250948 |

| Batch | Slices | Intra shift | kNN shift | Random shift | Constant loss | Random loss | K8 loss | Admissible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 0.789771 | 0.776908 | 0.799670 | 0.36569142 | 0.01597951 | -0.00109163 | True |
| 2 | 16 | 0.797845 | 0.791063 | 0.808781 | 0.36012459 | 0.01584197 | 0.00121879 | True |
| 3 | 16 | 0.786448 | 0.783818 | 0.792201 | 0.35412768 | 0.01543830 | -0.00079278 | True |
| 4 | 16 | 0.797582 | 0.785811 | 0.792504 | 0.34418523 | 0.01496909 | 0.00130329 | True |
| 5 | 16 | 0.797746 | 0.790364 | 0.812509 | 0.36161500 | 0.01594158 | 0.00147580 | True |
| 6 | 16 | 0.797280 | 0.787224 | 0.788597 | 0.34538007 | 0.01523620 | 0.00033080 | True |
| 7 | 16 | 0.787799 | 0.778791 | 0.784923 | 0.35332721 | 0.01549208 | 0.00057209 | True |
| 8 | 16 | 0.798805 | 0.789064 | 0.786989 | 0.34852237 | 0.01528781 | -0.00077671 | True |
| 9 | 16 | 0.805173 | 0.797418 | 0.809493 | 0.35314417 | 0.01555686 | 0.00152516 | True |
| 10 | 16 | 0.796797 | 0.789125 | 0.800305 | 0.35607249 | 0.01568322 | -0.00058881 | True |
| 11 | 16 | 0.792548 | 0.782128 | 0.800560 | 0.35852435 | 0.01583645 | 0.00039134 | True |
| 12 | 16 | 0.782900 | 0.777960 | 0.787172 | 0.36075306 | 0.01574665 | -0.00134347 | True |
| 13 | 8 | 0.792571 | 0.786055 | 0.809529 | 0.37325105 | 0.01663513 | 0.00419334 | True |

Each adaptive batch reaches 20%/20%/10% attraction to floating-point quantile precision; exact fractions and all unweighted component losses are in `batch_shifts.json`. Admissibility is constant loss > random loss AND K8 loss < random loss (lower loss is better).

| Calibration | Admissible batches | Fraction |
| --- | --- | --- |
| adaptive | 13/13 | 100.000% |
| fixed | 13/13 | 100.000% |

| Batch | Fixed constant loss | Fixed random loss | Fixed K8 loss | Fixed intra/kNN/random attraction % |
| --- | --- | --- | --- | --- |
| 1 | 0.36980069 | 0.01620720 | -0.00010508 | 19.350 / 18.820 / 10.235 |
| 2 | 0.34954536 | 0.01531078 | -0.00106859 | 20.461 / 20.737 / 11.524 |
| 3 | 0.36344123 | 0.01593940 | 0.00122116 | 18.931 / 19.705 / 9.292 |
| 4 | 0.34534985 | 0.01500332 | 0.00144310 | 20.468 / 19.967 / 9.325 |
| 5 | 0.34892827 | 0.01531262 | -0.00130321 | 20.487 / 20.674 / 11.924 |
| 6 | 0.34885514 | 0.01538146 | 0.00088919 | 20.449 / 20.182 / 8.857 |
| 7 | 0.36757803 | 0.01622064 | 0.00364999 | 19.057 / 18.920 / 8.383 |
| 8 | 0.35152823 | 0.01540097 | -0.00038784 | 20.658 / 20.460 / 8.652 |
| 9 | 0.33561850 | 0.01464292 | -0.00238935 | 21.660 / 21.782 / 11.593 |
| 10 | 0.35201961 | 0.01547224 | -0.00146187 | 20.369 / 20.481 / 10.346 |
| 11 | 0.35890734 | 0.01586904 | 0.00054548 | 19.759 / 19.448 / 10.360 |
| 12 | 0.37707645 | 0.01660863 | 0.00218621 | 18.474 / 18.817 / 8.762 |
| 13 | 0.36698619 | 0.01634756 | 0.00283323 | 19.753 / 20.010 / 11.491 |

The slice-weighted fixed-loss totals reproduce `audit/calibrated/objective.json` to <1e-7, asserted by `report.py`. No evidence here that per-batch calibration improves admissibility over fixed D448: both pass every sampled batch.

## Verdict and UNVERIFIED

Co-located adjacent-slice pairs have higher liver-target purity in these validation diagnostics: at 10% overall retention, ±1 gives 97.524% versus 79.559% same-image affinity and 77.059% kNN affinity, gains of 17.965 and 20.465 percentage points; affinity-filtered ±1 gives 99.041%. This supports higher pair purity, not improved learned segmentation.

Batch variation is modest on this sample: SD 0.00612 intra, 0.00588 kNN and 0.00987 random, close to the fixed D448 values; adaptive and fixed shifts both pass 13/13 sampled batches. UNVERIFIED: dropout-on calibration, gradient contributions and optimization stability, benefit of either idea to liver segmentation, generalization to other cohorts, and patient-bootstrap uncertainty. High purity can result from local anatomical continuity and does not establish semantic discrimination between organs; tiny structures and boundaries are reduced by patch-majority labeling. Body-box alignment is not anatomical registration. Small selected liver-anchor counts at strict cutoffs and correlated slices limit precision of the percentages. These four patients informed earlier choices.

Commands:
```sh
python3 -B audit/stego_ideas/pair_plan.py
python3 -B audit/stego_ideas/batch_shifts.py
python3 -B audit/stego_ideas/score.py
python3 -B audit/stego_ideas/report.py
```

Runtimes: pair plan 9.352s; label scoring 1.405s; batch calibration 7.402s. Plan SHA256 `e7951e90f2583848ed1bf66e0d919cc0d543e256ba59511ac66048d2ca8e76ec`. All label-use claims are supported by `pair_plan_access.json`, `score_access.json`, `batch_access.json`; source hashes in `plan.json`.


# Fixed F1/F2 refinement

Protocol: [PROTOCOL.md](PROTOCOL.md). No arm selected, no parameter tuning. All outputs restricted to audit/improve2. No source/dataset/cache/checkpoint changes, installs, model loading, frozen forward passes or k-means refitting. Cached B2-448 hard grids and saved soft weights were reused; inherited mappings unchanged.

## Main results

All 20 patients, 2,341 slices, seeds 0/1/2. Average seeds within patient, then report equally weighted patient mean, sample SD and [minimum,maximum]. Both strict-majority and optimal inherited subsets are identical for every seed/patient and both scoring resolutions; these rows apply to both. No reoptimization after filtering. F1/F2 are exploratory because the parent was historically selected using cohort labels.

Producer: `run.py --seed 0/1/2` → `seed0/1/2.json`; aggregation: `report.py` → `summary.json`.

| Arm | Scoring resolution | IoU % mean ± SD [min,max] | Dice % mean ± SD [min,max] | Mapping |
|---|---|---:|---:|---|
| B2_hard | 224 | 69.784 ± 2.959 [61.498, 74.452] | 82.168 ± 2.089 [76.159, 85.355] | Both inherited |
| B2_hard | 512 | 69.801 ± 2.956 [61.565, 74.539] | 82.180 ± 2.087 [76.211, 85.412] | Both inherited |
| B5_2D | 224 | 76.504 ± 2.137 [72.677, 80.278] | 86.672 ± 1.374 [84.177, 89.060] | Both inherited |
| B5_2D | 512 | 76.529 ± 2.132 [72.778, 80.303] | 86.688 ± 1.370 [84.245, 89.076] | Both inherited |
| F1 | 224 | 79.272 ± 2.849 [75.263, 84.156] | 88.409 ± 1.774 [85.885, 91.394] | Both inherited |
| F1 | 512 | 79.274 ± 2.856 [75.315, 84.184] | 88.410 ± 1.778 [85.918, 91.411] | Both inherited |
| F2 | 224 | 79.392 ± 2.834 [75.381, 84.290] | 88.484 ± 1.762 [85.962, 91.473] | Both inherited |
| F2 | 512 | 79.395 ± 2.840 [75.427, 84.322] | 88.486 ± 1.765 [85.992, 91.492] | Both inherited |

B2_hard and B5_2D exactly reproduce every IMPROVE patient IoU/Dice at both resolutions for all seeds (reference_max_metric_difference=0). B5_2D is the original largest component of hard predictions. F1/F2 instead apply largest-component removal after guided soft predictions, then holes/closing as declared.

## (a) No-liver slices

**There are 0 eligible native slices: all 2341 masks contain liver (smallest 25 pixels).** FP pixels per no-liver slice and the corresponding FP rate are **UNVERIFIED / not estimable**, not evidence of zero false positives. The number of observed qualifying slices with FP is 0 only because the qualifying set is empty. Labeled liver-absent CT slices would settle this.

| Arm | Eligible slices | Mean FP pixels/slice | Slices with any FP |
|---|---:|---|---|
| B2_hard | 0 | N/A | 0 of 0; rate N/A |
| B5_2D | 0 | N/A | 0 of 0; rate N/A |
| F1 | 0 | N/A | 0 of 0; rate N/A |
| F2 | 0 | N/A | 0 of 0; rate N/A |

## (b) Multi-component liver slices

Ground truth has >=2 eight-connected components in **203 slices**. Of these, 6 contain a component smaller than 10 pixels. No component-size threshold was used; component count does not necessarily correspond to anatomical lobes. Labels are used only to diagnose what each predeclared largest-component stage removed.

| Largest-component stage | Seed | Eligible slices | Slices losing captured liver | Captured liver pixels removed | % of eligible GT liver removed | Previously hit GT components entirely lost |
|---|---|---:|---:|---:|---:|---:|
| hard_to_B5 | 0 | 203 | 51 | 34117 | 0.854 | 40 / 253 |
| hard_to_B5 | 1 | 203 | 54 | 34717 | 0.869 | 40 / 253 |
| hard_to_B5 | 2 | 203 | 51 | 34117 | 0.854 | 40 / 253 |
| guided_to_lcc | 0 | 203 | 109 | 39129 | 0.979 | 34 / 233 |
| guided_to_lcc | 1 | 203 | 110 | 40169 | 1.005 | 35 / 234 |
| guided_to_lcc | 2 | 203 | 109 | 39128 | 0.979 | 34 / 233 |

The removal counts compare masks immediately before/after the relevant largest-component step. Thus they isolate that step’s loss of already-captured liver; they do not attribute every missed GT pixel to it. F1/F2 hole filling, closing and temporal filtering can subsequently alter these pixels.

**Yes, largest-component filtering drops true liver on multi-component slices.** The tables quantify removed captured pixels and entirely lost previously hit components for each seed; improved overall IoU does not eliminate this failure mode.

| Arm | Seed | Multi-component pooled IoU % | Multi-component pooled Dice % |
|---|---|---:|---:|
| B2_hard | 0 | 64.384 | 78.333 |
| B2_hard | 1 | 64.238 | 78.225 |
| B2_hard | 2 | 64.389 | 78.337 |
| B5_2D | 0 | 72.865 | 84.303 |
| B5_2D | 1 | 72.626 | 84.143 |
| B5_2D | 2 | 72.871 | 84.307 |
| F1 | 0 | 75.174 | 85.828 |
| F1 | 1 | 75.496 | 86.037 |
| F1 | 2 | 75.174 | 85.828 |
| F2 | 0 | 75.289 | 85.903 |
| F2 | 1 | 75.620 | 86.118 |
| F2 | 2 | 75.290 | 85.903 |

## (c) Native surface distances, per patient

Units: **native pixels**, symmetric 2D boundary-pixel distances pooled across each patient’s slices, then seed-average per patient. HD95 is the 95th percentile of the pooled bidirectional surface distances; MSD is their mean. Surface pixels use4-neighbour erosion. Distances are exact Euclidean cKDTree nearest distances, not 3D or millimetres. Labels provide GT surfaces for scoring only.

An empty prediction with nonempty GT contributes infinite distances for every unmatched GT surface point. Strict MSD is therefore infinite if any such point exists. HD95 can remain finite if the infinite tail lies above its95th percentile. Conditional finite-only distances are also shown; these exclude missing-object surface points and must be interpreted alongside the missed-slice counts. No finite diagonal penalty was substituted.

| Arm | Patient | HD95 px | MSD px | Finite-only HD95 px | Finite-only MSD px | One-empty slices seeds0/1/2 |
|---|---|---:|---:|---:|---:|---|
| B2_hard | 1 | 202.143 | 33.294 | 202.143 | 33.294 | [0, 0, 0] |
| B2_hard | 2 | 192.212 | ∞ | 174.699 | 22.966 | [10, 10, 10] |
| B2_hard | 5 | 161.292 | ∞ | 160.074 | 24.628 | [1, 1, 1] |
| B2_hard | 6 | 149.894 | ∞ | 149.632 | 26.863 | [1, 0, 1] |
| B2_hard | 8 | 197.269 | ∞ | 190.884 | 27.626 | [7, 7, 7] |
| B2_hard | 10 | 153.223 | ∞ | 150.700 | 22.667 | [2, 2, 2] |
| B2_hard | 14 | 178.519 | ∞ | 156.312 | 28.942 | [16, 16, 16] |
| B2_hard | 16 | 144.677 | 24.558 | 144.677 | 24.558 | [0, 0, 0] |
| B2_hard | 18 | 151.334 | ∞ | 150.866 | 25.936 | [0, 1, 0] |
| B2_hard | 19 | 144.380 | 26.963 | 144.380 | 26.963 | [0, 0, 0] |
| B2_hard | 21 | 166.007 | 29.950 | 166.007 | 29.950 | [0, 0, 0] |
| B2_hard | 22 | 117.847 | ∞ | 105.125 | 18.032 | [7, 7, 7] |
| B2_hard | 23 | 234.861 | ∞ | 230.342 | 39.062 | [12, 12, 12] |
| B2_hard | 24 | 160.472 | ∞ | 148.015 | 24.438 | [13, 13, 13] |
| B2_hard | 25 | 165.944 | 32.540 | 165.944 | 32.540 | [0, 0, 0] |
| B2_hard | 26 | 148.277 | ∞ | 133.976 | 21.354 | [13, 13, 13] |
| B2_hard | 27 | 155.103 | ∞ | 148.308 | 24.543 | [15, 15, 15] |
| B2_hard | 28 | 149.729 | 30.377 | 149.729 | 30.377 | [0, 0, 0] |
| B2_hard | 29 | 196.830 | ∞ | 193.199 | 30.184 | [10, 10, 10] |
| B2_hard | 30 | 170.617 | ∞ | 148.554 | 25.063 | [21, 21, 21] |
| B5_2D | 1 | 32.246 | 9.454 | 32.246 | 9.454 | [0, 0, 0] |
| B5_2D | 2 | 23.667 | ∞ | 17.735 | 6.873 | [10, 10, 10] |
| B5_2D | 5 | 32.765 | ∞ | 32.159 | 9.686 | [1, 1, 1] |
| B5_2D | 6 | 36.161 | ∞ | 35.767 | 10.490 | [1, 0, 1] |
| B5_2D | 8 | 41.667 | ∞ | 38.205 | 10.758 | [7, 7, 7] |
| B5_2D | 10 | 24.182 | ∞ | 23.419 | 9.007 | [2, 2, 2] |
| B5_2D | 14 | 56.258 | ∞ | 39.366 | 10.946 | [16, 16, 16] |
| B5_2D | 16 | 29.308 | 8.919 | 29.308 | 8.919 | [0, 0, 0] |
| B5_2D | 18 | 37.952 | ∞ | 37.685 | 10.855 | [0, 1, 0] |
| B5_2D | 19 | 32.924 | 9.967 | 32.924 | 9.967 | [0, 0, 0] |
| B5_2D | 21 | 33.111 | 9.345 | 33.111 | 9.345 | [0, 0, 0] |
| B5_2D | 22 | 33.531 | ∞ | 30.022 | 8.190 | [7, 7, 7] |
| B5_2D | 23 | 27.308 | ∞ | 25.160 | 8.227 | [12, 12, 12] |
| B5_2D | 24 | 30.653 | ∞ | 27.171 | 8.702 | [13, 13, 13] |
| B5_2D | 25 | 86.908 | 20.698 | 86.908 | 20.698 | [0, 0, 0] |
| B5_2D | 26 | 42.468 | ∞ | 37.121 | 10.120 | [13, 13, 13] |
| B5_2D | 27 | 39.266 | ∞ | 35.730 | 9.649 | [15, 15, 15] |
| B5_2D | 28 | 30.935 | 10.089 | 30.935 | 10.089 | [0, 0, 0] |
| B5_2D | 29 | 28.107 | ∞ | 26.401 | 7.837 | [10, 10, 10] |
| B5_2D | 30 | 34.557 | ∞ | 23.962 | 10.037 | [21, 21, 21] |
| F1 | 1 | 30.324 | ∞ | 28.656 | 8.336 | [4, 3, 4] |
| F1 | 2 | 33.388 | ∞ | 20.379 | 5.971 | [11, 11, 11] |
| F1 | 5 | 27.000 | ∞ | 26.114 | 8.697 | [2, 1, 2] |
| F1 | 6 | 45.269 | ∞ | 42.941 | 11.069 | [2, 2, 2] |
| F1 | 8 | 57.177 | ∞ | 40.381 | 10.873 | [13, 13, 13] |
| F1 | 10 | 26.185 | ∞ | 23.838 | 8.609 | [7, 7, 7] |
| F1 | 14 | 92.375 | ∞ | 41.406 | 10.855 | [19, 18, 19] |
| F1 | 16 | 21.422 | 7.368 | 21.422 | 7.368 | [0, 0, 0] |
| F1 | 18 | 34.449 | ∞ | 32.755 | 10.377 | [2, 2, 2] |
| F1 | 19 | 35.299 | ∞ | 33.692 | 9.881 | [2, 1, 2] |
| F1 | 21 | 25.852 | ∞ | 24.464 | 7.512 | [5, 4, 5] |
| F1 | 22 | 36.524 | ∞ | 26.588 | 7.168 | [9, 9, 9] |
| F1 | 23 | 26.185 | ∞ | 22.162 | 6.691 | [23, 23, 23] |
| F1 | 24 | 36.894 | ∞ | 29.214 | 8.900 | [15, 14, 15] |
| F1 | 25 | 109.167 | ∞ | 66.874 | 20.492 | [8, 7, 8] |
| F1 | 26 | 41.328 | ∞ | 28.103 | 8.079 | [19, 18, 19] |
| F1 | 27 | 46.887 | ∞ | 34.458 | 8.850 | [25, 20, 25] |
| F1 | 28 | 29.586 | ∞ | 28.908 | 9.888 | [1, 1, 1] |
| F1 | 29 | 41.680 | ∞ | 29.966 | 7.494 | [22, 22, 22] |
| F1 | 30 | 30.842 | ∞ | 21.409 | 8.671 | [26, 25, 26] |
| F2 | 1 | 30.344 | ∞ | 28.665 | 8.359 | [4, 3, 4] |
| F2 | 2 | 33.892 | ∞ | 20.587 | 5.999 | [11, 11, 11] |
| F2 | 5 | 26.814 | ∞ | 25.521 | 8.601 | [2, 2, 2] |
| F2 | 6 | 43.406 | ∞ | 41.059 | 10.917 | [2, 2, 2] |
| F2 | 8 | 56.813 | ∞ | 40.293 | 10.838 | [13, 13, 13] |
| F2 | 10 | 25.998 | ∞ | 23.775 | 8.581 | [7, 7, 7] |
| F2 | 14 | 91.895 | ∞ | 41.046 | 10.799 | [19, 18, 19] |
| F2 | 16 | 20.599 | 6.964 | 20.599 | 6.964 | [0, 0, 0] |
| F2 | 18 | 33.387 | ∞ | 31.644 | 10.231 | [2, 2, 2] |
| F2 | 19 | 35.206 | ∞ | 33.698 | 9.839 | [2, 1, 2] |
| F2 | 21 | 24.960 | ∞ | 23.223 | 7.346 | [6, 5, 6] |
| F2 | 22 | 37.503 | ∞ | 25.940 | 7.080 | [10, 10, 10] |
| F2 | 23 | 25.883 | ∞ | 21.675 | 6.518 | [24, 24, 24] |
| F2 | 24 | 36.855 | ∞ | 29.179 | 8.886 | [15, 14, 15] |
| F2 | 25 | 108.553 | ∞ | 66.818 | 20.457 | [8, 7, 8] |
| F2 | 26 | 41.459 | ∞ | 28.134 | 8.046 | [18, 20, 18] |
| F2 | 27 | 47.254 | ∞ | 33.962 | 8.763 | [26, 20, 26] |
| F2 | 28 | 28.682 | 9.914 | 28.682 | 9.914 | [0, 0, 0] |
| F2 | 29 | 41.685 | ∞ | 29.966 | 7.483 | [22, 22, 22] |
| F2 | 30 | 30.079 | ∞ | 20.935 | 8.561 | [26, 28, 26] |

## Patient IoU/Dice

### B2_hard

| Patient |224 IoU % |224 Dice % |512 IoU % |512 Dice % |
|---|---:|---:|---:|---:|
| 1 | 71.111 | 83.116 | 71.074 | 83.091 |
| 2 | 74.452 | 85.355 | 74.539 | 85.412 |
| 5 | 73.335 | 84.617 | 73.330 | 84.613 |
| 6 | 71.258 | 83.217 | 71.260 | 83.218 |
| 8 | 68.998 | 81.655 | 69.173 | 81.778 |
| 10 | 71.993 | 83.716 | 72.000 | 83.721 |
| 14 | 66.040 | 79.547 | 65.934 | 79.470 |
| 16 | 70.258 | 82.531 | 70.404 | 82.632 |
| 18 | 67.593 | 80.663 | 67.505 | 80.600 |
| 19 | 67.135 | 80.336 | 67.226 | 80.401 |
| 21 | 72.853 | 84.295 | 72.857 | 84.298 |
| 22 | 71.580 | 83.436 | 71.525 | 83.399 |
| 23 | 69.886 | 82.274 | 69.888 | 82.275 |
| 24 | 69.297 | 81.864 | 69.190 | 81.790 |
| 25 | 69.444 | 81.965 | 69.639 | 82.101 |
| 26 | 72.480 | 84.045 | 72.347 | 83.955 |
| 27 | 69.341 | 81.895 | 69.383 | 81.925 |
| 28 | 61.498 | 76.159 | 61.565 | 76.211 |
| 29 | 67.089 | 80.303 | 67.041 | 80.269 |
| 30 | 70.034 | 82.376 | 70.135 | 82.446 |

### B5_2D

| Patient |224 IoU % |224 Dice % |512 IoU % |512 Dice % |
|---|---:|---:|---:|---:|
| 1 | 78.358 | 87.866 | 78.326 | 87.846 |
| 2 | 78.384 | 87.882 | 78.478 | 87.941 |
| 5 | 76.459 | 86.659 | 76.473 | 86.669 |
| 6 | 78.978 | 88.254 | 78.975 | 88.253 |
| 8 | 74.242 | 85.217 | 74.456 | 85.358 |
| 10 | 76.795 | 86.874 | 76.771 | 86.860 |
| 14 | 73.453 | 84.695 | 73.436 | 84.684 |
| 16 | 77.234 | 87.155 | 77.437 | 87.284 |
| 18 | 73.316 | 84.604 | 73.226 | 84.544 |
| 19 | 74.295 | 85.252 | 74.311 | 85.262 |
| 21 | 80.278 | 89.060 | 80.303 | 89.076 |
| 22 | 76.872 | 86.924 | 76.878 | 86.928 |
| 23 | 79.910 | 88.833 | 79.904 | 88.829 |
| 24 | 76.038 | 86.388 | 75.991 | 86.357 |
| 25 | 76.511 | 86.690 | 76.644 | 86.776 |
| 26 | 76.130 | 86.448 | 75.961 | 86.339 |
| 27 | 77.089 | 87.062 | 77.168 | 87.113 |
| 28 | 72.677 | 84.177 | 72.778 | 84.245 |
| 29 | 77.602 | 87.389 | 77.502 | 87.325 |
| 30 | 75.467 | 86.019 | 75.559 | 86.078 |

### F1

| Patient |224 IoU % |224 Dice % |512 IoU % |512 Dice % |
|---|---:|---:|---:|---:|
| 1 | 81.966 | 90.089 | 82.008 | 90.114 |
| 2 | 82.813 | 90.598 | 82.938 | 90.673 |
| 5 | 79.332 | 88.474 | 79.300 | 88.454 |
| 6 | 80.666 | 89.297 | 80.632 | 89.276 |
| 8 | 76.334 | 86.575 | 76.341 | 86.580 |
| 10 | 78.349 | 87.859 | 78.398 | 87.889 |
| 14 | 75.739 | 86.194 | 75.719 | 86.181 |
| 16 | 81.316 | 89.695 | 81.302 | 89.686 |
| 18 | 75.300 | 85.908 | 75.315 | 85.918 |
| 19 | 75.520 | 86.049 | 75.538 | 86.061 |
| 21 | 84.156 | 91.394 | 84.184 | 91.411 |
| 22 | 80.882 | 89.430 | 80.927 | 89.457 |
| 23 | 83.270 | 90.870 | 83.273 | 90.872 |
| 24 | 77.526 | 87.337 | 77.510 | 87.327 |
| 25 | 77.209 | 87.129 | 77.144 | 87.087 |
| 26 | 80.005 | 88.890 | 79.917 | 88.835 |
| 27 | 80.496 | 89.194 | 80.464 | 89.174 |
| 28 | 75.263 | 85.885 | 75.324 | 85.925 |
| 29 | 81.702 | 89.929 | 81.670 | 89.910 |
| 30 | 77.605 | 87.389 | 77.587 | 87.378 |

### F2

| Patient |224 IoU % |224 Dice % |512 IoU % |512 Dice % |
|---|---:|---:|---:|---:|
| 1 | 81.990 | 90.103 | 82.035 | 90.130 |
| 2 | 82.797 | 90.588 | 82.919 | 90.662 |
| 5 | 79.434 | 88.538 | 79.394 | 88.513 |
| 6 | 80.944 | 89.467 | 80.912 | 89.448 |
| 8 | 76.449 | 86.649 | 76.467 | 86.661 |
| 10 | 78.410 | 87.897 | 78.464 | 87.931 |
| 14 | 75.812 | 86.241 | 75.803 | 86.236 |
| 16 | 81.575 | 89.851 | 81.556 | 89.840 |
| 18 | 75.614 | 86.113 | 75.629 | 86.122 |
| 19 | 75.646 | 86.131 | 75.666 | 86.143 |
| 21 | 84.290 | 91.473 | 84.322 | 91.492 |
| 22 | 80.920 | 89.453 | 80.970 | 89.484 |
| 23 | 83.385 | 90.939 | 83.389 | 90.941 |
| 24 | 77.603 | 87.386 | 77.580 | 87.371 |
| 25 | 77.371 | 87.232 | 77.305 | 87.190 |
| 26 | 80.071 | 88.930 | 79.989 | 88.880 |
| 27 | 80.630 | 89.276 | 80.594 | 89.254 |
| 28 | 75.381 | 85.962 | 75.427 | 85.992 |
| 29 | 81.811 | 89.996 | 81.775 | 89.974 |
| 30 | 77.717 | 87.460 | 77.696 | 87.446 |

## Labels and limitations

Labels enter only inherited other 19-patient mappings, held-out scoring, empty/multi-component/surface analyses and worst-slice selection; the original largest-liver figure selection also used labels. No labels tune F1/F2 or select between them. Parent selection in IMPROVE used cohort scores, so both arms remain exploratory. All cohort images entered the original clustering: transductive, single CT acquisition cohort, no untouched test set. External performance, liver-absent specificity and 3D/mm boundary metrics remain **UNVERIFIED**. The 3-slice median uses adjacent acquired slices and endpoint replication; it is not a fixed-millimetre filter.

Verification: `validate.py` → `validation.json` checks exact reference reproduction, all patient summary statistics, the temporal median against an independent Boolean majority implementation, and native-pixel surface distances against a synthetic 5-pixel displacement. Source inputs are read-only; their digests are recorded. `repository_final.json` records repository status.

## Figures

- /Users/navneetbavineni/STEGO/audit/improve2/eleven_slices_contours.png
- /Users/navneetbavineni/STEGO/audit/improve2/eleven_slices_contours.pdf
Same 8 earlier slices plus 3 worst B5_2D native per-slice IoUs over all 2341 slices, fixed seed 0. Worst-slice selection uses labels and is not representative sampling. Details and selected IDs: `figures.json`. Red=ground truth; green=prediction; each prediction column reports native per-slice IoU.
PDF rendered with installed pdftoppm and visually inspected (`pdf_render.png`). F1/F2 smooth stepped boundaries and fill internal holes, but connected spill and missed separate liver regions remain. All three worst reference slices still have zero IoU in both arms; neither arm is selected.

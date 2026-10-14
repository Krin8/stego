# P1 multi-cluster and disconnected-component evaluation

## Declared comparison

The existing P1 seed-0 checkpoint, train-only K=8 fit, 0.30 threshold, guided
filter, hole filling, and radius-3 closing are unchanged. This diagnostic varies:

1. cluster mapping: exhaustively search all 256 subsets of C0-C7 for the subset
   with the highest globally pooled validation liver IoU;
2. component retention: keep the largest one, largest two, or all thresholded
   connected components before hole filling and closing.

The subset search and every score use all 472 validation labels. This is an
optimistic label-assisted oracle diagnostic and is not a deployable label-free rule.

## Mapping result

The IoU-optimal subset is **C2 alone**. N2 also selects **C2**. Therefore, even an
exhaustive many-to-one search does not find a second cluster whose inclusion raises
pooled IoU. The cluster-selection part of the prediction remains unchanged.

## Refinement result

Scores are percentages. Patient mean first pools pixels per patient, then averages
the four validation patients.

| Components retained | Native patient-mean IoU | Native patient-mean Dice | Native globally pooled IoU | Native globally pooled Dice | 224 patient-mean IoU / Dice |
|---|---:|---:|---:|---:|---:|
| Largest 1 (current F1) | **83.7386** | **91.1324** | **83.9166** | **91.2550** | **83.7469 / 91.1379** |
| Largest 2 | 80.2729 | 88.9999 | 80.2141 | 89.0209 | 80.2797 / 89.0047 |
| All components | 80.4515 | 89.1157 | 80.4206 | 89.1479 | 80.4575 / 89.1201 |

Relative to the current one-component F1 result, keeping two components lowers
patient-mean native IoU by **3.4656 points**. Keeping all components lowers it by
**3.2870 points**.

## Why the score falls

| Variant | True-positive pixels | False-positive pixels | False-negative pixels | Slices improved / worsened / unchanged versus current F1 |
|---|---:|---:|---:|---:|
| Largest 1 | 8,192,483 | 307,090 | 1,263,080 | reference |
| Largest 2 | 8,198,600 | 765,336 | 1,256,963 | 137 / 239 / 96 |
| All components | 8,229,946 | 778,061 | 1,225,617 | 142 / 236 / 94 |

Extra components recover some liver pixels, but the false-positive increase is much
larger. Keeping two components adds 6,117 true-positive pixels and 458,246
false-positive pixels. Keeping all adds 37,463 true positives and 470,971 false
positives.

The ground-truth liver mask has at least two 8-connected components in 65 of 472
slices. On those 65 slices, retaining two predicted components improves 11, worsens
51, and leaves 3 unchanged; its mean per-slice IoU change is -4.4098 points. Keeping
all components improves 10, worsens 52, and leaves 3 unchanged; mean change is
-3.9936 points. Thus the model's secondary predicted components generally do not
correspond to the secondary annotated liver region.

## Conclusion

This test does not improve the P1 pipeline. The validation-oracle many-to-one mapping
still chooses only C2, and relaxing largest-component filtering reduces liver IoU.
The current N2 + single-largest-component F1 result remains higher at 83.7386%
patient-mean native IoU.

## Artifacts

- Producer: `audit/p1_hungarian/evaluate_many_to_one.py`
- Metrics: `audit/p1_hungarian/many_to_one_evaluation/metrics.json`
- Manifest: `audit/p1_hungarian/many_to_one_evaluation/manifest.json`
- Largest-two masks: `audit/p1_hungarian/many_to_one_evaluation/largest_2_components.npy`
- All-component masks: `audit/p1_hungarian/many_to_one_evaluation/all_components.npy`


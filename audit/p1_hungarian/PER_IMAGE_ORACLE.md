# P1 per-image cluster-subset oracle

## Protocol

For each of the 472 validation slices independently, all 256 subsets of the eight
raw P1 clusters were evaluated against that slice's ground-truth liver mask. The
subset with the highest native hard-mask IoU was selected, with ties resolved by
fewer clusters and then lower cluster IDs. That same selected subset was used to
sum soft cluster probabilities and run the unchanged F1 refinement.

This is an image-level label oracle. Each ground-truth mask changes its own
prediction. It is not label-free inference and is not a reportable deployment score.

## Results

All values are percentages.

| Method | Native all-slice mean hard IoU / Dice | Native all-slice mean F1 IoU / Dice | Native patient-pooled mean hard IoU / Dice | Native patient-pooled mean F1 IoU / Dice |
|---|---:|---:|---:|---:|
| N2, one global C2 choice | 58.1319 / 69.8939 | 71.9303 / 78.9115 | **68.6928 / 81.4216** | **83.7386 / 91.1324** |
| Per-image IoU-optimal subset oracle | **58.5745 / 70.6770** | **72.4002 / 79.4580** | 57.9576 / 73.0447 | 67.5196 / 80.0154 |

At 224, the per-image oracle has patient-pooled hard IoU/Dice of 57.9240/73.0142%
and F1 IoU/Dice of 67.5414/80.0349%.

## Selection behavior

- C2 alone: 418 slices
- C4 alone: 15 slices
- C5 alone: 16 slices
- C1 alone: 8 slices
- C7 alone: 1 slice
- C2+C4: 4 slices
- C2+C5: 8 slices
- C1+C5: 2 slices
- Multiple clusters selected: 14 slices
- Mean selected clusters per slice: 1.0297

Compared slice by slice with N2, the hard oracle improves 54 slices, worsens none,
and leaves 418 unchanged. Mean hard IoU rises by 0.4426 points. After unchanged F1
refinement, 37 slices improve, none worsen, and 435 are unchanged; mean F1 IoU rises
by 0.4699 points.

## Why patient-pooled IoU decreases despite per-slice improvement

Each subset maximizes its own slice IoU. Patient-pooled IoU is instead computed from
the sum of TP, FP, and FN over all slices before taking the ratio. Ratios optimized
independently do not necessarily maximize the ratio of pooled counts. The alternative
subsets occur on slices with different liver/union sizes, so their contributions are
weighted differently in pooled scoring. This is a ratio-aggregation effect; no slice's
hard IoU became worse.

The relevant oracle answer is therefore the all-slice mean: the labels raise mean
hard IoU from 58.1319% to 58.5745% and mean refined IoU from 71.9303% to 72.4002%.
The small gain shows that a global C2 identity is already consistent for 88.6% of
slices, and per-image relabeling exposes little additional recoverable structure.

## Artifacts

- Producer: `audit/p1_hungarian/evaluate_per_image_oracle.py`
- Metrics: `audit/p1_hungarian/per_image_oracle/metrics.json`
- Per-slice selected subsets: `audit/p1_hungarian/per_image_oracle/selections.json`
- Hard masks: `audit/p1_hungarian/per_image_oracle/hard.npy`
- Refined masks: `audit/p1_hungarian/per_image_oracle/refined.npy`
- Manifest: `audit/p1_hungarian/per_image_oracle/manifest.json`


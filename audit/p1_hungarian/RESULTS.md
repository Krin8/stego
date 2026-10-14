# P1 trained head: N2 versus STEGO Hungarian mapping

## Scope

This is an evaluation-only experiment. The P1 seed-0 checkpoint, preprocessing,
70-dimensional head, K=8 fit, soft assignments, and F1 refinement are unchanged.
Only the rule that names/maps clusters to the two semantic classes is varied.

- Checkpoint: `src2/checkpoints/p1_seed0.pt`
- Checkpoint SHA-256: `1454f1bb8d2261a9a56a0fbcd6bed1d8946bf8cd7ff319d697b108ec16d28f1b`
- Evaluation set: all 472 validation slices from patients 5, 6, 8, and 30
- Cluster fit: K=8, fit on the 16 training patients only
- Scores: validation labels are used for scoring. Hungarian additionally uses those
  same validation labels to choose the mapping.

## Result

All values below are percentages. "Patient mean" first pools pixels within each
patient, then averages the four patient scores. "Global" pools all validation pixels.

| Mapping | Liver cluster | Native hard IoU / Dice, patient mean | Native F1 IoU / Dice, patient mean | Native F1 IoU / Dice, global | 224 F1 IoU / Dice, patient mean |
|---|---:|---:|---:|---:|---:|
| N2, label-free anatomical naming | C2 | 68.6928 / 81.4216 | **83.7386 / 91.1324** | **83.9166 / 91.2550** | **83.7469 / 91.1379** |
| Hungarian over all 8 clusters (overcluster-aware adaptation) | C2 | 68.6928 / 81.4216 | **83.7386 / 91.1324** | **83.9166 / 91.2550** | **83.7469 / 91.1379** |
| Literal upstream STEGO metric behavior | C0 | 0.0033 / 0.0067 | **0.0042 / 0.0085** | **0.0040 / 0.0081** | **0.0045 / 0.0090** |

The overcluster-aware Hungarian assignment selects C2, exactly the same cluster as
N2. Its hard and refined masks are byte-identical to N2, so it cannot increase or
decrease the score in this run.

The literal upstream behavior produces the lookup `[1, 0, -1, -1, -1, -1, -1,
-1]`: C0 is mapped to liver, C1 to background, and C2-C7 are unmatched. This is why
the literal result is effectively zero.

## Why literal STEGO fails here

Upstream STEGO's metric allocates rows for `n_classes + extra_clusters`, but its
update validity mask admits only predicted IDs `< n_classes`. Its cluster mapping
then gives unmatched overclusters the value `-1`. This is harmless in the original
COCO-Stuff configuration when `extra_clusters=0` and K equals the 27 semantic
classes. It is incompatible with this pipeline's K=8 / two-class setup because it
discards C2-C7 before matching.

The literal behavior is reproduced at
`audit/p1_hungarian/evaluate_hungarian.py:36-50`. The mapping is applied without
changing raw cluster assignments at lines 68-74. Labels are opened at line 34 and
used to construct the mapping at lines 39-47, so this is label-assisted evaluation,
not label-free inference.

## Interpretation

Replacing N2 with a compatible Hungarian mapping does not improve the P1 result:
both choose C2 and produce the same 83.7386% patient-mean native F1 IoU. Replacing
N2 with upstream STEGO's implementation literally is invalid for this overclustered
binary setup and reduces patient-mean native F1 IoU to 0.0042%.

N2 remains usable without validation labels. Hungarian matching requires labeled
pixels from the evaluation dataset to establish the cluster-to-class lookup; it is
therefore suitable as the original STEGO evaluation protocol but not as the deployed
cluster-naming rule.

## Reproduction and artifacts

Commands used:

```text
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src2.fit_clusters --output audit/p1_hungarian/p1_fit
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src2.eval_segmentation --fit-dir audit/p1_hungarian/p1_fit --output audit/p1_hungarian/n2_baseline
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 audit/p1_hungarian/evaluate_hungarian.py
```

- N2 metrics and predictions: `audit/p1_hungarian/n2_baseline/`
- Overcluster-aware Hungarian metrics and predictions: `audit/p1_hungarian/hungarian_evaluation/`
- Literal STEGO metrics and predictions: `audit/p1_hungarian/literal_stego_evaluation/`
- Literal evaluator: `audit/p1_hungarian/evaluate_hungarian.py`

The follow-up test of exhaustive many-to-one mapping and retaining disconnected
components is reported in `audit/p1_hungarian/MANY_TO_ONE.md`.

The per-image label-oracle subset test is reported in
`audit/p1_hungarian/PER_IMAGE_ORACLE.md`.


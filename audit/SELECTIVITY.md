# Affinity selectivity: complete read-only investigation



## Verdict

The current affinity contains substantial liver/background discrimination; the positive shifts are too permissive to yield liver-specific attracting pairs. This establishes pair selectivity, not a successful segmentation objective or a training fix.

- Same image: AUC 0.9460, AP 0.7862. Liver-target precision exceeds 50% at the tested overall attraction rates 2.00%, 5.00%, 10.00%, 20.00%. At 10% attraction: precision 73.74%, liver-pair recall 66.57%. Producer: `score.py --pool intra`.

- Current kNN: AUC 0.9413, AP 0.7542. Liver-target precision exceeds 50% at the tested overall attraction rates 2.00%, 5.00%, 10.00%, 20.00%. At 10% attraction: precision 70.05%, liver-pair recall 65.79%. Producer: `score.py --pool knn`.

- Random cross-image: AUC 0.9360, AP 0.7049. Liver-target precision exceeds 50% at the tested overall attraction rates 2.00%, 5.00%, 10.00%. At 10% attraction: precision 64.73%, liver-pair recall 62.75%. Producer: `score.py --pool random`.

- Subtracting the unlabeled all-train mean then L2 normalizing improves precision at 75% recall by 3.95/4.43/5.58 percentage points for same-image/kNN/random pools. This is a consistent, modest diagnostic improvement, not evidence of improved trained segmentation. Block 11 helps less; block 9 does not beat the baseline. Removing loss centering alone raises AUC slightly but reduces AP and matched-recall precision. Producers: the three `score.py` runs.

- Hard frozen K8 co-membership is less precise than the current continuous affinity at the same liver-pair recall in all three pools; it is not an upper bound. All variant comparisons and matched-recall envelopes use validation labels for diagnosis only. No model, shift or neighbour set was selected or changed.

## Scope, provenance and label use

Source commit: `e3c3ab0e396c194a7314d64546c81e9ba4973b72`. Tracked source diff is empty. No training, fitting, installation, pipeline cache regeneration or checkpoint modification was performed. New files are confined to this report and `audit/scratch/selectivity/`. CPU only; all scripts enforce a 570-second timeout. `report.py` renders this report from the named JSON results.

`extract.py` verified 472 validation slices, patient counts 5: 95, 6: 82, 8: 98, 30: 197. Frozen feature shape `[472, 768, 14, 14]`; final output agrees exactly with saved FORENSICS features (max absolute difference 0.0). The all-train mean was recomputed and exactly matched across 1869 training images, without opening annotations. Existing BASELINE K8 seed-0 partition was recovered from its nearest-upsampled maps exactly, with no refit.

Labels: validation masks only, read by `score.py` and `verify.py`. ROC-AUC, AP, base rates, conditional precision/recall, same-label rates, matched-recall envelopes and majority-precision boundaries use these labels. Affinity thresholds, overall attraction rates, patient/image counts, feature checks and runtimes do not use segmentation labels. Extraction and affinity generation install an explicit annotation-read prohibition; all scripts prohibit writes outside this investigation. Training-label reads are forbidden.

### Exact reuse and sampling

`extract.py` imports the actual `LitUnsupervisedSegmenter`, dataset preprocessing and image-only loader. It calls `BiomedCLIPFeaturizer.forward` (`src/modules.py:200`), instrumenting its actual trunk call to capture normalized blocks 9, 11 and 12, with CLS excluded. Its final output is retained and verified. Blocks are one-based, final=12. The source checkpoint supplies the unchanged frozen backbone, not learned affinity scores.

`affinities.py` AST-imports CROSSPATIENT `batches`/`image_id_codes` and PAIRS `spy_sample`/`hard_labels`; calls the actual `ContrastiveCorrelationLoss.forward` and `helper` (`src/modules.py:473`, `src/modules.py:499`). Positive scalar ID codes have cosine one and expose affinity by inverting the helper return. No affinity or restored-mean equation is reimplemented. Uncentered variants call that same helper with a diagnostic config copy having `pointwise=False`. Project configs remain unchanged.

Seed 20260930. Seven deterministic shuffled passes through all slices, batches 16 (last 8), actual 11×11 sampled coordinates. The current config uses the top seven non-self neighbours; the actual dataset loader verified the cache identity, order and values. Every cached top-seven edge is used once. Random pairs retain the first actual nonzero roll per batch, including both same-patient and cross-patient images. This is the actual negative-image sampler, not exclusively cross-patient sampling. There are 3,304 image-pair draws and 2,048 label-free uniformly selected entries from each 121×121 matrix: 6,766,592 scored pairs per pool. All variants share identical image pairs, coordinates and retained pair indices. These are Monte Carlo pair estimates across the full cohort, not exhaustive enumeration of all possible pairs.

Loss centering is performed on the FULL actual batch correlation tensor before retaining sampled entries. Final/block features are LayerNorm-normalized trunk outputs; the actual helper applies channel L2 normalization after spatial sampling. The train-mean alternative subtracts the all-training raw-token mean, L2 normalizes each grid token, then uses the actual sampler and helper. Its primary result has no restored-row centering; the additional “+ loss centering” variant is reported separately. All features are evaluated with dropout off; source variants retain the SAME final-feature cache neighbours.

The K8 reference is `audit/scratch/baseline/k8_c0_s0_assignments.npy`, produced previously by `baseline/kmeans.py --k 8 --center 0 --seed 0` on all training slices without fit labels. No validation-derived cluster-to-organ mapping is used here: affinity is simply whether the two sampled nearest tokens share a cluster. It has only one nontrivial operating point. Continuous features are bilinearly sampled; discrete annotations/cluster membership follow PAIRS nearest sampling.

Patch annotation: majority of the corresponding 16×16 mask block, ties background. `score.py` reports liver grid occupancy 7.41% and mixed-occupancy grid patches 9.36%. These coarse binary targets do not measure boundary accuracy. Pair indices, score arrays, all counts, per-patient results and exact thresholds remain under `audit/scratch/selectivity/`.

## Task 1: current affinity as a liver-anchor pair classifier

Producer: `score.py --pool intra|knn|random` → `score_<pool>.json`. All classifier/base-rate numbers below use validation labels. ROC-AUC/AP are in [0,1]; AP uses the empirical target prevalence, without class balancing.

| Pool | Pairs | Liver-anchor pairs | Liver→liver pairs | Liver-target base rate | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- |
| Same image | 6,766,592 | 573,652 | 79,281 | 13.82% | 0.946049 | 0.786177 |
| Current kNN | 6,766,592 | 573,652 | 74,630 | 13.01% | 0.941304 | 0.754180 |
| Random cross-image | 6,766,592 | 573,652 | 48,684 | 8.49% | 0.936041 | 0.704864 |

## Task 2: label-free overall-attraction sweep

Threshold producers: `affinities.py`, before labels are opened. Score producer: `score.py --pool <pool>`. Attraction budgets are over ALL anchors/targets, not conditional on liver anchors. Each threshold uses strict `affinity > shift`; achieved rates match the displayed budgets to rounding. Precision = P(target liver | anchor liver, attracted); recall = P(attracted | liver→liver); L→B attraction = P(attracted | liver→background). Same-label rate uses all attracted pairs and is background dominated.

| Pool | Overall attraction | Shift | Liver-target precision | Liver-pair recall | L→B attraction | Overall same-label |
| --- | --- | --- | --- | --- | --- | --- |
| Same image | 2.00% | 0.978216 | 92.11% | 34.84% | 0.48% | 96.79% |
| Same image | 5.00% | 0.913861 | 83.28% | 52.90% | 1.70% | 95.55% |
| Same image | 10.00% | 0.861971 | 73.74% | 66.57% | 3.80% | 95.08% |
| Same image | 20.00% | 0.795970 | 60.43% | 81.12% | 8.52% | 94.55% |
| Same image | 30.00% | 0.739875 | 49.21% | 89.36% | 14.79% | 93.78% |
| Same image | 50.00% | 0.627210 | 31.09% | 97.43% | 34.63% | 91.59% |
| Same image | 75.00% | 0.481351 | 18.18% | 99.81% | 72.05% | 88.47% |
| Current kNN | 2.00% | 0.954688 | 88.72% | 35.32% | 0.67% | 95.41% |
| Current kNN | 5.00% | 0.897927 | 79.55% | 52.57% | 2.02% | 94.59% |
| Current kNN | 10.00% | 0.850069 | 70.05% | 65.79% | 4.21% | 94.47% |
| Current kNN | 20.00% | 0.787061 | 57.05% | 80.29% | 9.04% | 94.14% |
| Current kNN | 30.00% | 0.732269 | 46.32% | 88.82% | 15.39% | 93.44% |
| Current kNN | 50.00% | 0.621809 | 29.21% | 97.27% | 35.25% | 91.34% |
| Current kNN | 75.00% | 0.477711 | 17.06% | 99.83% | 72.60% | 88.20% |
| Random cross-image | 2.00% | 0.875429 | 87.76% | 35.56% | 0.46% | 96.64% |
| Random cross-image | 5.00% | 0.826812 | 77.00% | 49.91% | 1.38% | 96.11% |
| Random cross-image | 10.00% | 0.781172 | 64.73% | 62.75% | 3.17% | 95.62% |
| Random cross-image | 20.00% | 0.719449 | 47.53% | 78.58% | 8.04% | 94.58% |
| Random cross-image | 30.00% | 0.668669 | 34.60% | 87.86% | 15.40% | 93.28% |
| Random cross-image | 50.00% | 0.574500 | 18.21% | 96.61% | 40.25% | 90.17% |
| Random cross-image | 75.00% | 0.444842 | 10.56% | 99.67% | 78.27% | 87.09% |

### The unchanged C shifts

| Pool | C shift | Attraction | Liver-target precision | Liver-pair recall | L→B attraction |
| --- | --- | --- | --- | --- | --- |
| Same image | 0.458446622 | 78.57% | 17.05% | 99.89% | 77.92% |
| Current kNN | 0.452472180 | 78.95% | 15.91% | 99.92% | 78.99% |
| Random cross-image | 0.745958567 | 15.32% | 54.83% | 72.19% | 5.51% |

These observations isolate a selectivity issue: current positive cuts preserve almost all liver/liver pairs but also attract most liver/background pairs. Raising a shift changes this tradeoff; it does not demonstrate that the full learned-code/probe objective will improve.

### Empirical full-curve majority boundary (optimistic diagnostic)

To answer “any shift” beyond the requested seven budgets, `score.py` also finds the largest empirical liver-pair recall with precision strictly above 50%. Unlike the sweep, these boundaries are chosen WITH validation labels; they are not proposed label-free calibration settings or reliable generalization cutoffs. The near-50% precision is a boundary condition, not a safety margin. Here the cut is inclusive `affinity >= threshold` to retain exact tied-score groups.

| Pool | Inclusive cut | Overall attraction | Precision | Recall |
| --- | --- | --- | --- | --- |
| Same image | 0.743852 | 29.28% | 50.000355% | 88.94% |
| Current kNN | 0.751565 | 26.46% | 50.000389% | 86.16% |
| Random cross-image | 0.728697 | 18.31% | 50.000672% | 76.47% |

## Task 3: feature and centering comparisons

Producer: `score.py --pool <pool>`. Every performance number uses validation labels. P@R is the best empirical threshold precision among thresholds with recall at least R, an optimistic diagnostic envelope. The JSON records each achieved recall/cut; continuous achieved recalls remain very close to the requested values. Neighbours and sample pairs do not change between rows.

| Pool | Source | ROC-AUC | AP | Best precision at ≥50% recall | Best precision at ≥75% recall |
| --- | --- | --- | --- | --- | --- |
| Same image | Final + loss centering | 0.946049 | 0.786177 | 84.98% | 66.69% |
| Same image | Final, uncentered | 0.949557 | 0.783020 | 83.78% | 65.07% |
| Same image | Final − train mean, L2 | 0.960776 | 0.815431 | 86.88% | 70.65% |
| Same image | Final − train mean, L2 + loss centering | 0.957029 | 0.802567 | 85.75% | 68.75% |
| Same image | Block 9 + loss centering | 0.944494 | 0.777137 | 83.08% | 64.30% |
| Same image | Block 11 + loss centering | 0.950601 | 0.793650 | 85.59% | 69.18% |
| Current kNN | Final + loss centering | 0.941304 | 0.754180 | 81.10% | 62.32% |
| Current kNN | Final, uncentered | 0.944356 | 0.740926 | 78.67% | 60.55% |
| Current kNN | Final − train mean, L2 | 0.957507 | 0.783785 | 82.91% | 66.75% |
| Current kNN | Final − train mean, L2 + loss centering | 0.953665 | 0.769373 | 81.43% | 65.06% |
| Current kNN | Block 9 + loss centering | 0.940257 | 0.742478 | 78.59% | 60.60% |
| Current kNN | Block 11 + loss centering | 0.946333 | 0.761876 | 82.16% | 64.97% |
| Random cross-image | Final + loss centering | 0.936041 | 0.704864 | 76.91% | 51.69% |
| Random cross-image | Final, uncentered | 0.938676 | 0.668246 | 70.31% | 46.68% |
| Random cross-image | Final − train mean, L2 | 0.959777 | 0.737140 | 78.32% | 57.27% |
| Random cross-image | Final − train mean, L2 + loss centering | 0.954464 | 0.718385 | 76.80% | 54.49% |
| Random cross-image | Block 9 + loss centering | 0.934150 | 0.685336 | 73.28% | 49.52% |
| Random cross-image | Block 11 + loss centering | 0.940790 | 0.724122 | 79.61% | 56.48% |

### Hard K8 co-membership reference and matched recall

Producer: `score.py --pool <pool>`. The K8 score is binary: same cluster=1, different cluster=0; threshold 0.5 selects hard co-membership. No arbitrary-recall interpolation is claimed. AUC/AP account for these ties. Baseline precision is measured at at least the K8 point’s recall; it does not use a K8 liver lookup table.

| Pool | K8 AUC | K8 AP | K8 attraction | K8 precision | K8 recall | Current affinity precision at K8 recall | Mean-subtracted precision at K8 recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Same image | 0.812308 | 0.486232 | 20.58% | 64.65% | 68.46% | 72.27% | 75.64% |
| Current kNN | 0.798627 | 0.451797 | 19.78% | 61.92% | 65.78% | 70.06% | 73.62% |
| Random cross-image | 0.779845 | 0.392278 | 16.31% | 60.02% | 59.65% | 67.94% | 70.78% |

### Same-image self-pair sensitivity

Producer: `score.py --pool intra`. Remove exact same sampled coordinates, then more conservatively all pairs assigned to the same nearest token. Recompute the attraction quantiles on each reduced pool without using labels. This checks whether near-identity comparisons alone explain the selectivity.

| Same-image baseline pool | Excluded pairs | AUC | AP | Precision at ≥75% recall | Majority-precision attraction budgets |
| --- | --- | --- | --- | --- | --- |
| Original | 0% | 0.946049 | 0.786177 | 66.69% | 2.00%, 5.00%, 10.00%, 20.00% |
| exclude_same_coordinate | 0.93% | 0.942300 | 0.756593 | 62.99% | 2.00%, 5.00%, 10.00%, 20.00% |
| exclude_same_nearest_token | 1.47% | 0.939867 | 0.737017 | 60.42% | 2.00%, 5.00%, 10.00%, 20.00% |

### Anchor-patient consistency

Producer: `score.py --pool <pool>`. AUC scored separately by anchor patient; target pool membership stays unchanged. These are correlated image/patch observations, not independent-patient replication. All four patients have a higher AUC after train-mean subtraction in all three pools.

| Pool | Anchor patient | Current AUC | Mean-subtracted AUC | Block 11 AUC |
| --- | --- | --- | --- | --- |
| Same image | 5 | 0.937752 | 0.953836 | 0.941860 |
| Same image | 6 | 0.938912 | 0.953616 | 0.944871 |
| Same image | 8 | 0.950309 | 0.965107 | 0.954651 |
| Same image | 30 | 0.963911 | 0.973532 | 0.967335 |
| Current kNN | 5 | 0.932374 | 0.951112 | 0.937333 |
| Current kNN | 6 | 0.935519 | 0.951263 | 0.941622 |
| Current kNN | 8 | 0.944413 | 0.960916 | 0.949467 |
| Current kNN | 30 | 0.959014 | 0.969603 | 0.962810 |
| Random cross-image | 5 | 0.940266 | 0.960622 | 0.943477 |
| Random cross-image | 6 | 0.929665 | 0.961202 | 0.935870 |
| Random cross-image | 8 | 0.936018 | 0.958535 | 0.943934 |
| Random cross-image | 30 | 0.938480 | 0.958773 | 0.940856 |

## Verification and remaining UNVERIFIED items

`verify.py` independently checks the saved metrics using Boolean pair counts, verifies all 472 anchors appear seven times in every pool, checks non-self cross-image identities, recomputes AUC/AP, and confirms all attraction targets and exact extraction checks. `verify.json` records the results. All checks passed.

| Script | CPU wall seconds |
| --- | --- |
| extract.py | 31.90 |
| affinities.py | 40.86 |
| score.py --pool intra | 10.63 |
| score.py --pool knn | 7.19 |
| score.py --pool random | 7.01 |
| verify.py | 6.87 |

- UNVERIFIED: whether stricter positive shifts or mean subtraction improve learned segmentation. That needs a separately authorized controlled experiment; none was run, and no validation-derived setting is recommended as an unsupervised result. Affinity attraction is only one part of the objective; learned-code gates, negative repulsion and background dominance remain relevant.

- UNVERIFIED: generalization to new patients or a held-out cohort, uncertainty over patient sampling, and optimal train-only shifts. Only the existing four validation patients were scored. No training labels were opened. Pair draws are correlated and are not millions of independent clinical observations.

- UNVERIFIED: exact pixel/boundary purity or sensitivity to soft patch occupancy. The reused hard majority patch labels discard partial-volume information; self-token exclusion does not remove all nearby-patch similarity. Training-time dropout sensitivity was not tested.

- UNVERIFIED: exhaustive-pair results, other random pair seeds, other neighbour ranks or neighbours recalculated from alternative blocks. The declared fixed Monte Carlo sample and current top-seven cache are used throughout.

## Reproduction commands (already executed)

```sh
cd /Users/navneetbavineni/STEGO
PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/selectivity/extract.py
PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/selectivity/affinities.py
PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/selectivity/score.py --pool intra
PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/selectivity/score.py --pool knn
PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/selectivity/score.py --pool random
PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/selectivity/verify.py
PYTHONDONTWRITEBYTECODE=1 python3 audit/scratch/selectivity/report.py
```

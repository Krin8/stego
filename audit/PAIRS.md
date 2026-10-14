# CHAOS CT correspondence-pair purity and code-versus-probe diagnostic

## Verdict

Attracting positives are predominantly background, not liver-specific: kNN same-label purity is 87.687%, but liver/liver is only 1.403%. Among attracting liver-anchor pairs only 15.962% target liver; 79.017% of all liver→background pairs attract. Thus positive affinity enrichment is weak for liver isolation, even though it attracts 99.911% of liver/liver pairs. These validation-label diagnostics support objective misalignment; they do not prove a causal sufficiency threshold.

Positive image neighbours overwhelmingly stay within patient (99.667%). Exactly adjacent slices account for 28.299%, while within-three accounts for 78.208%. Liver-anchor-weighted values are 99.852% same-patient and 80.610% within-three. This is image-level serial-slice locality, not a stored organ-specific patch correspondence.

Primary matched-inference code k-means beats the own probe on optimal IoU for: none. Across all checkpoint/seed combinations, improvements occur for [('final', '1'), ('final', '2')]. Nearest-grid code k-means spans 28.447–34.601% optimal IoU, compared with 64.135% on the identical frozen features. The per-checkpoint table distinguishes assignment limitations from degradation of the code geometry; this is not a proof of complete information erasure.

Late-run interpretation, matched-inference oracle IoU: k-means seed 0, batch600 28.931% → batch5000 29.279%; k-means seed 1, batch600 28.931% → batch5000 30.181%; k-means seed 2, batch600 28.948% → batch5000 35.175%. Meanwhile the own probe falls from 44.400% to 29.807%. Thus these tests do not show a uniform loss of recoverable liver structure in the late codes. Final-probe assignment has room for improvement in repeat seeds, but selecting the 35%-level repeat by its IoU would use validation labels. The primary seed does not establish an improvement. All code-k-means majority mappings remain all-background; the gains discussed here require oracle subsets.

UNVERIFIED: training-pair label purity, a literal patch-kNN graph, causal gains from changing the objective, and whether another decoder could recover additional liver information.

## Protocol and label accounting

Producer `setup.py` verifies the validation cache through the project loader/manifest guard, ordered samples, and all validation images/masks against FORENSICS. Producer `purity.py` instruments the actual `ContrastiveCorrelationLoss.forward`/`helper` and the actual `modules.sample`; no affinity, centering, head or probe formula is replaced. All scripts import the project extractor/head/probe and run CPU/eval/inference only. All writes are confined to this report and `audit/scratch/pairs/`; no source, dataset, pipeline cache or checkpoint changes; no neural/supervised training or installs. A runtime guard rejects training annotation reads. K-means is the explicitly requested unsupervised fitting only.

All 472 validation slices: patient counts {'5': 95, '6': 82, '8': 98, '30': 197}. Validation cache `/Users/navneetbavineni/STEGO/src/pytorch_data_dir/nns/nns_chaos_v2_val_5ddb5797be0b7b014c0745edcc62538bc2a10917dfbb6946c04401ec448536df.npz`, shape [472, 30]; actual non-self ranks 1–7. Actual full-precision C shifts: `{'pos_intra_shift': 0.4584466218948364, 'pos_inter_shift': 0.45247218012809753, 'neg_inter_shift': 0.7459585666656494}`. Source checkpoint: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_C_seed0/checkpoints/last.ckpt`. Fresh validation features match the reused diagnostic features exactly (maximum absolute error 0); all validation masks match. No cache was generated or modified; `val_image_neighbors.npy` is a scratch copy of the existing verified indices.

**Labels used for each number:** pair purity, class-conditional attraction and liver-anchor-weighted locality use ONLY the 472 validation masks. Unweighted neighbour locality, image counts, shifts, feature equality, seeds, fit iterations/inertia and runtimes use no labels. Code k-means fitting uses all 1,869 training images/codes with NO training labels; majority and exhaustive optimal mapping and every IoU/Dice use all 472 validation masks for both mapping and scoring. Those scores are optimistic evaluation diagnostics, not label-free mapping results. No supervised classifier is fitted.

**Patch labels:** each 14×14 token gets the majority label from its 16×16 block of the actual 224×224 validation mask; a 50% tie is background. The loss samples bilinearly mixed features at 11×11 random coordinates. Diagnostic labels are nearest-sampled from the patch-majority map at exactly those coordinates, with the same coordinate permutation, reflection padding and align_corners=True. Boundaries make this a discrete approximation to a mixed feature. This differs deliberately from treating an image-cache entry as a patch correspondence.

**Pair sampling:** seven seeded passes enumerate every top-seven positive image edge once, shuffling all 472 anchors into batches of 16 (last batch 8) per pass. Each actual loss forward uses its own uniform coordinate sampling and min(20,batch size) random nonzero rolls. The random comparison is the entire unfiltered set of sampled spatial pairs for that same image-pair type, not a separately chosen easier distribution. Intra-image includes the sampled diagonal, as the actual loss does. Image edges and patch pairs repeat across negative rolls; the large pair counts are not independent observations.

To expose affinity without reimplementing it, positive scalar image-ID codes are passed through the actual loss. Their cosine is one, and the returned loss yields the shifted affinity exactly; stabilization is asserted off. Attracting means affinity > the corresponding saved shift. This measures the coefficient favoring attraction, not whether the learned-code clamp currently permits a nonzero gradient. Restored-global-mean centering is done by the actual helper over each real diagnostic batch. Because pointwise row centering is directional, liver→background and background→liver attraction rates need not match.

Background here includes every non-liver tissue and air. Binary purity therefore measures specificity for this liver task, not whether a correspondence could be meaningful for other anatomy. Pair counts are pooled within each role; differing numbers of negative draws are not estimates of relative gradient weight.

## Task 1: attracting-pair purity

Producer: `audit/scratch/pairs/purity.py` → `purity.json`. Percentages. Parentheses show the random/unfiltered spatial-pair baseline. Same-label = background/background + liver/liver. All these class rates use validation patch labels only.

| Pair type | Attraction rate % | Same-label % (random) | Liver/liver % (random) | Mixed % (random) |
|---|---:|---:|---:|---:|
| Intra-image | 78.526 | 88.000 (85.361) | 1.481 (1.164) | 12.000 (14.639) |
| kNN positive | 78.919 | 87.687 (85.097) | 1.403 (1.108) | 12.313 (14.903) |
| Random cross-image negative | 15.290 | 95.097 (84.402) | 3.413 (0.727) | 4.903 (15.598) |

The high same-label rate alone is misleading because background dominates. These directional quantities directly test whether liver anchors are pulled toward liver:

| Pair type | Target liver given liver anchor, attracted % | Same conditional rate, random % | P(attract given liver→liver) % | P(attract given liver→background) % | P(attract given background→liver) % |
|---|---:|---:|---:|---:|---:|
| Intra-image | 16.943 | 13.724 | 99.902 | 77.902 | 50.842 |
| kNN positive | 15.962 | 13.060 | 99.911 | 79.017 | 51.667 |
| Random cross-image negative | 55.198 | 8.571 | 71.763 | 5.460 | 4.159 |

Patch-majority liver prevalence is 7.409%; 9.363% of token footprints contain both labels. Sampled-coordinate prevalence differs because uniform align_corners=True sampling gives boundary tokens different mass from uniformly enumerating all tokens. Producer: `purity.py`; both quantities use validation masks.

### Patient/slice strata

Producer: `purity.py`, same validation labels. Locality uses native physical-position rank within the same patient and series; adjacent = rank difference exactly one; within-three includes differences ≤3. No self-positive image edges are allowed. Strata overlap (for example adjacent is part of within-three); do not add rows. A dash means no sampled image edges in that stratum.

| Pair type / stratum | Image-pair draws | Patch pairs | Attracted pairs | Attraction % | Same-label attracted % (random) | Liver/liver attracted % (random) | Mixed attracted % (random) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Intra-image / same_patient | 3304 | 48373864 | 37986015 | 78.526 | 88.000 (85.361) | 1.481 (1.164) | 12.000 (14.639) |
| Intra-image / cross_patient | 0 | 0 | 0 | — | — (—) | — (—) | — (—) |
| Intra-image / same_slice | 3304 | 48373864 | 37986015 | 78.526 | 88.000 (85.361) | 1.481 (1.164) | 12.000 (14.639) |
| Intra-image / adjacent_slice | 0 | 0 | 0 | — | — (—) | — (—) | — (—) |
| Intra-image / same_or_adjacent | 3304 | 48373864 | 37986015 | 78.526 | 88.000 (85.361) | 1.481 (1.164) | 12.000 (14.639) |
| Intra-image / within_3_slices | 3304 | 48373864 | 37986015 | 78.526 | 88.000 (85.361) | 1.481 (1.164) | 12.000 (14.639) |
| Intra-image / same_patient_beyond_3 | 0 | 0 | 0 | — | — (—) | — (—) | — (—) |
| kNN positive / same_patient | 3293 | 48212813 | 38048428 | 78.918 | 87.664 (85.069) | 1.405 (1.110) | 12.336 (14.931) |
| kNN positive / cross_patient | 11 | 161051 | 127571 | 79.212 | 94.486 (93.497) | 0.780 (0.618) | 5.514 (6.503) |
| kNN positive / same_slice | 0 | 0 | 0 | — | — (—) | — (—) | — (—) |
| kNN positive / adjacent_slice | 935 | 13689335 | 10816714 | 79.016 | 87.699 (85.091) | 1.401 (1.108) | 12.301 (14.909) |
| kNN positive / same_or_adjacent | 935 | 13689335 | 10816714 | 79.016 | 87.699 (85.091) | 1.401 (1.108) | 12.301 (14.909) |
| kNN positive / within_3_slices | 2584 | 37832344 | 29862464 | 78.934 | 87.362 (84.686) | 1.464 (1.156) | 12.638 (15.314) |
| kNN positive / same_patient_beyond_3 | 709 | 10380469 | 8185964 | 78.859 | 88.768 (86.466) | 1.190 (0.939) | 11.232 (13.534) |
| Random cross-image negative / same_patient | 14833 | 217169953 | 33347205 | 15.355 | 95.451 (86.047) | 3.035 (0.636) | 4.549 (13.953) |
| Random cross-image negative / cross_patient | 37583 | 550252703 | 83992489 | 15.264 | 94.957 (83.753) | 3.562 (0.763) | 5.043 (16.247) |
| Random cross-image negative / same_slice | 0 | 0 | 0 | — | — (—) | — (—) | — (—) |
| Random cross-image negative / adjacent_slice | 205 | 3001405 | 570723 | 19.015 | 94.217 (85.092) | 4.805 (1.155) | 5.783 (14.908) |
| Random cross-image negative / same_or_adjacent | 205 | 3001405 | 570723 | 19.015 | 94.217 (85.092) | 4.805 (1.155) | 5.783 (14.908) |
| Random cross-image negative / within_3_slices | 668 | 9780188 | 1878941 | 19.212 | 94.056 (84.654) | 4.887 (1.202) | 5.944 (15.346) |
| Random cross-image negative / same_patient_beyond_3 | 14165 | 207389765 | 31468264 | 15.173 | 95.534 (86.113) | 2.925 (0.609) | 4.466 (13.887) |

### What a liver patch’s cache neighbours actually mean

The existing cache has image IDs only, so a literal patch-kNN liver-neighbour percentage is **UNVERIFIED/not defined by this cache**. The loss compares sampled patch pairs across each cached image pair, without storing a patch match. The following measurements answer the implemented image-neighbour and all-patch-comparison questions explicitly.

| Locality | All image-cache edges % (no labels) | Edges weighted by source liver-patch count % (validation labels) |
|---|---:|---:|
| same_patient | 99.667 | 99.852 |
| same_slice | 0.000 | 0.000 |
| adjacent_slice | 28.299 | 28.542 |
| within_3_slices | 78.208 | 80.610 |
| same_patient_beyond_3 | 21.459 | 19.242 |
| cross_patient | 0.333 | 0.148 |

Exact counts from `purity.py`: 3304 directed non-self image edges; 6854 liver-majority patches among 92512 validation patches; 47978 liver-patch/image-neighbour combinations. Of the latter, 99.831% point to an image containing at least one liver patch. This does not establish patch alignment. Uniformly comparing each liver anchor patch with all 196 patches in each cached neighbour gives 1,056,201 liver/liver comparisons among 9,403,688, or **11.232% target-liver purity**. Actual loss-coordinate sampling and attraction conditioning instead give the directional percentages in the preceding table. All liver-weighted numbers use validation labels only.

## Task 2: learned codes versus the learned probe

Producer: `audit/scratch/pairs/codes.py --name NAME` → `codes_NAME.json`. All 1,869 train-image codes (366,324 patches), 70 channels, per-vector L2, no mean subtraction. K=8 Euclidean Lloyd k-means, three initializations, max_iter=100, tol=1e-4; primary seed 0 declared before results, seeds 1/2 repeated without selecting a winner. Actual learned heads/probes are imported and run eval/dropout-off. Every visual-trunk tensor equals the frozen reference; live train/val forward checks give zero feature/code error. Fresh actual-probe assignments match every FORENSICS pixel exactly.

Two inference controls: (1) patch-grid k-means assignments resized by nearest-neighbour, matching the frozen-feature BASELINE; (2) raw codes resized bilinearly to 224×224, then L2 and k-means assignment, matching the actual probe’s upsampling/normalization order. K-means uses Euclidean distance to its fitted centroids; the own probe uses cosine to normalized learned centroids, as implemented. No equality of these different assignment objectives is assumed.

Primary seed-0 scores, **IoU / Dice %**. All majority/optimal mappings fit AND score on validation labels. K=8 optimal uses all 256 subsets through the reused FORENSICS scoring routines.

| Checkpoint | Own probe majority | Own probe optimal | Code k-means grid majority | Code k-means grid optimal | Code k-means matched majority | Code k-means matched optimal | Frozen k-means majority = optimal |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed0 | 43.749 / 60.869 | 43.749 / 60.869 | 0.000 / 0.000 | 28.451 / 44.299 | 0.000 / 0.000 | 28.460 / 44.310 | 64.135 / 78.149 |
| seed1 | 0.000 / 0.000 | 32.853 / 49.457 | 0.000 / 0.000 | 28.762 / 44.675 | 0.000 / 0.000 | 28.567 / 44.439 | 64.135 / 78.149 |
| seed2 | 0.000 / 0.000 | 31.948 / 48.425 | 0.000 / 0.000 | 30.429 / 46.660 | 0.000 / 0.000 | 30.405 / 46.632 | 64.135 / 78.149 |
| best600 | 44.400 / 61.496 | 44.400 / 61.496 | 0.000 / 0.000 | 29.032 / 45.000 | 0.000 / 0.000 | 28.931 / 44.879 | 64.135 / 78.149 |
| final | 0.000 / 0.000 | 29.807 / 45.925 | 0.000 / 0.000 | 29.267 / 45.281 | 0.000 / 0.000 | 29.279 / 45.296 | 64.135 / 78.149 |

Frozen reference: `audit/scratch/baseline/kmeans.py --k 8 --center 0 --seed S`, `baseline/k8_c0_sS.json`. It is applicable to every checkpoint because the frozen visual weights and live activations were verified identical. Frozen fitting uses all training images and no labels; its mapping/scoring uses all validation labels. This is the linear-probe-free code-versus-feature comparison.

### K-means seed sensitivity

All scores use validation labels only for mapping/scoring. No seeds are selected by these scores. Producer: `codes.py`; iterations/inertia use no labels.

| Checkpoint | K-means seed | Iterations | Inertia | Grid majority IoU / Dice % | Grid optimal IoU / Dice % | Matched majority IoU / Dice % | Matched optimal IoU / Dice % |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed0 | 0 | 22 | 74116.140625 | 0.000 / 0.000 | 28.451 / 44.299 | 0.000 / 0.000 | 28.460 / 44.310 |
| seed0 | 1 | 24 | 74116.132812 | 0.000 / 0.000 | 28.452 / 44.300 | 0.000 / 0.000 | 28.460 / 44.309 |
| seed0 | 2 | 100 | 74116.078125 | 0.000 / 0.000 | 28.447 / 44.294 | 0.000 / 0.000 | 28.457 / 44.306 |
| seed1 | 0 | 22 | 84260.421875 | 0.000 / 0.000 | 28.762 / 44.675 | 0.000 / 0.000 | 28.567 / 44.439 |
| seed1 | 1 | 31 | 84260.796875 | 0.000 / 0.000 | 28.759 / 44.672 | 0.000 / 0.000 | 28.566 / 44.438 |
| seed1 | 2 | 100 | 84260.406250 | 0.000 / 0.000 | 28.757 / 44.668 | 0.000 / 0.000 | 28.564 / 44.435 |
| seed2 | 0 | 32 | 74596.593750 | 0.000 / 0.000 | 30.429 / 46.660 | 0.000 / 0.000 | 30.405 / 46.632 |
| seed2 | 1 | 34 | 74596.671875 | 0.000 / 0.000 | 30.415 / 46.643 | 0.000 / 0.000 | 30.402 / 46.628 |
| seed2 | 2 | 24 | 74596.515625 | 0.000 / 0.000 | 30.429 / 46.660 | 0.000 / 0.000 | 30.408 / 46.635 |
| best600 | 0 | 47 | 83275.437500 | 0.000 / 0.000 | 29.032 / 45.000 | 0.000 / 0.000 | 28.931 / 44.879 |
| best600 | 1 | 43 | 83275.593750 | 0.000 / 0.000 | 29.041 / 45.010 | 0.000 / 0.000 | 28.931 / 44.878 |
| best600 | 2 | 31 | 83275.617188 | 0.000 / 0.000 | 29.019 / 44.983 | 0.000 / 0.000 | 28.948 / 44.898 |
| final | 0 | 12 | 143811.187500 | 0.000 / 0.000 | 29.267 / 45.281 | 0.000 / 0.000 | 29.279 / 45.296 |
| final | 1 | 20 | 139215.546875 | 0.000 / 0.000 | 29.579 / 45.654 | 0.000 / 0.000 | 30.181 / 46.367 |
| final | 2 | 20 | 143820.156250 | 0.000 / 0.000 | 34.601 / 51.413 | 0.000 / 0.000 | 35.175 / 52.043 |

Selected-initialization fits reaching the predeclared iteration cap: seed0, k-means seed 2, seed1, k-means seed 2. Convergence beyond that cap is UNVERIFIED; returned fits are reported unchanged. All primary seed-0 fits terminated before the cap.

All confusion matrices, selected cluster subsets, and incidental LOPO results produced by the shared evaluator are retained in the JSON files. The tables above use the requested global mapping definitions. No linear probe is used or refitted.

## Limits and UNVERIFIED

- Pair purity describes this four-patient validation cohort/cache. Training-pair purity is UNVERIFIED because no training labels are opened; validation locality and class balance can differ from training.
- No universal numeric purity threshold establishes that correspondence can or cannot teach liver structure. These diagnostics measure permissive/mixed supervision, not a causal intervention on training.
- Literal patch-kNN neighbours are absent from the cache. A specified patch-matching rule/graph would be required to answer that distinct question; no new graph/cache was generated.
- Sampled interpolated features receive discrete nearest patch-majority labels. Boundary-sensitive alternative labeling is UNVERIFIED. The sampled-pair counts are not independent sample sizes.
- Code k-means tests unsupervised K=8 Euclidean separability, not all information carried in a 70-D representation. A failure to improve on the probe does not prove that every possible decoder would fail. Global mapping also remains optimistic.
- No hyperparameters, learned models, source, derivatives or pipeline caches are modified. No neural training or supervised fitting is run.

## Provenance and runtimes

Producer files are all under `audit/scratch/pairs/`. Setup runtime 30.397s; pair purity 32.065s. Each script has a 570-second alarm. Source hashes and full checkpoint/cache paths are in `setup.json`; checkpoint-specific equality checks and primary/repeat fits are in `codes_NAME.json`.

- `seed0`: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_C_seed0/checkpoints/last.ckpt`; producer `python3 -B audit/scratch/pairs/codes.py --name seed0`; runtime 32.560s.
- `seed1`: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_C_seed1/checkpoints/last.ckpt`; producer `python3 -B audit/scratch/pairs/codes.py --name seed1`; runtime 30.671s.
- `seed2`: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_C_seed2/checkpoints/last.ckpt`; producer `python3 -B audit/scratch/pairs/codes.py --name seed2`; runtime 30.181s.
- `best600`: `/Users/navneetbavineni/STEGO/checkpoints/chaos/chaos_fixes3_full_C_date_Sep30_19-47-58/checkpoint-epoch=05-step=1800-mIoU=test/cluster/mIoU=69.3704.ckpt`; producer `python3 -B audit/scratch/pairs/codes.py --name best600`; runtime 30.416s.
- `final`: `/Users/navneetbavineni/STEGO/checkpoints/chaos/chaos_fixes3_full_C_date_Sep30_19-47-58/checkpoint-epoch=42-step=15000-mIoU=test/cluster/mIoU=46.1792.ckpt`; producer `python3 -B audit/scratch/pairs/codes.py --name final`; runtime 30.336s.

Source commit `e3c3ab0e396c194a7314d64546c81e9ba4973b72`. Tracked source changes: none. Existing audit/baseline artifacts are read-only inputs.

Final checks: `audit/scratch/pairs/verify.py` → `verification.json`; {"script": "/Users/navneetbavineni/STEGO/audit/scratch/pairs/verify.py", "status": "PASS", "affinity_exposure_max_errors": {"pos_intra_shift": 2.9802322387695312e-08, "pos_inter_shift": 2.9802322387695312e-08, "neg_inter_shift": 2.9802322387695312e-08}, "source_hashes_unchanged": true, "no_tracked_source_changes": true, "purity_conservation_checks": "PASS", "all_five_checkpoint_head_probe_reproductions": "PASS", "code_fit_zero_majority_cases": [["seed0", "0", "nearest"], ["seed0", "0", "matched_bilinear"], ["seed0", "1", "nearest"], ["seed0", "1", "matched_bilinear"], ["seed0", "2", "nearest"], ["seed0", "2", "matched_bilinear"], ["seed1", "0", "nearest"], ["seed1", "0", "matched_bilinear"], ["seed1", "1", "nearest"], ["seed1", "1", "matched_bilinear"], ["seed1", "2", "nearest"], ["seed1", "2", "matched_bilinear"], ["seed2", "0", "nearest"], ["seed2", "0", "matched_bilinear"], ["seed2", "1", "nearest"], ["seed2", "1", "matched_bilinear"], ["seed2", "2", "nearest"], ["seed2", "2", "matched_bilinear"], ["best600", "0", "nearest"], ["best600", "0", "matched_bilinear"], ["best600", "1", "nearest"], ["best600", "1", "matched_bilinear"], ["best600", "2", "nearest"], ["best600", "2", "matched_bilinear"], ["final", "0", "nearest"], ["final", "0", "matched_bilinear"], ["final", "1", "nearest"], ["final", "1", "matched_bilinear"], ["final", "2", "nearest"], ["final", "2", "matched_bilinear"]], "code_script_runtimes": {"seed0": 32.56021320800937, "seed1": 30.670654499990633, "seed2": 30.18052758400154, "best600": 30.415551415993832, "final": 30.336174875003053}, "setup_runtime": 30.396979459008435, "purity_runtime": 32.06529199999932, "elapsed_seconds": 4.533747749999748}

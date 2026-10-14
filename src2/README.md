# Optimized P1 CT liver segmentation — seed 0, N2 only

`src2` is the dedicated implementation of the selected optimized P1 pipeline. It contains its own Python modules and does not import code from `src/`, `audit/` or `experiments/`. Large existing data, cached features and the local BiomedCLIP weight/config files are external inputs named in configuration. Original project source, datasets and experiment artifacts remain untouched.

## The exact method this package represents

CT HU → [-150,250] window → uint16-equivalent quantization → native 512 frame → intensity-based square body crop → RGB → 448 input → BiomedCLIP normalization once → frozen final-block LayerNorm patch tokens, CLS excluded → patch L2 → 28×28×768 → **seed-0 P1 nonlinear branch only, without output L2** → K=8 train-only k-means → **N2 anatomical naming only** → soft assignment → inverse crop → guided filter size 25 / epsilon .01 → threshold .30 → largest 8-connected component → fill holes → closing disk radius 3.

This is the later optimized variant. The checkpoint was trained with the **full linear + nonlinear P1 head**, not the branch-only inference head. `modules.py` strict-loads all checkpoint keys, preserves both branches, and explicitly exposes branch-only `embed()` for optimized inference. It does not silently ignore state keys. `embed_full()` is available for parity checks but is not used by evaluation.

There is no text naming, agreement naming, label-based cluster mapping, learned cluster probe or CRF in the inference route. N2 is selected once using training images and then fixed for validation and new-image inference. If no cluster satisfies N2, the fit records abstention and predictions are empty.

## File layout

```text
src2/
  configs/
    pipeline.yaml          shared method, external paths and backbone/checkpoint identities
    train_config.yaml      original P1 training settings and new output location
    train_knn_config.yaml  optional feature-kNN comparison; same shifts and budget
    fit_config.yaml        train-only clustering/N2 output location
    eval_config.yaml       fit artifact and validation output location
    demo_config.yaml       new-image inference settings
    plot_config.yaml       four fixed validation IDs and plot output
    chaos_splits_v2.json    copied frozen patient assignment
  checkpoints/
    p1_seed0.pt             byte-identical copy of the existing 500-batch checkpoint
  preprocessing.py         raw CT/windowed PNG loading, crop, normalization and inverse geometry
  preprocess.py            dry-run/default or explicit windowed-PNG generation CLI
  backbone.py              strict local-only frozen BiomedCLIP loading and dense extraction
  extract_features.py      optional explicit feature-cache creation CLI
  data.py                  ordered feature cache and patient/series metadata
  modules.py               full training head and explicit optimized inference head
  reference_loss.py        pinned restored-global-mean STEGO loss helper
  loss.py                  P1 adjacent-slice, co-located partner term
  correspondence.py        selectable P1 or train-only feature-kNN pairing/loss
  train_segmentation.py    fresh seed-0 training; final state only
  collapse_monitor.py      original train-only monitor definitions
  state.py                 optimizer/model/RNG serialization
  clustering.py            squared distances, soft assignments and fit validation
  naming.py                N2 body/HU/centroid/area computation
  fit_clusters.py          all-train K=8 fit, temperature and N2 selection
  postprocessing.py        guided filtering and binary morphology
  eval_segmentation.py     seal validation predictions, then score annotations
  metrics.py               pooled/per-slice/per-patient binary metrics
  demo_segmentation.py     CT folder → native masks, clusters and filled overlays
  visualize_segmentation.py  plots actual saved arrays without refitting
  utils.py                 config, identities, output protection and label-access guard
  tests/                   numerical parity tests and one-real-slice backbone smoke
  verification/            logs of checks actually run during implementation
  PROVENANCE.json          copied/reused source identities and verification scope
  requirements-tested.txt  versions present for testing; no packages installed by this task
```

Configs are JSON-formatted YAML (valid YAML), parsed with the Python JSON reader. Preserve that syntax when editing. Shared sections live in `pipeline.yaml`; stage files name a `base` and add their own settings. Stage overrides replace an entire top-level section, not individual nested fields. Paths in configs are project-root relative or absolute. Existing image-record JSON files retain their original absolute dataset paths; moving datasets to another computer requires updating those records/paths explicitly, without changing sample order or crop geometry.

## Fixed parameters and geometry

- Head training: seed 0, Adam LR 0.0005, batch 16, **500 batches**, no scheduler, Dropout2d .1; full 768→70 linear plus 768→768→70 nonlinear branches. No text/salience/reconstruction/CRF/alignment loss.
- Correspondence weights: self .67, adjacent partner .25, random .63. Shifts: .794418, .8869667053222656, .797651. Five random negative repetitions; 11×11 sampled coordinates. Restored-global-mean affinity centering; code similarities clamped below at zero. P1 selects the diagonal **after full spatial centering**.
- Partners: one uniform offset from {-2,-1,+1,+2} in each patient's physical-position-sorted series. Out-of-range draws skip the partner term for that sample. This is body-frame co-location, not deformable registration.
- K-means: K=8, seed 0, n_init=3, max_iter=100, tol=.0001, Lloyd; all 784 patches of all 1,869 train slices. No mean subtraction or L2 normalization of the optimized 70-D codes.
- Soft temperature: median of the nearest/second-nearest **squared-distance** gaps on train codes, divided by log(4), minimum 1e-6.
- Feature crop: windowed intensity >.05, largest 8-connected component, holes filled; square bounding box with side multiplied by 1.1 and zero padding. This differs from N2's body mask.
- N2: raw HU >-500, disk-radius-5 opening, largest component and hole fill. Candidates have body-relative centroid x<.5, y<.6 and mean HU in [40,180]; choose largest body-patch area. Coordinates are native image-left/top to right/bottom, not a new DICOM reorientation.
- BiomedCLIP mean [.48145466,.4578275,.40821073], std [.26862954,.26130258,.27577711], once. Position embeddings are interpolated by the dynamically sized vision transformer; the 28×28 map is extracted directly at 448, not upsampled from 14×14.
- Native PNG contract: uint16 grayscale, already windowed; ambiguous 8-bit/RGB input is rejected. Native CT frame is 512×512. No slice-spacing resampling or flips.

## Commands — not run automatically

Run from the STEGO root. Direct script execution (`python src2/fit_clusters.py ...`) and module execution both work. Every output directory must be new; existing results are never overwritten.

### Existing checkpoint → fit → evaluation → visualization

```bash
python -m src2.fit_clusters
python -m src2.eval_segmentation
python -m src2.visualize_segmentation
```

The first command performs the full train-only k-means fit and N2 naming. It can be substantial CPU work and was **not** executed during package construction. Evaluation refuses a mismatched checkpoint, feature cache, sample order or fit contract. It saves `raw_clusters.npy`, `N2_hard.npy`, `N2_refined.npy` and a hashed `prediction_manifest.json` **before** opening annotation arrays. Only then does it write `metrics.json`.

Metrics cover all 472 validation slices and four patients, hard and refined predictions, native 512 and 224 scales. Values in JSON are fractions. The report distinguishes globally pooled pixels, per-patient pooled pixels, mean slice scores, and mean/sample-SD across patients. An empty/empty mask receives IoU/Dice zero to match the existing denominator convention. There is no label-assisted mapping; selected_cluster comes from the saved N2 fit.

### Predict new slices without annotations

```bash
python -m src2.demo_segmentation --input /path/to/flat_CT_folder --output src2/runs/new_CT_predictions
```

Requires the saved fit from `fit_clusters`. Reads only the top level of the supplied folder, with unique stems and `.dcm`/`.png` extensions; it does not search raw dataset annotation subfolders. Saves native cluster arrays, hard-mask PNGs, refined-mask PNGs, filled overlays and a manifest. No ground truth is needed.

### Inspect preprocessing, or explicitly generate derivatives

```bash
python -m src2.preprocess --input /path/to/flat_CT_folder
# Explicit optional write to a NEW folder:
python -m src2.preprocess --input /path/to/flat_CT_folder --output src2/runs/new_windowed_CT --write
```

The first command reads/checks inputs and prints a count without writing images. It does not replace the existing cohort's image/mask pairing or split manifest. Training uses the already frozen cohort records.

### Optional training — not needed to use the existing checkpoint

```bash
python -m src2.train_segmentation --output src2/runs/p1_seed0_new_training
```

This creates a fresh 500-batch run, not a continuation. It checks for overlapping training processes, uses a package lock, reads no annotations, serializes optimizer and RNG state, and writes the final checkpoint only. It reports the original collapse monitors every 50 batches; it does not add a new early-stop policy. A failed/interrupted run must restart into another new directory. The first ten batches estimate runtime; a projection above 90 minutes stops the run.

The shipped inference config intentionally remains pinned to the existing checkpoint digest. To evaluate a newly trained checkpoint, create a separate configuration with its path and SHA from `complete.json`, then use fresh fit/evaluation directories. Training does not silently replace the inference model. Bit-identical long-run reproduction is **not established** by this packaging task; the original loss/gradient and head computations have been checked numerically.

### Optional feature extraction — not needed with the existing cache

```bash
python -m src2.extract_features --output src2/runs/new_feature_cache
```

This explicitly creates a new array from the configured, ordered image records. It never replaces the old cache. Point a separate configuration's `paths.features` and `paths.feature_metadata` at those new outputs to use them. No extractor runs automatically during cached-feature training/fit/evaluation. New-image demo inference does run the frozen image encoder.

## Optional comparison: original feature-kNN positive correspondence

The default `train_config.yaml` remains P1. To **replace adjacency during a new
training run**, use the separate configuration:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
caffeinate -i python3 -m src2.train_segmentation \
  --config src2/configs/train_knn_config.yaml
```

The output is `src2/runs/knn_seed0_training`, which must not already exist. This
command has not been run during implementation. It reads the configured cached
448 features; it does not regenerate features or use validation images for pairing.
At startup it builds train-only image neighbours from L2-normalized means of the
frozen normalized patch tokens, cosine similarity, stable tie-breaking and explicit
self exclusion. It records 30 neighbours per image and draws uniformly from the
top seven, matching the earlier 448 feature-kNN recipe. The neighbour array and
its ordering, method, pooled-feature digest and file digest are saved only inside
the new run. No old neighbour cache is silently reused or overwritten.

The loss uses `reference_loss.ContrastiveCorrelationLoss` directly: independent
sampled coordinates in the two images and the full 121×121 patch-pair matrix,
with restored-global-mean centering. It does not take the P1 diagonal. Same-image
and random-image terms remain present; a neighbour image is not assumed to be
uniformly positive, since each patch affinity is still compared with its shift.

Only partner selection and the partner correspondence construction change.
The seed, weights, shifts **(.794418, .8869667053222656, .797651)**, optimizer,
head, dropout and 500-batch budget are unchanged. In particular, the P1-tuned
partner shift is intentionally retained for this isolated comparison; it has
not been recalibrated for kNN. This restores the original positive-pair mechanism,
not every component or hyperparameter of the original STEGO paper.

The bundled checkpoint is still the historical **P1** model; changing a config
does not turn that checkpoint into a kNN-trained model. After a successful new run,
point separate fit/eval configs at `knn_seed0_training/final.pt` and its SHA from
`complete.json`, and give each stage a fresh output directory. Then rerun clustering,
N2 naming, evaluation and visualization. N2 and guided refinement are unchanged.
No segmentation-quality improvement is established until that experiment is run.

## Verification performed

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s src2/tests -v
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m src2.tests.smoke_backbone
```

Tests cover optimized-head parity, preprocessing/inverse geometry, exact guided-refinement masks, N2 statistics, original loss and gradients, patient ordering, adjacency boundary behavior, annotation-access rejection, metrics and fit-identity rejection. The real-slice check compares DICOM→PNG-equivalent→448→backbone tokens with the existing cache, tests determinism and checks the 70-D code shape. No training run, k-means fit, full validation scoring, derivative regeneration or feature-cache regeneration was performed to build this package.

The local OpenCLIP constructor can print a temporary random-initialization warning before `backbone.py` immediately strict-loads the verified local checkpoint. The strict load and cache-parity check establish that random weights are not used for extraction.

## Limitations and claim boundaries

This package preserves the selected branch-only optimization; it does not make that variant equivalent to the full trained P1 head. The claimed 83.92% has not been newly established. Full clustering/evaluation must be run to produce a new attributable result. The four validation patients informed earlier method choices; resulting scores are not from an untouched test cohort. The N2 spatial prior assumes the existing native CHAOS orientation. CT only; arbitrary image sizes and ambiguous PNG intensity scales are rejected.

No performance superiority, external-cohort validity or causal claim about dimensionality/overconfidence is implied by a successful software check. All original project files remain available unchanged.

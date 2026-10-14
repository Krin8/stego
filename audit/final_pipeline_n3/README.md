# Frozen-feature CT F1 package

CPU inference; no gradient training. Run from the existing STEGO environment. `segment_ct.py` is the single inference entrypoint. It imports the project's actual `BiomedCLIPFeaturizer` and approved DICOM HU preprocessing; it does not invoke the training entrypoint. Local BiomedCLIP weights and installed dependencies are required; no network/download/install is performed. Spatial/filter functions are frozen verbatim from IMPROVE; `function_provenance.json` records their source lines and hashes.

## Method

1. DICOM: apply RescaleSlope/Intercept, clip HU [-150,250], scale [0,1], quantize through uint16 to exactly match existing derivatives. PNG: accept only already-windowed uint16 grayscale under this contract; no second HU window. Native 512×512, orientation unchanged, no spatial resampling.
2. Image-only body mask >0.05, largest 8-connected component, hole filling, square bounding box with 5% margin on each side and zero padding. Resize crop to448 with bilinear antialiasing, replicate grayscale to3 channels, apply BiomedCLIP mean/std exactly once.
3. Frozen eval-mode project extractor: final-block LayerNorm patch tokens, CLS excluded, interpolated positional embeddings, 28×28×768, per-patch L2 normalization, no mean subtraction. No learned STEGO projection head/probe is used.
4. K8 Euclidean k-means; distances to fixed centres determine soft weights. Temperature is the fitting-feature median nearest/second-nearest squared-distance gap divided by log4. Sum weights of liver-mapped clusters.
5. Bilinear resize/invert body crop into native512 (outside crop score0), guided filter radius8/epsilon0.01, threshold >=0.5, largest8-connected component, fill holes with4-connected background, disk-radius3 closing with zero padding. No parameters chosen from current validation scores.
6. Optional `--f2`: median across3 physically ordered same-series slices, replicated endpoints. OFF by default. No mm resampling. PNG requires an explicit physical-order manifest; DICOM verifies one series and orders by projected physical position.

## Inference commands

Run from the project root:

```sh
python3 audit/final_pipeline/segment_ct.py --input /path/to/ct_slices --output /path/to/new_output
python3 audit/final_pipeline/segment_ct.py --check
python3 audit/final_pipeline/segment_ct.py --check --seed 1
python3 audit/final_pipeline/segment_ct.py --check --seed 2
```

Output: `<slice>_mask.png` (uint8 0/255), `<slice>_overlay.png` (red fill/contour), `run.json` (full parameters, seed and input paths). Existing non-empty output directories are rejected. Seed0 is the predetermined default, not selected by evaluation score; `--seed 1/2` selects the other saved fits.

PNG input must have documented [-150,250] HU windowing and uint16 storage round(65535*x). The script cannot infer a PNG's HU window from its pixels. 8-bit/RGB inputs and non512 images explicitly fail instead of silently using incorrect preprocessing. DICOM accepts `.dcm`, `.dicom`, and extensionless files; it rejects non-CT.

Optional F2 on one DICOM series:

```sh
python3 audit/final_pipeline/segment_ct.py --input /path/to/one_dicom_series --output /path/to/new_f2_output --f2
```

For PNG F2, also pass `--order-manifest order.json`, a JSON array of records with `file` (input basename), `series_id`, and `physical_position` (number), covering every input exactly once. One series and unique positions are mandatory. Filename ordering alone is never treated as physical ordering.

## Label-Use Ledger

This final pipeline uses **no label or mask files** during clustering, naming, or inference.
- **BiomedCLIP** provides language-to-vision alignment based on its pretraining corpus, used here as a zero-shot anatomical prior (N1 naming).
- **N2 anatomical rules** (Hounsfield units, body masking, and area thresholds) are predefined medical priors.
- **Hyperparameters** ($K=8$, $448 \times 448$ resolution, $[-150, 250]$ HU window, and closing/filling post-processing) were selected historically during development using ground-truth labels on the validation patients.
- The runtime inference reads exclusively DICOM and PNG intensity data. No annotation paths are accessed.

- `all20_seed{0,1,2}_centers.npy/.json`: exact existing all20 IMPROVE K8 fits reused. Mapped fully label-free using the N3 naming rule (requiring N1=N2 agreement). These are the deployment artefacts. Metadata includes backbone weight-file SHA256, centres SHA256, preprocessing, K, seed, N1/N2 cluster selections, git commit, and library versions.
- `check_reference.json`: five fixed validation images (four existing fixed slices plus first remaining validation record). `--check` recomputes REAL features and inference and compares native IoU within absolute1e-4 (0.01 percentage point).
- `package_verification.json`: live-vs-cached feature check, inference comparison and DICOM/PNG equivalence. `cli_verification.json`: actual CLI behavior and file outputs.

## Reproducing the study

The following scripts write only into this package directory and reuse immutable IMPROVE features. Existing completed fits are not overwritten. The current results are already generated; do not rerun into populated output names.

```sh
python3 audit/final_pipeline/study.py fit --seed 0
python3 audit/final_pipeline/study.py mapping --seed 0 --scope inductive
python3 audit/final_pipeline/study.py evaluate --seed 0
# Repeat for seeds 1 and 2, then:
python3 audit/final_pipeline/report.py
```

Exact executed commands are in `commands.txt`; fitted temperatures and subsets are in the per-seed JSON files. KMeans settings are unchanged from BASELINE/IMPROVE. The `copy_x=False` working array is a new disposable file in this directory, never the original feature cache.

## Fixed parameters

| Parameter | Fixed value |
|---|---|
| modality | "CT" |
| window_hu | [-150, 250] |
| intensity | "clip HU then linear [0,1]; round(65535*x)/65535" |
| png | "uint16 grayscale already windowed [-150,250]; reject 8-bit/RGB" |
| orientation | "native; no flip/transpose" |
| spatial_resampling | "none" |
| native_shape | [512, 512] |
| body_threshold | 0.05 |
| body_connectivity | 8 |
| body_box | "largest component; fill holes; square bounding box with 5% each-side margin; zero padding" |
| input_resolution | 448 |
| resize | "torchvision bilinear antialias=True" |
| mean | [0.48145466, 0.4578275, 0.40821073] |
| std | [0.26862954, 0.26130258, 0.27577711] |
| normalization_application_count | 1 |
| grayscale_channels | 3 |
| backbone | "BiomedCLIP ViT-B/16" |
| layer | "final block" |
| normalization | "final LayerNorm then patch L2; uncentered" |
| cls_excluded | true |
| grid | [28, 28] |
| eval | true |
| device | "cpu" |
| soft_assignment | "softmax(-squared_euclidean_distance/temperature)" |
| temperature | "median nearest/second-nearest squared-distance gap over fitting patches / log(4); floor 1e-6" |
| inverse_crop | "bilinear align_corners=False; outside=0" |
| guided_radius | 8 |
| guided_epsilon | 0.01 |
| guided_boundary | "reflect" |
| threshold | 0.5 |
| largest_component_connectivity | 8 |
| hole_fill_background_connectivity | 4 |
| closing_disk_radius | 3 |
| closing_padding | "3 zero pixels then crop" |
| f2_default | false |
| f2 | "median(3,1,1), nearest endpoints, one series at a time" |
| K | 8 |
| kmeans_init | "k-means++" |
| kmeans_n_init | 3 |
| kmeans_max_iter | 100 |
| kmeans_tolerance | 0.0001 |
| kmeans_algorithm | "lloyd" |
| seeds | [0, 1, 2] |
| default_deployment_seed | 0 |
| cluster_mapping | "native512 strict majority (>50% liver), ties/background outside crop" |
| check_iou_absolute_tolerance | 0.0001 |
| primary_scoring_resolution | 224 |
| mask_resize | "nearest-neighbour" |

## Evaluation definitions and limitations

`FINAL_PIPELINE.md` reports both pooled-pixel per-patient IoU/Dice and arithmetic mean of slice IoUs/Dices, at native512 and nearest-resized224. Aggregate seeds within patient first; then report equal-patient mean, sample SD (ddof1), min/max across the4 patients. JSON retains every seed and slice. Tiny-slice strata use native GT area <2,000 pixels, and compare against >=2,000. Boundary distances are symmetric2D native pixel distances pooled per patient; one-empty masks contribute infinity. Finite-only summaries are explicitly conditional, not substituted for missed objects. Surface metrics are not mm or3D distances.

CT only;20 labeled patients from one cohort; only4 validation patients. Tiny liver slices can fail completely. Liver-absent slice FP behavior is UNVERIFIED if no such labeled slices occur. Do not infer clinical validity or performance on other scanners/protocols. Arbitrary PNG intensity history is UNVERIFIABLE from PNG data alone. The parent F1 pipeline was historically selected using cohort labels; this inductive recomputation removes fitting/mapping leakage for this split but DOES NOT make it an untouched independent test set.

## Do not run the default STEGO trainer for this pipeline

**Do not run `src/train_segmentation.py` with its defaults expecting F1. It trains a different STEGO head and the recorded default run collapsed.** Evidence: `logs/chaos/chaos_chaos_date_Oct01_23-36-02/lightning_logs/version_0/validation_diagnostics.jsonl` line4: batch217, cosine0.9985517473, cluster4 usage1.0, liver IoU0; early stopping disabled. `logs/wandb/run-20261001_233607-98vsvgl7/logs/debug.log` line20 records shifts0.18/0.12/0.46 and `collapse_early_stop=False`.

Prior method/evaluation evidence: `audit/improve/IMPROVE.md`, `audit/improve2/IMPROVE2.md`, `audit/improve2/PROTOCOL.md`. Current inductive evidence: `FINAL_PIPELINE.md`. No source files, existing datasets, caches or checkpoints were modified for this package.

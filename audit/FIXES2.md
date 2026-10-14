# Approved contract implementation

Branch: `audit-fixes-2`, from snapshot `5639388` on `audit-fixes`. Historical results are void for this corrected baseline. No training, derivative generation, cache generation, package installation or existing dataset/checkpoint changes are authorized in this turn.

Task 3 is implemented first because the dry-run plan and subsequent loaders need a common frozen split. MRI retains its current processed-directory patient assignment; only in-phase T1 and T2 are eligible. Each task has a separate commit. Tests are bounded CPU checks, with before/after results recorded below.

Current follow-ups F1–F4 below supersede historical qualifications: CRF is optional, headline loss excludes diagnostic CE, and cache identity is independent of Git commit and machine location.

## Tasks

Task | Status | Commit | Verification
--- | --- | --- | ---
1 | APPLIED; dry-run only | 38a2880 | 3333 unique image/mask pairs; zero pairing errors
2 | APPLIED | 3344fbe | Shared raw/PNG tensors; one normalization; real CT/MRI loader checks
3 | APPLIED | bd72ab1 | Frozen patient IDs; overlap/unknown-patient rejection
4 | APPLIED; no cache generated | 1395021 | UTF-8 metadata and10 rejection cases; exclusive write/archive spy
5 | PASS | 1a16287 | Six native overlays visually checked; finite inside/outside means
6 | VERIFIED WITH QUALIFICATION | 0a3aeac | Label/mask perturbation gives zero head/cluster gradient change; aggregate diagnostic scalar is label-dependent
7 | PASS | commit containing fixes2_smoke.py (use git log below) | Real BiomedCLIP shape/loss/backward/metric pass
m1 | APPLIED; comment only | 255f3e0 | Reference centering numeric values unchanged

Task7 commit lookup: `git log -1 --format=%h -- audit/scratch/fixes2_smoke.py`. One commit per task; this log is included in each task commit. The prior investigation snapshot was made on audit-fixes before branching.

## Task evidence


### Task 3 — APPLIED

`src/configs/chaos_splits_v2.json`, `src/chaos_splits.py:1`, `src/data.py:12,116,129,383`: froze exactly the current CT_ patient assignment, and retained current MRI assignments for both eligible sequences. CT validation patients are 5,6,8,30; MRI validation patients are 5,8,38,39. CT and MR identifiers are separate namespaces. Loaders validate membership; the cache writer uses these loaders.

Before: `fixes2_split_check.py` demonstrates adding patient99 silently changes the old computed validation set from {30,5,6,8} to {5,6,8,99}. After: the same check with `--expect-fixed` verifies exact equality to current directory membership, rejects unknown patient99 and a duplicate train/val patient. PASS; no data files written. Generator and cache identity integration follow in tasks1/4.

### Task 1 — APPLIED (dry-run only)

`src/chaos_preprocessing.py:1`, `src/preprocess.py`: approved CT [-150,250] parameter; T1 in-phase/T2 whole-slice 1–99.9 percentile scaling; finite/constant checks; excluded out-phase; native planes without 3D resampling. Pairing uses CT `InstanceNumber - 1` → `liver_GT_ID`, MRI exact basename, then physical-position sorting. All pairs validate one-to-one IDs, series identity, orientation and matching image/mask dimensions. Output defaults to NEW `preprocessed_v2`, requires `--write`, and refuses existing output. Derivatives store normalized grayscale in uint16 PNG at native dimensions; model resizing follows in task2. The quantization error is at most 1/65535, and raw/derived routes use the same quantization.

Before: CT input[-150,50,250] became uint8[80,143,207]; old mask association zipped independently sorted lists. After: same test verifies [0,.5,1], slope/intercept, constant MRI→zero, MRI quantiles, excluded sequence and missing-mask rejection; PASS. Initial test harness needed tuple-assignment extraction; initial full dry run encountered a non-patient `CHAOS` directory, so discovery now selects numeric patient directories and still rejects any unknown/missing patient against the frozen manifest.

Full guarded dry run: **3333 images, exactly one mask each, zero pairing problems**. Train: CT1869, T1DUAL395, T2SPIR393; val: CT472, T1DUAL105, T2SPIR99. Native CT counts differ from old resampled CT_ counts (2606/629); patient assignment is unchanged. No derivative directory or manifest was created. Existing historical resampling helpers remain but are not used by the v2 CLI.

### Task 2 — APPLIED

`src/data.py`, `src/chaos_preprocessing.py:ChaosTransform`, train/cache/eval/demo callers and eval/demo resolution defaults: shared224 short-edge bilinear resize + center crop, grayscale→3 channels, exactly one BiomedCLIP normalization. Masks take the same geometry with nearest-neighbor. Raw and uint16-PNG routes use identical normalized intensities; no body ROI crop, histogram equalization or in-plane physical resampling. Loader reads only `preprocessed_v2/manifest.json`, validates its preprocessing/frozen split, excludes legacy families/out-phase and keeps the mixed-family guard. Missing v2 derivatives fail explicitly instead of falling back to old PNGs. The unsupported-format stub was preserved verbatim.

Before: MRI stub and demo equalization reproduced; different entrypoint geometry and old-family discovery found. After: `fixes2_loader_check.py --expect-fixed` PASS for all3 modalities' raw/PNG tensor equality (PNG roundtrip in memory), exact one mean/std application, nearest labels, missing-v2 rejection and all entrypoint callsites. Additional real raw-loader reads: CT1869/T1DUAL395/T2SPIR393 training images; first sample from each produces finite(3,224,224) and224² integer labels. Source parses. No derivative generated. Default CT still uses2 classes; MRI runs must request5 classes. Historical checkpoints/configs are incompatible with this preprocessing contract.

### Task 4 — APPLIED (no cache generation)

`src/knn_cache.py:1`, `src/data.py:ContrastiveSegDataset`, `src/precompute_knns.py`, `src/configs/train_config.yaml`: lean UTF-8 JSON manifest stored as a uint8 byte array (no pickle), exact ordered IDs/hash, approved transform, frozen split hash, freshly recomputed image-content hashes, backbone/config file hashes, final normalized token/eval/dropout identity, k/self rule, array shape/dtype/digest and git commit. Filename contains the identity SHA256. Reader rejects mismatches/invalid indices and never falls back to an unverified legacy cache. Including image and split hashes prevents stale content under unchanged filenames; git changes conservatively invalidate identity even for documentation-only commits.

Before: actual current reader with mocked dataset accepted arbitrary metadata and array[[0,999]]. After: `fixes2_cache_check.py --expect-fixed` PASS for in-memory valid roundtrip and rejection of reordered IDs, weights, preprocessing, split, git, missing metadata, bounds, self-slot, duplicate positives and digest corruption. Writer's actual tensor expressions tested on tied similarities produce [[0,1,2],[1,0,2],[2,0,1],[3,0,1]]: self first, non-self positives, stable index tie-breaking. Existing target refusal and explicit archival tested with a rename spy; no files moved. Changed sources parse.

Writer requires `archive_existing_cache=true` to move an existing *same-identity target* to a unique .old-* path; otherwise it raises. Exclusive new-file creation prevents overwrite. Legacy filenames remain untouched because new filenames differ. All NPZ tests were in BytesIO; no real cache was written or regenerated. Config uses knn_k30 (including self),7 selectable non-self positives. Regenerate only after user-run derivative generation, on the final intended commit.

### Task 5 — APPLIED

`audit/scratch/fixes2_overlays.py`, `audit/scratch/fixes2/overlays/`: before `--verify-only` failed because no v2-pair overlays existed; after creating only audit images, the same check PASS (six overlays and finite numerical records). Native CT1/22/30 and MRI T1-in8/37,T2-37 masks visually agree with organ anatomy; no flip/transposition applied. Mean approved[0,1] intensity inside/outside: CT1 .759925/.220232; CT22 .725182/.150100; CT30 .684984/.184135; T1-in8 .427775/.261999; T1-in37 .403841/.178320; T2-37 .295132/.050886. All six have100% foreground on nonzero image intensity. This is sampled evidence, not a full anatomical audit; tissue overlap alone cannot prove orientation. Exact paired paths/IDs are in `overlays/numbers.json`. No derivative, cache, dataset or checkpoint was written.

### m1 decision — APPLIED, comment only

`src/modules.py:479`: identifies pinned STEGO eb4d6b5 restored-global-mean centering, explicitly distinguishes literal Paper A Eq(3). `fixes2_centering_check.py` before/after gives identical self/neighbor/negative losses[-.2071744949,-.0324866250,.1538662761]; after also checks the comment. No centering or loss change.

### Task 6 — VERIFIED WITH QUALIFICATION (read-only)

`audit/scratch/fixes2_unsupervised_check.py` executes the unchanged production `training_step` and actual correspondence/cluster-loss definitions on tiny CPU tensors, with no-op optimizer spies (zero updates, all parameter values unchanged). A frozen tiny backbone isolates the training method; the real pretrained forward/freeze check is task7. Changing labels0→1 and masks/positive labels together gives **max segmentation-head gradient delta0 and cluster-gradient delta0**, while linear-probe gradients change. Backbone grads remainNone. With correspondence temporarily disabled in the test only, head grads areNone despite linear+cluster backward. PASS(a). Baseline flags disable true-label features, salience, text prompts and all experimental auxiliary losses. PASS(b) for the unsupervised representation/cluster gradients; **literal(b) for the aggregate scalar is false**: `loss/total` includes isolated supervised linear CE and changed .55918628→.40883285. This is a diagnostic-loss aggregation, not a gradient leak. No production objective edits were made. Initial harness attempted to extract an unused scripted helper without a module name; removing that unused harness dependency resolved the test.

Label-use inventory for the active workflow:

- **Data eligibility/pairing:** `chaos_preprocessing.py:86–129,174–179` requires one mask and matching dimensions, checks label encoding when generating, and copies masks. It does not select samples by organ occupancy or use mask values to compute image intensity normalization (`normalize_pixels:45–63`). Current slice eligibility is label-availability informed.
- **Preprocessing choice:** the approved CT window/MRI percentiles were chosen using organ-mask clipping statistics (`audit/DECISIONS.md:66–111`), so preprocessing hyperparameter selection is label-informed even though its runtime transforms are mask-independent. Task5 uses nonempty masks only to choose illustrative overlays, never to select training examples.
- **Decode/resize/ignore:** `data.py:175–181,234–235` maps CT/MRI codes and applies nearest-neighbor masks; unsupported codes become ignore. Valid/salience mask is image-derived at237, and baseline `use_salience=False`. `data.py:394–396,416–418` transports labels and masks to batches; baseline correspondence ignores them.
- **Diagnostic linear training:** `train_segmentation.py:269–281` filters valid labels, clones detached code and computes foreground-weighted supervised CE; `linear_probe_weights` at106–111. Only the separate linear optimizer receives those gradients (`configure_optimizers:521–529`). Combined logged loss includes this diagnostic. Linear probe is NOT an unsupervised result.
- **Inactive label/text routes:** `train_segmentation.py:179–181` true-label correspondence,192–193 salience;60–89 organ-name text prompts/cluster initialization,256–267 text alignment. All disabled in baseline (`configs/train_config.yaml:33–38,53`); restoring these flags would invalidate this check.
- **Known class count:** `configs/train_config.yaml:13,32` chooses CT2 clusters (extra0); `train_segmentation.py:83–84,91` uses n_classes for cluster/linear architecture. MRI must use5. This is class-count prior information, not pixel-label gradient supervision.
- **Metrics/Hungarian/plots:** `utils.py:239–300` accumulates a ground-truth confusion matrix and computes Hungarian matching/mIoU/accuracy; `train_segmentation.py:319–405` compares both probes to labels and logs cross-tabs/figures, with label shape used for output resize. `eval_segmentation.py:116–180` uses labels for metric updates, Hungarian colors, plots and saved comparisons. These evaluation uses are supervised measurement, not training gradients.
- **Checkpoint/model selection:** `train_segmentation.py:658–665` monitors `test/cluster/mIoU`; this is the primary result, not linear mIoU. `save_top_k=-1` retains all scheduled checkpoints rather than automatically keeping a best-only subset. Any user choice by validation mIoU is label-informed model selection and should be disclosed. No held-out test result is established here.
- **Demo semantic names:** `demo_segmentation.py:119–167` derives cluster-to-class mapping from validation labels; a legacy train fallback exists at140–147 if validation were empty. V2 loaders reject empty/missing data. This mapping labels predictions; it does not train features. Missing mapping fails explicitly at201.
- **Cache:** `precompute_knns.py:17–23` consumes only images/backbone features; returned labels are unused in similarity. Runtime normalization depends only on the image, not organ masks.

Primary-cluster evaluation PASS(c): checkpoint monitor, cluster confusion mapping and saved cluster predictions use cluster probe outputs. Linear metrics are retained as a labeled diagnostic. Qualification to confirm if literal scalar purity is required: **Should the combined/logged loss exclude linear CE while the diagnostic probe is trained separately, or is verified isolation of segmentation/cluster gradients the intended condition?** No patch applied to this read-only task.

### Task 7 — PASS (bounded CPU smoke)

`audit/scratch/fixes2_smoke.py`: config loads (dim70, CT2, extra0, text off). Real paired CT/1 `i0049,0000b.dcm` through approved raw preprocessing → input(1,3,224,224), frozen normalized BiomedCLIP features(1,768,14,14), trainable code(1,70,14,14). Diagnostic CE backward on the real model leaves every backbone/head gradientNone. Configured correspondence loss on a two-view synthetic batch from those real features is−0.3521715701; backward reaches the head and not the frozen backbone. Synthetic Hungarian metric mIoU/accuracy100% with expected swapped lookup. No optimizer updates, training run, derivative/cache generation or checkpoint write.

Final lightweight regression: split/preprocessing/loader/cache/centering checks all PASS; changed source and scratch scripts parse; `git diff 5639388 --check` clean. The loader's missing-v2 guard still fires because generation is deliberately left to the user. Full Lightning training/evaluation lifecycle and a trained-head checkpoint roundtrip remain UNVERIFIED; component tests do not certify those.

### New observations — not patched outside scope

- Guarded full-entrypoint import checks: train, cache and demo import successfully after redirecting Matplotlib/W&B temporary state into `audit/scratch/fixes2/`. The first attempt correctly blocked a Matplotlib cache write outside scratch. A scratch-only Matplotlib font-cache JSON remains untracked; it is not a dataset/cache artifact used by the model.
- **Historical observation, RESOLVED by F1 below:** `src/eval_segmentation.py:8` imports `crf`, which imports absent `pydensecrf`. Setting run_crf=False does not bypass this unconditional import. No package was installed, and this newly discovered dependency issue was not patched. Exact question: **May the CRF import be made conditional so run_crf=False works without installation, or which existing Python environment with pydensecrf should evaluation use?**
- Baseline CT train1869 with batch16 has remainder13. A future combined-modality train2657 with batch16 has remainder1, which triggers the existing within-batch-negative batch>1 assertion (`modules.py:555`). This turn does not change that pre-existing last-batch policy. For combined-modality experiments, choose a batch size without a singleton tail or separately authorize drop_last handling.
- Default eval/demo configs still contain historical example checkpoint paths. They must be overridden with a newly trained v2 checkpoint; no historical result is valid for this baseline. The exact commands below require that explicit path.

## Exact next commands — NOT RUN

Use branch audit-fixes-2 containing all four follow-up commits FIRST on the execution machine. No commits have been pushed or transferred by this task. The commands below are relative to that checkout and override the old machine-specific data path. Use its existing Python environment with the project dependencies, raw data and backbone files already available. No installation is proposed. Generation refuses an existing preprocessed_v2 directory; caches refuse an existing target unless explicitly archived. Default training/evaluation is CT; derivatives include all three approved modalities.

```sh
# 0. These follow-up commits first, before any generation.
git switch audit-fixes-2
git log --oneline -4
cd "$(git rev-parse --show-toplevel)/src"
AUDIT_PY=python3
DATA_ROOT="$PWD/pytorch_data_dir"
export WANDB_MODE=offline

# 1. Derivatives (first actual generation; native geometry).
"$AUDIT_PY" -B preprocess.py --write

# 2. New identity-keyed CT train/validation caches.
"$AUDIT_PY" -B precompute_knns.py "pytorch_data_dir=$DATA_ROOT"

# 3. GPU smoke: require CUDA, then one batch/head update (three optimizer steps).
"$AUDIT_PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required"; print(torch.cuda.get_device_name(0))' &&
"$AUDIT_PY" -B train_segmentation.py "pytorch_data_dir=$DATA_ROOT" max_steps=1 batch_size=2 experiment_name=chaos_v2_gpu_smoke

# 4. Full corrected CT baseline, only after smoke succeeds.
"$AUDIT_PY" -B train_segmentation.py "pytorch_data_dir=$DATA_ROOT" experiment_name=chaos_v2_ct

# 5. Enter an absolute NEW corrected checkpoint path when read waits.
read -r CORRECTED_CHECKPOINT
"$AUDIT_PY" -B eval_segmentation.py "pytorch_data_dir=$DATA_ROOT" res=224 run_crf=False "model_paths=[\"$CORRECTED_CHECKPOINT\"]" experiment_name=chaos_v2_ct
```

Run these stages in order and stop on any failure. `archive_existing_cache=true` is the explicit opt-in to move an existing cache target aside before replacement; it was not executed. Use the same derivative bytes, weights and feature source files across machines. Later unrelated Git commits do not invalidate caches. Old schema2 caches and historical checkpoints are not the corrected baseline. Primary evaluation is the cluster probe without CRF; linear metrics remain a supervised diagnostic.

## Verification limits and unresolved questions

- No approval questions remain for these four follow-ups. F1 resolves the optional-import blocker; F2 resolves the supervised headline-loss qualification.
- No derivative/cache generation, GPU smoke or training/evaluation command above was executed. Existing datasets, model weights and checkpoints are unchanged. Full GPU/Lightning training, evaluation lifecycle and trained-checkpoint roundtrip remain UNVERIFIED until the user runs these stages. Historical reported numbers remain void.

## Approved follow-ups (supersede the earlier blockers/command policy)

### F1 — APPLIED: optional CRF

`src/eval_segmentation.py:_load_crf,my_app,_apply_crf`; `src/configs/eval_config.yaml:run_crf`: no import of crf/pydensecrf for primary non-CRF evaluation, and run_crf defaultsFalse. Explicit run_crf=True loads the optional implementation before model loading/output creation and raises an actionable RuntimeError when pydensecrf is absent; workers import lazily as well. Before/after `audit/scratch/followup_crf_check.py`: previously importing the evaluator failed with ModuleNotFoundError; now full evaluator import and disabled-CRF dispatch pass without importing crf, while enabled-CRF dispatch raises the clear error. No installation, model inference or result-file generation. The earlier evaluation-import blocker is resolved.

### F2 — APPLIED: label-free headline loss

`src/train_segmentation.py:269–289`: diagnostic CE is logged only as `loss/linear_probe` and receives its own backward call on detached codes. It is excluded from `loss/total` and the returned training loss; optimizer ownership is unchanged. Approved baseline loss/total contains only image/code correspondence and cluster-probe losses. The true-label correspondence, salience and text paths remain disabled; no other runtime label-dependent term reaches this scalar in the baseline. Label-informed preprocessing choices, known class count, diagnostic training and evaluation/model selection remain as disclosed in task6; enabling experimental label/text flags would invalidate this baseline claim.

Reproduced before with the unchanged tiny training-step harness: labels0→1 changed headline .55918628→.40883285. After `fixes2_unsupervised_check.py --expect-fixed`, both are exactly−.2297102809; all-ignored labels also leave that headline and head/cluster gradients unchanged, with diagnostic CE0. Non-ignored labels change only diagnostic CE/linear gradients. Head/cluster gradient delta0; frozen-backbone gradsNone; isolated linear-only backward gives no head gradient. Optimizer spies performed zero updates. Earlier task6 scalar qualification is resolved.

### F3 — APPLIED: portable cache identity

`src/knn_cache.py:git_commit,expected_identity,cache_metadata,read_cache`: schema3 moves Git commit to top-level descriptive `provenance.git_commit`, outside the identity/hash and reader equality check. A Git-less source copy records null rather than blocking generation. Code identity hashes only `modules.py` (feature extraction), `chaos_preprocessing.py`, `data.py` (feature-input loading/order), and `precompute_knns.py` (pooling/search). Keys are relative filenames; values are content SHA256. Training/evaluation scripts and audit/README documentation are not hashed. Whole-file hashing is conservative: a change even to a comment inside one of those four files changes code identity.

**Machine-specific fields in the identity: NONE.** Absolute checkout/data/weight paths are used only to locate files, never serialized. Cache directory/root is outside the identity. Device/backend, GPU model, machine/host name, OS, timestamps and library versions are absent. `backbone.dtype=float32`, eval/dropout state and feature dimensions are algorithm requirements, not device identity. Complete identity fields are schema_version, split, ordered_sample_ids, ordered_sample_ids_sha256, ordered_image_sha256, split_manifest_sha256, feature_code_sha256, preprocessing (approved modality/intensity/resize/crop/mean/std/application count), backbone (weight/config hashes, layer/norm/CLS/eval/dropout/dtype/pooling), k and self_exclusion_rule. Array shape/dtype/digest remain validated artifact metadata.

Before `followup_cache_portability.py`: a Git-only change alters filename/identity. After `--expect-fixed`: identical content at simulated /Users/... and /content/drive/... roots with different commits produces identical identity and filename; changing provenance alone does not reject the NPZ; changing any hashed feature file, backbone weights or image bytes is rejected. Git-less provenance passes. Existing synthetic cache validation suite also PASS with code-change rejection replacing commit-change rejection. NPZ fixtures stayed in memory; no real cache was generated, migrated, moved or edited. Portability requires the same derivative bytes/ordered IDs, weights and four source files; it does not assert bit-identical CPU/GPU or cross-library numerical execution. No machine-specific runtime versions are used to invalidate the cache.

This replaces the earlier task4 policy that tied identity to HEAD. Old schema2 identities are not silently upgraded. The new schema/hash contract is the one to use for future generation.

### F4 — VERIFIED: native derivative counts/geometry; behavior unchanged

`src/preprocess.py:286–290` dispatches only to the v2 planner/writer. It does NOT call the retained old resampling helpers. The old helper default is isotropic1mm plus256×256 (`src/preprocess.py:22–23,158–164`); the active v2 writer emits one native plane per eligible raw DICOM, with no extra axial planes and no in-plane resize (`src/chaos_preprocessing.py:132–157,171–184`). Thus **1mm resampling is dropped** in v2. Native slice spacing is retained. The old CT_2606/629 counts cannot be expected from this native-plane contract. Equality with legacy CT_p1869/472 counts does not mean reuse: v2 reads raw DICOM and paired PNG masks, uses HU[-150,250], emits normalized uint16 images, and retains the frozen CT_ patient assignment. It does not read legacy derivative images.

Bounded read-only `audit/scratch/followup_geometry_check.py` validates all3333 header-level pairings and decodes one image per modality in memory. Results:

Split/modality | Train | Validation
--- | --- | ---
CT | 1869 | 472
T1 in-phase | 395 | 105
T2 | 393 | 99

Written image/mask dimensions are the native paired dimensions: all2341 CT512×512; T1:325 at256×256,175 at288×288; T2:318 at256×256,145 at320×320,29 at288×288. No pairing errors. Adjacent physical-position intervals (mm, rounded4decimals) are CT{1,1.5,1.6,1.6001,2}; T1{5.5,6,8,9}; T2{7.7,8,8.7,8.8,9}. These are observed header-position differences, not a new spacing target. Original in-plane pixel spacing and position/orientation are preserved in each manifest sample's metadata; no physical regridding is performed.

The manifest's preprocessing contract already explicitly records `spatial_resampling="none"`, `orientation="native; no flip or transpose"` and normalized uint16 storage (`src/chaos_preprocessing.py:37–42,188–191`). Exact native dimensions/geometry are stored per sample (`src/chaos_preprocessing.py:117–119`), rather than a single global output-size field because MRI sizes differ. The PNG arrays are saved directly (`:179–184`). Separately, the runtime transform resizes the short edge to224 and center-crops224×224, images bilinear/antialiased and masks nearest-neighbour, followed by exactly one BiomedCLIP normalization (`:194–220`); that transform contract also enters cache identity (`src/knn_cache.py:38–39`). No production preprocessing/manifest behavior was changed in F4, and no derivative files were written. The old file families' full historical provenance remains UNVERIFIED; these conclusions establish the current source behavior and raw-plane counts, not the command that originally created every historical PNG.

F4 also updates the command order above: follow-up commits first → derivatives → caches → CUDA GPU smoke → training → non-CRF primary cluster evaluation. Geometry verification is read-only, so no before/after behavior change is claimed.

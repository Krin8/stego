# Decisions requiring confirmation

Investigation: 2026-09-30, `/Users/navneetbavineni/STEGO`. **Recommendations only; no decisions or production changes applied.** The evidence ledger below retains commands, sample paths, logs, weight inventories and numerical results. All evidence paths without a leading slash are relative to this project. Scope is m6, M5, M4 and m1, using CHAOS DICOM/PNG.

## Recommendations and exact confirmations

| Question | Recommendation | Confidence | Exact confirmation needed |
| --- | --- | --- | --- |
| m6: reported results | Treat 46.8485 as a historical **DINO** log value, not a BiomedCLIP result. Preserve its matching cloud artifact as a recovery candidate. Treat historical scores as unvalidated for the corrected baseline until the exact head, source and split are recoverable and evaluated. | High for run attribution; medium for cloud candidate; low for complete reproduction | “Are 46.8485 cluster mIoU and/or 30.7159 the numbers you reported? Do you agree to withdraw them as validated BiomedCLIP-baseline results, while preserving the logs and recovery candidates? Which locally available checkpoint and source/split record, if any, should be used for recovery?” |
| M5: CT window | Use HU window **[−150, 250]**, linearly mapped to [0,1], consistently across future training/cache/evaluation preprocessing. | Medium; sampled clipping/contrast only | “Do you approve [−150,250] HU as the new CT contract, accepting that existing PNG intensities cannot establish or implement this new raw-HU contract?” |
| M5: MRI normalization | Use each slice's whole-image **1st–99.9th percentiles**, clip and scale to [0,1], computed independently per sequence/phase; explicitly handle constant slices. | Medium; 20 images/sequence, 18 usable out-phase pairs | “Do you approve per-slice 1–99.9 percentile MRI normalization, and should out-phase images with no unique physical match to a labeled in-phase image be excluded from supervised evaluation and recorded as unmatched?” |
| M5: family/source | Raw DICOM plus its paired PNG mask is authoritative. Prefer the **CT_<patient>_<slice>** naming family for a future explicit derivative manifest; do not mix it with legacy **CT_p...**. Neither existing derivative family has complete recovered lineage. | High against mixing; medium for preferred naming family | “Do you approve raw DICOM/PNG as authority and a single CT_ derivative family, with legacy CT_p files retained but excluded from the new baseline? Can you supply the legacy conversion recipe/manifest if it must remain eligible?” |
| M5: orientation/labels | Keep native image row/column orientation and apply the same geometric transform to the mask; no blanket flip/transpose. Pair before sorting by physical position. Associate out-phase masks through the physically matching in-phase image, never by list rank. Keep MRI codes 63/126/189/252 distinct. | Medium for sampled anatomical alignment; high that out-phase sorted pairing is unsafe | “Do you approve native image/mask orientation, explicit physical pairing, nearest-neighbor mask transforms, and the code mapping 0 background, 63 liver, 126 right kidney, 189 left kidney, 252 spleen? Are there any manual corrections or sequence-specific exceptions?” |
| M4: cache manifest | Adopt the manifest schema/validation contract below. Reject absent or mismatched identity. Do not add dummy metadata to old caches. | High for required identity; actual source-to-slice lineage remains UNVERIFIED | “Do you approve the proposed manifest and strict rejection rules, with ordered IDs derived from the approved single-family source manifest and split, and no reuse of unproven existing caches?” |
| m1: correspondence centering | Retain the pinned reference's restored-global-mean variant for the reference-implementation baseline; label that choice explicitly. Literal Eq(3) should be a separate chosen contract, not an incidental fix. | Medium; scientific objective determines the choice | “Should the primary baseline retain pinned-reference centering (recommended), or implement literal Paper A Eq(3) row-mean centering? If literal is chosen, do you accept that the correspondence shifts must be reconsidered?” |

The numerical and source evidence for each recommendation follows. No model-quality gain is established by these small checks.

## m6 — Recoverability and attribution

Evidence commands: `python3 -B audit/scratch/decision_runs.py`, `decision_inventory.py`, `decision_cloud_weights.py`, and `decision_followup.py` (each script is under `audit/scratch/`). The inventory covers project/Downloads/Documents and targeted STEGO/checkpoint/model folders on both mounted Google Drive accounts. A broader cloud traversal stalled and was interrupted: **exhaustive coverage of all cloud folders is UNVERIFIED**. No online-only file was opened or fetched. The ledger lists path, bytes and modification date for every discovered weight file.

| Saved run | Saved configuration | Last recorded results and progress | Recoverable artifact assessment |
| --- | --- | --- | --- |
| `logs/wandb/run-20260426_000459-5e6nhit6/files` | DINO ViT-small, patch8, dim40, extra_clusters5, configured classes5, CT, batch16, res320, max_steps7500; preprocessing flag absent | Cluster mIoU46.8485298, accuracy93.6970596; linear mIoU52.8696175, accuracy90.8768997; epoch21, logged `trainer/global_step=2499` | Strong filename/run-time/epoch match to the cloud epoch21 candidate below. Contents, optimizer-step interpretation and historical split UNVERIFIED. |
| `logs/wandb/run-20260909_044210-y3l0vxcn/files` | BiomedCLIP, declared dim70, text prompts enabled, extra_clusters10, classes2, CT, preprocessed data, shallow_layer_n2, batch16, res224, max_steps5000 | Cluster mIoU30.7159061, accuracy92.1427460; linear mIoU87.4687500, accuracy98.0036926; epoch5, logged `trainer/global_step=1659` | Saved log/config recovered; no matching September checkpoint identified in searched locations. Declared dim70 is not proof of actual head shape with text mode enabled. Trained head and exact dirty source/split UNVERIFIED. |

Sources: each run's `config.yaml` (`cfg.value`), `wandb-summary.json:1` and `wandb-metadata.json`. All 162 saved run records and selected output-log lines are retained in the ledger. Metrics are final values in saved summaries, not recomputed metrics or necessarily the best checkpoint score. April's configured five classes and September's two classes also prevent treating the two scores as the same experimental contract.

Best 46.8485 recovery candidate:

`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=21-step=step=7500-miou=test/cluster/mIoU=46.8485.ckpt`

It is 88,987,475 bytes; mtime **2026-04-26 00:12:06.421733 UTC**. Its run folder is three seconds before the W&B run start in local time, and epoch21, configured max_steps7500 and rounded score agree. This is **a candidate, not a state-dictionary-verified match**. The alias on the other mounted Drive (`...GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/mIoU=46.8485.ckpt`, 88,979,833 bytes) has mtime2026-04-25 15:38:17.282 UTC, before that run started; do not equate the alias with the candidate without a digest/key comparison. Evidence: `decision_cloud_weights.py`, `decision_followup.py` and the run metadata.

| Weight candidate/class | Size / modified UTC | Key-only inspection outcome |
| --- | --- | --- |
| 253 cloud `.ckpt` paths in ledger | Individual sizes listed; dates range 2026-04-05 to 2026-05-27 | All online-only. **UNVERIFIED** whether each is STEGO head/backbone/other. Names are not evidence of state contents. |
| `models/model_epoch_0720_iter_085000.pth` | 502,621,184 bytes / 2026-04-02 13:58:16.173 | Torch metadata loading fails: `failed finding central directory`. A later static scan of the first15,586-byte pickle entry finds `latents.weight`, `img_siren.net.*`, `per_element_factor/bias`, and none of the expected STEGO/BiomedCLIP key strings. Its available metadata indicates an unrelated SIREN-style model, not a trained STEGO head or the BiomedCLIP backbone; complete archive integrity remains UNVERIFIED. No pickle execution or tensor-storage read. |
| `src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin` | 783,705,670 bytes / 2026-08-18 15:21:00.952298 | Pretrained backbone: 352 keys, visual transformer/text model, 768-wide patch embedding; no cluster_probe/linear_probe/segmentation-head keys. It cannot supply a trained STEGO head. |
| `/Users/navneetbavineni/Documents/vscode/Python/Cosmetics/SkinCareModel.safetensors` | 29,764 bytes / 2025-06-26 22:55:45.558575 | Four MLP keys with 109→64→5 shapes; neither STEGO head nor BiomedCLIP backbone. |

Metadata inspection used Torch FakeTensorMode/mmap/weights_only, the safetensors header, or bounded static pickle-opcode scanning (`decision_pickle_keys.py`): no tensor payload was materialized. `.bin` was inspected explicitly because the active pretrained backbone uses that extension, in addition to the requested filename suffix search (`decision_cloud_weights.py`).

**Command/source/split limits.** Both focal runs record executable `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3` and program `/Users/navneetbavineni/STEGO/src/train_segmentation.py`, but no argument list. The exact shell command is **UNVERIFIED**; program plus resolved config cannot reconstruct original overrides, working directory or dirty files. April records commit `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d`, unavailable to local `git cat-file -t`. September records locally available commit `c859758193f15f37f2a4d5139555f7f4997ee22a`, but its W&B `files/code` snapshot is absent. Read-only `git show <September-commit>:src/train_segmentation.py` shows train split at588, val at609, and `trainer.fit(..., val_loader)` at676. That commit's `src/data.py:377` sorts processed filenames. This supports **intended train/val directory evaluation**, not proof that historical data files or dirty source matched the commit. A source snapshot plus content-hashed ordered train/val manifests and the actual trained checkpoint would settle it. Current split membership must not be retroactively substituted.

**Recommendation:** withdraw any attribution of46.8485 to BiomedCLIP (high confidence); retain the log as historical DINO evidence and the cloud candidate for possible recovery. Neither focal score is currently a certified result for the corrected baseline. Treat scores lacking a recoverable matching artifact/source/split as **void for reproducible performance claims**, while retaining records. It would be premature to claim no recoverable DINO artifact exists, because the strong cloud candidate has not been opened.

## M5 — Preprocessing evidence

Sampling seed **20260930**:20 raw images each for CT, T1 in-phase, T1 out-phase and T2. All selected paths, headers and masks are in the ledger (`decision_samples.py`). The initial selection follows sorted-list pairing, so out-phase sampling is conditional on the prefix for which sorted masks existed; it is not an unbiased sample of all out-phase images. Physical verification accepts18/20 out-phase pairs and rejects two. Corrected out-phase organ statistics below supersede the initial wrong-pair diagnostic (`decision_geometry.py`). Results are descriptive samples, not population estimates or segmentation benchmarks.

### Filename families

| Split | Current CT_ | Legacy CT_p | Legacy T1DUAL | Legacy T2SPIR |
| --- | ---: | ---: | ---: | ---: |
| train | 2,606 | 1,869 | 395 | 393 |
| val | 629 | 472 | 105 | 99 |

Evidence: `decision_families.py` enumerates `src/pytorch_data_dir/preprocessed/{train,val}/images`. Canonical patient IDs have zero current train/val overlap within each modality. This does not establish historical splits or shared patient identity between CT and MR.

Twenty proposed current/legacy CT pairs are256×256 versus512×512. After common256×256 bilinear image resizing and nearest-neighbor mask resizing: **0/20 exact pixel matches**, median absolute difference **14.8983/255**, correlations **0.9050–0.9666**, mask IoU **0.7564–0.9702**. Pairing uses patient ID plus a hypothesized nearest physical slice for the current1mm index and legacy filename rank; no legacy source manifest was recovered. **Classification: unknown exact slice equivalence, related patient volumes with different transformations; not established formatting-only duplicates.** These comparisons do not prove each hypothesized pair is the same slice.

`src/preprocess.py:56–58,79–81,99–103,128–142` implements a [-400,400] HU,1mm resampling,256-pixel pipeline and initially associates masks by filename rank before GDCM reordering. Its presence does not prove it produced every CT_ PNG. Raw DICOM/PNG should be authority; the preferred CT_ family needs explicit source lineage. Do not attempt to recover raw HU or a new window from already clipped8-bit PNGs. The requested new window would be a future raw-data derivative contract, not a claim about current files.

### CT window choice

Twenty CT slices contain436,505 liver pixels. HU percentiles in order **0.5,1,5,50,95,99,99.5**:

| Region | HU percentiles |
| --- | --- |
| Whole slice | −1024, −1024, −1003, −304,154,302,425 |
| Liver | 10,36,71,126,187,238,253 |

| Window HU | Liver clipped | Mean liver [0,1] | Mean non-liver [0,1] | Mean intensity difference |
| --- | ---: | ---: | ---: | ---: |
| [−160,240] | 0.869864% | 0.708184 | 0.175693 | 0.532491 |
| **[−150,250]** | **0.564026%** | **0.683433** | **0.165157** | **0.518276** |
| [−400,400] | 0% | 0.654511 | 0.224960 | 0.429551 |

Evidence: `decision_samples.py`, CT aggregate. Clipping is pooled liver pixels; mean contrasts are averaged over nonempty-mask slices. Non-liver includes air, so this is not a liver-vs-adjacent-tissue contrast measure. [−150,250] preserves more liver pixels than [−160,240] with similar sampled contrast and more contrast than the wider window. Confidence is medium; organ-boundary accuracy or training benefit is UNVERIFIED and would require a separate authorized validation experiment.

### MRI intensity and labels

MRI histograms are positive stored intensities, not CT HU. Whole-image percentiles for T1-in are **0,0,0,3,511,776,867**; T2 **0,0,0,5,455,838,942**. T1-in liver median307 and T2 kidney medians708/705 already exceed the250 upper bound of the recommended CT window. Applying [-150,250] clips **62.13%** of sampled T1-in liver and **96.34% /96.29%** of sampled T2 kidney-code126/189 pixels. Evidence: `decision_samples.py`, per-sequence aggregates. Detailed per-organ percentiles for all four sequences are retained; corrected physical-pair results replace out-phase diagnostic values.

Sampled mask pixel counts (includes background, paired to the indicated sequence):

| Sequence | Usable slices | 0 | 63 | 126 | 189 | 252 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T1-in | 20 | 1,362,122 | 59,452 | 9,062 | 10,146 | 9,202 |
| T1-out | 18 | 1,188,695 | 54,567 | 6,225 | 6,802 | 10,399 |
| T2 | 20 | 1,296,170 | 73,093 | 9,542 | 9,643 | 13,408 |

Source: `decision_followup.py`; per-slice paired paths and counts appear in the ledger. All sampled MRI values are exactly0/63/126/189/252; sampled CT values are0/255. MRI clipping percentages, using each slice's whole-image thresholds (lower1st percentile in both columns):

| Sequence / label | Upper99th: organ pixels clipped | Upper99.9th: organ pixels clipped |
| --- | ---: | ---: |
| T1-in / 63 | 0.0101% | 0.0050% |
| T1-in / 126 | 0.1104% | 0.0000% |
| T1-in / 189 | 0.0099% | 0.0000% |
| T1-in / 252 | 0.1630% | 0.0000% |
| T1-out / 63 | 0.0183% | 0.0037% |
| T1-out / 126 | 0.0964% | 0.0000% |
| T1-out / 189 | 0.0147% | 0.0000% |
| T1-out / 252 | 0.0288% | 0.0000% |
| T2 / 63 | 0.7183% | 0.0424% |
| T2 / 126 | 21.0543% | 0.5869% |
| T2 / 189 | 17.4323% | 0.8192% |
| T2 / 252 | 12.9922% | 0.1641% |

Source: `decision_geometry.py`; same image sample, corrected out-phase associations. The99.9 choice reduces high-signal organ clipping, especially T2, without assuming a CT intensity scale. Recommendation: per-slice1–99.9 percentile clip/scale, no label-dependent thresholds, explicit constant-slice handling, then repeat grayscale to RGB and apply model mean/std once. This is a provisional preprocessing choice, not proof of BiomedCLIP performance. Whole-volume normalization and all-dataset intensity behavior remain UNVERIFIED.

### Orientation and pairing

Six original overlays plus one corrected out-phase overlay are saved under `audit/scratch/overlays/`; their exact image/mask source paths are in the ledger. Visual inspection supports native orientation in the three CT examples, T1-in and T2, and the corrected T1-out example. **No universal flip or transpose is supported.** Foreground tissue overlap alone cannot prove correct pairing: the wrong out-phase example still has96.65% overlap with nonzero image tissue.

| Overlay | Mean image intensity inside / outside mask | Finding |
| --- | --- | --- |
| CT1, CT2, CT3 | HU129.14/−483.28;136.06/−553.08;140.53/−529.81 | Liver overlays agree visually; sampled identity tissue overlap100%. |
| T1-in (MR8) | 108.74/71.86 | Native overlay fits. |
| T1-out (MR38), **corrected physical pair** | 245.17/93.37 | Correct mask `IMG-0023-00026.png` for image `IMG-0023-00025.dcm`; sorted pairing had incorrectly selected mask00044. |
| T2 (MR37) | 552.01/95.31 | Native overlay fits. |

Evidence: `decision_samples.py`, `decision_overlay_check.py`, and images `decision_CT_1.png`, `decision_CT_2.png`, `decision_CT_3.png`, `decision_T1-in_1.png`, `decision_T2_1.png`, `decision_T1-out_physical.png`. MRI intensities are arbitrary stored units. The original `decision_T1-out_1.png` intentionally preserves the wrong-pair diagnostic.

For18 physically matched out-phase samples, **7 masks differ from sorted-order associations**; two additional sampled images have no matching in-phase position. Accepted pairs agree in position (<0.01mm), orientation (<1e-5), matrix and pixel spacing. MR38 has24 in-phase images/24 masks but34 out-phase images: a rank-based correspondence cannot be assumed. `src/data.py:192–208` currently selects InPhase, so this out-phase failure is a warning for extending sequence support, not evidence that the current loader actually loads OutPhase. Evidence: `decision_samples.py` series inventory and `decision_geometry.py`, `decision_overlay_check.py`.

Current raw collection pairs sorted filenames (`src/data.py:171–180,199–208`); processed collection pairs equal PNG basenames (`src/data.py:377–391`). The inspected CT series run physically descending in filename order, MRI ascending (`decision_followup.py`). Monotonic order does not prove cross-sequence slice identity. Recommendation: first establish image→mask associations by explicit ID and sequence geometry, then sort paired records by projected physical slice position. For out-phase, use a unique in-phase physical match and its PNG; unmatched or ambiguous positions should fail matching, not get a guessed rank or flip.

Label126 is physically right of189 in the sampled T1-in and corrected out-phase images (patient-X centroid −83.12 vs82.92mm; −47.46 vs64.57mm). The overlays support liver63 and spleen252 locations. This corroborates the proposed organ convention on these examples; complete-dataset anatomical labeling remains UNVERIFIED. No mask edits or orientation corrections were applied.

## M4 — Proposed cache manifest contract

Observed implementation: filenames encode only model_type, dataset, split, crop label and224 (`src/precompute_knns.py:75–77`; `src/data.py:554–558`). The writer skips existing paths and saves only `nns` (`precompute_knns.py:79,105–108`). The loader rejects missing metadata but does **not** validate metadata content (`data.py:560–563`). Therefore an arbitrary metadata field would not establish identity.

Read-only inspection (`decision_cache_centering.py`) found:

| Cache under `src/pytorch_data_dir/nns/` | Shape / dtype | Range | Self in first column | Provenance |
| --- | --- | --- | --- | --- |
| `nns_vit_base_chaos_train_five_224.npz` | (4475,30),int64 | 0…4474 | 100% | Only `nns`; metadata absent |
| `nns_vit_base_chaos_val_five_224.npz` | (1101,30),int64 | 0…1100 | 100% | Only `nns`; metadata absent |

Those row counts equal the **mixed** current+legacy CT inventories. Equality does not prove cache ordering or its backbone. The mixed-family guard currently rejects that dataset (`data.py:393–395`). Observed lexical-name JSON hashes are train `31b0f284f689e02e0e7737ce66f510c92efe60f9cd4b4fd0758db50a014a99ed`, val `97fb27b98ec76a89c89fea6cf155491075b4cbd8ea322e3f459e15242ec60c6b`; these describe today's names, not historical cache identities.

The future manifest should contain the following **explicit fields**, serialized as canonical JSON with a schema version and SHA256. These are proposed requirements, not an implemented or verified manifest:

| Field group | Required fields / values | Why and evidence |
| --- | --- | --- |
| Schema and provenance | `schema_version`, `manifest_sha256`, generation command/resolved config, code-file hashes and git commit plus dirty-diff digest, library versions (torch,timm,open_clip,torchvision,Pillow,pydicom,SimpleITK,numpy), UTC creation time | Code and preprocessing can change while path stays fixed. Time/path is descriptive, never the identity by itself. |
| Ordered sample identity | `split`, `sample_count`, `ordered_samples[]`, `ordered_samples_sha256`; each entry has stable `sample_id`, canonical patient ID plus modality namespace, sequence/phase, relative input path, input content SHA256, derivative family/version, slice/crop ID | Current order is sorted filenames per split, filtered by modality (`data.py:369–391`); precompute uses `shuffle=False` (`precompute_knns.py:91`). Future dataset row i and manifest row i must be exactly the same sequence, not merely the same set/count. |
| Raw-to-derivative lineage | Source DICOM content digest, SeriesInstanceUID/SOPInstanceUID, position/orientation/spacing, source and output shapes; any resampling origin/direction/spacing, interpolation and source-slice contributors; paired mask path/digest, original label encoding and output mapping | Existing CT families differ. A resampled slice may derive from more than one raw slice; record contributors/weights or enough deterministic geometry to reconstruct them. Mask association must precede reordering (`preprocess.py:99–111,118–142`). |
| Split contract | Explicit patient→split assignment, split-generation version/seed if used, split hash, separate query/reference ordered lists if different | Current raw fallback uses lexicographic last20% patients (`data.py:129–135`); processed data trusts directories. Cache rows must not silently cross train/val pools. Do not assume same numeric CT/MR ID means same person without a crosswalk. |
| Intensity preprocessing | Modality/phase dispatch; CT slope/intercept handling and approved HU limits; MRI percentile bounds, interpolation method for quantiles, per-slice scope, background inclusion, constant-slice policy; scale/range/dtype, clipping/rounding, RGB replication | Needed to distinguish the recorded [−400,400] conversion recipe from a future [−150,250] contract; the exact lineage of existing PNGs remains unverified. Record actual operations, not a generic “CHAOS preprocessing” label. |
| Spatial/model preprocessing | Input and resize sizes, interpolation/antialias settings, aspect-ratio policy, crop kind/coordinates/ratio, actual crop lineage, transform order, mask-nearest policy; mean `[0.48145466,0.4578275,0.40821073]`, std `[0.26862954,0.26130258,0.27577711]`, application count1 | Cache generation currently uses short-edge resize plus center crop224 (`precompute_knns.py:43,86`; `utils.py:191–209`; local `src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json`, `preprocess_cfg`). Filename `five` does not prove five crops were applied. Record actual geometry separately from config labels. |
| Backbone identity | Hub/model ID, architecture ViT-B/16, patch16, feature width768, local weight file SHA256, model/preprocess-config digest, loader and library versions | Current local weight SHA256 is `52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be` (`decision_cache_centering.py`). `model_type=vit_base` alone is not a weight identity. |
| Feature extraction | Final block12/index11, normalized patch tokens (`norm=True`), CLS excluded, no image projection/pooling before map extraction, map axis BCHW, expected14×14 at224, positional-embedding/dynamic-size policy | Actual current extraction: `modules.py:161–165,173,204–215`. Record the resolved final block, not the currently unused `shallow_layer_n` setting. |
| Inference state | Whole featurizer/model eval state, requires_gradFalse for backbone, no_grad, dropout inactive, dtype/autocast policy, compute device/backend, deterministic flags and seed | `precompute_knns.py:34,67,93`; `modules.py:167–171,230`. Eval must include outer dropout, not only the transformer. Record both configured dropout probability and effective inactive state. |
| Cache vector and search | Spatial mean over H,W of the feature map, then channel L2 normalization; cosine/dot-product similarity, vector width768, stored topk30, chosen positive-neighbor count, explicit self slot/exclusion, tie-breaking rule and index order | `precompute_knns.py:19,100–101`; loader samples ranks1…k (`data.py:586–590`). Assert self at column0 and no self among eligible positives, or explicitly change/review the sampling contract. Do not rely on topk tie order. |
| Stored artifact | `nns_shape`, `nns_dtype`, array-byte digest, row count, per-row uniqueness, index bounds, query/reference list hashes; optional similarities with their dtype/digest | Existing bounds alone cannot detect stale ordering. Validate integer indices, shape[N,k], 0≤index<N_reference, enough non-self neighbors and correct self placement before exposing the cache. |

Keep label pairing/split provenance in the manifest. For cache reuse, distinguish fields that affect the actual input samples/features/search from descriptive run settings: segmentation-head checkpoint, loss shifts/weights and m1 centering do not enter this backbone-only nearest-neighbor computation (`precompute_knns.py:59–62,19`). Mask contents affect cache identity if they alter slice selection/cropping; otherwise they remain dataset-provenance fields rather than feature dependencies.

**Validation recommendation:** compute expected identity from the selected source manifest and actual transforms/model before loading; reject missing metadata, schema mismatch, sample-order/hash mismatch, preprocessing/weight/extraction mismatch and invalid indices. Encode JSON as a plain UTF-8 string/byte array compatible with `allow_pickle=False`. Use an identity digest in the cache name and reject a conflicting existing target instead of silently skipping it. Publish only a completely generated/validated artifact in future authorized work. The present writer/reader do not implement this contract; no cache was generated or edited. A complete source-to-slice manifest is **UNVERIFIED** until the M5 choices and lineage are supplied.

## m1 — Literal equation versus pinned implementation

The ground-truth ledger already distinguishes literal row-centering `Fsc = F − mean_target(F)` (`audit/AUDIT.md:263,290`) from upstream commit `eb4d6b521740bd4265681b353547f2ffca65d673`, which restores the global mean. Current `src/modules.py:478–481` retains that restoration. No paper was reread.

| Synthetic term | Current/reference loss | Literal Eq(3) loss |
| --- | ---: | ---: |
| self | −0.207174495 | −0.075054839 |
| neighbor | −0.032486625 | −0.006922718 |
| negative | 0.153866276 | 0.128302366 |

Evidence: `decision_cache_centering.py`, seed11,2×3×2×2 features, pointwise centering and zero clamp enabled, stabilization disabled. If `mu=mean(F)` and `Splus=max(S,0)`, then `L_reference = L_literal − mu*mean(Splus)`; maximum numerical residual here is1.49e−8. With constant affinity0.7 and shift0.18, literal centering gives +0.18*Splus while restored-mean gives −0.52*Splus. Thus keeping the same shifts after switching equations does not preserve the objective.

**Recommendation (medium): retain reference centering for the pinned-reference baseline**, explicitly describe that departure from literal Eq(3), and keep its existing shifts provisional for the backbone/domain change. This minimizes differences from the executable reference; it does not make the formula literally paper-faithful. If your priority is exact Eq(3), choose that instead and accept a distinct objective requiring shift reassessment. No evidence here establishes which choice yields better CHAOS segmentation. No change has been applied; your choice remains required.

## Investigation limits and next authorized step

The remaining evidence gaps are: actual contents of online-only candidate checkpoints; complete integrity of the unrelated local `.pth` archive; historical command arguments/dirty source/data split; derivative source lineage; complete-dataset anatomical alignment; and model-performance validation of the proposed preprocessing/centering choices. Confirming a recommendation alone does not fill these evidence gaps. After your choices, a separate implementation request can define a single data manifest and metadata-validating cache writer/reader. This investigation has not run training, regenerated caches, installed packages or changed existing project files.

## Detailed evidence ledger

<details>
<summary>Expand inventories, per-run records, sample paths, exact numerical outputs and overlays</summary>

# Evidence for remaining audit decisions

Investigation date: 2026-09-30. Scope: m6, M5, M4, m1; CHAOS DICOM/PNG only. Recommendations are provisional and require the confirmations listed at the end. No training, cache regeneration, package installation, dataset/checkpoint edits or production-code changes are performed.

Evidence scripts live under `audit/scratch/decision_*.py`; each CPU check has a110-second limit. A broader initial filename search was interrupted when traversal stalled; the bounded inventory below records its own coverage and failures. Cloud files marked `stat.SF_DATALESS` are listed but not opened, to avoid fetching file contents or altering their local availability. For those files, state-dictionary classification is UNVERIFIED until a locally available copy can be inspected.

Existing audit/fix reports are historical evidence, not proof of current data provenance. Source references are project-relative unless an absolute path is given. Numeric sampling results identify their actual source paths and fixed seed.

## m6 — Saved-run evidence
Command: `python3 -B audit/scratch/decision_runs.py`. Metrics below are saved records, not recomputed scores. Config dim is the declared value; text mode can override it, so checkpoint shape is required to prove the effective dimension.
Run / source | Script + recorded arguments | Git commit | Config: architecture; extra clusters; dim; text; max steps; data | Last logged step/epoch; metrics
--- | --- | --- | --- | ---
`logs/wandb/run-20260426_000028-e173ffm5/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 5, 'dim': 40, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 7500, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'CT', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260426_000459-5e6nhit6/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 5, 'dim': 40, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 7500, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'CT', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'test/linear/Accuracy': 90.87689971923828, 'test/cluster/Accuracy': 93.69705963134766, 'test/cluster/mIoU': 46.84852981567383, 'epoch': 21, 'test/linear/mIoU': 52.8696174621582, 'trainer/global_step': 2499}`
`logs/wandb/run-20260426_131506-nsdx88r7/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260426_190916-wlylyubd/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 5, 'dim': 40, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 4000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260426_191043-vyqdmn72/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260429_180630-chi7gjaf/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260429_185409-mhhpep50/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 5, 'dim': 40, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 4000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260429_185953-6y5vxhz3/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260429_190707-uhj6sf1i/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260429_194316-qesvetrm/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260429_221938-7bdhwffl/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260513_124500-veybx750/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260513_140325-cwqmbo0k/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260513_143331-wl0489e9/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260513_145833-jyengzag/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260513_151746-wky3tmdd/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260513_162847-q3dhnpox/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'test/linear/mIoU': 19.344585418701172, 'test/cluster/foreground_mIoU': 2.5682120323181152, 'test/cluster/Accuracy': 78.3818130493164, 'test/linear/foreground_mIoU': 4.393045425415039, 'epoch': 9, 'trainer/global_step': 1659, 'test/cluster/mIoU': 17.68351936340332, 'test/linear/Accuracy': 78.36811828613281}`
`logs/wandb/run-20260513_215815-h4x8du3t/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'epoch': 4, 'test/linear/Accuracy': 95.3387451171875, 'test/cluster/Accuracy': 96.34615325927734, 'test/cluster/foreground_mIoU': 0, 'test/cluster/mIoU': 19.26923179626465, 'test/linear/foreground_mIoU': 4.103417873382568, 'trainer/global_step': 879, 'test/linear/mIoU': 22.35017967224121}`
`logs/wandb/run-20260513_230248-dulde3tl/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260514_002744-sxegqqea/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260514_003053-uzqx3xos/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260514_003254-ebzy5asa/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260514_003438-rr95zx5v/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260514_003537-3hb37vyg/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260514_003900-npyiqwzx/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260514_004053-4y5h48wc/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260514_004331-4e2pwwo2/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260514_004456-kirp4ece/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260514_004652-bd0e17oq/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'test/cluster/Accuracy': 88.06433868408203, 'test/linear/Accuracy': 98.10073852539062, 'test/linear/foreground_mIoU': 3.327712059020996, 'test/linear/mIoU': 22.290590286254883, 'trainer/global_step': 1659, 'test/cluster/mIoU': 18.26106834411621, 'test/cluster/foreground_mIoU': 0.7660632729530334, 'epoch': 35}`
`logs/wandb/run-20260515_140032-ydei1uz3/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': ['max_steps=5', 'val_freq=2']}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260515_140221-hi8w1916/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': ['max_steps=5', 'val_freq=2']}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'MR', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260515_140338-6hjzvjyh/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': ['max_steps=5', 'val_freq=2']}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260515_140710-mvvg1cy8/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260515_143901-gmzfz6c2/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260525_112050-h6l9i84h/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260525_151019-d6y6b9jl/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260525_153056-dmnz1j3b/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260525_161141-0mgooh66/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'test/cluster/Accuracy': 73.63668823242188, 'test/linear/mIoU': 21.663724899291992, 'test/cluster/mIoU': 14.978655815124512, 'test/linear/Accuracy': 95.0328140258789, 'epoch': 0, 'test/linear/foreground_mIoU': 3.3215489387512207, 'test/cluster/foreground_mIoU': 0.32360005378723145, 'trainer/global_step': 175}`
`logs/wandb/run-20260525_162059-o8z7oyn5/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260525_170457-l6hig6jf/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260525_190310-ii3tcem8/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260525_200805-v0s49bgm/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260525_215807-a1pf9skj/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260525_220232-udv9pz68/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260525_220412-bgvcjqqd/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 2, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260525_220544-v6tpkvb2/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260525_224130-pkahol1n/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260525_224423-4xa2o3op/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 1659, 'test/linear/mIoU': 22.488788604736328, 'test/linear/Accuracy': 83.97742462158203, 'test/cluster/mIoU': 18.687152862548828, 'epoch': 9, 'test/cluster/Accuracy': 93.43576049804688}`
`logs/wandb/run-20260525_224445-sg9j4a81/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260525_224749-fv9f3fuv/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'epoch': 9, 'test/cluster/mIoU': 18.687152862548828, 'test/linear/mIoU': 23.91607666015625, 'trainer/global_step': 1659, 'test/linear/Accuracy': 89.64020538330078, 'test/cluster/Accuracy': 93.43576049804688}`
`logs/wandb/run-20260526_132101-wlc9oj7x/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260526_143913-nmxoexnb/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260526_145925-bwbfm44z/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'epoch': 7, 'test/linear/Accuracy': 95.87371063232422, 'trainer/global_step': 1659, 'test/cluster/Accuracy': 34.708457946777344, 'test/linear/mIoU': 28.99094581604004, 'test/cluster/mIoU': 10.88235855102539}`
`logs/wandb/run-20260526_163714-uovczt8s/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260526_165051-u3yvfgk7/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260526_165422-rsl611hc/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260526_173034-4boxw7qb/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260526_182846-e3b1kn1a/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260526_183223-rl42sq05/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260526_215929-tvefzrqx/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'trainer/global_step': 1319, 'test/cluster/mIoU': 10.759109497070312, 'epoch': 5, 'test/linear/Accuracy': 96.1423110961914, 'test/linear/mIoU': 30.079326629638672, 'test/cluster/Accuracy': 34.594200134277344}`
`logs/wandb/run-20260527_064151-ypsb20vx/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'test/cluster/Accuracy': 34.74179458618164, 'trainer/global_step': 1659, 'test/linear/mIoU': 28.875520706176758, 'test/linear/Accuracy': 95.84915161132812, 'test/cluster/mIoU': 10.851913452148438, 'epoch': 7}`
`logs/wandb/run-20260527_081333-sgyqypti/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': 'UNVERIFIED'}` | `{'test/cluster/Accuracy': 47.15835189819336, 'test/cluster/mIoU': 14.223752975463867, 'trainer/global_step': 1659, 'test/linear/mIoU': 27.418193817138672, 'epoch': 7, 'test/linear/Accuracy': 95.34632110595703}`
`logs/wandb/run-20260527_085650-m1acipkf/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'epoch': 1, 'test/cluster/mIoU': 8.322930335998535, 'test/linear/mIoU': 19.071866989135742, 'trainer/global_step': 409, 'test/cluster/Accuracy': 41.61465072631836, 'test/linear/Accuracy': 95.35932922363281}`
`logs/wandb/run-20260527_091425-w6gg6pyl/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/linear/Accuracy': 95.35932922363281, 'test/cluster/Accuracy': 41.61465072631836, 'test/linear/mIoU': 19.071866989135742, 'epoch': 1, 'test/cluster/mIoU': 8.322930335998535, 'trainer/global_step': 409}`
`logs/wandb/run-20260527_092609-07aaeay9/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/cluster/mIoU': 13.177812576293945, 'test/linear/mIoU': 29.763307571411133, 'trainer/global_step': 1663, 'epoch': 7, 'test/linear/Accuracy': 96.52677154541016, 'test/cluster/Accuracy': 44.56776428222656}`
`logs/wandb/run-20260527_101718-i2azfj9y/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `a61797f23e0dcc9222f7dedf05637ad68aeb5a1d` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'epoch': 7, 'test/linear/Accuracy': 96.52961730957031, 'test/cluster/mIoU': 13.286956787109375, 'trainer/global_step': 1663, 'test/cluster/Accuracy': 44.95936584472656, 'test/linear/mIoU': 29.86551284790039}`
`logs/wandb/run-20260527_110607-7pbaan39/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/cluster/mIoU': 17.657249450683594, 'epoch': 7, 'test/linear/mIoU': 29.12730598449707, 'test/cluster/Accuracy': 50.14491271972656, 'test/linear/Accuracy': 96.44046783447266, 'trainer/global_step': 1663}`
`logs/wandb/run-20260527_123926-b2nm9qdn/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{}`
`logs/wandb/run-20260527_124119-u2kj1pkk/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'trainer/global_step': 1663, 'test/cluster/mIoU': 18.005414962768555, 'test/linear/mIoU': 18.76439094543457, 'epoch': 7, 'test/linear/Accuracy': 93.82048034667969, 'test/cluster/Accuracy': 90.0270767211914}`
`logs/wandb/run-20260527_200740-y892q5lw/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260527_221335-ucaes56v/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260527_233129-0ejbk7n0/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'dino', 'extra_clusters': 5, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': False}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260527_233317-bs8f1ara/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'dino', 'extra_clusters': 5, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': False}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260527_233505-3nc3lfej/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'dino', 'extra_clusters': 5, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': False}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260527_233817-1se9blcm/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260527_235644-3nposycj/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'dino', 'extra_clusters': 5, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': False}` | `{'test/cluster/Accuracy': 93.1186294555664, 'test/linear/Accuracy': 93.39197540283203, 'trainer/global_step': 1659, 'epoch': 7, 'test/cluster/mIoU': 18.62372398376465, 'test/linear/mIoU': 18.678396224975586}`
`logs/wandb/run-20260529_201821-jzo57l7k/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260529_204920-6k5qaplc/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{}`
`logs/wandb/run-20260529_205257-mgvn3ond/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'dino', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{}`
`logs/wandb/run-20260529_205551-co8xob97/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260530_161941-t2yh3n1q/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'biomedclip', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{}`
`logs/wandb/run-20260530_162149-gob60epe/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'biomedclip', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{}`
`logs/wandb/run-20260530_163243-24r3oek0/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260530_170115-9831fww8/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260530_172410-41p3c26x/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260530_222136-abijx33a/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'biomedclip', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{}`
`logs/wandb/run-20260530_223018-x2chyaps/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260530_224955-0owte4z5/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260530_225914-64yxk4om/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260530_231941-uqnmjhw4/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'biomedclip', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260530_232140-m0tn2l5t/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'biomedclip', 'extra_clusters': 0, 'dim': 70, 'use_text_prompts': 'UNVERIFIED', 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260530_232403-inomefky/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260818_205130-jstjxc8i/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'biomedclip', 'extra_clusters': 1, 'dim': 70, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'CT', 'use_preprocessed_data': True}` | `{'test/linear/Accuracy': 97.4002685546875, 'test/linear/mIoU': 83.05570220947266, 'test/cluster/mIoU': 30.8116397857666, 'test/cluster/Accuracy': 92.41861724853516, 'trainer/global_step': 1659, 'epoch': 11}`
`logs/wandb/run-20260818_235059-xnqkrmqm/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'biomedclip', 'extra_clusters': 20, 'dim': 70, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'CT', 'use_preprocessed_data': True}` | `{'test/cluster/mIoU': 15.134758949279785, 'trainer/global_step': 1659, 'test/linear/mIoU': 85.34717559814453, 'test/linear/Accuracy': 97.91583251953125, 'epoch': 11, 'test/cluster/Accuracy': 37.42141342163086}`
`logs/wandb/run-20260819_103004-6n3r0uc9/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'biomedclip', 'extra_clusters': 20, 'dim': 70, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'CT', 'use_preprocessed_data': True}` | `{'epoch': 11, 'test/cluster/Accuracy': 37.42141342163086, 'trainer/global_step': 1659, 'test/linear/mIoU': 85.34717559814453, 'test/linear/Accuracy': 97.91583251953125, 'test/cluster/mIoU': 15.134758949279785}`
`logs/wandb/run-20260819_170051-spkj1v8g/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `ff94353cb6c072efd711873c1b8947e6d68e3533` | `{'arch': 'biomedclip', 'extra_clusters': 1, 'dim': 70, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/cluster/mIoU': 8.074823379516602, 'trainer/global_step': 1659, 'test/linear/Accuracy': 95.69263458251953, 'epoch': 7, 'test/linear/mIoU': 43.64595413208008, 'test/cluster/Accuracy': 46.31901550292969}`
`logs/wandb/run-20260819_200842-qthlmgbm/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260819_232912-eutfogcf/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'biomedclip', 'extra_clusters': 5, 'dim': 70, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/cluster/Accuracy': 37.02806854248047, 'test/cluster/mIoU': 7.511361598968506, 'trainer/global_step': 1659, 'test/linear/mIoU': 43.40403747558594, 'epoch': 7, 'test/linear/Accuracy': 96.03894805908203}`
`logs/wandb/run-20260820_190926-737tny13/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260820_213607-zzbyml7w/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260821_020228-yxjqete3/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'biomedclip', 'extra_clusters': 4, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/linear/Accuracy': 95.94556427001953, 'test/linear/mIoU': 44.835758209228516, 'test/cluster/mIoU': 7.411355495452881, 'trainer/global_step': 1659, 'epoch': 7, 'test/cluster/Accuracy': 27.3843994140625}`
`logs/wandb/run-20260821_083435-fwg0piup/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/cluster/mIoU': 11.931325912475586, 'epoch': 7, 'test/cluster/Accuracy': 43.538307189941406, 'test/linear/Accuracy': 96.04226684570312, 'trainer/global_step': 1499, 'test/linear/mIoU': 44.72077941894531}`
`logs/wandb/run-20260821_152910-z88zqq3t/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260821_202106-zkkj5bo7/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260821_233058-ct8o1ley/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260822_014333-2cell07t/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/cluster/Accuracy': 19.439380645751953, 'test/cluster/mIoU': 6.180226802825928, 'trainer/global_step': 1659, 'test/linear/mIoU': 48.14246368408203, 'epoch': 7, 'test/linear/Accuracy': 96.54019165039062}`
`logs/wandb/run-20260822_103526-n5j74e89/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260822_110726-78uio9rv/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260822_203112-9s93ss0m/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260822_210224-h1p0r9cj/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260822_231332-6y5cokod/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260822_234518-8b7maf8y/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/linear/Accuracy': 96.05752563476562, 'test/cluster/Accuracy': 42.52362823486328, 'test/cluster/mIoU': 11.684138298034668, 'test/linear/mIoU': 46.411014556884766, 'epoch': 7, 'trainer/global_step': 1659}`
`logs/wandb/run-20260823_034747-bqt6znpa/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'trainer/global_step': 199, 'epoch': 0, 'test/cluster/Accuracy': 41.60906982421875, 'test/cluster/mIoU': 9.457324028015137, 'test/linear/Accuracy': 95.44648742675781, 'test/linear/mIoU': 29.645263671875}`
`logs/wandb/run-20260823_120147-qw1j0jpd/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `7794588c8c68f27c4c15e9ca835a339aeb72038e` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260823_142413-whzb9cqe/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c61559c5a79c452a891a775bc12967a545428411` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/cluster/Accuracy': 52.96223068237305, 'test/linear/mIoU': 44.02423095703125, 'test/cluster/mIoU': 9.201027870178223, 'test/linear/Accuracy': 96.80110931396484, 'epoch': 7, 'trainer/global_step': 1659}`
`logs/wandb/run-20260824_011127-5cipd35z/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260824_025227-duuzymio/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/linear/mIoU': 44.02423095703125, 'test/cluster/Accuracy': 94.16049194335938, 'trainer/global_step': 1659, 'epoch': 7, 'test/cluster/mIoU': 15.693469047546387, 'test/linear/Accuracy': 96.80110931396484}`
`logs/wandb/run-20260824_103128-whsxmp9n/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260824_132701-4letlnov/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260824_163916-o7g8i6th/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/cluster/Accuracy': 76.4452896118164, 'trainer/global_step': 1659, 'epoch': 7, 'test/cluster/mIoU': 14.059011459350586, 'test/linear/Accuracy': 96.80110931396484, 'test/linear/mIoU': 44.02423095703125}`
`logs/wandb/run-20260824_211309-icmoohdi/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260824_211612-ineqwge2/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'trainer/global_step': 0}`
`logs/wandb/run-20260824_211836-0q1urba0/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260825_012931-tp6f50bk/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260825_021540-7buagh9s/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/linear/Accuracy': 95.51036071777344, 'test/cluster/Accuracy': 18.771190643310547, 'test/linear/mIoU': 37.93598556518555, 'test/cluster/mIoU': 6.366879463195801, 'epoch': 7, 'trainer/global_step': 1659}`
`logs/wandb/run-20260825_092445-fetktyu5/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260825_111358-kywepbkz/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'trainer/global_step': 1659, 'test/cluster/mIoU': 14.048995018005371, 'test/linear/mIoU': 44.02423095703125, 'test/linear/Accuracy': 96.80110931396484, 'epoch': 7, 'test/cluster/Accuracy': 76.484619140625}`
`logs/wandb/run-20260825_155541-20vptvsw/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{}`
`logs/wandb/run-20260825_161954-k4sfe9j1/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260825_171825-vrtyi6wz/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260825_185352-7sax9op6/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260825_191512-oe1a0brw/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260825_191533-q3qu1bvt/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `5e68b97135bfeb1509115c59afdb1801f7f1d591` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260825_201534-qoidav7b/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `227657a09e30c65ab7baf65030919d3976e8d4e3` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260825_212249-q5g4qy1e/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `227657a09e30c65ab7baf65030919d3976e8d4e3` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260825_215956-3u1a20n1/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `227657a09e30c65ab7baf65030919d3976e8d4e3` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'trainer/global_step': 1659, 'test/cluster/Accuracy': 18.996562957763672, 'test/linear/Accuracy': 95.1456069946289, 'test/linear/mIoU': 36.67020034790039, 'test/cluster/mIoU': 5.28690242767334, 'epoch': 7}`
`logs/wandb/run-20260826_131327-d96kuivn/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `227657a09e30c65ab7baf65030919d3976e8d4e3` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260826_170833-6xkx7xxw/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `227657a09e30c65ab7baf65030919d3976e8d4e3` | `{'arch': 'biomedclip', 'extra_clusters': 0, 'dim': 128, 'use_text_prompts': True, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/cluster/mIoU': 15.880374908447266, 'test/linear/Accuracy': 96.09416961669922, 'trainer/global_step': 399, 'test/linear/mIoU': 40.744136810302734, 'test/cluster/Accuracy': 64.65734100341797, 'epoch': 1}`
`logs/wandb/run-20260826_193406-78lvu81i/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `227657a09e30c65ab7baf65030919d3976e8d4e3` | `{'arch': 'biomedclip', 'extra_clusters': 0, 'dim': 128, 'use_text_prompts': True, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/cluster/Accuracy': 33.60818099975586, 'test/linear/mIoU': 48.778778076171875, 'test/cluster/mIoU': 9.377954483032227, 'epoch': 7, 'test/linear/Accuracy': 97.53829193115234, 'trainer/global_step': 1659}`
`logs/wandb/run-20260828_200520-950ml5zu/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `227657a09e30c65ab7baf65030919d3976e8d4e3` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260828_234121-ggwfmoqc/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `9e8d4db76714eb64e3f0b8b1ecf5e324c102aa1c` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260907_174657-wls99dok/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `9e8d4db76714eb64e3f0b8b1ecf5e324c102aa1c` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260907_191331-ckb78tko/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `9e8d4db76714eb64e3f0b8b1ecf5e324c102aa1c` | `{'arch': 'biomedclip', 'extra_clusters': 0, 'dim': 128, 'use_text_prompts': True, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/linear/mIoU': 30.296737670898438, 'epoch': 1, 'test/linear/Accuracy': 96.08960723876953, 'trainer/global_step': 399, 'test/cluster/mIoU': 11.641268730163574, 'test/cluster/Accuracy': 43.47063446044922}`
`logs/wandb/run-20260907_203241-6094aa1i/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `9911411f40493978f1af9ed4f31c046f7c1d3ff4` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260907_211127-pwaekmsa/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `9911411f40493978f1af9ed4f31c046f7c1d3ff4` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260907_224406-ln55dwfl/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `9911411f40493978f1af9ed4f31c046f7c1d3ff4` | `{'arch': 'biomedclip', 'extra_clusters': 2, 'dim': 128, 'use_text_prompts': False, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'all', 'use_preprocessed_data': True}` | `{'test/cluster/Accuracy': 19.243450164794922, 'trainer/global_step': 1659, 'epoch': 7, 'test/linear/mIoU': 41.105690002441406, 'test/linear/Accuracy': 96.9137954711914, 'test/cluster/mIoU': 6.769985675811768}`
`logs/wandb/run-20260908_014053-wrjaph1n/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `9911411f40493978f1af9ed4f31c046f7c1d3ff4` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260908_021035-vehng4je/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c809b332ca3da4ec561d7bb24136eaf5fb84f2a6` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260908_023403-o879wblt/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c809b332ca3da4ec561d7bb24136eaf5fb84f2a6` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260908_025959-15xor701/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c809b332ca3da4ec561d7bb24136eaf5fb84f2a6` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260908_032246-sjn6n26g/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c809b332ca3da4ec561d7bb24136eaf5fb84f2a6` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260908_033113-8ishmyza/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c809b332ca3da4ec561d7bb24136eaf5fb84f2a6` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260908_033705-gnlyaxt5/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c809b332ca3da4ec561d7bb24136eaf5fb84f2a6` | `{'arch': 'biomedclip', 'extra_clusters': 0, 'dim': 128, 'use_text_prompts': True, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/src/pytorch_data_dir', 'chaos_modality': 'CT', 'use_preprocessed_data': False}` | `{'test/linear/Accuracy': 97.78999328613281, 'test/linear/mIoU': 86.97398376464844, 'epoch': 1, 'test/cluster/mIoU': 33.48676300048828, 'trainer/global_step': 179, 'test/cluster/Accuracy': 66.76146697998047}`
`logs/wandb/run-20260908_035506-4o1zbesc/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c809b332ca3da4ec561d7bb24136eaf5fb84f2a6` | `{'arch': 'biomedclip', 'extra_clusters': 0, 'dim': 128, 'use_text_prompts': True, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'CT', 'use_preprocessed_data': False}` | `{'test/linear/Accuracy': 97.60437774658203, 'test/linear/mIoU': 86.5045166015625, 'test/cluster/Accuracy': 69.04815673828125, 'test/cluster/mIoU': 34.524078369140625, 'epoch': 14, 'trainer/global_step': 1659}`
`logs/wandb/run-20260908_092253-n8kw50b1/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c859758193f15f37f2a4d5139555f7f4997ee22a` | `{'arch': 'biomedclip', 'extra_clusters': 0, 'dim': 128, 'use_text_prompts': True, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'CT', 'use_preprocessed_data': True}` | `{'test/cluster/mIoU': 28.146955490112305, 'epoch': 14, 'trainer/global_step': 1659, 'test/linear/mIoU': 81.58818817138672, 'test/cluster/Accuracy': 56.29390335083008, 'test/linear/Accuracy': 96.33332824707031}`
`logs/wandb/run-20260908_130818-iz7z60w4/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c859758193f15f37f2a4d5139555f7f4997ee22a` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260909_023853-gnn4l5f9/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c859758193f15f37f2a4d5139555f7f4997ee22a` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260909_033448-wicrfwro/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c859758193f15f37f2a4d5139555f7f4997ee22a` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260909_035333-6pmajvfa/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c859758193f15f37f2a4d5139555f7f4997ee22a` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260909_035925-75cq6wm7/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c859758193f15f37f2a4d5139555f7f4997ee22a` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`
`logs/wandb/run-20260909_044210-y3l0vxcn/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c859758193f15f37f2a4d5139555f7f4997ee22a` | `{'arch': 'biomedclip', 'extra_clusters': 10, 'dim': 70, 'use_text_prompts': True, 'max_steps': 5000, 'pytorch_data_dir': '/Users/navneetbavineni/STEGO/src/pytorch_data_dir', 'chaos_modality': 'CT', 'use_preprocessed_data': True}` | `{'test/cluster/Accuracy': 92.14274597167969, 'epoch': 5, 'test/linear/Accuracy': 98.00369262695312, 'test/linear/mIoU': 87.46875, 'trainer/global_step': 1659, 'test/cluster/mIoU': 30.715906143188477}`
`logs/wandb/run-20260909_173533-67xgrz2i/files` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{'executable': '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3', 'script': '/Users/navneetbavineni/STEGO/src/train_segmentation.py', 'args': 'UNVERIFIED: absent'}` | `c859758193f15f37f2a4d5139555f7f4997ee22a` | `{'arch': 'UNVERIFIED', 'extra_clusters': 'UNVERIFIED', 'dim': 'UNVERIFIED', 'use_text_prompts': 'UNVERIFIED', 'max_steps': 'UNVERIFIED', 'pytorch_data_dir': 'UNVERIFIED', 'chaos_modality': 'UNVERIFIED', 'use_preprocessed_data': 'UNVERIFIED'}` | `{}`

Saved run records found: 162.

### Selected output-log provenance lines

`logs/wandb/run-20260426_000028-e173ffm5/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260426_000028-e173ffm5/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260426_000028-e173ffm5/files/output.log:90` RecursionError: maximum recursion depth exceeded

`logs/wandb/run-20260426_000028-e173ffm5/files/output.log:92` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260426_000459-5e6nhit6/files/output.log:1142` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260426_000459-5e6nhit6/files/output.log:1143` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260426_000459-5e6nhit6/files/output.log:1197` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260426_000459-5e6nhit6/files/output.log:1198` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260426_000459-5e6nhit6/files/output.log:1213` `Trainer.fit` stopped: `max_steps=7500` reached.

`logs/wandb/run-20260426_131506-nsdx88r7/files/output.log:42` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260426_131506-nsdx88r7/files/output.log:85` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260426_131506-nsdx88r7/files/output.log:86` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260426_190916-wlylyubd/files/output.log:38` Error executing job with overrides: []

`logs/wandb/run-20260426_190916-wlylyubd/files/output.log:39` Traceback (most recent call last):

`logs/wandb/run-20260426_190916-wlylyubd/files/output.log:67` raise ValueError(

`logs/wandb/run-20260426_190916-wlylyubd/files/output.log:70` ValueError:  `val_check_interval` (100) must be less than or equal to the number of the training batches (50). If you want to disable validation set `limit_val_batches` to 0.0 instead. If you want to validate based on the total training batches, set `check_val_every_n_epoch=None`.

`logs/wandb/run-20260426_190916-wlylyubd/files/output.log:72` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260426_191043-vyqdmn72/files/output.log:42` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260426_191043-vyqdmn72/files/output.log:110` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260426_191043-vyqdmn72/files/output.log:111` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260429_180630-chi7gjaf/files/output.log:37` The ``compute`` method of metric UnsupervisedMetrics was called before the ``update`` method which may lead to errors, as metric

`logs/wandb/run-20260429_180630-chi7gjaf/files/output.log:105` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:40m [3mv_num: gjaf[0mv_num: gjaf[0mgjaf[0m

`logs/wandb/run-20260429_180630-chi7gjaf/files/output.log:106` 1: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint

`logs/wandb/run-20260429_185409-mhhpep50/files/output.log:65` which may lead to errors, as metric states have not yet been

`logs/wandb/run-20260429_185409-mhhpep50/files/output.log:72` Error executing job with overrides: []

`logs/wandb/run-20260429_185409-mhhpep50/files/output.log:73` Traceback (most recent call last):

`logs/wandb/run-20260429_185409-mhhpep50/files/output.log:121` NameError: name 'img_aug' is not defined

`logs/wandb/run-20260429_185409-mhhpep50/files/output.log:123` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260429_185953-6y5vxhz3/files/output.log:66` which may lead to errors, as metric states have not yet been

`logs/wandb/run-20260429_190707-uhj6sf1i/files/output.log:57` which may lead to errors, as metric states have not yet been

`logs/wandb/run-20260429_190707-uhj6sf1i/files/output.log:222` 4/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:

`logs/wandb/run-20260429_190707-uhj6sf1i/files/output.log:223` 401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260429_190707-uhj6sf1i/files/output.log:225` The checkpoint will contain the model state AFTER optimization.

`logs/wandb/run-20260429_194316-qesvetrm/files/output.log:57` which may lead to errors, as metric states have not yet been

`logs/wandb/run-20260429_194316-qesvetrm/files/output.log:704` 4/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:

`logs/wandb/run-20260429_194316-qesvetrm/files/output.log:705` 401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260429_194316-qesvetrm/files/output.log:707` The checkpoint will contain the model state AFTER optimization.

`logs/wandb/run-20260429_221938-7bdhwffl/files/output.log:10622` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260429_221938-7bdhwffl/files/output.log:11543` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260429_221938-7bdhwffl/files/output.log:11545` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260429_221938-7bdhwffl/files/output.log:12668` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260429_221938-7bdhwffl/files/output.log:12670` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260513_124500-veybx750/files/output.log:156` with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization.

`logs/wandb/run-20260513_124500-veybx750/files/output.log:213` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260513_124500-veybx750/files/output.log:214` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260513_124500-veybx750/files/output.log:271` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260513_124500-veybx750/files/output.log:272` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260513_140325-cwqmbo0k/files/output.log:49` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260513_140325-cwqmbo0k/files/output.log:99` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260513_140325-cwqmbo0k/files/output.log:100` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260513_143331-wl0489e9/files/output.log:49` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260513_143331-wl0489e9/files/output.log:99` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260513_143331-wl0489e9/files/output.log:100` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260513_145833-jyengzag/files/output.log:49` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260513_145833-jyengzag/files/output.log:99` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260513_145833-jyengzag/files/output.log:100` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260513_151746-wky3tmdd/files/output.log:158` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260513_151746-wky3tmdd/files/output.log:215` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260513_151746-wky3tmdd/files/output.log:216` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260513_151746-wky3tmdd/files/output.log:273` ightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260513_151746-wky3tmdd/files/output.log:275` checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260513_162847-q3dhnpox/files/output.log:505` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260513_162847-q3dhnpox/files/output.log:506` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260513_162847-q3dhnpox/files/output.log:563` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260513_162847-q3dhnpox/files/output.log:564` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260513_162847-q3dhnpox/files/output.log:579` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260513_215815-h4x8du3t/files/output.log:319` Traceback (most recent call last):

`logs/wandb/run-20260513_215815-h4x8du3t/files/output.log:378` FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.

`logs/wandb/run-20260513_215815-h4x8du3t/files/output.log:379` Original Traceback (most recent call last):

`logs/wandb/run-20260513_215815-h4x8du3t/files/output.log:393` FileNotFoundError: [Errno 2] No such file or directory: '/Users/navneetbavineni/STEGO/src/pytorch_data_dir/CHAOS_preprocessed_png/val/images/T1DUAL_p8_s3139.png'

`logs/wandb/run-20260513_215815-h4x8du3t/files/output.log:396` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260513_230248-dulde3tl/files/output.log:216` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260513_230248-dulde3tl/files/output.log:273` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260513_230248-dulde3tl/files/output.log:274` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260513_230248-dulde3tl/files/output.log:331` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260513_230248-dulde3tl/files/output.log:332` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260514_002744-sxegqqea/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260514_002744-sxegqqea/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260514_002744-sxegqqea/files/output.log:80` IndexError: index 5 is out of bounds for axis 0 with size 5

`logs/wandb/run-20260514_002744-sxegqqea/files/output.log:82` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260514_003053-uzqx3xos/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260514_003053-uzqx3xos/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260514_003053-uzqx3xos/files/output.log:102` ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (7,) and arg 1 with shape (6,).

`logs/wandb/run-20260514_003053-uzqx3xos/files/output.log:104` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260514_003254-ebzy5asa/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260514_003254-ebzy5asa/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260514_003254-ebzy5asa/files/output.log:102` ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (5,) and arg 1 with shape (6,).

`logs/wandb/run-20260514_003254-ebzy5asa/files/output.log:104` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260514_003438-rr95zx5v/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260514_003438-rr95zx5v/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260514_003438-rr95zx5v/files/output.log:96` ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (5,) and arg 1 with shape (6,).

`logs/wandb/run-20260514_003438-rr95zx5v/files/output.log:98` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260514_003537-3hb37vyg/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260514_003537-3hb37vyg/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260514_003537-3hb37vyg/files/output.log:102` ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (5,) and arg 1 with shape (6,).

`logs/wandb/run-20260514_003537-3hb37vyg/files/output.log:104` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260514_003900-npyiqwzx/files/output.log:49` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260514_004053-4y5h48wc/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260514_004053-4y5h48wc/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260514_004053-4y5h48wc/files/output.log:80` IndexError: index 5 is out of bounds for axis 0 with size 5

`logs/wandb/run-20260514_004053-4y5h48wc/files/output.log:82` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260514_004331-4e2pwwo2/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260514_004331-4e2pwwo2/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260514_004331-4e2pwwo2/files/output.log:102` ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (5,) and arg 1 with shape (6,).

`logs/wandb/run-20260514_004331-4e2pwwo2/files/output.log:104` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260514_004456-kirp4ece/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260514_004456-kirp4ece/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260514_004456-kirp4ece/files/output.log:102` ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (5,) and arg 1 with shape (6,).

`logs/wandb/run-20260514_004456-kirp4ece/files/output.log:104` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260514_004652-bd0e17oq/files/output.log:994` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260514_004652-bd0e17oq/files/output.log:995` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260514_004652-bd0e17oq/files/output.log:1079` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260514_004652-bd0e17oq/files/output.log:1080` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260514_004652-bd0e17oq/files/output.log:1118` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260515_140032-ydei1uz3/files/output.log:35` Error executing job with overrides: ['max_steps=5', 'val_freq=2']

`logs/wandb/run-20260515_140032-ydei1uz3/files/output.log:36` Traceback (most recent call last):

`logs/wandb/run-20260515_140032-ydei1uz3/files/output.log:96` AssertionError: Input height (320) doesn't match model (224).

`logs/wandb/run-20260515_140032-ydei1uz3/files/output.log:98` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260515_140221-hi8w1916/files/output.log:35` Error executing job with overrides: ['max_steps=5', 'val_freq=2']

`logs/wandb/run-20260515_140221-hi8w1916/files/output.log:36` Traceback (most recent call last):

`logs/wandb/run-20260515_140221-hi8w1916/files/output.log:88` NameError: name 'math' is not defined. Did you forget to import 'math'?

`logs/wandb/run-20260515_140221-hi8w1916/files/output.log:90` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260515_140338-6hjzvjyh/files/output.log:53` lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260515_140710-mvvg1cy8/files/output.log:52` UnsupervisedMetrics was called before the ``update`` method which may lead to errors, as

`logs/wandb/run-20260515_140710-mvvg1cy8/files/output.log:109` ightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260515_140710-mvvg1cy8/files/output.log:111` checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260515_140710-mvvg1cy8/files/output.log:172` ightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260515_140710-mvvg1cy8/files/output.log:174` checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260515_143901-gmzfz6c2/files/output.log:194` checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260515_143901-gmzfz6c2/files/output.log:247` ightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260515_143901-gmzfz6c2/files/output.log:249` checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260515_143901-gmzfz6c2/files/output.log:302` ightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260515_143901-gmzfz6c2/files/output.log:304` checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260525_112050-h6l9i84h/files/output.log:52` UnsupervisedMetrics was called before the ``update`` method which may lead to errors, as

`logs/wandb/run-20260525_112050-h6l9i84h/files/output.log:82` ightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260525_112050-h6l9i84h/files/output.log:84` checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260525_151019-d6y6b9jl/files/output.log:52` UnsupervisedMetrics was called before the ``update`` method which may lead to errors, as

`logs/wandb/run-20260525_151019-d6y6b9jl/files/output.log:109` ightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260525_151019-d6y6b9jl/files/output.log:111` checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260525_153056-dmnz1j3b/files/output.log:52` UnsupervisedMetrics was called before the ``update`` method which may lead to errors, as

`logs/wandb/run-20260525_153056-dmnz1j3b/files/output.log:105` ightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260525_153056-dmnz1j3b/files/output.log:107` checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260525_161141-0mgooh66/files/output.log:33` /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: The ``compute`` method of metric UnsupervisedMetrics was called before the ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260525_162059-o8z7oyn5/files/output.log:49` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260525_162059-o8z7oyn5/files/output.log:99` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260525_162059-o8z7oyn5/files/output.log:100` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260525_162059-o8z7oyn5/files/output.log:157` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260525_162059-o8z7oyn5/files/output.log:158` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260525_170457-l6hig6jf/files/output.log:100` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260525_170457-l6hig6jf/files/output.log:157` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260525_170457-l6hig6jf/files/output.log:158` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260525_170457-l6hig6jf/files/output.log:215` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260525_170457-l6hig6jf/files/output.log:216` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260525_190310-ii3tcem8/files/output.log:100` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260525_190310-ii3tcem8/files/output.log:158` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260525_190310-ii3tcem8/files/output.log:159` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260525_190310-ii3tcem8/files/output.log:217` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260525_190310-ii3tcem8/files/output.log:218` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260525_200805-v0s49bgm/files/output.log:281` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260525_200805-v0s49bgm/files/output.log:339` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260525_200805-v0s49bgm/files/output.log:340` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260525_200805-v0s49bgm/files/output.log:398` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260525_200805-v0s49bgm/files/output.log:399` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260525_215807-a1pf9skj/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260525_215807-a1pf9skj/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260525_215807-a1pf9skj/files/output.log:80` IndexError: index 5 is out of bounds for axis 0 with size 5

`logs/wandb/run-20260525_215807-a1pf9skj/files/output.log:82` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260525_220232-udv9pz68/files/output.log:50` was called before the ``update`` method which may lead to errors, as metric states have not

`logs/wandb/run-20260525_220232-udv9pz68/files/output.log:55` Error executing job with overrides: []

`logs/wandb/run-20260525_220232-udv9pz68/files/output.log:56` Traceback (most recent call last):

`logs/wandb/run-20260525_220232-udv9pz68/files/output.log:125` RuntimeError: MPS: Unsupported Border padding mode

`logs/wandb/run-20260525_220232-udv9pz68/files/output.log:127` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260525_220412-bgvcjqqd/files/output.log:50` was called before the ``update`` method which may lead to errors, as metric states have not

`logs/wandb/run-20260525_220412-bgvcjqqd/files/output.log:55` Error executing job with overrides: []

`logs/wandb/run-20260525_220412-bgvcjqqd/files/output.log:56` Traceback (most recent call last):

`logs/wandb/run-20260525_220412-bgvcjqqd/files/output.log:140` NotImplementedError: The operator 'aten::grid_sampler_2d_backward' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://github.com/pytorch/pytorch/issues/141287 and mention use-case, that resulted in missing op as well as commit hash 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0. As a temporary fix, you can set the environment vari

`logs/wandb/run-20260525_220412-bgvcjqqd/files/output.log:142` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260525_220544-v6tpkvb2/files/output.log:50` was called before the ``update`` method which may lead to errors, as metric states have not

`logs/wandb/run-20260525_220544-v6tpkvb2/files/output.log:103` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260525_220544-v6tpkvb2/files/output.log:104` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260525_224130-pkahol1n/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260525_224130-pkahol1n/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260525_224130-pkahol1n/files/output.log:90` RecursionError: maximum recursion depth exceeded

`logs/wandb/run-20260525_224130-pkahol1n/files/output.log:92` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260525_224423-4xa2o3op/files/output.log:115` /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the model state in training_step before optimizer

`logs/wandb/run-20260525_224423-4xa2o3op/files/output.log:130` /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the model state in training_step before optimizer

`logs/wandb/run-20260525_224423-4xa2o3op/files/output.log:145` /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the model state in training_step before optimizer

`logs/wandb/run-20260525_224423-4xa2o3op/files/output.log:160` /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the model state in training_step before optimizer

`logs/wandb/run-20260525_224423-4xa2o3op/files/output.log:173` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260525_224445-sg9j4a81/files/output.log:50` was called before the ``update`` method which may lead to errors, as metric states have not

`logs/wandb/run-20260525_224749-fv9f3fuv/files/output.log:516` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260525_224749-fv9f3fuv/files/output.log:517` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260525_224749-fv9f3fuv/files/output.log:575` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260525_224749-fv9f3fuv/files/output.log:576` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260525_224749-fv9f3fuv/files/output.log:592` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260526_132101-wlc9oj7x/files/output.log:50` was called before the ``update`` method which may lead to errors, as metric states have not

`logs/wandb/run-20260526_132101-wlc9oj7x/files/output.log:103` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260526_132101-wlc9oj7x/files/output.log:104` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260526_143913-nmxoexnb/files/output.log:50` was called before the ``update`` method which may lead to errors, as metric states have not

`logs/wandb/run-20260526_143913-nmxoexnb/files/output.log:96` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260526_143913-nmxoexnb/files/output.log:97` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260526_145925-bwbfm44z/files/output.log:389` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260526_145925-bwbfm44z/files/output.log:390` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260526_145925-bwbfm44z/files/output.log:447` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260526_145925-bwbfm44z/files/output.log:448` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260526_145925-bwbfm44z/files/output.log:463` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260526_163714-uovczt8s/files/output.log:49` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260526_163714-uovczt8s/files/output.log:53` Error executing job with overrides: []

`logs/wandb/run-20260526_163714-uovczt8s/files/output.log:54` Traceback (most recent call last):

`logs/wandb/run-20260526_163714-uovczt8s/files/output.log:123` RuntimeError: MPS: Unsupported Border padding mode

`logs/wandb/run-20260526_163714-uovczt8s/files/output.log:125` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260526_165051-u3yvfgk7/files/output.log:49` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260526_165051-u3yvfgk7/files/output.log:53` Error executing job with overrides: []

`logs/wandb/run-20260526_165051-u3yvfgk7/files/output.log:54` Traceback (most recent call last):

`logs/wandb/run-20260526_165051-u3yvfgk7/files/output.log:138` NotImplementedError: The operator 'aten::grid_sampler_2d_backward' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://github.com/pytorch/pytorch/issues/141287 and mention use-case, that resulted in missing op as well as commit hash 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0. As a temporary fix, you can set the environment vari

`logs/wandb/run-20260526_165051-u3yvfgk7/files/output.log:140` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260526_165422-rsl611hc/files/output.log:49` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260526_165422-rsl611hc/files/output.log:99` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260526_165422-rsl611hc/files/output.log:100` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260526_173034-4boxw7qb/files/output.log:49` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260526_173034-4boxw7qb/files/output.log:94` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260526_173034-4boxw7qb/files/output.log:95` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260526_173034-4boxw7qb/files/output.log:152` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260526_173034-4boxw7qb/files/output.log:153` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260526_182846-e3b1kn1a/files/output.log:49` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260526_182846-e3b1kn1a/files/output.log:53` Error executing job with overrides: []

`logs/wandb/run-20260526_182846-e3b1kn1a/files/output.log:54` Traceback (most recent call last):

`logs/wandb/run-20260526_182846-e3b1kn1a/files/output.log:102` KeyError: 'img_aug'

`logs/wandb/run-20260526_182846-e3b1kn1a/files/output.log:104` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260526_183223-rl42sq05/files/output.log:49` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260526_215929-tvefzrqx/files/output.log:344` Traceback (most recent call last):

`logs/wandb/run-20260526_215929-tvefzrqx/files/output.log:403` AttributeError: Caught AttributeError in DataLoader worker process 0.

`logs/wandb/run-20260526_215929-tvefzrqx/files/output.log:404` Original Traceback (most recent call last):

`logs/wandb/run-20260526_215929-tvefzrqx/files/output.log:413` AttributeError: 'ContrastiveSegDataset' object has no attribute 'pos_images'

`logs/wandb/run-20260526_215929-tvefzrqx/files/output.log:416` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260527_064151-ypsb20vx/files/output.log:384` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260527_064151-ypsb20vx/files/output.log:385` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260527_064151-ypsb20vx/files/output.log:442` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260527_064151-ypsb20vx/files/output.log:443` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260527_064151-ypsb20vx/files/output.log:458` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260527_081333-sgyqypti/files/output.log:384` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260527_081333-sgyqypti/files/output.log:385` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260527_081333-sgyqypti/files/output.log:442` odel_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260527_081333-sgyqypti/files/output.log:443` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260527_081333-sgyqypti/files/output.log:458` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260527_085650-m1acipkf/files/output.log:108` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260527_085650-m1acipkf/files/output.log:175` Error executing job with overrides: []

`logs/wandb/run-20260527_085650-m1acipkf/files/output.log:176` Traceback (most recent call last):

`logs/wandb/run-20260527_085650-m1acipkf/files/output.log:228` IndexError: index 63 is out of bounds for axis 0 with size 5

`logs/wandb/run-20260527_085650-m1acipkf/files/output.log:230` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260527_091425-w6gg6pyl/files/output.log:108` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260527_091425-w6gg6pyl/files/output.log:175` Error executing job with overrides: []

`logs/wandb/run-20260527_091425-w6gg6pyl/files/output.log:176` Traceback (most recent call last):

`logs/wandb/run-20260527_091425-w6gg6pyl/files/output.log:228` IndexError: index 63 is out of bounds for axis 0 with size 5

`logs/wandb/run-20260527_091425-w6gg6pyl/files/output.log:230` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260527_092609-07aaeay9/files/output.log:463` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260527_092609-07aaeay9/files/output.log:464` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_092609-07aaeay9/files/output.log:535` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260527_092609-07aaeay9/files/output.log:536` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_092609-07aaeay9/files/output.log:607` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260527_101718-i2azfj9y/files/output.log:469` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260527_101718-i2azfj9y/files/output.log:470` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_101718-i2azfj9y/files/output.log:541` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260527_101718-i2azfj9y/files/output.log:542` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_101718-i2azfj9y/files/output.log:613` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260527_110607-7pbaan39/files/output.log:469` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260527_110607-7pbaan39/files/output.log:470` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_110607-7pbaan39/files/output.log:541` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260527_110607-7pbaan39/files/output.log:542` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_110607-7pbaan39/files/output.log:613` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260527_123926-b2nm9qdn/files/output.log:35` Error executing job with overrides: []

`logs/wandb/run-20260527_123926-b2nm9qdn/files/output.log:36` Traceback (most recent call last):

`logs/wandb/run-20260527_123926-b2nm9qdn/files/output.log:104` AssertionError: Input height (320) doesn't match model (224).

`logs/wandb/run-20260527_123926-b2nm9qdn/files/output.log:106` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260527_124119-u2kj1pkk/files/output.log:469` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260527_124119-u2kj1pkk/files/output.log:470` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_124119-u2kj1pkk/files/output.log:541` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260527_124119-u2kj1pkk/files/output.log:542` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_124119-u2kj1pkk/files/output.log:613` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260527_200740-y892q5lw/files/output.log:156` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_200740-y892q5lw/files/output.log:214` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260527_200740-y892q5lw/files/output.log:215` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_200740-y892q5lw/files/output.log:273` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260527_200740-y892q5lw/files/output.log:274` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_221335-ucaes56v/files/output.log:156` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_221335-ucaes56v/files/output.log:214` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260527_221335-ucaes56v/files/output.log:215` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260527_221335-ucaes56v/files/output.log:319` _checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260527_221335-ucaes56v/files/output.log:320` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260527_233129-0ejbk7n0/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260527_233129-0ejbk7n0/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260527_233129-0ejbk7n0/files/output.log:80` IndexError: index 5 is out of bounds for axis 0 with size 5

`logs/wandb/run-20260527_233129-0ejbk7n0/files/output.log:82` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260527_233317-bs8f1ara/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260527_233317-bs8f1ara/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260527_233317-bs8f1ara/files/output.log:102` ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (10,) and arg 1 with shape (6,).

`logs/wandb/run-20260527_233317-bs8f1ara/files/output.log:104` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260527_233505-3nc3lfej/files/output.log:36` Error executing job with overrides: []

`logs/wandb/run-20260527_233505-3nc3lfej/files/output.log:37` Traceback (most recent call last):

`logs/wandb/run-20260527_233505-3nc3lfej/files/output.log:102` ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (10,) and arg 1 with shape (6,).

`logs/wandb/run-20260527_233505-3nc3lfej/files/output.log:104` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260527_233817-1se9blcm/files/output.log:95` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260527_233817-1se9blcm/files/output.log:152` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260527_233817-1se9blcm/files/output.log:153` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260527_233817-1se9blcm/files/output.log:210` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260527_233817-1se9blcm/files/output.log:211` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260527_235644-3nposycj/files/output.log:384` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260527_235644-3nposycj/files/output.log:385` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260527_235644-3nposycj/files/output.log:442` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260527_235644-3nposycj/files/output.log:443` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260527_235644-3nposycj/files/output.log:458` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260529_201821-jzo57l7k/files/output.log:56` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260529_201821-jzo57l7k/files/output.log:101` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260529_201821-jzo57l7k/files/output.log:102` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260529_204920-6k5qaplc/files/output.log:35` Error executing job with overrides: []

`logs/wandb/run-20260529_204920-6k5qaplc/files/output.log:36` Traceback (most recent call last):

`logs/wandb/run-20260529_204920-6k5qaplc/files/output.log:104` AssertionError: Input height (320) doesn't match model (224).

`logs/wandb/run-20260529_204920-6k5qaplc/files/output.log:106` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260529_205257-mgvn3ond/files/output.log:35` Error executing job with overrides: []

`logs/wandb/run-20260529_205257-mgvn3ond/files/output.log:36` Traceback (most recent call last):

`logs/wandb/run-20260529_205257-mgvn3ond/files/output.log:96` ValueError: not enough values to unpack (expected 4, got 3)

`logs/wandb/run-20260529_205257-mgvn3ond/files/output.log:98` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260529_205551-co8xob97/files/output.log:49` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260529_205551-co8xob97/files/output.log:107` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260529_205551-co8xob97/files/output.log:108` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260529_205551-co8xob97/files/output.log:178` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260529_205551-co8xob97/files/output.log:179` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260530_161941-t2yh3n1q/files/output.log:35` Error executing job with overrides: []

`logs/wandb/run-20260530_161941-t2yh3n1q/files/output.log:36` Traceback (most recent call last):

`logs/wandb/run-20260530_161941-t2yh3n1q/files/output.log:110` AssertionError: Input height (320) doesn't match model (224).

`logs/wandb/run-20260530_161941-t2yh3n1q/files/output.log:112` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260530_162149-gob60epe/files/output.log:38` Error executing job with overrides: []

`logs/wandb/run-20260530_162149-gob60epe/files/output.log:39` Traceback (most recent call last):

`logs/wandb/run-20260530_162149-gob60epe/files/output.log:113` AssertionError: Input height (320) doesn't match model (224).

`logs/wandb/run-20260530_162149-gob60epe/files/output.log:115` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260530_163243-24r3oek0/files/output.log:57` method which may lead to errors, as metric states have not yet been

`logs/wandb/run-20260530_163243-24r3oek0/files/output.log:331` kages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using

`logs/wandb/run-20260530_163243-24r3oek0/files/output.log:332` ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260530_163243-24r3oek0/files/output.log:333` pre-optimization state was saved. The checkpoint will contain the model

`logs/wandb/run-20260530_170115-9831fww8/files/output.log:57` method which may lead to errors, as metric states have not yet been

`logs/wandb/run-20260530_170115-9831fww8/files/output.log:626` checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260530_170115-9831fww8/files/output.log:628` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260530_170115-9831fww8/files/output.log:1228` checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260530_170115-9831fww8/files/output.log:1230` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260530_172410-41p3c26x/files/output.log:56` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260530_172410-41p3c26x/files/output.log:114` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260530_172410-41p3c26x/files/output.log:115` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260530_222136-abijx33a/files/output.log:35` Error executing job with overrides: []

`logs/wandb/run-20260530_222136-abijx33a/files/output.log:36` Traceback (most recent call last):

`logs/wandb/run-20260530_222136-abijx33a/files/output.log:103` AssertionError: Input height (320) doesn't match model (224).

`logs/wandb/run-20260530_222136-abijx33a/files/output.log:105` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260530_223018-x2chyaps/files/output.log:49` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260530_223018-x2chyaps/files/output.log:107` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260530_223018-x2chyaps/files/output.log:108` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260530_224955-0owte4z5/files/output.log:49` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260530_224955-0owte4z5/files/output.log:107` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260530_224955-0owte4z5/files/output.log:108` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260530_225914-64yxk4om/files/output.log:49` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260530_225914-64yxk4om/files/output.log:107` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260530_225914-64yxk4om/files/output.log:108` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260530_231941-uqnmjhw4/files/output.log:56` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260530_231941-uqnmjhw4/files/output.log:60` Error executing job with overrides: []

`logs/wandb/run-20260530_231941-uqnmjhw4/files/output.log:61` Traceback (most recent call last):

`logs/wandb/run-20260530_231941-uqnmjhw4/files/output.log:130` RuntimeError: MPS: Unsupported Border padding mode

`logs/wandb/run-20260530_231941-uqnmjhw4/files/output.log:132` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260530_232140-m0tn2l5t/files/output.log:56` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260530_232140-m0tn2l5t/files/output.log:60` Error executing job with overrides: []

`logs/wandb/run-20260530_232140-m0tn2l5t/files/output.log:61` Traceback (most recent call last):

`logs/wandb/run-20260530_232140-m0tn2l5t/files/output.log:145` NotImplementedError: The operator 'aten::grid_sampler_2d_backward' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://github.com/pytorch/pytorch/issues/141287 and mention use-case, that resulted in missing op as well as commit hash 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0. As a temporary fix, you can set the environment vari

`logs/wandb/run-20260530_232140-m0tn2l5t/files/output.log:147` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260530_232403-inomefky/files/output.log:56` ``update`` method which may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260530_232403-inomefky/files/output.log:119` /model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no

`logs/wandb/run-20260530_232403-inomefky/files/output.log:120` pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save

`logs/wandb/run-20260818_205130-jstjxc8i/files/output.log:676` orch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260818_205130-jstjxc8i/files/output.log:678` checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260818_205130-jstjxc8i/files/output.log:753` orch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual

`logs/wandb/run-20260818_205130-jstjxc8i/files/output.log:755` checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260818_205130-jstjxc8i/files/output.log:771` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260818_235059-xnqkrmqm/files/output.log:793` heckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260818_235059-xnqkrmqm/files/output.log:794` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260818_235059-xnqkrmqm/files/output.log:901` heckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260818_235059-xnqkrmqm/files/output.log:902` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260818_235059-xnqkrmqm/files/output.log:917` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260819_103004-6n3r0uc9/files/output.log:793` heckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260819_103004-6n3r0uc9/files/output.log:794` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260819_103004-6n3r0uc9/files/output.log:901` heckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260819_103004-6n3r0uc9/files/output.log:902` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260819_103004-6n3r0uc9/files/output.log:917` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260819_170051-spkj1v8g/files/output.log:1307` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with num: 1v8g[0mg[0mm[0m

`logs/wandb/run-20260819_170051-spkj1v8g/files/output.log:1308` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260819_170051-spkj1v8g/files/output.log:1373` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with [3mv_num: 1v8g[0m[0m

`logs/wandb/run-20260819_170051-spkj1v8g/files/output.log:1374` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260819_170051-spkj1v8g/files/output.log:1440` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260819_200842-qthlmgbm/files/output.log:492` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260819_200842-qthlmgbm/files/output.log:595` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with _num: mgbm[0mm[0m[0m

`logs/wandb/run-20260819_200842-qthlmgbm/files/output.log:596` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260819_200842-qthlmgbm/files/output.log:700` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with num: mgbm[0mm[0mm[0m

`logs/wandb/run-20260819_200842-qthlmgbm/files/output.log:701` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260819_232912-eutfogcf/files/output.log:1287` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with num: ogcf[0mf[0mm[0m

`logs/wandb/run-20260819_232912-eutfogcf/files/output.log:1288` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260819_232912-eutfogcf/files/output.log:1353` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with [3mv_num: ogcf[0m[0m

`logs/wandb/run-20260819_232912-eutfogcf/files/output.log:1354` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260819_232912-eutfogcf/files/output.log:1420` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260820_190926-737tny13/files/output.log:567` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260820_190926-737tny13/files/output.log:667` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260820_190926-737tny13/files/output.log:668` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260820_190926-737tny13/files/output.log:734` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260820_190926-737tny13/files/output.log:735` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260820_213607-zzbyml7w/files/output.log:1202` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260820_213607-zzbyml7w/files/output.log:1268` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260820_213607-zzbyml7w/files/output.log:1269` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260820_213607-zzbyml7w/files/output.log:1335` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260820_213607-zzbyml7w/files/output.log:1336` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_020228-yxjqete3/files/output.log:1310` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260821_020228-yxjqete3/files/output.log:1311` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_020228-yxjqete3/files/output.log:1377` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260821_020228-yxjqete3/files/output.log:1378` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_020228-yxjqete3/files/output.log:1444` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260821_083435-fwg0piup/files/output.log:1138` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_083435-fwg0piup/files/output.log:1204` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260821_083435-fwg0piup/files/output.log:1205` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_083435-fwg0piup/files/output.log:1271` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260821_083435-fwg0piup/files/output.log:1272` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_152910-z88zqq3t/files/output.log:929` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_152910-z88zqq3t/files/output.log:1030` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260821_152910-z88zqq3t/files/output.log:1031` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_152910-z88zqq3t/files/output.log:1132` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260821_152910-z88zqq3t/files/output.log:1133` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_202106-zkkj5bo7/files/output.log:664` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_202106-zkkj5bo7/files/output.log:730` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260821_202106-zkkj5bo7/files/output.log:731` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_202106-zkkj5bo7/files/output.log:797` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260821_202106-zkkj5bo7/files/output.log:798` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_233058-ct8o1ley/files/output.log:1032` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_233058-ct8o1ley/files/output.log:1131` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260821_233058-ct8o1ley/files/output.log:1132` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260821_233058-ct8o1ley/files/output.log:1198` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260821_233058-ct8o1ley/files/output.log:1199` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260822_014333-2cell07t/files/output.log:1275` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260822_014333-2cell07t/files/output.log:1276` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260822_014333-2cell07t/files/output.log:1342` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260822_014333-2cell07t/files/output.log:1343` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260822_014333-2cell07t/files/output.log:1409` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260822_103526-n5j74e89/files/output.log:74` to errors, as metric states have not yet been updated.

`logs/wandb/run-20260822_103526-n5j74e89/files/output.log:162` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260822_103526-n5j74e89/files/output.log:163` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260822_103526-n5j74e89/files/output.log:263` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260822_103526-n5j74e89/files/output.log:264` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260822_110726-78uio9rv/files/output.log:1012` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260822_110726-78uio9rv/files/output.log:1109` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260822_110726-78uio9rv/files/output.log:1110` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260822_110726-78uio9rv/files/output.log:1176` oint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260822_110726-78uio9rv/files/output.log:1177` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the

`logs/wandb/run-20260822_203112-9s93ss0m/files/output.log:76` may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260822_203112-9s93ss0m/files/output.log:161` checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260822_203112-9s93ss0m/files/output.log:162` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260822_203112-9s93ss0m/files/output.log:257` checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260822_203112-9s93ss0m/files/output.log:258` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260822_210224-h1p0r9cj/files/output.log:1882` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260822_210224-h1p0r9cj/files/output.log:1949` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260822_210224-h1p0r9cj/files/output.log:1950` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260822_210224-h1p0r9cj/files/output.log:2017` tning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and

`logs/wandb/run-20260822_210224-h1p0r9cj/files/output.log:2018` every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain

`logs/wandb/run-20260822_231332-6y5cokod/files/output.log:161` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260822_231332-6y5cokod/files/output.log:259` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 43[0m [2;4m0.29it/s[0m [3mv_num: okod[0mm[0m

`logs/wandb/run-20260822_231332-6y5cokod/files/output.log:260` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260822_231332-6y5cokod/files/output.log:358` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: :03[0m [2;4m0.27it/s[0m [3mv_num: okod[0m[0m

`logs/wandb/run-20260822_231332-6y5cokod/files/output.log:359` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260822_234518-8b7maf8y/files/output.log:1270` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: m [2;4m0.24it/s[0m [3mv_num: af8y[0my[0mm[0m

`logs/wandb/run-20260822_234518-8b7maf8y/files/output.log:1271` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260822_234518-8b7maf8y/files/output.log:1337` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: :18[0m [2;4m0.25it/s[0m [3mv_num: af8y[0m[0m

`logs/wandb/run-20260822_234518-8b7maf8y/files/output.log:1338` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260822_234518-8b7maf8y/files/output.log:1405` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260823_034747-bqt6znpa/files/output.log:896` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260823_034747-bqt6znpa/files/output.log:1310` Error executing job with overrides: []

`logs/wandb/run-20260823_034747-bqt6znpa/files/output.log:1311` Traceback (most recent call last):

`logs/wandb/run-20260823_034747-bqt6znpa/files/output.log:1379` RuntimeError: grid_sampler(): expected grid and input to have same batch size, but got input with sizes [16, 768, 20, 20] and grid with sizes [12, 25, 25, 2]

`logs/wandb/run-20260823_034747-bqt6znpa/files/output.log:1381` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260823_120147-qw1j0jpd/files/output.log:1023` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260823_120147-qw1j0jpd/files/output.log:1122` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with [3mv_num: 0jpd[0mm[0m

`logs/wandb/run-20260823_120147-qw1j0jpd/files/output.log:1123` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260823_120147-qw1j0jpd/files/output.log:1188` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with _num: 0jpd[0md[0m[0m

`logs/wandb/run-20260823_120147-qw1j0jpd/files/output.log:1189` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260823_142413-whzb9cqe/files/output.log:1244` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with num: 9cqe[0me[0mm[0m

`logs/wandb/run-20260823_142413-whzb9cqe/files/output.log:1245` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260823_142413-whzb9cqe/files/output.log:1310` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint with [3mv_num: 9cqe[0m[0m

`logs/wandb/run-20260823_142413-whzb9cqe/files/output.log:1311` manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To save the

`logs/wandb/run-20260823_142413-whzb9cqe/files/output.log:1377` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260824_011127-5cipd35z/files/output.log:354` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_011127-5cipd35z/files/output.log:453` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Usingm [2;4m0.21it/s[0m [3mv_num: d35z[0mm[0m

`logs/wandb/run-20260824_011127-5cipd35z/files/output.log:454` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_011127-5cipd35z/files/output.log:554` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using2;4m0.22it/s[0m [3mv_num: d35z[0mz[0m[0m

`logs/wandb/run-20260824_011127-5cipd35z/files/output.log:555` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_025227-duuzymio/files/output.log:1241` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using;4m0.19it/s[0m [3mv_num: ymio[0mo[0mm[0m

`logs/wandb/run-20260824_025227-duuzymio/files/output.log:1242` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_025227-duuzymio/files/output.log:1308` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using0m [2;4m0.21it/s[0m [3mv_num: ymio[0m[0m

`logs/wandb/run-20260824_025227-duuzymio/files/output.log:1309` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_025227-duuzymio/files/output.log:1376` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260824_103128-whsxmp9n/files/output.log:1204` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_103128-whsxmp9n/files/output.log:1270` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using;4m0.21it/s[0m [3mv_num: mp9n[0mn[0mm[0m

`logs/wandb/run-20260824_103128-whsxmp9n/files/output.log:1271` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_103128-whsxmp9n/files/output.log:1337` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using0m [2;4m0.19it/s[0m [3mv_num: mp9n[0m[0m

`logs/wandb/run-20260824_103128-whsxmp9n/files/output.log:1338` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_132701-4letlnov/files/output.log:557` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_132701-4letlnov/files/output.log:657` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using;4m0.20it/s[0m [3mv_num: lnov[0mv[0mm[0m

`logs/wandb/run-20260824_132701-4letlnov/files/output.log:658` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_132701-4letlnov/files/output.log:724` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using0m [2;4m0.17it/s[0m [3mv_num: lnov[0m[0m

`logs/wandb/run-20260824_132701-4letlnov/files/output.log:725` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_163916-o7g8i6th/files/output.log:1249` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using;4m0.21it/s[0m [3mv_num: i6th[0mh[0mm[0m

`logs/wandb/run-20260824_163916-o7g8i6th/files/output.log:1250` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_163916-o7g8i6th/files/output.log:1316` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using0m [2;4m0.21it/s[0m [3mv_num: i6th[0m[0m

`logs/wandb/run-20260824_163916-o7g8i6th/files/output.log:1317` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260824_163916-o7g8i6th/files/output.log:1384` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260824_211309-icmoohdi/files/output.log:94` lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260824_211309-icmoohdi/files/output.log:98` Error executing job with overrides: []

`logs/wandb/run-20260824_211309-icmoohdi/files/output.log:99` Traceback (most recent call last):

`logs/wandb/run-20260824_211309-icmoohdi/files/output.log:183` NotImplementedError: The operator 'aten::grid_sampler_2d_backward' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://github.com/pytorch/pytorch/issues/141287 and mention use-case, that resulted in missing op as well as commit hash 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0. As a temporary fix, you can set the environment vari

`logs/wandb/run-20260824_211309-icmoohdi/files/output.log:185` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260824_211612-ineqwge2/files/output.log:94` lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260824_211612-ineqwge2/files/output.log:98` Error executing job with overrides: []

`logs/wandb/run-20260824_211612-ineqwge2/files/output.log:99` Traceback (most recent call last):

`logs/wandb/run-20260824_211612-ineqwge2/files/output.log:183` NotImplementedError: The operator 'aten::grid_sampler_2d_backward' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://github.com/pytorch/pytorch/issues/141287 and mention use-case, that resulted in missing op as well as commit hash 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0. As a temporary fix, you can set the environment vari

`logs/wandb/run-20260824_211612-ineqwge2/files/output.log:185` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260824_211836-0q1urba0/files/output.log:662` was saved. The checkpoint will contain the model state AFTER optimization. To

`logs/wandb/run-20260824_211836-0q1urba0/files/output.log:774` /pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint

`logs/wandb/run-20260824_211836-0q1urba0/files/output.log:776` was saved. The checkpoint will contain the model state AFTER optimization. To

`logs/wandb/run-20260824_211836-0q1urba0/files/output.log:843` /pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint

`logs/wandb/run-20260824_211836-0q1urba0/files/output.log:845` was saved. The checkpoint will contain the model state AFTER optimization. To

`logs/wandb/run-20260825_012931-tp6f50bk/files/output.log:1509` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_012931-tp6f50bk/files/output.log:2111` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260825_012931-tp6f50bk/files/output.log:2113` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_012931-tp6f50bk/files/output.log:2717` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260825_012931-tp6f50bk/files/output.log:2719` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_021540-7buagh9s/files/output.log:8061` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260825_021540-7buagh9s/files/output.log:8063` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_021540-7buagh9s/files/output.log:8546` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260825_021540-7buagh9s/files/output.log:8548` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_021540-7buagh9s/files/output.log:8964` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260825_092445-fetktyu5/files/output.log:4369` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_092445-fetktyu5/files/output.log:4854` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260825_092445-fetktyu5/files/output.log:4856` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_092445-fetktyu5/files/output.log:5339` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260825_092445-fetktyu5/files/output.log:5341` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_111358-kywepbkz/files/output.log:7975` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260825_111358-kywepbkz/files/output.log:7977` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_111358-kywepbkz/files/output.log:8462` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260825_111358-kywepbkz/files/output.log:8464` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_111358-kywepbkz/files/output.log:8879` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260825_155541-20vptvsw/files/output.log:39` Error executing job with overrides: []

`logs/wandb/run-20260825_155541-20vptvsw/files/output.log:40` Traceback (most recent call last):

`logs/wandb/run-20260825_155541-20vptvsw/files/output.log:83` raise AttributeError(

`logs/wandb/run-20260825_155541-20vptvsw/files/output.log:86` AttributeError: 'LitUnsupervisedSegmenter' object has no attribute '_add_spatial_coords'

`logs/wandb/run-20260825_155541-20vptvsw/files/output.log:88` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260825_161954-k4sfe9j1/files/output.log:114` method which may lead to errors, as metric states have

`logs/wandb/run-20260825_161954-k4sfe9j1/files/output.log:896` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260825_161954-k4sfe9j1/files/output.log:898` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_161954-k4sfe9j1/files/output.log:1495` model_checkpoint.py:401: Using ModelCheckpoint with

`logs/wandb/run-20260825_161954-k4sfe9j1/files/output.log:1497` pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260825_171825-vrtyi6wz/files/output.log:745` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260825_171825-vrtyi6wz/files/output.log:811` checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260825_171825-vrtyi6wz/files/output.log:812` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260825_171825-vrtyi6wz/files/output.log:878` checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260825_171825-vrtyi6wz/files/output.log:879` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260825_185352-7sax9op6/files/output.log:78` may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260825_185352-7sax9op6/files/output.log:163` checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260825_185352-7sax9op6/files/output.log:164` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260825_191512-oe1a0brw/files/output.log:78` may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260825_191533-q3qu1bvt/files/output.log:78` may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260825_191533-q3qu1bvt/files/output.log:167` checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260825_191533-q3qu1bvt/files/output.log:168` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260825_191533-q3qu1bvt/files/output.log:269` checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260825_191533-q3qu1bvt/files/output.log:270` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260825_201534-qoidav7b/files/output.log:353` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260825_201534-qoidav7b/files/output.log:449` checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260825_201534-qoidav7b/files/output.log:450` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260825_201534-qoidav7b/files/output.log:546` checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260825_201534-qoidav7b/files/output.log:547` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization

`logs/wandb/run-20260825_212249-q5g4qy1e/files/output.log:78` may lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260825_212249-q5g4qy1e/files/output.log:166` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Using2;4m0.22it/s[0m [3mv_num: qy1e[0me[0m[0m

`logs/wandb/run-20260825_212249-q5g4qy1e/files/output.log:167` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260825_212249-q5g4qy1e/files/output.log:268` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: Usingm [2;4m0.21it/s[0m [3mv_num: qy1e[0mm[0m

`logs/wandb/run-20260825_212249-q5g4qy1e/files/output.log:269` ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will contain the

`logs/wandb/run-20260825_215956-3u1a20n1/files/output.log:1280` nt.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260825_215956-3u1a20n1/files/output.log:1281` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the model

`logs/wandb/run-20260825_215956-3u1a20n1/files/output.log:1347` nt.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260825_215956-3u1a20n1/files/output.log:1348` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the model

`logs/wandb/run-20260825_215956-3u1a20n1/files/output.log:1414` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260826_131327-d96kuivn/files/output.log:74` to errors, as metric states have not yet been updated.

`logs/wandb/run-20260826_131327-d96kuivn/files/output.log:163` nt.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260826_131327-d96kuivn/files/output.log:164` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the model

`logs/wandb/run-20260826_131327-d96kuivn/files/output.log:265` nt.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260826_131327-d96kuivn/files/output.log:266` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the model

`logs/wandb/run-20260826_170833-6xkx7xxw/files/output.log:696` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the model

`logs/wandb/run-20260826_170833-6xkx7xxw/files/output.log:794` Error executing job with overrides: []

`logs/wandb/run-20260826_170833-6xkx7xxw/files/output.log:795` Traceback (most recent call last):

`logs/wandb/run-20260826_170833-6xkx7xxw/files/output.log:847` IndexError: index 4 is out of bounds for dimension 0 with size 4

`logs/wandb/run-20260826_170833-6xkx7xxw/files/output.log:849` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260826_193406-78lvu81i/files/output.log:2222` nt.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260826_193406-78lvu81i/files/output.log:2223` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the model

`logs/wandb/run-20260826_193406-78lvu81i/files/output.log:2340` nt.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was

`logs/wandb/run-20260826_193406-78lvu81i/files/output.log:2341` saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state, save the model

`logs/wandb/run-20260826_193406-78lvu81i/files/output.log:2458` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260828_200520-950ml5zu/files/output.log:1290` no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To

`logs/wandb/run-20260828_200520-950ml5zu/files/output.log:1407` ks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but

`logs/wandb/run-20260828_200520-950ml5zu/files/output.log:1408` no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To

`logs/wandb/run-20260828_200520-950ml5zu/files/output.log:1525` ks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but

`logs/wandb/run-20260828_200520-950ml5zu/files/output.log:1526` no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To

`logs/wandb/run-20260828_234121-ggwfmoqc/files/output.log:855` no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To

`logs/wandb/run-20260828_234121-ggwfmoqc/files/output.log:1028` ks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but

`logs/wandb/run-20260828_234121-ggwfmoqc/files/output.log:1029` no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To

`logs/wandb/run-20260828_234121-ggwfmoqc/files/output.log:1174` ks/model_checkpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but

`logs/wandb/run-20260828_234121-ggwfmoqc/files/output.log:1175` no pre-optimization state was saved. The checkpoint will contain the model state AFTER optimization. To

`logs/wandb/run-20260907_174657-wls99dok/files/output.log:150` which may lead to errors, as metric

`logs/wandb/run-20260907_174657-wls99dok/files/output.log:666` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 0m [2;4m0.20it/s[0m [3mv_num: 9dok[0mk[0m[0m

`logs/wandb/run-20260907_174657-wls99dok/files/output.log:667` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260907_191331-ckb78tko/files/output.log:695` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260907_191331-ckb78tko/files/output.log:794` Error executing job with overrides: []

`logs/wandb/run-20260907_191331-ckb78tko/files/output.log:795` Traceback (most recent call last):

`logs/wandb/run-20260907_191331-ckb78tko/files/output.log:847` IndexError: index 4 is out of bounds for dimension 0 with size 4

`logs/wandb/run-20260907_191331-ckb78tko/files/output.log:849` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260907_203241-6094aa1i/files/output.log:163` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260907_203241-6094aa1i/files/output.log:264` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 52[0m [2;4m0.22it/s[0m [3mv_num: aa1i[0mm[0m

`logs/wandb/run-20260907_203241-6094aa1i/files/output.log:265` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260907_203241-6094aa1i/files/output.log:366` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: :18[0m [2;4m0.22it/s[0m [3mv_num: aa1i[0m[0m

`logs/wandb/run-20260907_203241-6094aa1i/files/output.log:367` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260907_211127-pwaekmsa/files/output.log:740` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260907_211127-pwaekmsa/files/output.log:806` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 36[0m [2;4m0.22it/s[0m [3mv_num: kmsa[0mm[0m

`logs/wandb/run-20260907_211127-pwaekmsa/files/output.log:807` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260907_211127-pwaekmsa/files/output.log:873` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 0m [2;4m0.22it/s[0m [3mv_num: kmsa[0ma[0m[0m

`logs/wandb/run-20260907_211127-pwaekmsa/files/output.log:874` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260907_224406-ln55dwfl/files/output.log:1278` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: m [2;4m0.22it/s[0m [3mv_num: dwfl[0ml[0mm[0m

`logs/wandb/run-20260907_224406-ln55dwfl/files/output.log:1279` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260907_224406-ln55dwfl/files/output.log:1345` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: :51[0m [2;4m0.22it/s[0m [3mv_num: dwfl[0m[0m

`logs/wandb/run-20260907_224406-ln55dwfl/files/output.log:1346` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260907_224406-ln55dwfl/files/output.log:1413` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260908_014053-wrjaph1n/files/output.log:64` ``compute`` method of metric UnsupervisedMetrics was called before the ``update`` method which may lead to errors, as metric states

`logs/wandb/run-20260908_014053-wrjaph1n/files/output.log:122` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: m [2;4m0.20it/s[0m [3mv_num: ph1n[0mn[0mm[0m

`logs/wandb/run-20260908_014053-wrjaph1n/files/output.log:123` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_014053-wrjaph1n/files/output.log:193` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 34[0m [2;4m0.19it/s[0m [3mv_num: ph1n[0mm[0m

`logs/wandb/run-20260908_014053-wrjaph1n/files/output.log:194` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_021035-vehng4je/files/output.log:60` ``compute`` method of metric UnsupervisedMetrics was called before the ``update`` method which may lead to errors, as metric states

`logs/wandb/run-20260908_021035-vehng4je/files/output.log:113` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 00[0m [2;4m0.30it/s[0m [3mv_num: g4je[0m[0m

`logs/wandb/run-20260908_021035-vehng4je/files/output.log:114` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_021035-vehng4je/files/output.log:231` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: m [2;4m0.26it/s[0m [3mv_num: g4je[0me[0m[0m

`logs/wandb/run-20260908_021035-vehng4je/files/output.log:232` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_023403-o879wblt/files/output.log:114` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_023403-o879wblt/files/output.log:231` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: m [2;4m0.27it/s[0m [3mv_num: wblt[0mt[0m[0m

`logs/wandb/run-20260908_023403-o879wblt/files/output.log:232` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_023403-o879wblt/files/output.log:349` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: m [2;4m0.28it/s[0m [3mv_num: wblt[0mt[0m[0m

`logs/wandb/run-20260908_023403-o879wblt/files/output.log:350` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_025959-15xor701/files/output.log:66` ``compute`` method of metric UnsupervisedMetrics was called before the ``update`` method which may lead to errors, as metric states

`logs/wandb/run-20260908_025959-15xor701/files/output.log:124` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 09[0m [2;4m0.26it/s[0m [3mv_num: r701[0m[0m

`logs/wandb/run-20260908_025959-15xor701/files/output.log:125` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_025959-15xor701/files/output.log:246` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: m [2;4m0.27it/s[0m [3mv_num: r701[0m1[0m[0m

`logs/wandb/run-20260908_025959-15xor701/files/output.log:247` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_032246-sjn6n26g/files/output.log:62` ``compute`` method of metric UnsupervisedMetrics was called before the ``update`` method which may lead to errors, as metric states

`logs/wandb/run-20260908_032246-sjn6n26g/files/output.log:117` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 06[0m [2;4m0.28it/s[0m [3mv_num: n26g[0m[0m

`logs/wandb/run-20260908_032246-sjn6n26g/files/output.log:118` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_033113-8ishmyza/files/output.log:60` ``compute`` method of metric UnsupervisedMetrics was called before the ``update`` method which may lead to errors, as metric states

`logs/wandb/run-20260908_033705-gnlyaxt5/files/output.log:233` Error executing job with overrides: []

`logs/wandb/run-20260908_033705-gnlyaxt5/files/output.log:234` Traceback (most recent call last):

`logs/wandb/run-20260908_033705-gnlyaxt5/files/output.log:300` RuntimeError: Exception thrown in SimpleITK ImageFileReader_Execute: /Users/ec2-user/actions-runner/_work/SimpleITK/SimpleITK/Code/IO/src/sitkImageReaderBase.cxx:91:

`logs/wandb/run-20260908_033705-gnlyaxt5/files/output.log:301` sitk::ERROR: The file "/Users/navneetbavineni/src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/23/DICOM_anon/IMG-0006-00078.dcm" does not exist.

`logs/wandb/run-20260908_033705-gnlyaxt5/files/output.log:303` Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

`logs/wandb/run-20260908_035506-4o1zbesc/files/output.log:1231` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 0m [2;4m0.31it/s[0m [3mv_num: besc[0mc[0m[0m

`logs/wandb/run-20260908_035506-4o1zbesc/files/output.log:1232` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_035506-4o1zbesc/files/output.log:1305` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: :06[0m [2;4m0.31it/s[0m [3mv_num: besc[0m[0m

`logs/wandb/run-20260908_035506-4o1zbesc/files/output.log:1306` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_035506-4o1zbesc/files/output.log:1351` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260908_092253-n8kw50b1/files/output.log:1231` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 0m [2;4m0.31it/s[0m [3mv_num: 50b1[0m1[0m[0m

`logs/wandb/run-20260908_092253-n8kw50b1/files/output.log:1232` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_092253-n8kw50b1/files/output.log:1305` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: :06[0m [2;4m0.31it/s[0m [3mv_num: 50b1[0m[0m

`logs/wandb/run-20260908_092253-n8kw50b1/files/output.log:1306` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_092253-n8kw50b1/files/output.log:1351` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260908_130818-iz7z60w4/files/output.log:491` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_130818-iz7z60w4/files/output.log:564` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: 35[0m [2;4m0.23it/s[0m [3mv_num: 60w4[0mm[0m

`logs/wandb/run-20260908_130818-iz7z60w4/files/output.log:565` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260908_130818-iz7z60w4/files/output.log:682` [2K/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:401: :28[0m [2;4m0.11it/s[0m [3mv_num: 60w4[0m[0m

`logs/wandb/run-20260908_130818-iz7z60w4/files/output.log:683` Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization state was saved. The checkpoint will

`logs/wandb/run-20260909_023853-gnn4l5f9/files/output.log:742` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_023853-gnn4l5f9/files/output.log:937` eckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260909_023853-gnn4l5f9/files/output.log:938` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_023853-gnn4l5f9/files/output.log:1043` eckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260909_023853-gnn4l5f9/files/output.log:1044` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_033448-wicrfwro/files/output.log:63` lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260909_033448-wicrfwro/files/output.log:155` eckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260909_033448-wicrfwro/files/output.log:156` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_033448-wicrfwro/files/output.log:351` eckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260909_033448-wicrfwro/files/output.log:352` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_035333-6pmajvfa/files/output.log:63` lead to errors, as metric states have not yet been updated.

`logs/wandb/run-20260909_035925-75cq6wm7/files/output.log:353` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_035925-75cq6wm7/files/output.log:547` eckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260909_035925-75cq6wm7/files/output.log:548` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_035925-75cq6wm7/files/output.log:741` eckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260909_035925-75cq6wm7/files/output.log:742` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_044210-y3l0vxcn/files/output.log:2612` eckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260909_044210-y3l0vxcn/files/output.log:2613` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_044210-y3l0vxcn/files/output.log:2839` eckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260909_044210-y3l0vxcn/files/output.log:2840` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_044210-y3l0vxcn/files/output.log:3067` `Trainer.fit` stopped: `max_steps=5000` reached.

`logs/wandb/run-20260909_173533-67xgrz2i/files/output.log:1159` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_173533-67xgrz2i/files/output.log:1284` eckpoint.py:401: Using ModelCheckpoint with manual optimization and every_n_train_steps, but no pre-optimization

`logs/wandb/run-20260909_173533-67xgrz2i/files/output.log:1285` state was saved. The checkpoint will contain the model state AFTER optimization. To save the pre-optimization state,

`logs/wandb/run-20260909_173533-67xgrz2i/files/output.log:1520` s/pytorch_lightning/callbacks/model_checkpoint.py:401: Using ModelCheckpoint

`logs/wandb/run-20260909_173533-67xgrz2i/files/output.log:1522` state was saved. The checkpoint will contain the model state AFTER

## M5 — Seeded raw-slice intensity and pairing evidence
Command: `python3 -B audit/scratch/decision_samples.py`; seed=20260930;20 slices per sequence (T1 phases sampled separately). Percentiles are in order[0.5, 1, 5, 50, 95, 99, 99.5]. CT values use slope/intercept; MRI values are stored pixel intensities. Pairing follows the current loader: sorted DICOM filenames zipped to sorted mask filenames. Thus intensity/organ statistics are conditional on that pairing; no anatomical label identity is inferred from a numeric value.
### Series counts (DICOM, PNG)
`[('CT', '1', 93, 93), ('CT', '10', 107, 107), ('CT', '14', 93, 93), ('CT', '16', 89, 89), ('CT', '18', 106, 106), ('CT', '19', 90, 90), ('CT', '2', 81, 81), ('CT', '21', 170, 170), ('CT', '22', 101, 101), ('CT', '23', 180, 180), ('CT', '24', 96, 96), ('CT', '25', 118, 118), ('CT', '26', 107, 107), ('CT', '27', 182, 182), ('CT', '28', 102, 102), ('CT', '29', 154, 154), ('CT', '30', 197, 197), ('CT', '5', 95, 95), ('CT', '6', 82, 82), ('CT', '8', 98, 98), ('T1-in', '1', 25, 25), ('T1-out', '1', 35, 25), ('T2', '1', 24, 24), ('T1-in', '10', 33, 33), ('T1-out', '10', 50, 33), ('T2', '10', 25, 25), ('T1-in', '13', 27, 27), ('T1-out', '13', 30, 27), ('T2', '13', 28, 28), ('T1-in', '15', 25, 25), ('T1-out', '15', 26, 25), ('T2', '15', 26, 26), ('T1-in', '19', 25, 25), ('T1-out', '19', 30, 25), ('T2', '19', 25, 25), ('T1-in', '2', 21, 21), ('T1-out', '2', 26, 21), ('T2', '2', 23, 23), ('T1-in', '20', 21, 21), ('T1-out', '20', 26, 21), ('T2', '20', 24, 24), ('T1-in', '21', 21, 21), ('T1-out', '21', 34, 21), ('T2', '21', 21, 21), ('T1-in', '22', 24, 24), ('T1-out', '22', 35, 24), ('T2', '22', 27, 27), ('T1-in', '3', 21, 21), ('T1-out', '3', 30, 21), ('T2', '3', 21, 21), ('T1-in', '31', 20, 20), ('T1-out', '31', 30, 20), ('T2', '31', 20, 20), ('T1-in', '32', 29, 29), ('T1-out', '32', 33, 29), ('T2', '32', 30, 30), ('T1-in', '33', 24, 24), ('T1-out', '33', 33, 24), ('T2', '33', 27, 27), ('T1-in', '34', 35, 35), ('T1-out', '34', 35, 35), ('T2', '34', 25, 25), ('T1-in', '36', 21, 21), ('T1-out', '36', 38, 21), ('T2', '36', 23, 23), ('T1-in', '37', 23, 23), ('T1-out', '37', 30, 23), ('T2', '37', 24, 24), ('T1-in', '38', 24, 24), ('T1-out', '38', 34, 24), ('T2', '38', 24, 24), ('T1-in', '39', 21, 21), ('T1-out', '39', 26, 21), ('T2', '39', 22, 22), ('T1-in', '5', 24, 24), ('T1-out', '5', 30, 24), ('T2', '5', 24, 24), ('T1-in', '8', 36, 36), ('T1-out', '8', 36, 36), ('T2', '8', 29, 29)]`
### CT
Image / mask | Patient / filename-order index | Shape | Stored labels/counts | Header: instance; position; orientation; spacing; slope/intercept
--- | --- | --- | --- | ---
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00067.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/Ground/liver_GT_066.png` | 30/26 | (512, 512) | `{0: 242593, 255: 19551}` | `{'InstanceNumber': '67', 'ImagePositionPatient': '[-213.5, -64.5, -84.7000122]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.732421875, 0.732421875]', 'RescaleSlope': '1', 'RescaleIntercept': '-1024'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/28/DICOM_anon/IMG-0011-00048.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/28/Ground/liver_GT_047.png` | 28/47 | (512, 512) | `{0: 239500, 255: 22644}` | `{'InstanceNumber': '48', 'ImagePositionPatient': '[-167.5, -92.5, 1494.80005]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.693359375, 0.693359375]', 'RescaleSlope': '1', 'RescaleIntercept': '-1000'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/22/DICOM_anon/IMG-0005-00082.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/22/Ground/liver_GT_081.png` | 22/81 | (512, 512) | `{0: 256165, 255: 5979}` | `{'InstanceNumber': '82', 'ImagePositionPatient': '[-183, -89, 1443.90002]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.72265625, 0.72265625]', 'RescaleSlope': '1', 'RescaleIntercept': '-1000'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/25/DICOM_anon/IMG-0008-00073.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/25/Ground/liver_GT_072.png` | 25/54 | (512, 512) | `{0: 203923, 255: 58221}` | `{'InstanceNumber': '73', 'ImagePositionPatient': '[-149, 4, -179.5]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.6015625, 0.6015625]', 'RescaleSlope': '1', 'RescaleIntercept': '-1024'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00227.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/Ground/liver_GT_226.png` | 30/186 | (512, 512) | `{0: 260259, 255: 1885}` | `{'InstanceNumber': '227', 'ImagePositionPatient': '[-213.5, -64.5, -244.700012]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.732421875, 0.732421875]', 'RescaleSlope': '1', 'RescaleIntercept': '-1024'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/28/DICOM_anon/IMG-0011-00102.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/28/Ground/liver_GT_101.png` | 28/101 | (512, 512) | `{0: 257663, 255: 4481}` | `{'InstanceNumber': '102', 'ImagePositionPatient': '[-167.5, -92.5, 1408.40002]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.693359375, 0.693359375]', 'RescaleSlope': '1', 'RescaleIntercept': '-1000'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/1/DICOM_anon/i0093,0000b.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/1/Ground/liver_GT_093.png` | 1/90 | (512, 512) | `{0: 254241, 255: 7903}` | `{'InstanceNumber': '94', 'ImagePositionPatient': '[-167.7859, -176.0438, -223.5]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.687671, 0.687671]', 'RescaleSlope': '1', 'RescaleIntercept': '-1200'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/14/DICOM_anon/i0006,0000b.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/14/Ground/liver_GT_006.png` | 14/4 | (512, 512) | `{0: 246635, 255: 15509}` | `{'InstanceNumber': '7', 'ImagePositionPatient': '[-139.5, -32.5, 1659.40002]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.568359375, 0.568359375]', 'RescaleSlope': '1', 'RescaleIntercept': '-1000'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/8/DICOM_anon/i0087,0000b.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/8/Ground/liver_GT_087.png` | 8/77 | (512, 512) | `{0: 257033, 255: 5111}` | `{'InstanceNumber': '88', 'ImagePositionPatient': '[-176, -95, 1540.30005]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.734375, 0.734375]', 'RescaleSlope': '1', 'RescaleIntercept': '-1000'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/24/DICOM_anon/IMG-0007-00033.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/24/Ground/liver_GT_032.png` | 24/23 | (512, 512) | `{0: 218335, 255: 43809}` | `{'InstanceNumber': '33', 'ImagePositionPatient': '[-176, -1, -299.200012]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.63671875, 0.63671875]', 'RescaleSlope': '1', 'RescaleIntercept': '-1024'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00084.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/Ground/liver_GT_083.png` | 30/43 | (512, 512) | `{0: 229548, 255: 32596}` | `{'InstanceNumber': '84', 'ImagePositionPatient': '[-213.5, -64.5, -101.700012]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.732421875, 0.732421875]', 'RescaleSlope': '1', 'RescaleIntercept': '-1024'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/29/DICOM_anon/IMG-0012-00099.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/29/Ground/liver_GT_098.png` | 29/45 | (512, 512) | `{0: 228119, 255: 34025}` | `{'InstanceNumber': '99', 'ImagePositionPatient': '[-196, -28, -166.500031]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.68359375, 0.68359375]', 'RescaleSlope': '1', 'RescaleIntercept': '-1024'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/10/DICOM_anon/i0019,0000b.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/10/Ground/liver_GT_019.png` | 10/16 | (512, 512) | `{0: 231754, 255: 30390}` | `{'InstanceNumber': '20', 'ImagePositionPatient': '[-159.5, -98.5, 1652.09998]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.693359375, 0.693359375]', 'RescaleSlope': '1', 'RescaleIntercept': '-1000'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/10/DICOM_anon/i0073,0000b.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/10/Ground/liver_GT_073.png` | 10/70 | (512, 512) | `{0: 246106, 255: 16038}` | `{'InstanceNumber': '74', 'ImagePositionPatient': '[-159.5, -98.5, 1565.69995]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.693359375, 0.693359375]', 'RescaleSlope': '1', 'RescaleIntercept': '-1000'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00209.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/Ground/liver_GT_208.png` | 30/168 | (512, 512) | `{0: 257670, 255: 4474}` | `{'InstanceNumber': '209', 'ImagePositionPatient': '[-213.5, -64.5, -226.700012]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.732421875, 0.732421875]', 'RescaleSlope': '1', 'RescaleIntercept': '-1024'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/16/DICOM_anon/i0055,0000b.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/16/Ground/liver_GT_055.png` | 16/53 | (512, 512) | `{0: 244925, 255: 17219}` | `{'InstanceNumber': '56', 'ImagePositionPatient': '[-190.5, -103.5, 1591]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.791015625, 0.791015625]', 'RescaleSlope': '1', 'RescaleIntercept': '-1000'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/21/DICOM_anon/IMG-0001-00110.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/21/Ground/liver_GT_109.png` | 21/67 | (512, 512) | `{0: 227164, 255: 34980}` | `{'InstanceNumber': '110', 'ImagePositionPatient': '[-211.325459, -74.0918655, -99.4999847]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.78125, 0.78125]', 'RescaleSlope': '1', 'RescaleIntercept': '-1024'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/6/DICOM_anon/i0027,0000b.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/6/Ground/liver_GT_027.png` | 6/21 | (512, 512) | `{0: 216005, 255: 46139}` | `{'InstanceNumber': '28', 'ImagePositionPatient': '[-169.2873, -166.2845, -81]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.649549, 0.649549]', 'RescaleSlope': '1', 'RescaleIntercept': '-1200'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/26/DICOM_anon/IMG-0009-00109.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/26/Ground/liver_GT_108.png` | 26/106 | (512, 512) | `{0: 262119, 255: 25}` | `{'InstanceNumber': '109', 'ImagePositionPatient': '[-190, -106, 1425.69995]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.76171875, 0.76171875]', 'RescaleSlope': '1', 'RescaleIntercept': '-1000'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/16/DICOM_anon/i0023,0000b.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/16/Ground/liver_GT_023.png` | 16/21 | (512, 512) | `{0: 226618, 255: 35526}` | `{'InstanceNumber': '24', 'ImagePositionPatient': '[-190.5, -103.5, 1655]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[0.791015625, 0.791015625]', 'RescaleSlope': '1', 'RescaleIntercept': '-1000'}`

Aggregate result (pooled pixels; contrast averaged across nonempty-mask slices):
```json
{
  "sequence": "CT",
  "n": 20,
  "whole_percentiles": [
    -1024.0,
    -1024.0,
    -1003.0,
    -304.0,
    154.0,
    302.0,
    425.0
  ],
  "mask_counts": {
    "0": 4806375,
    "255": 436505
  },
  "organs": {
    "255": {
      "count": 436505,
      "percentiles": [
        10.0,
        36.0,
        71.0,
        126.0,
        187.0,
        238.0,
        253.0
      ],
      "clip_1_99_counts": [
        0,
        30,
        436505
      ],
      "clip_1_99_fraction": 6.872773507749052e-05
    }
  },
  "windows": {
    "(-160, 240)": {
      "clip_counts": [
        6,
        3791,
        436505
      ],
      "clipped_fraction": 0.008698640336307717,
      "mean_liver": 0.7081837713718414,
      "mean_nonliver": 0.17569311559200287,
      "mean_contrast": 0.5324906557798386
    },
    "(-150, 250)": {
      "clip_counts": [
        6,
        2456,
        436505
      ],
      "clipped_fraction": 0.005640256125359388,
      "mean_liver": 0.6834334418177604,
      "mean_nonliver": 0.16515738926827908,
      "mean_contrast": 0.5182760525494814
    },
    "(-400, 400)": {
      "clip_counts": [
        0,
        0,
        436505
      ],
      "clipped_fraction": 0.0,
      "mean_liver": 0.6545110329985618,
      "mean_nonliver": 0.22496037632226945,
      "mean_contrast": 0.42955065667629244
    }
  }
}
```
### T1-in
Image / mask | Patient / filename-order index | Shape | Stored labels/counts | Header: instance; position; orientation; spacing; slope/intercept
--- | --- | --- | --- | ---
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/DICOM_anon/InPhase/IMG-0034-00020.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/Ground/IMG-0034-00020.png` | 8/9 | (288, 288) | `{0: 77853, 63: 1768, 126: 1120, 189: 962, 252: 1241}` | `{'InstanceNumber': '20', 'ImagePositionPatient': '[-200.20083713531, -228.6872882843, -2.9766197204589]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/21/T1DUAL/DICOM_anon/InPhase/IMG-0004-00056.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/21/T1DUAL/Ground/IMG-0004-00056.png` | 21/15 | (256, 256) | `{0: 60187, 63: 4451, 252: 898}` | `{'InstanceNumber': '56', 'ImagePositionPatient': '[-247.71958260703, -231.80079654464, 28.230947835946]', 'ImageOrientationPatient': '[0.9997271895409, 0.0094338627532, -0.02136612311, -0.0094312336295, 0.9999554753304, 0.0002238224406]', 'PixelSpacing': '[1.81640625, 1.81640625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T1DUAL/DICOM_anon/InPhase/IMG-0005-00060.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T1DUAL/Ground/IMG-0005-00060.png` | 34/29 | (256, 256) | `{0: 60648, 63: 4888}` | `{'InstanceNumber': '60', 'ImagePositionPatient': '[-219.34210515022, -207.76315784454, 29.167442321777]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.73828125, 1.73828125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/20/T1DUAL/DICOM_anon/InPhase/IMG-0027-00026.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/20/T1DUAL/Ground/IMG-0027-00026.png` | 20/8 | (288, 288) | `{0: 76384, 63: 3448, 126: 912, 189: 1588, 252: 612}` | `{'InstanceNumber': '26', 'ImagePositionPatient': '[-207.49999809265, -219.77059459686, 6.2036304473877]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/InPhase/IMG-0023-00038.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/Ground/IMG-0023-00038.png` | 38/9 | (288, 288) | `{0: 78865, 63: 3013, 126: 103, 189: 819, 252: 144}` | `{'InstanceNumber': '38', 'ImagePositionPatient': '[-196.60490894317, -200.13157701492, -17.185522079467]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/39/T1DUAL/DICOM_anon/InPhase/IMG-0027-00030.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/39/T1DUAL/Ground/IMG-0027-00030.png` | 39/12 | (256, 256) | `{0: 60598, 63: 2961, 126: 214, 189: 64, 252: 1699}` | `{'InstanceNumber': '30', 'ImagePositionPatient': '[-215.25214416347, -206.80334907211, -1.9482051935519]', 'ImageOrientationPatient': '[0.999110519886, -0.0421353019773, 0.0016553440364, 0.0421352423728, 0.9991118907928, 6.98104195E-05]', 'PixelSpacing': '[1.69921875, 1.69921875]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/10/T1DUAL/DICOM_anon/InPhase/IMG-0046-00026.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/10/T1DUAL/Ground/IMG-0046-00026.png` | 10/8 | (256, 256) | `{0: 63861, 63: 625, 126: 615, 189: 435}` | `{'InstanceNumber': '26', 'ImagePositionPatient': '[-243.11873376369, -237.73130226135, -89.690040588378]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.89453125, 1.89453125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/InPhase/IMG-0023-00030.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/Ground/IMG-0023-00030.png` | 38/5 | (288, 288) | `{0: 81193, 63: 125, 126: 948, 189: 678}` | `{'InstanceNumber': '30', 'ImagePositionPatient': '[-196.60490894317, -200.13157701492, -53.185523986816]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/DICOM_anon/InPhase/IMG-0016-00028.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/Ground/IMG-0016-00028.png` | 5/11 | (256, 256) | `{0: 60254, 63: 3286, 126: 946, 189: 1050}` | `{'InstanceNumber': '28', 'ImagePositionPatient': '[-223.20717573165, -235.50464630126, -40.849029541015]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.69921875, 1.69921875]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/DICOM_anon/InPhase/IMG-0034-00030.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/Ground/IMG-0034-00030.png` | 8/14 | (288, 288) | `{0: 77011, 63: 2789, 126: 908, 189: 736, 252: 1500}` | `{'InstanceNumber': '30', 'ImagePositionPatient': '[-200.20083713531, -228.6872882843, 27.023380279541]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T1DUAL/DICOM_anon/InPhase/IMG-0005-00026.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T1DUAL/Ground/IMG-0005-00026.png` | 22/8 | (256, 256) | `{0: 63895, 63: 438, 126: 631, 189: 572}` | `{'InstanceNumber': '26', 'ImagePositionPatient': '[-244.4476052504, -277.02752803266, -124.14252062514]', 'ImageOrientationPatient': '[0.9999698996544, 0.0050446353853, -0.0058990521356, -0.0051399474032, 0.999854683876, -0.0162551701068]', 'PixelSpacing': '[2.03125, 2.03125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/DICOM_anon/InPhase/IMG-0016-00044.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/Ground/IMG-0016-00044.png` | 5/19 | (256, 256) | `{0: 59581, 63: 5150, 252: 805}` | `{'InstanceNumber': '44', 'ImagePositionPatient': '[-223.20717573165, -235.50464630126, 31.150970458984]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.69921875, 1.69921875]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T1DUAL/DICOM_anon/InPhase/IMG-0004-00068.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T1DUAL/Ground/IMG-0004-00068.png` | 1/23 | (256, 256) | `{0: 59131, 63: 5982, 189: 58, 252: 365}` | `{'InstanceNumber': '68', 'ImagePositionPatient': '[-232.99694347381, -238.76329636573, 142.26080322266]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.89453125, 1.89453125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/19/T1DUAL/DICOM_anon/InPhase/IMG-0022-00032.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/19/T1DUAL/Ground/IMG-0022-00032.png` | 19/14 | (288, 288) | `{0: 73218, 63: 7837, 189: 369, 252: 1520}` | `{'InstanceNumber': '32', 'ImagePositionPatient': '[-202.23243951797, -211.3795967102, 41.151153564453]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T1DUAL/DICOM_anon/InPhase/IMG-0004-00028.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T1DUAL/Ground/IMG-0004-00028.png` | 1/3 | (256, 256) | `{0: 64335, 63: 519, 126: 542, 189: 140}` | `{'InstanceNumber': '28', 'ImagePositionPatient': '[-232.99694347381, -238.76329636573, 32.260795593262]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.89453125, 1.89453125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T1DUAL/DICOM_anon/InPhase/IMG-0029-00042.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T1DUAL/Ground/IMG-0029-00042.png` | 31/15 | (256, 256) | `{0: 59926, 63: 5192, 252: 418}` | `{'InstanceNumber': '42', 'ImagePositionPatient': '[-216.70591408014, -205.4257787764, 314.16548328102]', 'ImageOrientationPatient': '[1, 0, 0, 0, 0.9999789595604, -0.0064869225025]', 'PixelSpacing': '[1.69921875, 1.69921875]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/InPhase/IMG-0010-00020.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/Ground/IMG-0010-00020.png` | 2/7 | (288, 288) | `{0: 76816, 63: 3939, 126: 966, 189: 1223}` | `{'InstanceNumber': '20', 'ImagePositionPatient': '[-189.17794892194, -242.46268653869, 22.529375062992]', 'ImageOrientationPatient': '[0.9999886155128, 0, -0.0047766352072, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/3/T1DUAL/DICOM_anon/InPhase/IMG-0004-00014.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/3/T1DUAL/Ground/IMG-0004-00014.png` | 3/5 | (288, 288) | `{0: 79584, 63: 1110, 126: 1038, 189: 1212}` | `{'InstanceNumber': '14', 'ImagePositionPatient': '[-185.04817581176, -219.65718841552, -1.7775039672851]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/DICOM_anon/InPhase/IMG-0016-00014.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/Ground/IMG-0016-00014.png` | 5/4 | (256, 256) | `{0: 63928, 63: 1249, 126: 119, 189: 240}` | `{'InstanceNumber': '14', 'ImagePositionPatient': '[-223.20717573165, -235.50464630126, -103.84902954101]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.69921875, 1.69921875]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T1DUAL/DICOM_anon/InPhase/IMG-0029-00050.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T1DUAL/Ground/IMG-0029-00050.png` | 31/19 | (256, 256) | `{0: 64854, 63: 682}` | `{'InstanceNumber': '50', 'ImagePositionPatient': '[-216.70591408014, -205.19224968552, 350.16472034156]', 'ImageOrientationPatient': '[1, 0, 0, 0, 0.9999789595604, -0.0064869225025]', 'PixelSpacing': '[1.69921875, 1.69921875]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`

Aggregate result (pooled pixels; contrast averaged across nonempty-mask slices):
```json
{
  "sequence": "T1-in",
  "n": 20,
  "whole_percentiles": [
    0.0,
    0.0,
    0.0,
    3.0,
    511.0,
    776.0,
    867.0
  ],
  "mask_counts": {
    "0": 1362122,
    "63": 59452,
    "126": 9062,
    "189": 10146,
    "252": 9202
  },
  "organs": {
    "63": {
      "count": 59452,
      "percentiles": [
        91.0,
        98.0,
        125.0,
        307.0,
        472.0,
        537.0,
        566.0
      ],
      "clip_1_99_counts": [
        0,
        6,
        59452
      ],
      "clip_1_99_fraction": 0.00010092175200161475
    },
    "126": {
      "count": 9062,
      "percentiles": [
        59.305,
        63.0,
        76.0,
        237.0,
        549.0,
        921.39,
        1011.39
      ],
      "clip_1_99_counts": [
        0,
        10,
        9062
      ],
      "clip_1_99_fraction": 0.0011035091591260207
    },
    "189": {
      "count": 10146,
      "percentiles": [
        76.0,
        81.0,
        97.0,
        229.0,
        541.75,
        729.55,
        860.925
      ],
      "clip_1_99_counts": [
        0,
        1,
        10146
      ],
      "clip_1_99_fraction": 9.856100926473487e-05
    },
    "252": {
      "count": 9202,
      "percentiles": [
        74.0,
        79.0,
        91.0,
        222.0,
        359.0,
        526.99,
        614.99
      ],
      "clip_1_99_counts": [
        0,
        15,
        9202
      ],
      "clip_1_99_fraction": 0.0016300804173005868
    }
  },
  "ct_window_clipping": {
    "(-160, 240)": {
      "63": 0.6697335665747157,
      "126": 0.4844405208563231,
      "189": 0.45722452197910507,
      "252": 0.4846772440773745
    },
    "(-150, 250)": {
      "63": 0.6213415864899414,
      "126": 0.4512248951666299,
      "189": 0.429726000394244,
      "252": 0.4663116713757879
    },
    "(-400, 400)": {
      "63": 0.22122048038753953,
      "126": 0.14533215625689694,
      "189": 0.140153755174453,
      "252": 0.035427081069332754
    }
  }
}
```
### T1-out
Image / mask | Patient / filename-order index | Shape | Stored labels/counts | Header: instance; position; orientation; spacing; slope/intercept
--- | --- | --- | --- | ---
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/OutPhase/IMG-0023-00025.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/Ground/IMG-0023-00044.png` | 38/12 | (288, 288) | `{0: 76857, 63: 4856, 189: 164, 252: 1067}` | `{'InstanceNumber': '25', 'ImagePositionPatient': '[-196.60490894317, -200.13157701492, -71.185523986816]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00031.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/Ground/IMG-0031-00032.png` | 32/15 | (256, 256) | `{0: 58108, 63: 6011, 189: 759, 252: 658}` | `{'InstanceNumber': '31', 'ImagePositionPatient': '[-240.41821098327, -272.5, -6.6946635246276]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.81640625, 1.81640625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/10/T1DUAL/DICOM_anon/OutPhase/IMG-0046-00053.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/10/T1DUAL/Ground/IMG-0046-00062.png` | 10/26 | (256, 256) | `{0: 61425, 63: 3414, 252: 697}` | `{'InstanceNumber': '53', 'ImagePositionPatient': '[-243.11873376369, -237.73130226135, -12.690042495727]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.89453125, 1.89453125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T1DUAL/DICOM_anon/OutPhase/IMG-0075-00037.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T1DUAL/Ground/IMG-0075-00040.png` | 15/18 | (256, 256) | `{0: 57184, 63: 7070, 252: 1282}` | `{'InstanceNumber': '37', 'ImagePositionPatient': '[-240.67453575134, -221.65929222106, 75.725639343262]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.69921875, 1.69921875]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00017.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/Ground/IMG-0017-00018.png` | 37/8 | (256, 256) | `{0: 60765, 63: 1918, 126: 1157, 189: 1111, 252: 585}` | `{'InstanceNumber': '17', 'ImagePositionPatient': '[-172.4342470169, -185.87104082107, -59.189704895019]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.46484375, 1.46484375]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/OutPhase/IMG-0023-00001.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/Ground/IMG-0023-00020.png` | 38/0 | (288, 288) | `{0: 82770, 126: 174}` | `{'InstanceNumber': '1', 'ImagePositionPatient': '[-196.60490894317, -200.13157701492, -179.18551635742]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T1DUAL/DICOM_anon/OutPhase/IMG-0005-00001.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T1DUAL/Ground/IMG-0005-00002.png` | 34/0 | (256, 256) | `{0: 65488, 189: 48}` | `{'InstanceNumber': '1', 'ImagePositionPatient': '[-219.34210515022, -207.76315784454, -130.33255004882]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.73828125, 1.73828125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00019.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/Ground/IMG-0031-00020.png` | 32/9 | (256, 256) | `{0: 61170, 63: 2940, 126: 667, 189: 759}` | `{'InstanceNumber': '19', 'ImagePositionPatient': '[-240.41821098327, -272.5, -60.694664001464]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.81640625, 1.81640625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00019.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/Ground/IMG-0017-00020.png` | 37/9 | (256, 256) | `{0: 59852, 63: 2878, 126: 864, 189: 870, 252: 1072}` | `{'InstanceNumber': '19', 'ImagePositionPatient': '[-172.4342470169, -185.87104082107, -50.189704895019]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.46484375, 1.46484375]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/DICOM_anon/OutPhase/IMG-0064-00053.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/Ground/IMG-0064-00054.png` | 13/26 | (288, 288) | `{0: 81625, 63: 1319}` | `{'InstanceNumber': '53', 'ImagePositionPatient': '[-196.36827278137, -219.67068481445, 125.49282836914]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/DICOM_anon/OutPhase/IMG-0016-00033.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/Ground/IMG-0016-00038.png` | 5/16 | (256, 256) | `{0: 58157, 63: 5951, 252: 1428}` | `{'InstanceNumber': '33', 'ImagePositionPatient': '[-223.20717573165, -235.50464630126, -13.849029541015]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.69921875, 1.69921875]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/OutPhase/IMG-0010-00005.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/Ground/IMG-0010-00010.png` | 2/2 | (288, 288) | `{0: 80997, 63: 507, 126: 836, 189: 604}` | `{'InstanceNumber': '5', 'ImagePositionPatient': '[-189.47887703778, -242.46268653869, -40.469903959224]', 'ImageOrientationPatient': '[0.9999886155128, 0, -0.0047766352072, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00025.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/Ground/IMG-0017-00026.png` | 37/12 | (256, 256) | `{0: 57543, 63: 5816, 252: 2177}` | `{'InstanceNumber': '25', 'ImagePositionPatient': '[-172.4342470169, -185.87104082107, -23.189704895019]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.46484375, 1.46484375]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/21/T1DUAL/DICOM_anon/OutPhase/IMG-0004-00003.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/21/T1DUAL/Ground/IMG-0004-00028.png` | 21/1 | (256, 256) | `{0: 65269, 189: 267}` | `{'InstanceNumber': '3', 'ImagePositionPatient': '[-252.16397767234, -231.79616812476, -179.72156109044]', 'ImageOrientationPatient': '[0.9997271895409, 0.0094338627532, -0.02136612311, -0.0094312336295, 0.9999554753304, 0.0002238224406]', 'PixelSpacing': '[1.81640625, 1.81640625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00045.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/Ground/IMG-0017-00046.png` | 37/22 | (256, 256) | `{0: 65250, 63: 286}` | `{'InstanceNumber': '45', 'ImagePositionPatient': '[-172.4342470169, -185.87104082107, 66.81029510498]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.46484375, 1.46484375]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00043.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/Ground/IMG-0031-00044.png` | 32/21 | (256, 256) | `{0: 56851, 63: 7159, 252: 1526}` | `{'InstanceNumber': '43', 'ImagePositionPatient': '[-240.41821098327, -272.5, 47.305335998535]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.81640625, 1.81640625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T1DUAL/DICOM_anon/OutPhase/IMG-0005-00031.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T1DUAL/Ground/IMG-0005-00040.png` | 22/15 | (256, 256) | `{0: 61468, 63: 3125, 252: 943}` | `{'InstanceNumber': '31', 'ImagePositionPatient': '[-244.29056846909, -276.58783267438, -97.146556574851]', 'ImageOrientationPatient': '[0.9999698996544, 0.0050446353853, -0.0058990521356, -0.0051399474032, 0.999854683876, -0.0162551701068]', 'PixelSpacing': '[2.03125, 2.03125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00015.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/Ground/IMG-0031-00016.png` | 32/7 | (256, 256) | `{0: 61577, 63: 2391, 126: 887, 189: 681}` | `{'InstanceNumber': '15', 'ImagePositionPatient': '[-240.41821098327, -272.5, -78.694664001464]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.81640625, 1.81640625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/OutPhase/IMG-0010-00013.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/Ground/IMG-0010-00018.png` | 2/6 | (288, 288) | `{0: 77567, 63: 3199, 126: 1061, 189: 1117}` | `{'InstanceNumber': '13', 'ImagePositionPatient': '[-189.30691811444, -242.46268653869, -4.4703159465291]', 'ImageOrientationPatient': '[0.9999886155128, 0, -0.0047766352072, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/DICOM_anon/OutPhase/IMG-0064-00033.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/Ground/IMG-0064-00034.png` | 13/16 | (288, 288) | `{0: 76689, 63: 4766, 252: 1489}` | `{'InstanceNumber': '33', 'ImagePositionPatient': '[-196.36827278137, -219.67068481445, 35.492828369141]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.44097220897674, 1.44097220897674]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`

Aggregate result (pooled pixels; contrast averaged across nonempty-mask slices):
```json
{
  "sequence": "T1-out",
  "n": 20,
  "whole_percentiles": [
    0.0,
    0.0,
    0.0,
    2.0,
    501.0,
    701.0,
    785.0
  ],
  "mask_counts": {
    "0": 1326612,
    "63": 63606,
    "189": 6380,
    "252": 12924,
    "126": 5646
  },
  "organs": {
    "63": {
      "count": 63606,
      "percentiles": [
        1.025,
        28.0,
        89.0,
        211.0,
        528.0,
        719.0,
        882.0
      ],
      "clip_1_99_counts": [
        0,
        156,
        63606
      ],
      "clip_1_99_fraction": 0.002452598811432884
    },
    "189": {
      "count": 6380,
      "percentiles": [
        0.0,
        0.0,
        39.0,
        177.0,
        381.0,
        674.0,
        711.0
      ],
      "clip_1_99_counts": [
        0,
        8,
        6380
      ],
      "clip_1_99_fraction": 0.0012539184952978057
    },
    "252": {
      "count": 12924,
      "percentiles": [
        0.0,
        0.0,
        48.0,
        197.0,
        739.0,
        925.0,
        947.385
      ],
      "clip_1_99_counts": [
        0,
        24,
        12924
      ],
      "clip_1_99_fraction": 0.0018570102135561746
    },
    "126": {
      "count": 5646,
      "percentiles": [
        0.0,
        0.0,
        40.0,
        172.5,
        349.0,
        385.55,
        409.55
      ],
      "clip_1_99_counts": [
        0,
        22,
        5646
      ],
      "clip_1_99_fraction": 0.0038965639390719092
    }
  },
  "ct_window_clipping": {
    "(-160, 240)": {
      "63": 0.33655629972015216,
      "189": 0.33510971786833854,
      "252": 0.4155060352831941,
      "126": 0.2410556145944031
    },
    "(-150, 250)": {
      "63": 0.3101122535609848,
      "189": 0.31865203761755484,
      "252": 0.3865676261219437,
      "126": 0.22051009564293306
    },
    "(-400, 400)": {
      "63": 0.20946137156871994,
      "189": 0.039655172413793106,
      "252": 0.12550294026617145,
      "126": 0.0060219624512929506
    }
  }
}
```
### T2
Image / mask | Patient / filename-order index | Shape | Stored labels/counts | Header: instance; position; orientation; spacing; slope/intercept
--- | --- | --- | --- | ---
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/DICOM_anon/IMG-0020-00019.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/Ground/IMG-0020-00019.png` | 37/12 | (256, 256) | `{0: 57482, 63: 5920, 126: 128, 252: 2006}` | `{'InstanceNumber': '19', 'ImagePositionPatient': '[-172.4342470169, -185.87104082107, 21.81029510498]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.46484375, 1.46484375]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T2SPIR/DICOM_anon/IMG-0073-00016.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T2SPIR/Ground/IMG-0073-00016.png` | 15/15 | (256, 256) | `{0: 57500, 63: 6502, 189: 466, 252: 1068}` | `{'InstanceNumber': '16', 'ImagePositionPatient': '[-232.28246498107, -206.65929222106, 52.690395355225]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.58203125, 1.58203125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T2SPIR/DICOM_anon/IMG-0031-00015.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T2SPIR/Ground/IMG-0031-00015.png` | 8/12 | (288, 288) | `{0: 76907, 63: 2482, 126: 1220, 189: 887, 252: 1448}` | `{'InstanceNumber': '15', 'ImagePositionPatient': '[-197.23244190216, -223.68729019165, 58.37100982666]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.40625, 1.40625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/DICOM_anon/IMG-0002-00030.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/Ground/IMG-0002-00030.png` | 1/17 | (256, 256) | `{0: 56363, 63: 8562, 189: 147, 252: 464}` | `{'InstanceNumber': '30', 'ImagePositionPatient': '[-192.65368556976, -192.7957458496, 146.20783996582]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.54296875, 1.54296875]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T2SPIR/DICOM_anon/IMG-0014-00006.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T2SPIR/Ground/IMG-0014-00006.png` | 5/1 | (256, 256) | `{0: 65152, 63: 384}` | `{'InstanceNumber': '6', 'ImagePositionPatient': '[-218.20717573165, -230.50464630126, -98.581092834472]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.66015625, 1.66015625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T2SPIR/DICOM_anon/IMG-0008-00011.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T2SPIR/Ground/IMG-0008-00011.png` | 22/5 | (256, 256) | `{0: 63789, 126: 947, 189: 800}` | `{'InstanceNumber': '11', 'ImagePositionPatient': '[-212.44267146359, -239.46105495654, -105.87566296802]', 'ImageOrientationPatient': '[0.9999698996544, 0.0050446353853, -0.0058990521356, -0.0051399474032, 0.999854683876, -0.0162551701068]', 'PixelSpacing': '[1.73828125, 1.73828125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/DICOM_anon/IMG-0002-00033.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/Ground/IMG-0002-00033.png` | 1/20 | (256, 256) | `{0: 61377, 63: 4159}` | `{'InstanceNumber': '33', 'ImagePositionPatient': '[-192.65368556976, -192.7957458496, 169.30783081055]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.54296875, 1.54296875]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T2SPIR/DICOM_anon/IMG-0026-00015.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T2SPIR/Ground/IMG-0026-00015.png` | 31/8 | (256, 256) | `{0: 61238, 63: 1952, 126: 480, 189: 721, 252: 1145}` | `{'InstanceNumber': '15', 'ImagePositionPatient': '[-206.70591408014, -195.74709227681, 264.60165165365]', 'ImageOrientationPatient': '[1, 0, 0, 0, 0.9999789595604, -0.0064869225025]', 'PixelSpacing': '[1.62109375, 1.62109375]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T2SPIR/DICOM_anon/IMG-0008-00020.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T2SPIR/Ground/IMG-0008-00020.png` | 34/16 | (256, 256) | `{0: 59891, 63: 5231, 252: 414}` | `{'InstanceNumber': '20', 'ImagePositionPatient': '[-219.34210515022, -207.76315784454, 25.925338745117]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.73828125, 1.73828125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T2SPIR/DICOM_anon/IMG-0008-00013.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T2SPIR/Ground/IMG-0008-00013.png` | 34/9 | (256, 256) | `{0: 62533, 63: 979, 126: 702, 189: 591, 252: 731}` | `{'InstanceNumber': '13', 'ImagePositionPatient': '[-219.34210515022, -207.76315784454, -35.674663543701]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.73828125, 1.73828125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/DICOM_anon/IMG-0030-00038.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/Ground/IMG-0030-00038.png` | 32/29 | (256, 256) | `{0: 65024, 63: 512}` | `{'InstanceNumber': '38', 'ImagePositionPatient': '[-245.61324787139, -272.5, 138.09573364258]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.81640625, 1.81640625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/DICOM_anon/IMG-0030-00017.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/Ground/IMG-0030-00017.png` | 32/8 | (256, 256) | `{0: 61555, 63: 2480, 126: 769, 189: 732}` | `{'InstanceNumber': '17', 'ImagePositionPatient': '[-245.61324787139, -272.5, -50.904258728027]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.81640625, 1.81640625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T2SPIR/DICOM_anon/IMG-0007-00010.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T2SPIR/Ground/IMG-0007-00010.png` | 2/7 | (320, 320) | `{0: 96685, 63: 2812, 126: 1411, 189: 1492}` | `{'InstanceNumber': '10', 'ImagePositionPatient': '[-200.01738891005, -252.46268844604, 30.133110662457]', 'ImageOrientationPatient': '[0.9999886155128, 0, -0.0047766352072, 0, 1, 0]', 'PixelSpacing': '[1.359375, 1.359375]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/36/T2SPIR/DICOM_anon/IMG-0016-00029.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/36/T2SPIR/Ground/IMG-0016-00029.png` | 36/13 | (256, 256) | `{0: 57694, 63: 6188, 252: 1654}` | `{'InstanceNumber': '29', 'ImagePositionPatient': '[-214.11024570465, -201.26826715469, 324.76312255859]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.62109375, 1.62109375]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/DICOM_anon/IMG-0002-00019.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/Ground/IMG-0002-00019.png` | 1/6 | (256, 256) | `{0: 61330, 63: 2132, 126: 1069, 189: 807, 252: 198}` | `{'InstanceNumber': '19', 'ImagePositionPatient': '[-192.65368556976, -192.7957458496, 61.507835388184]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.54296875, 1.54296875]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/DICOM_anon/IMG-0030-00016.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/Ground/IMG-0030-00016.png` | 32/7 | (256, 256) | `{0: 61671, 63: 2231, 126: 883, 189: 751}` | `{'InstanceNumber': '16', 'ImagePositionPatient': '[-245.61324787139, -272.5, -59.904258728027]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.81640625, 1.81640625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/20/T2SPIR/DICOM_anon/IMG-0029-00019.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/20/T2SPIR/Ground/IMG-0029-00019.png` | 20/16 | (320, 320) | `{0: 89964, 63: 9660, 252: 2776}` | `{'InstanceNumber': '19', 'ImagePositionPatient': '[-217.5, -229.77059650421, 60.203628540039]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.359375, 1.359375]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T2SPIR/DICOM_anon/IMG-0014-00014.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T2SPIR/Ground/IMG-0014-00014.png` | 5/9 | (256, 256) | `{0: 61229, 63: 2480, 126: 838, 189: 989}` | `{'InstanceNumber': '14', 'ImagePositionPatient': '[-218.20717573165, -230.50464630126, -26.581090927124]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.66015625, 1.66015625]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/DICOM_anon/IMG-0020-00013.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/Ground/IMG-0020-00013.png` | 37/6 | (256, 256) | `{0: 63085, 126: 1095, 189: 1260, 252: 96}` | `{'InstanceNumber': '13', 'ImagePositionPatient': '[-172.4342470169, -185.87104082107, -32.189704895019]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.46484375, 1.46484375]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`
`src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T2SPIR/DICOM_anon/IMG-0073-00021.dcm` / `src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T2SPIR/Ground/IMG-0073-00021.png` | 15/20 | (256, 256) | `{0: 55701, 63: 8427, 252: 1408}` | `{'InstanceNumber': '21', 'ImagePositionPatient': '[-232.28246498107, -206.65929222106, 97.690399169922]', 'ImageOrientationPatient': '[1, 0, 0, 0, 1, 0]', 'PixelSpacing': '[1.58203125, 1.58203125]', 'RescaleSlope': 'None', 'RescaleIntercept': 'None'}`

Aggregate result (pooled pixels; contrast averaged across nonempty-mask slices):
```json
{
  "sequence": "T2",
  "n": 20,
  "whole_percentiles": [
    0.0,
    0.0,
    0.0,
    5.0,
    455.0,
    838.0,
    942.0
  ],
  "mask_counts": {
    "0": 1296170,
    "63": 73093,
    "126": 9542,
    "252": 13408,
    "189": 9643
  },
  "organs": {
    "63": {
      "count": 73093,
      "percentiles": [
        56.0,
        65.0,
        92.0,
        328.0,
        522.0,
        742.08,
        851.0
      ],
      "clip_1_99_counts": [
        0,
        525,
        73093
      ],
      "clip_1_99_fraction": 0.007182630347639309
    },
    "126": {
      "count": 9542,
      "percentiles": [
        132.0,
        165.41,
        294.0,
        708.0,
        1007.0,
        1116.36,
        1242.59
      ],
      "clip_1_99_counts": [
        0,
        2009,
        9542
      ],
      "clip_1_99_fraction": 0.21054286313141898
    },
    "252": {
      "count": 13408,
      "percentiles": [
        80.0,
        88.0,
        112.35,
        660.0,
        889.0,
        935.0,
        988.965
      ],
      "clip_1_99_counts": [
        0,
        1742,
        13408
      ],
      "clip_1_99_fraction": 0.12992243436754178
    },
    "189": {
      "count": 9643,
      "percentiles": [
        136.0,
        165.0,
        281.0,
        705.0,
        1035.0,
        1530.0,
        1580.79
      ],
      "clip_1_99_counts": [
        0,
        1681,
        9643
      ],
      "clip_1_99_fraction": 0.1743233433578762
    }
  },
  "ct_window_clipping": {
    "(-160, 240)": {
      "63": 0.7630963293338623,
      "126": 0.9681408509746384,
      "252": 0.820032816229117,
      "189": 0.9665041999377787
    },
    "(-150, 250)": {
      "63": 0.7455159864829738,
      "126": 0.9634248585202264,
      "252": 0.8166020286396182,
      "189": 0.9628746240796433
    },
    "(-400, 400)": {
      "63": 0.21405606556031356,
      "126": 0.9116537413540138,
      "252": 0.7778192124105012,
      "189": 0.8675723322617442
    }
  }
}
```
### Six overlay/tissue checks
Numerical tissue overlap cannot prove left/right organ identity. Different flips may all remain inside the body; assess the image overlays and source slice correspondence as well.
```json
[
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00067.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/Ground/liver_GT_066.png",
    "inside_mean": 129.14450073242188,
    "outside_mean": -483.2759094238281,
    "tissue_overlap": {
      "identity": 1.0,
      "flip_x": 0.5970027108587797,
      "flip_y": 0.9457316761290983,
      "transpose": 1.0
    },
    "overlay": "/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_CT_1.png"
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/28/DICOM_anon/IMG-0011-00048.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/28/Ground/liver_GT_047.png",
    "inside_mean": 136.06124877929688,
    "outside_mean": -553.0792236328125,
    "tissue_overlap": {
      "identity": 1.0,
      "flip_x": 0.8771860095389508,
      "flip_y": 0.9822469528351881,
      "transpose": 0.6784137078254725
    },
    "overlay": "/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_CT_2.png"
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/22/DICOM_anon/IMG-0005-00082.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/22/Ground/liver_GT_081.png",
    "inside_mean": 140.53236389160156,
    "outside_mean": -529.8145141601562,
    "tissue_overlap": {
      "identity": 1.0,
      "flip_x": 0.935440709148687,
      "flip_y": 1.0,
      "transpose": 1.0
    },
    "overlay": "/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_CT_3.png"
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/DICOM_anon/InPhase/IMG-0034-00020.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/Ground/IMG-0034-00020.png",
    "inside_mean": 108.74484252929688,
    "outside_mean": 71.86256408691406,
    "tissue_overlap": {
      "identity": 1.0,
      "flip_x": 1.0,
      "flip_y": 1.0,
      "transpose": 0.9785896680416422
    },
    "overlay": "/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_T1-in_1.png"
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/OutPhase/IMG-0023-00025.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/Ground/IMG-0023-00044.png",
    "inside_mean": 281.2350769042969,
    "outside_mean": 81.52081298828125,
    "tissue_overlap": {
      "identity": 0.9664859536717595,
      "flip_x": 0.9617217019878429,
      "flip_y": 0.9669788072942336,
      "transpose": 0.9630359783144407
    },
    "overlay": "/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_T1-out_1.png"
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/DICOM_anon/IMG-0020-00019.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/Ground/IMG-0020-00019.png",
    "inside_mean": 552.01416015625,
    "outside_mean": 95.3088607788086,
    "tissue_overlap": {
      "identity": 1.0,
      "flip_x": 0.9998758380928731,
      "flip_y": 0.9998758380928731,
      "transpose": 0.9956543332505587
    },
    "overlay": "/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_T2_1.png"
  }
]
```
![CT overlay](/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_CT_1.png)
![CT overlay](/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_CT_2.png)
![CT overlay](/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_CT_3.png)
![T1-in overlay](/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_T1-in_1.png)
![T1-out overlay](/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_T1-out_1.png)
![T2 overlay](/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_T2_1.png)

## m6 — Weight-path inventory
Command: `python3 -B audit/scratch/decision_inventory.py` (no tensor reads).
Completed roots: `['/Users/navneetbavineni/STEGO', '/Users/navneetbavineni/Downloads', '/Users/navneetbavineni/Documents']`. Found 22 files. Elapsed 85.03s.
Scope note: cloud traversal targeted first-level STEGO/CHAOS/EAGLE/checkpoint directories. Dependency environments and unrelated cloud trees were excluded after a broad traversal stalled; unsearched trees remain UNVERIFIED.
Incomplete/inaccessible paths: `['85-second traversal budget reached']`.
Path | Bytes | Modified UTC | Local contents
--- | ---: | --- | ---
`/Users/navneetbavineni/Documents/vscode/Python/Cosmetics/SkinCareModel.safetensors` | 29764 | 2025-06-26T22:55:45.558575+00:00 | available; classification pending
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr11_20-18-15/epoch=1-step=1200.ckpt` | 89305473 | 2026-04-11T20:46:03+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr11_20-18-15/epoch=3-step=2400.ckpt` | 89305601 | 2026-04-11T20:53:26+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=1-step=750.ckpt` | 89305219 | 2026-04-13T10:08:28+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=2-step=1500.ckpt` | 89305665 | 2026-04-13T10:12:10+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=3-step=2250.ckpt` | 89306047 | 2026-04-13T10:15:51+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=4-step=3000.ckpt` | 89306429 | 2026-04-13T10:19:35+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=5-step=3750.ckpt` | 89306811 | 2026-04-13T10:23:18+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=6-step=4500.ckpt` | 89307193 | 2026-04-13T10:27:01+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/last.ckpt` | 89307193 | 2026-04-13T10:27:13.252000+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=1-step=750.ckpt` | 89305537 | 2026-04-13T14:32:57+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=3-step=2250.ckpt` | 89306557 | 2026-04-13T14:41:42+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=4-step=3000.ckpt` | 89307003 | 2026-04-13T14:46:06+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=5-step=3750.ckpt` | 89307131 | 2026-04-13T14:50:28+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=6-step=4500.ckpt` | 89307131 | 2026-04-13T14:54:50+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/last.ckpt` | 89307131 | 2026-04-13T14:55:03.944000+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_15-33-24/epoch=1-step=900.ckpt` | 89305347 | 2026-04-13T15:38:31+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/epoch=1-step=600.ckpt` | 89306883 | 2026-04-24T00:13:29.129000+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/epoch=2-step=1200.ckpt` | 89302659 | 2026-04-22T23:18:47.951000+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/mIoU=18.6872.ckpt` | 89312551 | 2026-04-24T10:01:49.195000+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/mIoU=46.8485.ckpt` | 88979833 | 2026-04-25T15:38:17.282000+00:00 | ONLINE-ONLY; keys UNVERIFIED
`/Users/navneetbavineni/STEGO/models/model_epoch_0720_iter_085000.pth` | 502621184 | 2026-04-02T13:58:16.173000+00:00 | available; classification pending

## M5 — Physical-position pairing correction and MRI percentile alternatives

Command: `python3 -B audit/scratch/decision_geometry.py`; reuses the same selected image paths. Earlier sorted-order T1-out organ statistics/overlay are a diagnostic of wrong pairing, **not valid organ intensity estimates** where this check changes/rejects the pair. No source files or masks were corrected. T1-out masks are associated with the in-phase image of the same position/orientation (tolerance.01mm/1e-5), requiring exactly one match.

```json
[
  {
    "sequence": "CT",
    "matched_samples": 20,
    "unmatched_samples": [],
    "changed_pairings": [],
    "all_sampled_image_mask_stems_equal": null,
    "organ_percentiles": {
      "255": [
        10.0,
        36.0,
        71.0,
        126.0,
        187.0,
        238.0,
        253.0
      ]
    },
    "clipping_fraction": {
      "(255, 99)": 6.872773507749052e-05,
      "(255, 99.5)": 1.603647151808112e-05,
      "(255, 99.9)": 0.0
    }
  },
  {
    "sequence": "T1-in",
    "matched_samples": 20,
    "unmatched_samples": [],
    "changed_pairings": [],
    "all_sampled_image_mask_stems_equal": true,
    "organ_percentiles": {
      "63": [
        91.0,
        98.0,
        125.0,
        307.0,
        472.0,
        537.0,
        566.0
      ],
      "126": [
        59.305,
        63.0,
        76.0,
        237.0,
        549.0,
        921.39,
        1011.39
      ],
      "189": [
        76.0,
        81.0,
        97.0,
        229.0,
        541.75,
        729.55,
        860.925
      ],
      "252": [
        74.0,
        79.0,
        91.0,
        222.0,
        359.0,
        526.99,
        614.99
      ]
    },
    "clipping_fraction": {
      "(63, 99)": 0.00010092175200161475,
      "(63, 99.5)": 0.00010092175200161475,
      "(63, 99.9)": 5.0460876000807376e-05,
      "(126, 99)": 0.0011035091591260207,
      "(126, 99.5)": 0.00022070183182520416,
      "(126, 99.9)": 0.0,
      "(189, 99)": 9.856100926473487e-05,
      "(189, 99.5)": 0.0,
      "(189, 99.9)": 0.0,
      "(252, 99)": 0.0016300804173005868,
      "(252, 99.5)": 0.00021734405564007825,
      "(252, 99.9)": 0.0
    }
  },
  {
    "sequence": "T1-out",
    "matched_samples": 18,
    "unmatched_samples": [
      "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/OutPhase/IMG-0023-00001.dcm",
      "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/21/T1DUAL/DICOM_anon/OutPhase/IMG-0004-00003.dcm"
    ],
    "changed_pairings": [
      [
        "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/OutPhase/IMG-0023-00025.dcm",
        "IMG-0023-00044.png",
        "IMG-0023-00026.png"
      ],
      [
        "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/10/T1DUAL/DICOM_anon/OutPhase/IMG-0046-00053.dcm",
        "IMG-0046-00062.png",
        "IMG-0046-00054.png"
      ],
      [
        "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T1DUAL/DICOM_anon/OutPhase/IMG-0075-00037.dcm",
        "IMG-0075-00040.png",
        "IMG-0075-00038.png"
      ],
      [
        "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/DICOM_anon/OutPhase/IMG-0016-00033.dcm",
        "IMG-0016-00038.png",
        "IMG-0016-00034.png"
      ],
      [
        "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/OutPhase/IMG-0010-00005.dcm",
        "IMG-0010-00010.png",
        "IMG-0010-00006.png"
      ],
      [
        "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T1DUAL/DICOM_anon/OutPhase/IMG-0005-00031.dcm",
        "IMG-0005-00040.png",
        "IMG-0005-00032.png"
      ],
      [
        "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/OutPhase/IMG-0010-00013.dcm",
        "IMG-0010-00018.png",
        "IMG-0010-00014.png"
      ]
    ],
    "all_sampled_image_mask_stems_equal": null,
    "organ_percentiles": {
      "126": [
        0.0,
        0.0,
        40.0,
        175.0,
        333.0,
        439.0,
        467.76
      ],
      "189": [
        0.0,
        0.0,
        42.0,
        184.0,
        416.95,
        529.98,
        565.0
      ],
      "63": [
        24.0,
        42.0,
        99.0,
        206.0,
        515.0,
        571.0,
        592.0
      ],
      "252": [
        0.0,
        0.0,
        50.0,
        155.0,
        361.0,
        456.0,
        499.01
      ]
    },
    "clipping_fraction": {
      "(126, 99)": 0.0009638554216867469,
      "(126, 99.5)": 0.0001606425702811245,
      "(126, 99.9)": 0.0,
      "(189, 99)": 0.00014701558365186709,
      "(189, 99.5)": 0.0,
      "(189, 99.9)": 0.0,
      "(63, 99)": 0.00018326094525995565,
      "(63, 99.5)": 0.00010995656715597339,
      "(63, 99.9)": 3.665218905199113e-05,
      "(252, 99)": 0.0002884892778151745,
      "(252, 99.5)": 9.616309260505818e-05,
      "(252, 99.9)": 0.0
    }
  },
  {
    "sequence": "T2",
    "matched_samples": 20,
    "unmatched_samples": [],
    "changed_pairings": [],
    "all_sampled_image_mask_stems_equal": true,
    "organ_percentiles": {
      "63": [
        56.0,
        65.0,
        92.0,
        328.0,
        522.0,
        742.08,
        851.0
      ],
      "126": [
        132.0,
        165.41,
        294.0,
        708.0,
        1007.0,
        1116.36,
        1242.59
      ],
      "252": [
        80.0,
        88.0,
        112.35,
        660.0,
        889.0,
        935.0,
        988.965
      ],
      "189": [
        136.0,
        165.0,
        281.0,
        705.0,
        1035.0,
        1530.0,
        1580.79
      ]
    },
    "clipping_fraction": {
      "(63, 99)": 0.007182630347639309,
      "(63, 99.5)": 0.002845689737731383,
      "(63, 99.9)": 0.0004241172205272735,
      "(126, 99)": 0.21054286313141898,
      "(126, 99.5)": 0.0693774889960176,
      "(126, 99.9)": 0.005868790609935024,
      "(252, 99)": 0.12992243436754178,
      "(252, 99.5)": 0.04825477326968974,
      "(252, 99.9)": 0.0016408114558472554,
      "(189, 99)": 0.1743233433578762,
      "(189, 99.5)": 0.062325002592554184,
      "(189, 99.9)": 0.008192471222648554
    }
  }
]
```

### Sample physical geometry evidence

```json
[
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00067.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/Ground/liver_GT_066.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      0.9999997999999977,
      1.0
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -213.5,
      -64.5,
      -84.7000122
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/28/DICOM_anon/IMG-0011-00048.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/28/Ground/liver_GT_047.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.5999699999999848,
      1.6001000000001113
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -167.5,
      -92.5,
      1494.80005
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/22/DICOM_anon/IMG-0005-00082.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/22/Ground/liver_GT_081.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.5999699999999848,
      1.6001000000001113
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -183.0,
      -89.0,
      1443.90002
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/25/DICOM_anon/IMG-0008-00073.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/25/Ground/liver_GT_072.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.0,
      1.0
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -149.0,
      4.0,
      -179.5
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00227.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/Ground/liver_GT_226.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      0.9999997999999977,
      1.0
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -213.5,
      -64.5,
      -244.700012
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/28/DICOM_anon/IMG-0011-00102.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/28/Ground/liver_GT_101.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.5999699999999848,
      1.6001000000001113
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -167.5,
      -92.5,
      1408.40002
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/1/DICOM_anon/i0093,0000b.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/1/Ground/liver_GT_093.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.5,
      1.5
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -167.7859,
      -176.0438,
      -223.5
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/14/DICOM_anon/i0006,0000b.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/14/Ground/liver_GT_006.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.5999699999999848,
      1.6001000000001113
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -139.5,
      -32.5,
      1659.40002
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/8/DICOM_anon/i0087,0000b.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/8/Ground/liver_GT_087.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.5999699999999848,
      1.6001000000001113
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -176.0,
      -95.0,
      1540.30005
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/24/DICOM_anon/IMG-0007-00033.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/24/Ground/liver_GT_032.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      2.0,
      2.0000000000000284
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -176.0,
      -1.0,
      -299.200012
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00084.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/Ground/liver_GT_083.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      0.9999997999999977,
      1.0
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -213.5,
      -64.5,
      -101.700012
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/29/DICOM_anon/IMG-0012-00099.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/29/Ground/liver_GT_098.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      0.9999999999999716,
      1.0
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -196.0,
      -28.0,
      -166.500031
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/10/DICOM_anon/i0019,0000b.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/10/Ground/liver_GT_019.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.5999699999999848,
      1.6001000000001113
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -159.5,
      -98.5,
      1652.09998
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/10/DICOM_anon/i0073,0000b.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/10/Ground/liver_GT_073.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.5999699999999848,
      1.6001000000001113
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -159.5,
      -98.5,
      1565.69995
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00209.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/Ground/liver_GT_208.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      0.9999997999999977,
      1.0
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -213.5,
      -64.5,
      -226.700012
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/16/DICOM_anon/i0055,0000b.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/16/Ground/liver_GT_055.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      2.0,
      2.0
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -190.5,
      -103.5,
      1591.0
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/21/DICOM_anon/IMG-0001-00110.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/21/Ground/liver_GT_109.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.0,
      1.0000002999999964
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -211.325459,
      -74.0918655,
      -99.4999847
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/6/DICOM_anon/i0027,0000b.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/6/Ground/liver_GT_027.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.5,
      1.5
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -169.2873,
      -166.2845,
      -81.0
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/26/DICOM_anon/IMG-0009-00109.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/26/Ground/liver_GT_108.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      1.5999699999999848,
      1.6001000000001113
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -190.0,
      -106.0,
      1425.69995
    ]
  },
  {
    "sequence": "CT",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/16/DICOM_anon/i0023,0000b.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/16/Ground/liver_GT_023.png",
    "filename_order_physical_direction": "descending",
    "physical_step_range": [
      2.0,
      2.0
    ],
    "matrix": [
      512,
      512
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -190.5,
      -103.5,
      1655.0
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/DICOM_anon/InPhase/IMG-0034-00020.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/Ground/IMG-0034-00020.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      5.999996185303004,
      6.000000000000002
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -200.20083713531,
      -228.6872882843,
      -2.9766197204589
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/21/T1DUAL/DICOM_anon/InPhase/IMG-0004-00056.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/21/T1DUAL/Ground/IMG-0004-00056.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      7.999995058427899,
      8.000002686085253
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      0.9997271895409,
      0.0094338627532,
      -0.02136612311,
      -0.0094312336295,
      0.9999554753304,
      0.0002238224406
    ],
    "position": [
      -247.71958260703,
      -231.80079654464,
      28.230947835946
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T1DUAL/DICOM_anon/InPhase/IMG-0005-00060.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T1DUAL/Ground/IMG-0005-00060.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      5.499992370599998,
      5.5000000000000036
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -219.34210515022,
      -207.76315784454,
      29.167442321777
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/20/T1DUAL/DICOM_anon/InPhase/IMG-0027-00026.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/20/T1DUAL/Ground/IMG-0027-00026.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999998092650998,
      9.000001907348
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -207.49999809265,
      -219.77059459686,
      6.2036304473877
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/InPhase/IMG-0023-00038.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/Ground/IMG-0023-00038.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999998092652,
      9.000001907349002
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -196.60490894317,
      -200.13157701492,
      -17.185522079467
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/39/T1DUAL/DICOM_anon/InPhase/IMG-0027-00030.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/39/T1DUAL/Ground/IMG-0027-00030.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999996681666389,
      9.00000431263463
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      0.999110519886,
      -0.0421353019773,
      0.0016553440364,
      0.0421352423728,
      0.9991118907928,
      6.98104195e-05
    ],
    "position": [
      -215.25214416347,
      -206.80334907211,
      -1.9482051935519
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/10/T1DUAL/DICOM_anon/InPhase/IMG-0046-00026.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/10/T1DUAL/Ground/IMG-0046-00026.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      5.499998092650998,
      5.500007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -243.11873376369,
      -237.73130226135,
      -89.690040588378
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/InPhase/IMG-0023-00030.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/Ground/IMG-0023-00030.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999998092652,
      9.000001907349002
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -196.60490894317,
      -200.13157701492,
      -53.185523986816
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/DICOM_anon/InPhase/IMG-0016-00028.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/Ground/IMG-0016-00028.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999999999994998,
      9.000000000000007
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -223.20717573165,
      -235.50464630126,
      -40.849029541015
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/DICOM_anon/InPhase/IMG-0034-00030.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/Ground/IMG-0034-00030.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      5.999996185303004,
      6.000000000000002
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -200.20083713531,
      -228.6872882843,
      27.023380279541
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T1DUAL/DICOM_anon/InPhase/IMG-0005-00026.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T1DUAL/Ground/IMG-0005-00026.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999988083024306,
      9.000003370584437
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      0.9999698996544,
      0.0050446353853,
      -0.0058990521356,
      -0.0051399474032,
      0.999854683876,
      -0.0162551701068
    ],
    "position": [
      -244.4476052504,
      -277.02752803266,
      -124.14252062514
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/DICOM_anon/InPhase/IMG-0016-00044.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/Ground/IMG-0016-00044.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999999999994998,
      9.000000000000007
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -223.20717573165,
      -235.50464630126,
      31.150970458984
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T1DUAL/DICOM_anon/InPhase/IMG-0004-00068.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T1DUAL/Ground/IMG-0004-00068.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      5.499999999997996,
      5.500007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -232.99694347381,
      -238.76329636573,
      142.26080322266
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/19/T1DUAL/DICOM_anon/InPhase/IMG-0022-00032.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/19/T1DUAL/Ground/IMG-0022-00032.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999999999997002,
      9.000000000000002
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -202.23243951797,
      -211.3795967102,
      41.151153564453
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T1DUAL/DICOM_anon/InPhase/IMG-0004-00028.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T1DUAL/Ground/IMG-0004-00028.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      5.499999999997996,
      5.500007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -232.99694347381,
      -238.76329636573,
      32.260795593262
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T1DUAL/DICOM_anon/InPhase/IMG-0029-00042.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T1DUAL/Ground/IMG-0029-00042.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999975737220382,
      9.000006260354724
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      0.9999789595604,
      -0.0064869225025
    ],
    "position": [
      -216.70591408014,
      -205.4257787764,
      314.16548328102
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/InPhase/IMG-0010-00020.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/Ground/IMG-0010-00020.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999996075569072,
      9.000011334179277
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      0.9999886155128,
      0.0,
      -0.0047766352072,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -189.17794892194,
      -242.46268653869,
      22.529375062992
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/3/T1DUAL/DICOM_anon/InPhase/IMG-0004-00014.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/3/T1DUAL/Ground/IMG-0004-00014.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999999999994998,
      9.000007629400002
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -185.04817581176,
      -219.65718841552,
      -1.7775039672851
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/DICOM_anon/InPhase/IMG-0016-00014.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/Ground/IMG-0016-00014.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999999999994998,
      9.000000000000007
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -223.20717573165,
      -235.50464630126,
      -103.84902954101
    ]
  },
  {
    "sequence": "T1-in",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T1DUAL/DICOM_anon/InPhase/IMG-0029-00050.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T1DUAL/Ground/IMG-0029-00050.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999975737220382,
      9.000006260354724
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      0.9999789595604,
      -0.0064869225025
    ],
    "position": [
      -216.70591408014,
      -205.19224968552,
      350.16472034156
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/OutPhase/IMG-0023-00025.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/Ground/IMG-0023-00026.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.000001907349002
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -196.60490894317,
      -200.13157701492,
      -71.185523986816
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00031.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/Ground/IMG-0031-00032.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.000007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -240.41821098327,
      -272.5,
      -6.6946635246276
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/10/T1DUAL/DICOM_anon/OutPhase/IMG-0046-00053.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/10/T1DUAL/Ground/IMG-0046-00054.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      5.499998092650998,
      5.500007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -243.11873376369,
      -237.73130226135,
      -12.690042495727
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T1DUAL/DICOM_anon/OutPhase/IMG-0075-00037.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T1DUAL/Ground/IMG-0075-00038.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999999999997996,
      9.000007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -240.67453575134,
      -221.65929222106,
      75.725639343262
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00017.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/Ground/IMG-0017-00018.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.000007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -172.4342470169,
      -185.87104082107,
      -59.189704895019
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T1DUAL/DICOM_anon/OutPhase/IMG-0005-00001.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T1DUAL/Ground/IMG-0005-00002.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      5.499992370599998,
      5.5000000000000036
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -219.34210515022,
      -207.76315784454,
      -130.33255004882
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00019.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/Ground/IMG-0031-00020.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.000007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -240.41821098327,
      -272.5,
      -60.694664001464
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00019.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/Ground/IMG-0017-00020.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.000007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -172.4342470169,
      -185.87104082107,
      -50.189704895019
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/DICOM_anon/OutPhase/IMG-0064-00053.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/Ground/IMG-0064-00054.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999998092652003,
      9.000001907348999
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -196.36827278137,
      -219.67068481445,
      125.49282836914
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/DICOM_anon/OutPhase/IMG-0016-00033.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/Ground/IMG-0016-00034.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999999999994998,
      9.000000000000007
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -223.20717573165,
      -235.50464630126,
      -13.849029541015
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/OutPhase/IMG-0010-00005.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/Ground/IMG-0010-00006.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999996075563075,
      9.000011334179277
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      0.9999886155128,
      0.0,
      -0.0047766352072,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -189.47887703778,
      -242.46268653869,
      -40.469903959224
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00025.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/Ground/IMG-0017-00026.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.000007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -172.4342470169,
      -185.87104082107,
      -23.189704895019
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00045.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/Ground/IMG-0017-00046.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.000007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -172.4342470169,
      -185.87104082107,
      66.81029510498
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00043.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/Ground/IMG-0031-00044.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.000007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -240.41821098327,
      -272.5,
      47.305335998535
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T1DUAL/DICOM_anon/OutPhase/IMG-0005-00031.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T1DUAL/Ground/IMG-0005-00032.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999988083024306,
      9.000003370584437
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      0.9999698996544,
      0.0050446353853,
      -0.0058990521356,
      -0.0051399474032,
      0.999854683876,
      -0.0162551701068
    ],
    "position": [
      -244.29056846909,
      -276.58783267438,
      -97.146556574851
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00015.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/Ground/IMG-0031-00016.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.000007629400002
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -240.41821098327,
      -272.5,
      -78.694664001464
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/OutPhase/IMG-0010-00013.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/Ground/IMG-0010-00014.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999996075563075,
      9.000011334179277
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      0.9999886155128,
      0.0,
      -0.0047766352072,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -189.30691811444,
      -242.46268653869,
      -4.4703159465291
    ]
  },
  {
    "sequence": "T1-out",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/DICOM_anon/OutPhase/IMG-0064-00033.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/Ground/IMG-0064-00034.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999998092652003,
      9.000001907348999
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -196.36827278137,
      -219.67068481445,
      35.492828369141
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/DICOM_anon/IMG-0020-00019.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/Ground/IMG-0020-00019.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999999999999499,
      9.000000000000007
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -172.4342470169,
      -185.87104082107,
      21.81029510498
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T2SPIR/DICOM_anon/IMG-0073-00016.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T2SPIR/Ground/IMG-0073-00016.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999996185302997,
      9.000003814697003
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -232.28246498107,
      -206.65929222106,
      52.690395355225
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T2SPIR/DICOM_anon/IMG-0031-00015.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T2SPIR/Ground/IMG-0031-00015.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      7.999992370599998,
      8.000000000000002
    ],
    "matrix": [
      288,
      288
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -197.23244190216,
      -223.68729019165,
      58.37100982666
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/DICOM_anon/IMG-0002-00030.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/Ground/IMG-0002-00030.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      7.699996948239004,
      7.7000122070299994
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -192.65368556976,
      -192.7957458496,
      146.20783996582
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T2SPIR/DICOM_anon/IMG-0014-00006.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T2SPIR/Ground/IMG-0014-00006.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999998092651001,
      9.000001907348004
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -218.20717573165,
      -230.50464630126,
      -98.581092834472
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T2SPIR/DICOM_anon/IMG-0008-00011.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T2SPIR/Ground/IMG-0008-00011.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.699997374740555,
      8.700005018524678
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      0.9999698996544,
      0.0050446353853,
      -0.0058990521356,
      -0.0051399474032,
      0.999854683876,
      -0.0162551701068
    ],
    "position": [
      -212.44267146359,
      -239.46105495654,
      -105.87566296802
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/DICOM_anon/IMG-0002-00033.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/Ground/IMG-0002-00033.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      7.699996948239004,
      7.7000122070299994
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -192.65368556976,
      -192.7957458496,
      169.30783081055
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T2SPIR/DICOM_anon/IMG-0026-00015.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/31/T2SPIR/Ground/IMG-0026-00015.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999975737220382,
      9.000006260354667
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      0.9999789595604,
      -0.0064869225025
    ],
    "position": [
      -206.70591408014,
      -195.74709227681,
      264.60165165365
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T2SPIR/DICOM_anon/IMG-0008-00020.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T2SPIR/Ground/IMG-0008-00020.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.799995422360993,
      8.800003051760001
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -219.34210515022,
      -207.76315784454,
      25.925338745117
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T2SPIR/DICOM_anon/IMG-0008-00013.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T2SPIR/Ground/IMG-0008-00013.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.799995422360993,
      8.800003051760001
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -219.34210515022,
      -207.76315784454,
      -35.674663543701
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/DICOM_anon/IMG-0030-00038.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/Ground/IMG-0030-00038.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.0000000000003
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -245.61324787139,
      -272.5,
      138.09573364258
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/DICOM_anon/IMG-0030-00017.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/Ground/IMG-0030-00017.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.0000000000003
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -245.61324787139,
      -272.5,
      -50.904258728027
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T2SPIR/DICOM_anon/IMG-0007-00010.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T2SPIR/Ground/IMG-0007-00010.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.99999607556299,
      9.000011334189367
    ],
    "matrix": [
      320,
      320
    ],
    "iop": [
      0.9999886155128,
      0.0,
      -0.0047766352072,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -200.01738891005,
      -252.46268844604,
      30.133110662457
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/36/T2SPIR/DICOM_anon/IMG-0016-00029.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/36/T2SPIR/Ground/IMG-0016-00029.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      9.0,
      9.00001525879
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -214.11024570465,
      -201.26826715469,
      324.76312255859
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/DICOM_anon/IMG-0002-00019.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/1/T2SPIR/Ground/IMG-0002-00019.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      7.699996948239004,
      7.7000122070299994
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -192.65368556976,
      -192.7957458496,
      61.507835388184
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/DICOM_anon/IMG-0030-00016.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T2SPIR/Ground/IMG-0030-00016.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999992370610002,
      9.0000000000003
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -245.61324787139,
      -272.5,
      -59.904258728027
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/20/T2SPIR/DICOM_anon/IMG-0029-00019.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/20/T2SPIR/Ground/IMG-0029-00019.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999998092650998,
      9.000001907348
    ],
    "matrix": [
      320,
      320
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -217.5,
      -229.77059650421,
      60.203628540039
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T2SPIR/DICOM_anon/IMG-0014-00014.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T2SPIR/Ground/IMG-0014-00014.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999998092651001,
      9.000001907348004
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -218.20717573165,
      -230.50464630126,
      -26.581090927124
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/DICOM_anon/IMG-0020-00013.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/Ground/IMG-0020-00013.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999999999999499,
      9.000000000000007
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -172.4342470169,
      -185.87104082107,
      -32.189704895019
    ]
  },
  {
    "sequence": "T2",
    "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T2SPIR/DICOM_anon/IMG-0073-00021.dcm",
    "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T2SPIR/Ground/IMG-0073-00021.png",
    "filename_order_physical_direction": "ascending",
    "physical_step_range": [
      8.999996185302997,
      9.000003814697003
    ],
    "matrix": [
      256,
      256
    ],
    "iop": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0
    ],
    "position": [
      -232.28246498107,
      -206.65929222106,
      97.690399169922
    ]
  }
]
```

![Out-phase physical match](/Users/navneetbavineni/STEGO/audit/scratch/overlays/decision_T1-out_physical.png)

## m6 — Targeted cloud checkpoint inventory
Command: `python3 -B audit/scratch/decision_cloud_weights.py`. Targeted checkpoint/model folders avoid stalled traversal of unrelated cloud logs.
Path | Bytes | Modified UTC | State keys
--- | ---: | --- | ---
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_20-09-06_vit_small_chaos_EAGLE/epoch=00-step=00000700-test/cluster/mIoU=11.48.ckpt` | 122479839 | 2026-04-09T11:28:25+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_20-09-06_vit_small_chaos_EAGLE/epoch=01-step=00002100-test/cluster/mIoU=13.89.ckpt` | 122480795 | 2026-04-09T11:56:36+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_20-09-06_vit_small_chaos_EAGLE/epoch=00-step=00001400-test/cluster/mIoU=11.48.ckpt` | 122480349 | 2026-04-09T11:42:28+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr22_20-58-15_vit_small_chaos_EAGLE/epoch=00-step=00003500-test/cluster/mIoU=5.88.ckpt` | 122481177 | 2026-04-22T12:36:29+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr22_20-58-15_vit_small_chaos_EAGLE/epoch=00-step=00000700-test/cluster/mIoU=6.20.ckpt` | 122479775 | 2026-04-22T12:07:16+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr22_20-58-15_vit_small_chaos_EAGLE/epoch=00-step=00004200-test/cluster/mIoU=6.00.ckpt` | 122481177 | 2026-04-22T12:45:27+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr22_20-58-15_vit_small_chaos_EAGLE/epoch=00-step=00002100-test/cluster/mIoU=5.78.ckpt` | 122480603 | 2026-04-22T12:18:39+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr22_20-58-15_vit_small_chaos_EAGLE/epoch=00-step=00001400-test/cluster/mIoU=6.20.ckpt` | 122480157 | 2026-04-22T12:09:41+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr06_02-46-05_vit_small_chaos_EAGLE/epoch=00-step=00000700-test/cluster/mIoU=11.39.ckpt` | 122479519 | 2026-04-05T17:48:36+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr06_02-46-05_vit_small_chaos_EAGLE/epoch=01-step=00002800-test/cluster/mIoU=11.17.ckpt` | 122480857 | 2026-04-05T18:14:23+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr06_02-46-05_vit_small_chaos_EAGLE/epoch=01-step=00002100-test/cluster/mIoU=11.15.ckpt` | 122480411 | 2026-04-05T18:11:57+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr06_02-46-05_vit_small_chaos_EAGLE/epoch=00-step=00001400-test/cluster/mIoU=11.39.ckpt` | 122479965 | 2026-04-05T18:00:31+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr06_02-46-05_vit_small_chaos_EAGLE/epoch=01-step=00003500-test/cluster/mIoU=10.56.ckpt` | 122481113 | 2026-04-05T18:16:48+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_07-17-34_vit_small_chaos_EAGLE/epoch=00-step=00000700-test/cluster/mIoU=11.05.ckpt` | 122479839 | 2026-04-08T22:19:54+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_07-17-34_vit_small_chaos_EAGLE/epoch=01-step=00002100-test/cluster/mIoU=11.44.ckpt` | 122480795 | 2026-04-08T22:23:45+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_07-17-34_vit_small_chaos_EAGLE/epoch=02-step=00004200-test/cluster/mIoU=10.81.ckpt` | 122481497 | 2026-04-08T22:29:54+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_07-17-34_vit_small_chaos_EAGLE/epoch=00-step=00001400-test/cluster/mIoU=11.05.ckpt` | 122480349 | 2026-04-08T22:21:25+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_07-17-34_vit_small_chaos_EAGLE/epoch=01-step=00003500-test/cluster/mIoU=10.81.ckpt` | 122481497 | 2026-04-08T22:28:23+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr10_07-56-17_vit_small_chaos_EAGLE/epoch=02-step=00008400-test/cluster/mIoU=11.03.ckpt` | 122481177 | 2026-04-09T23:22:59+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr10_07-56-17_vit_small_chaos_EAGLE/epoch=02-step=00007700-test/cluster/mIoU=11.20.ckpt` | 122481177 | 2026-04-09T23:20:41+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr10_07-56-17_vit_small_chaos_EAGLE/epoch=00-step=00000700-test/cluster/mIoU=11.14.ckpt` | 122479775 | 2026-04-09T22:58:41+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr10_07-56-17_vit_small_chaos_EAGLE/epoch=00-step=00001400-test/cluster/mIoU=11.14.ckpt` | 122480157 | 2026-04-09T23:00:10+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr10_07-56-17_vit_small_chaos_EAGLE/epoch=02-step=00007000-test/cluster/mIoU=11.11.ckpt` | 122481177 | 2026-04-09T23:18:24+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr10_04-14-00_vit_small_chaos_EAGLE/epoch=02-step=00008400-test/cluster/mIoU=11.70.ckpt` | 122481177 | 2026-04-09T20:03:27+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr10_04-14-00_vit_small_chaos_EAGLE/epoch=01-step=00004900-test/cluster/mIoU=11.34.ckpt` | 122481177 | 2026-04-09T19:51:56+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr10_04-14-00_vit_small_chaos_EAGLE/epoch=02-step=00009100-test/cluster/mIoU=11.46.ckpt` | 122481177 | 2026-04-09T20:05:44+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr10_04-14-00_vit_small_chaos_EAGLE/epoch=03-step=00009800-test/cluster/mIoU=11.46.ckpt` | 122481177 | 2026-04-09T20:07:12+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr10_04-14-00_vit_small_chaos_EAGLE/epoch=01-step=00005600-test/cluster/mIoU=12.38.ckpt` | 122481177 | 2026-04-09T19:54:15+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_04-58-18_vit_small_chaos_EAGLE/epoch=00-step=00000700-test/cluster/mIoU=11.51.ckpt` | 122479839 | 2026-04-08T20:00:38+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_04-58-18_vit_small_chaos_EAGLE/epoch=01-step=00002800-test/cluster/mIoU=10.47.ckpt` | 122481241 | 2026-04-08T20:26:35+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_04-58-18_vit_small_chaos_EAGLE/epoch=01-step=00002100-test/cluster/mIoU=11.21.ckpt` | 122480795 | 2026-04-08T20:24:19+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_04-58-18_vit_small_chaos_EAGLE/epoch=00-step=00001400-test/cluster/mIoU=11.51.ckpt` | 122480349 | 2026-04-08T20:12:44+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints/chaos/chaos_Apr09_04-58-18_vit_small_chaos_EAGLE/epoch=01-step=00003500-test/cluster/mIoU=10.21.ckpt` | 122481497 | 2026-04-08T20:28:55+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/epoch=1-step=600.ckpt` | 89306883 | 2026-04-24T00:13:29.129000+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/mIoU=18.6872.ckpt` | 89312551 | 2026-04-24T10:01:49.195000+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/epoch=2-step=1200.ckpt` | 89302659 | 2026-04-22T23:18:47.951000+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/mIoU=46.8485.ckpt` | 88979833 | 2026-04-25T15:38:17.282000+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=4-step=3000.ckpt` | 89306429 | 2026-04-13T10:19:35+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=6-step=4500.ckpt` | 89307193 | 2026-04-13T10:27:01+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/last.ckpt` | 89307193 | 2026-04-13T10:27:13.252000+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=3-step=2250.ckpt` | 89306047 | 2026-04-13T10:15:51+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=5-step=3750.ckpt` | 89306811 | 2026-04-13T10:23:18+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=2-step=1500.ckpt` | 89305665 | 2026-04-13T10:12:10+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=1-step=750.ckpt` | 89305219 | 2026-04-13T10:08:28+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_15-33-24/epoch=1-step=900.ckpt` | 89305347 | 2026-04-13T15:38:31+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=4-step=3000.ckpt` | 89307003 | 2026-04-13T14:46:06+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=6-step=4500.ckpt` | 89307131 | 2026-04-13T14:54:50+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/last.ckpt` | 89307131 | 2026-04-13T14:55:03.944000+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=3-step=2250.ckpt` | 89306557 | 2026-04-13T14:41:42+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=5-step=3750.ckpt` | 89307131 | 2026-04-13T14:50:28+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=1-step=750.ckpt` | 89305537 | 2026-04-13T14:32:57+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr11_20-18-15/epoch=1-step=1200.ckpt` | 89305473 | 2026-04-11T20:46:03+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr11_20-18-15/epoch=3-step=2400.ckpt` | 89305601 | 2026-04-11T20:53:26+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=1-step=step=2400-miou=test/cluster/mIoU=18.0785.ckpt` | 89305525 | 2026-04-29T21:48:39.782825+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=2-step=step=3900-miou=test/cluster/mIoU=18.0785.ckpt` | 89307563 | 2026-04-30T00:34:16.265982+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=0-step=step=600-miou=test/cluster/mIoU=14.8776.ckpt` | 89303105 | 2026-04-29T17:25:47.577215+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=2-step=step=3000-miou=test/cluster/mIoU=18.0785.ckpt` | 89306353 | 2026-04-29T22:31:55.568938+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=0-step=step=300-miou=test/cluster/mIoU=14.8776.ckpt` | 89302723 | 2026-04-29T17:24:57.378928+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=1-step=step=2100-miou=test/cluster/mIoU=18.0785.ckpt` | 89305143 | 2026-04-29T21:02:19.663217+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=3-step=step=4500-miou=test/cluster/mIoU=18.0785.ckpt` | 89308327 | 2026-04-30T01:15:50.594778+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=0-step=step=900-miou=test/cluster/mIoU=17.6641.ckpt` | 89303551 | 2026-04-29T18:02:27.873847+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=2-step=step=3600-miou=test/cluster/mIoU=18.0785.ckpt` | 89307181 | 2026-04-29T23:53:59.587387+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=2-step=step=3300-miou=test/cluster/mIoU=18.0785.ckpt` | 89306735 | 2026-04-29T23:14:20.457784+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=1-step=step=2700-miou=test/cluster/mIoU=18.0785.ckpt` | 89305971 | 2026-04-29T22:30:49.545033+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=1-step=step=1500-miou=test/cluster/mIoU=18.0785.ckpt` | 89304315 | 2026-04-29T19:33:49.608767+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=2-step=step=4200-miou=test/cluster/mIoU=18.0785.ckpt` | 89307945 | 2026-04-30T01:14:44.381222+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=3-step=step=4800-miou=test/cluster/mIoU=18.0785.ckpt` | 89308773 | 2026-04-30T04:59:58.888795+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=0-step=step=1200-miou=test/cluster/mIoU=18.0785.ckpt` | 89303933 | 2026-04-29T18:45:55.802633+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr29_22-19-28/epoch=epoch=1-step=step=1800-miou=test/cluster/mIoU=18.0788.ckpt` | 89304761 | 2026-04-29T20:20:07.620787+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May27_22-13-30/checkpoint-epoch=02-step=1500-mIoU=test/cluster/mIoU=17.9574.ckpt` | 356008257 | 2026-05-27T17:22:57.387168+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_23-02-46/epoch=epoch=2-step=step=1200-miou=test/cluster/mIoU=19.1536.ckpt` | 89303871 | 2026-05-13T17:58:30.528039+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_23-02-46/epoch=epoch=3-step=step=1800-miou=test/cluster/mIoU=19.0795.ckpt` | 89304699 | 2026-05-13T18:12:55.570107+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_23-02-46/epoch=epoch=4-step=step=2400-miou=test/cluster/mIoU=19.0428.ckpt` | 89305463 | 2026-05-13T18:28:15.740449+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_23-02-46/epoch=epoch=5-step=step=2700-miou=test/cluster/mIoU=19.0430.ckpt` | 89305845 | 2026-05-13T18:42:15.688761+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_23-02-46/epoch=epoch=5-step=step=3000-miou=test/cluster/mIoU=19.0430.ckpt` | 89306291 | 2026-05-13T18:43:06.326186+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_23-02-46/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=18.0626.ckpt` | 89303043 | 2026-05-13T17:45:21.360964+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_23-02-46/epoch=epoch=2-step=step=1500-miou=test/cluster/mIoU=19.1536.ckpt` | 89304253 | 2026-05-13T17:59:21.881604+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_23-02-46/epoch=epoch=3-step=step=2100-miou=test/cluster/mIoU=19.0795.ckpt` | 89305081 | 2026-05-13T18:13:47.113252+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_23-02-46/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=18.0626.ckpt` | 89303489 | 2026-05-13T17:46:11.827686+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=4-step=3000.ckpt` | 89306429 | 2026-04-13T10:19:35+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=6-step=4500.ckpt` | 89307193 | 2026-04-13T10:27:01+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/last.ckpt` | 89307193 | 2026-04-13T10:27:13.252000+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=3-step=2250.ckpt` | 89306047 | 2026-04-13T10:15:51+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=5-step=3750.ckpt` | 89306811 | 2026-04-13T10:23:18+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=2-step=1500.ckpt` | 89305665 | 2026-04-13T10:12:10+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_09-43-44/epoch=1-step=750.ckpt` | 89305219 | 2026-04-13T10:08:28+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May27_09-14-24/checkpoint-epoch=01-step=0900-mIoU=test/cluster/mIoU=8.3229.ckpt` | 89284419 | 2026-05-27T03:47:51.522102+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_07-05-25/epoch=1-step=600.ckpt` | 89306883 | 2026-04-24T01:55:40.919365+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_15-17-43/epoch=epoch=2-step=step=1200-miou=test/cluster/mIoU=18.6876.ckpt` | 89303871 | 2026-05-13T10:14:38.077608+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_15-17-43/epoch=epoch=3-step=step=1800-miou=test/cluster/mIoU=18.6872.ckpt` | 89304699 | 2026-05-13T10:28:59.287026+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_15-17-43/epoch=epoch=4-step=step=2400-miou=test/cluster/mIoU=18.6872.ckpt` | 89305463 | 2026-05-13T10:43:52.154567+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_15-17-43/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=18.9320.ckpt` | 89303043 | 2026-05-13T10:00:22.149134+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_15-17-43/epoch=epoch=2-step=step=1500-miou=test/cluster/mIoU=18.6876.ckpt` | 89304253 | 2026-05-13T10:15:27.984592+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_15-17-43/epoch=epoch=3-step=step=2100-miou=test/cluster/mIoU=18.6872.ckpt` | 89305081 | 2026-05-13T10:29:50.147209+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_15-17-43/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=18.9320.ckpt` | 89303489 | 2026-05-13T10:01:12.551522+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=19-step=step=6900-miou=test/cluster/mIoU=46.8485.ckpt` | 88986711 | 2026-04-25T23:44:01.768585+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=14-step=step=5100-miou=test/cluster/mIoU=46.8485.ckpt` | 88984291 | 2026-04-25T22:25:46.231306+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=4-step=step=1500-miou=test/cluster/mIoU=46.8485.ckpt` | 88979451 | 2026-04-25T19:37:43.359575+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=0-step=step=300-miou=test/cluster/mIoU=28.5900.ckpt` | 88977859 | 2026-04-25T18:48:20.237377+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=11-step=step=4200-miou=test/cluster/mIoU=46.8485.ckpt` | 88983081 | 2026-04-25T21:51:39.642112+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=18-step=step=6600-miou=test/cluster/mIoU=46.8485.ckpt` | 88986265 | 2026-04-25T23:29:54.981814+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=5-step=step=2100-miou=test/cluster/mIoU=46.8485.ckpt` | 88980279 | 2026-04-25T20:11:00.761369+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=20-step=step=7200-miou=test/cluster/mIoU=46.8485.ckpt` | 88987093 | 2026-04-25T23:58:11.452568+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=15-step=step=5400-miou=test/cluster/mIoU=46.8485.ckpt` | 88984673 | 2026-04-25T22:43:05.686162+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=3-step=step=1200-miou=test/cluster/mIoU=46.8485.ckpt` | 88979069 | 2026-04-25T19:21:35.366109+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=12-step=step=4500-miou=test/cluster/mIoU=46.8485.ckpt` | 88983463 | 2026-04-25T21:55:54.228026+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=2-step=step=900-miou=test/cluster/mIoU=46.8485.ckpt` | 88978623 | 2026-04-25T19:07:52.673201+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=16-step=step=5700-miou=test/cluster/mIoU=46.8485.ckpt` | 88985055 | 2026-04-25T22:59:27.970819+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=13-step=step=4800-miou=test/cluster/mIoU=46.8485.ckpt` | 88983845 | 2026-04-25T22:10:38.133908+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=7-step=step=2700-miou=test/cluster/mIoU=46.8485.ckpt` | 88981043 | 2026-04-25T20:32:19.549948+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=21-step=step=7500-miou=test/cluster/mIoU=46.8485.ckpt` | 88987475 | 2026-04-26T00:12:06.421733+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=28.5900.ckpt` | 88978241 | 2026-04-25T18:52:08.165729+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=17-step=step=6300-miou=test/cluster/mIoU=46.8485.ckpt` | 88985883 | 2026-04-25T23:26:11.076349+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=6-step=step=2400-miou=test/cluster/mIoU=46.8485.ckpt` | 88980661 | 2026-04-25T20:15:20.247764+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=9-step=step=3300-miou=test/cluster/mIoU=46.8485.ckpt` | 88981871 | 2026-04-25T21:04:38.807526+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=10-step=step=3600-miou=test/cluster/mIoU=46.8485.ckpt` | 88982253 | 2026-04-25T21:20:42.921808+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=17-step=step=6000-miou=test/cluster/mIoU=46.8485.ckpt` | 88985501 | 2026-04-25T23:12:16.482472+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=8-step=step=3000-miou=test/cluster/mIoU=46.8485.ckpt` | 88981489 | 2026-04-25T20:49:04.564876+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=5-step=step=1800-miou=test/cluster/mIoU=46.8485.ckpt` | 88979833 | 2026-04-25T19:54:50.336873+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=11-step=step=3900-miou=test/cluster/mIoU=46.8485.ckpt` | 88982635 | 2026-04-25T21:36:36.277757+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May26_14-39-11/checkpoint-epoch=01-step=0900-mIoU=test/cluster/mIoU=12.2417.ckpt` | 89302723 | 2026-05-26T09:21:36.959225+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=14-step=step=2100-miou=test/cluster/mIoU=17.6922.ckpt` | 89307191 | 2026-05-13T20:46:02.497356+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=25-step=step=3600-miou=test/cluster/mIoU=18.1066.ckpt` | 89309229 | 2026-05-13T21:53:10.367557+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=12-step=step=1800-miou=test/cluster/mIoU=17.6140.ckpt` | 89306809 | 2026-05-13T20:33:55.245313+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=19-step=step=2700-miou=test/cluster/mIoU=18.0638.ckpt` | 89308019 | 2026-05-13T21:16:24.208887+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=4-step=step=600-miou=test/cluster/mIoU=19.9434.ckpt` | 89305217 | 2026-05-13T19:43:43.870231+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=21-step=step=3000-miou=test/cluster/mIoU=18.2829.ckpt` | 89308401 | 2026-05-13T21:28:39.989869+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=10-step=step=1500-miou=test/cluster/mIoU=17.4494.ckpt` | 89306363 | 2026-05-13T20:21:46.893369+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=34-step=step=4800-miou=test/cluster/mIoU=18.3203.ckpt` | 89310821 | 2026-05-13T22:47:35.678415+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=29-step=step=4200-miou=test/cluster/mIoU=18.3154.ckpt` | 89309993 | 2026-05-13T22:17:38.348198+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=17-step=step=2400-miou=test/cluster/mIoU=18.2528.ckpt` | 89307573 | 2026-05-13T21:04:11.314531+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=31-step=step=4500-miou=test/cluster/mIoU=18.2982.ckpt` | 89310439 | 2026-05-13T22:29:54.009964+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=27-step=step=3900-miou=test/cluster/mIoU=18.2646.ckpt` | 89309611 | 2026-05-13T22:05:21.597312+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=6-step=step=900-miou=test/cluster/mIoU=18.7405.ckpt` | 89305599 | 2026-05-13T19:57:19.039491+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=2-step=step=300-miou=test/cluster/mIoU=19.6632.ckpt` | 89304771 | 2026-05-13T19:31:05.864218+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=8-step=step=1200-miou=test/cluster/mIoU=17.8569.ckpt` | 89305981 | 2026-05-13T20:09:39.386571+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/MR/epoch=epoch=23-step=step=3300-miou=test/cluster/mIoU=18.3268.ckpt` | 89308783 | 2026-05-13T21:40:55.917671+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_16-20-58/epoch=epoch=2-step=step=1200-miou=test/cluster/mIoU=19.2694.ckpt` | 89305599 | 2026-05-25T11:24:24.705597+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_16-20-58/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=19.2840.ckpt` | 89304771 | 2026-05-25T11:10:48.985656+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_16-20-58/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=19.2840.ckpt` | 89305217 | 2026-05-25T11:11:38.606432+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=9-step=step=4800-miou=test/cluster/mIoU=18.6872.ckpt` | 89312551 | 2026-04-24T10:01:49.195266+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=2-step=step=1200-miou=test/cluster/mIoU=18.6872.ckpt` | 89307775 | 2026-04-24T07:52:46.696818+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=3-step=step=1800-miou=test/cluster/mIoU=18.6872.ckpt` | 89308539 | 2026-04-24T08:12:40.603086+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=6-step=step=3600-miou=test/cluster/mIoU=18.6872.ckpt` | 89310959 | 2026-04-24T09:11:07.944822+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=6-step=step=3300-miou=test/cluster/mIoU=18.6872.ckpt` | 89310577 | 2026-04-24T09:08:39.497755+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=7-step=step=3900-miou=test/cluster/mIoU=18.6872.ckpt` | 89311341 | 2026-04-24T09:27:03.918348+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=4-step=step=2400-miou=test/cluster/mIoU=18.6872.ckpt` | 89309367 | 2026-04-24T08:33:13.160082+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=5-step=step=2700-miou=test/cluster/mIoU=18.6872.ckpt` | 89309749 | 2026-04-24T08:50:15.540170+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=5-step=step=3000-miou=test/cluster/mIoU=18.6872.ckpt` | 89310131 | 2026-04-24T08:52:45.832622+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=19.0118.ckpt` | 89306947 | 2026-04-24T07:34:48.555653+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=7-step=step=4200-miou=test/cluster/mIoU=18.6872.ckpt` | 89311787 | 2026-04-24T09:29:36.702489+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=2-step=step=1500-miou=test/cluster/mIoU=18.6872.ckpt` | 89308157 | 2026-04-24T07:56:03.619626+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=3-step=step=2100-miou=test/cluster/mIoU=18.6872.ckpt` | 89308921 | 2026-04-24T08:16:19.507859+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=8-step=step=4500-miou=test/cluster/mIoU=18.6872.ckpt` | 89312169 | 2026-04-24T09:46:02.963439+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_12-47-31/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=19.0118.ckpt` | 89307329 | 2026-04-24T07:37:21.919229+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May27_10-17-17/checkpoint-epoch=07-step=4500-mIoU=test/cluster/mIoU=13.1158.ckpt` | 89284419 | 2026-05-27T05:06:14.208207+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr25_19-54-22/epoch=epoch=4-step=step=1500-miou=test/cluster/mIoU=46.8485.ckpt` | 88979451 | 2026-04-25T15:22:43.456492+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr25_19-54-22/epoch=epoch=0-step=step=300-miou=test/cluster/mIoU=38.9664.ckpt` | 88977859 | 2026-04-25T14:36:05.606404+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr25_19-54-22/epoch=epoch=3-step=step=1200-miou=test/cluster/mIoU=46.8485.ckpt` | 88979069 | 2026-04-25T15:07:32.618628+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr25_19-54-22/epoch=epoch=2-step=step=900-miou=test/cluster/mIoU=46.8485.ckpt` | 88978623 | 2026-04-25T14:53:44.047558+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr25_19-54-22/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=38.9664.ckpt` | 88978241 | 2026-04-25T14:39:47.272407+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr25_19-54-22/epoch=epoch=5-step=step=1800-miou=test/cluster/mIoU=46.8485.ckpt` | 88979833 | 2026-04-25T15:38:17.282002+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May27_11-06-05/checkpoint-epoch=06-step=3900-mIoU=test/cluster/mIoU=17.9822.ckpt` | 89302851 | 2026-05-27T06:32:21.923927+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_20-08-03/epoch=epoch=2-step=step=1200-miou=test/cluster/mIoU=18.6875.ckpt` | 89303871 | 2026-05-25T15:06:14.488180+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_20-08-03/epoch=epoch=3-step=step=1800-miou=test/cluster/mIoU=18.6872.ckpt` | 89304699 | 2026-05-25T15:21:02.447207+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_20-08-03/epoch=epoch=6-step=step=3300-miou=test/cluster/mIoU=18.6872.ckpt` | 89306673 | 2026-05-25T16:03:29.865282+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_20-08-03/epoch=epoch=4-step=step=2400-miou=test/cluster/mIoU=18.6872.ckpt` | 89305463 | 2026-05-25T15:35:20.856870+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_20-08-03/epoch=epoch=5-step=step=2700-miou=test/cluster/mIoU=18.6872.ckpt` | 89305845 | 2026-05-25T15:49:03.902434+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_20-08-03/epoch=epoch=5-step=step=3000-miou=test/cluster/mIoU=18.6872.ckpt` | 89306291 | 2026-05-25T15:49:53.569453+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_20-08-03/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=18.9084.ckpt` | 89303043 | 2026-05-25T14:51:26.017018+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_20-08-03/epoch=epoch=2-step=step=1500-miou=test/cluster/mIoU=18.6875.ckpt` | 89304253 | 2026-05-25T15:07:04.258603+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_20-08-03/epoch=epoch=3-step=step=2100-miou=test/cluster/mIoU=18.6872.ckpt` | 89305081 | 2026-05-25T15:21:53.998159+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_20-08-03/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=18.9084.ckpt` | 89303489 | 2026-05-25T14:52:15.755876+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May27_08-13-32/checkpoint-epoch=05-step=3600-mIoU=test/cluster/mIoU=14.3426.ckpt` | 89284291 | 2026-05-27T02:57:00.299350+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=9-step=step=4800-miou=test/cluster/mIoU=17.6835.ckpt` | 89308647 | 2026-05-13T13:04:15.120397+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=2-step=step=1200-miou=test/cluster/mIoU=17.2332.ckpt` | 89303871 | 2026-05-13T11:30:20.363959+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=3-step=step=1800-miou=test/cluster/mIoU=15.2937.ckpt` | 89304699 | 2026-05-13T11:43:11.587802+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=6-step=step=3600-miou=test/cluster/mIoU=13.8255.ckpt` | 89307055 | 2026-05-13T12:25:56.116221+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=6-step=step=3300-miou=test/cluster/mIoU=13.8255.ckpt` | 89306673 | 2026-05-13T12:22:32.859863+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=7-step=step=3900-miou=test/cluster/mIoU=15.7992.ckpt` | 89307501 | 2026-05-13T12:39:16.408414+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=4-step=step=2400-miou=test/cluster/mIoU=14.4625.ckpt` | 89305463 | 2026-05-13T11:55:58.056663+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=5-step=step=2700-miou=test/cluster/mIoU=15.7813.ckpt` | 89305845 | 2026-05-13T12:07:57.621661+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=5-step=step=3000-miou=test/cluster/mIoU=15.7813.ckpt` | 89306291 | 2026-05-13T12:08:48.088716+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=17.1877.ckpt` | 89303043 | 2026-05-13T11:16:51.846902+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=7-step=step=4200-miou=test/cluster/mIoU=15.7992.ckpt` | 89307883 | 2026-05-13T12:40:06.773272+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=2-step=step=1500-miou=test/cluster/mIoU=17.2332.ckpt` | 89304253 | 2026-05-13T11:31:10.950523+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=3-step=step=2100-miou=test/cluster/mIoU=15.2937.ckpt` | 89305081 | 2026-05-13T11:44:02.960696+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=8-step=step=4500-miou=test/cluster/mIoU=17.0368.ckpt` | 89308265 | 2026-05-13T12:52:00.798431+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/ALL/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=17.1877.ckpt` | 89303489 | 2026-05-13T11:18:22.869898+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_21-58-13/epoch=epoch=2-step=step=1200-miou=test/cluster/mIoU=19.2795.ckpt` | 89303871 | 2026-05-13T16:55:58.490203+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_21-58-13/epoch=epoch=3-step=step=1800-miou=test/cluster/mIoU=19.2692.ckpt` | 89304699 | 2026-05-13T17:09:51.186294+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_21-58-13/epoch=epoch=4-step=step=2400-miou=test/cluster/mIoU=19.2692.ckpt` | 89305463 | 2026-05-13T17:22:44.051821+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_21-58-13/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=19.3562.ckpt` | 89303043 | 2026-05-13T16:41:52.650900+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_21-58-13/epoch=epoch=2-step=step=1500-miou=test/cluster/mIoU=19.2795.ckpt` | 89304253 | 2026-05-13T16:56:49.076828+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_21-58-13/epoch=epoch=3-step=step=2100-miou=test/cluster/mIoU=19.2692.ckpt` | 89305081 | 2026-05-13T17:10:44.465540+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_21-58-13/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=19.3562.ckpt` | 89303489 | 2026-05-13T16:42:43.103555+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_14-33-29/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=18.6872.ckpt` | 89303043 | 2026-05-13T09:17:06.081134+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_14-33-29/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=18.6872.ckpt` | 89303489 | 2026-05-13T09:17:57.174055+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_15-10-17/epoch=epoch=2-step=step=300-miou=test/cluster/mIoU=17.8926.ckpt` | 89304771 | 2026-05-25T09:54:21.723514+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_22-44-21/epoch=epoch=2-step=step=1200-mIoU=test/cluster/mIoU=18.6872.ckpt` | 89302723 | 2026-05-25T18:00:52.076602+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_15-30-55/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=11.3828.ckpt` | 89304771 | 2026-05-25T10:14:37.048158+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_15-30-55/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=11.3828.ckpt` | 89305217 | 2026-05-25T10:15:27.178024+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May27_06-41-48/checkpoint-epoch=01-step=0900-mIoU=test/cluster/mIoU=12.4304.ckpt` | 89302723 | 2026-05-27T01:22:26.954486+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_22-05-42/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=18.6872.ckpt` | 89304771 | 2026-05-25T16:50:22.605621+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_22-05-42/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=18.6872.ckpt` | 89305217 | 2026-05-25T16:51:12.478196+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May27_08-56-48/checkpoint-epoch=01-step=0900-mIoU=test/cluster/mIoU=8.3229.ckpt` | 89284419 | 2026-05-27T03:30:16.850717+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_17-04-55/epoch=epoch=2-step=step=1200-miou=test/cluster/mIoU=19.0437.ckpt` | 89305599 | 2026-05-25T12:07:27.809083+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_17-04-55/epoch=epoch=3-step=step=1800-miou=test/cluster/mIoU=19.0434.ckpt` | 89306427 | 2026-05-25T12:24:20.480454+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_17-04-55/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=20.1545.ckpt` | 89304771 | 2026-05-25T11:52:33.521287+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_17-04-55/epoch=epoch=2-step=step=1500-miou=test/cluster/mIoU=19.0437.ckpt` | 89305981 | 2026-05-25T12:08:18.669608+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_17-04-55/epoch=epoch=3-step=step=2100-miou=test/cluster/mIoU=19.0434.ckpt` | 89306809 | 2026-05-25T12:25:20.235379+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_17-04-55/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=20.1545.ckpt` | 89305217 | 2026-05-25T11:54:26.617167+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May27_09-26-08/checkpoint-epoch=07-step=4500-mIoU=test/cluster/mIoU=13.0377.ckpt` | 89284419 | 2026-05-27T04:14:57.399175+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May26_14-59-23/checkpoint-epoch=01-step=0900-mIoU=test/cluster/mIoU=12.2417.ckpt` | 89302723 | 2026-05-26T09:40:14.629262+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_19-03-07/epoch=epoch=2-step=step=1200-miou=test/cluster/mIoU=18.6872.ckpt` | 89305599 | 2026-05-25T14:02:48.828003+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_19-03-07/epoch=epoch=3-step=step=1800-miou=test/cluster/mIoU=18.6872.ckpt` | 89306427 | 2026-05-25T14:18:23.446233+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_19-03-07/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=18.6872.ckpt` | 89304771 | 2026-05-25T13:48:37.182986+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_19-03-07/epoch=epoch=2-step=step=1500-miou=test/cluster/mIoU=18.6872.ckpt` | 89305981 | 2026-05-25T14:03:39.117442+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_19-03-07/epoch=epoch=3-step=step=2100-miou=test/cluster/mIoU=18.6872.ckpt` | 89306809 | 2026-05-25T14:19:18.181251+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_19-03-07/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=18.6872.ckpt` | 89305217 | 2026-05-25T13:49:27.654553+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May26_13-20-58/epoch=epoch=1-step=step=600-mIoU=test/cluster/mIoU=18.6873.ckpt` | 89304451 | 2026-05-26T08:07:00.337367+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May26_17-30-33/checkpoint-epoch=01-step=0900-mIoU=test/cluster/mIoU=12.2417.ckpt` | 89302723 | 2026-05-26T12:11:34.071526+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May26_16-54-20/checkpoint-epoch=01-step=0900-mIoU=test/cluster/mIoU=18.6872.ckpt` | 89302723 | 2026-05-26T11:41:31.225601+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May26_14-11-33/epoch=1-step=900.ckpt` | 89302659 | 2026-05-26T09:07:16.945815+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_15-33-24/epoch=1-step=900.ckpt` | 89305347 | 2026-04-13T15:38:31+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=4-step=3000.ckpt` | 89307003 | 2026-04-13T14:46:06+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=6-step=4500.ckpt` | 89307131 | 2026-04-13T14:54:50+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/last.ckpt` | 89307131 | 2026-04-13T14:55:03.944000+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=3-step=2250.ckpt` | 89306557 | 2026-04-13T14:41:42+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=5-step=3750.ckpt` | 89307131 | 2026-04-13T14:50:28+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr13_14-09-12/epoch=1-step=750.ckpt` | 89305537 | 2026-04-13T14:32:57+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr23_16-04-38/epoch=1-step=600.ckpt` | 89306883 | 2026-04-23T10:51:50.401592+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_11-20-47/epoch=epoch=1-step=step=300-miou=test/cluster/mIoU=15.5573.ckpt` | 89304771 | 2026-05-25T05:57:46.335581+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May26_21-59-27/checkpoint-epoch=01-step=0900-mIoU=test/cluster/mIoU=12.2417.ckpt` | 89302723 | 2026-05-26T16:44:56.687009+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_19-10-41/epoch=epoch=7-step=step=1200-miou=test/cluster/mIoU=19.0465.ckpt` | 88979069 | 2026-04-26T15:06:27.807081+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_19-10-41/epoch=epoch=3-step=step=600-miou=test/cluster/mIoU=19.0465.ckpt` | 88978241 | 2026-04-26T14:22:12.696099+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_19-10-41/epoch=epoch=9-step=step=1500-miou=test/cluster/mIoU=19.0465.ckpt` | 88979451 | 2026-04-26T15:29:08.792446+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_19-10-41/epoch=epoch=1-step=step=300-miou=test/cluster/mIoU=8.5882.ckpt` | 88977859 | 2026-04-26T13:59:04.215338+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_19-10-41/epoch=epoch=5-step=step=900-miou=test/cluster/mIoU=19.0465.ckpt` | 88978623 | 2026-04-26T14:45:00.093431+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_14-58-31/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=18.9281.ckpt` | 89303043 | 2026-05-13T09:45:09.018053+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr11_20-18-15/epoch=1-step=1200.ckpt` | 89305473 | 2026-04-11T20:46:03+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr11_20-18-15/epoch=3-step=2400.ckpt` | 89305601 | 2026-04-11T20:53:26+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May25_22-47-47/epoch=epoch=3-step=step=1800-mIoU=test/cluster/mIoU=18.6872.ckpt` | 89302723 | 2026-05-25T18:38:11.905915+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_12-44-58/epoch=epoch=2-step=step=1200-miou=test/cluster/mIoU=18.6876.ckpt` | 89303679 | 2026-05-13T07:42:22.133245+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_12-44-58/epoch=epoch=3-step=step=1800-miou=test/cluster/mIoU=18.6872.ckpt` | 89304507 | 2026-05-13T07:56:58.944933+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_12-44-58/epoch=epoch=4-step=step=2400-miou=test/cluster/mIoU=18.6872.ckpt` | 89305271 | 2026-05-13T08:12:16.160458+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_12-44-58/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=18.9290.ckpt` | 89302915 | 2026-05-13T07:27:56.058707+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_12-44-58/epoch=epoch=2-step=step=1500-miou=test/cluster/mIoU=18.6876.ckpt` | 89304125 | 2026-05-13T07:43:13.195561+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_12-44-58/epoch=epoch=3-step=step=2100-miou=test/cluster/mIoU=18.6872.ckpt` | 89304889 | 2026-05-13T07:57:51.548056+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_12-44-58/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=18.9290.ckpt` | 89303297 | 2026-05-13T07:28:46.102145+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_14-03-23/epoch=epoch=1-step=step=600-miou=test/cluster/mIoU=18.6872.ckpt` | 89303043 | 2026-05-13T08:48:15.044609+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_May13_14-03-23/epoch=epoch=1-step=step=900-miou=test/cluster/mIoU=18.6872.ckpt` | 89303489 | 2026-05-13T08:49:05.909198+00:00 | ONLINE-ONLY; UNVERIFIED
`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr24_05-24-58/epoch=1-step=600.ckpt` | 89306883 | 2026-04-24T00:13:29.129400+00:00 | ONLINE-ONLY; UNVERIFIED

Completed targeted roots/counts: `[('/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/checkpoints', 33), ('/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/checkpoints', 20), ('/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-bn147@snu.edu.in/My Drive/STEGO/models', 0), ('/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints', 200), ('/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/models', 0)]`.

### Local weight metadata

```json
{
  "path": "/Users/navneetbavineni/STEGO/models/model_epoch_0720_iter_085000.pth",
  "bytes": 502621184,
  "mtime_utc": "2026-04-02T13:58:16.173000+00:00",
  "classification": "UNVERIFIED: safe metadata read failed",
  "error": "RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory"
}
```

### Local weight metadata

```json
{
  "path": "/Users/navneetbavineni/Documents/vscode/Python/Cosmetics/SkinCareModel.safetensors",
  "bytes": 29764,
  "mtime_utc": "2025-06-26T22:55:45.558575+00:00",
  "key_count": 4,
  "sample_keys": [
    "model.0.bias",
    "model.0.weight",
    "model.2.bias",
    "model.2.weight"
  ],
  "head_keys": [],
  "shapes": {
    "model.0.bias": [
      64
    ],
    "model.0.weight": [
      64,
      109
    ],
    "model.2.bias": [
      5
    ],
    "model.2.weight": [
      5,
      64
    ]
  },
  "classification": "No STEGO head keys; inspect architecture keys below"
}
```

### Local weight metadata

```json
{
  "path": "/Users/navneetbavineni/STEGO/src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin",
  "bytes": 783705670,
  "mtime_utc": "2026-08-18T15:21:00.952298+00:00",
  "top_keys": [
    "logit_scale",
    "visual.trunk.cls_token",
    "visual.trunk.pos_embed",
    "visual.trunk.patch_embed.proj.weight",
    "visual.trunk.patch_embed.proj.bias",
    "visual.trunk.blocks.0.norm1.weight",
    "visual.trunk.blocks.0.norm1.bias",
    "visual.trunk.blocks.0.attn.qkv.weight",
    "visual.trunk.blocks.0.attn.qkv.bias",
    "visual.trunk.blocks.0.attn.proj.weight",
    "visual.trunk.blocks.0.attn.proj.bias",
    "visual.trunk.blocks.0.norm2.weight",
    "visual.trunk.blocks.0.norm2.bias",
    "visual.trunk.blocks.0.mlp.fc1.weight",
    "visual.trunk.blocks.0.mlp.fc1.bias",
    "visual.trunk.blocks.0.mlp.fc2.weight",
    "visual.trunk.blocks.0.mlp.fc2.bias",
    "visual.trunk.blocks.1.norm1.weight",
    "visual.trunk.blocks.1.norm1.bias",
    "visual.trunk.blocks.1.attn.qkv.weight"
  ],
  "key_count": 352,
  "sample_keys": [
    "logit_scale",
    "visual.trunk.cls_token",
    "visual.trunk.pos_embed",
    "visual.trunk.patch_embed.proj.weight",
    "visual.trunk.patch_embed.proj.bias",
    "visual.trunk.blocks.0.norm1.weight",
    "visual.trunk.blocks.0.norm1.bias",
    "visual.trunk.blocks.0.attn.qkv.weight",
    "visual.trunk.blocks.0.attn.qkv.bias",
    "visual.trunk.blocks.0.attn.proj.weight",
    "visual.trunk.blocks.0.attn.proj.bias",
    "visual.trunk.blocks.0.norm2.weight",
    "visual.trunk.blocks.0.norm2.bias",
    "visual.trunk.blocks.0.mlp.fc1.weight",
    "visual.trunk.blocks.0.mlp.fc1.bias",
    "visual.trunk.blocks.0.mlp.fc2.weight",
    "visual.trunk.blocks.0.mlp.fc2.bias",
    "visual.trunk.blocks.1.norm1.weight",
    "visual.trunk.blocks.1.norm1.bias",
    "visual.trunk.blocks.1.attn.qkv.weight"
  ],
  "head_keys": [],
  "shapes": {
    "logit_scale": [],
    "visual.trunk.cls_token": [
      1,
      1,
      768
    ],
    "visual.trunk.pos_embed": [
      1,
      197,
      768
    ],
    "visual.trunk.patch_embed.proj.weight": [
      768,
      3,
      16,
      16
    ],
    "visual.trunk.patch_embed.proj.bias": [
      768
    ],
    "visual.trunk.blocks.0.norm1.weight": [
      768
    ],
    "visual.trunk.blocks.0.norm1.bias": [
      768
    ],
    "visual.trunk.blocks.0.attn.qkv.weight": [
      2304,
      768
    ],
    "visual.trunk.blocks.0.attn.qkv.bias": [
      2304
    ],
    "visual.trunk.blocks.0.attn.proj.weight": [
      768,
      768
    ],
    "visual.trunk.blocks.0.attn.proj.bias": [
      768
    ],
    "visual.trunk.blocks.0.norm2.weight": [
      768
    ]
  },
  "classification": "No STEGO head keys; inspect architecture keys below"
}
```

## M5 — Processed-family counts and20-pair comparison

Command: `python3 -B audit/scratch/decision_families.py`; seed20260930. Images compared after bilinear resize to256×256; masks use nearest. Pairing is an explicit hypothesis, not a recovered conversion manifest: current index maps to generator1mm geometry, legacy file rank maps to raw filename rank within the same patient. Headers/GDCM order only were read for source geometry; no additional raw image pixels were sampled.

```json
{
  "counts": {
    "('train', 'CT', 'current')": 2606,
    "('train', 'CT', 'legacy')": 1869,
    "('train', 'T1DUAL', 'legacy')": 395,
    "('train', 'T2SPIR', 'legacy')": 393,
    "('val', 'CT', 'current')": 629,
    "('val', 'CT', 'legacy')": 472,
    "('val', 'T1DUAL', 'legacy')": 105,
    "('val', 'T2SPIR', 'legacy')": 99
  },
  "patient_ids": {
    "('train', 'CT')": [
      "1",
      "2",
      "10",
      "14",
      "16",
      "18",
      "19",
      "21",
      "22",
      "23",
      "24",
      "25",
      "26",
      "27",
      "28",
      "29"
    ],
    "('train', 'T1DUAL')": [
      "1",
      "2",
      "3",
      "10",
      "13",
      "15",
      "19",
      "20",
      "21",
      "22",
      "31",
      "32",
      "33",
      "34",
      "36",
      "37"
    ],
    "('train', 'T2SPIR')": [
      "1",
      "2",
      "3",
      "10",
      "13",
      "15",
      "19",
      "20",
      "21",
      "22",
      "31",
      "32",
      "33",
      "34",
      "36",
      "37"
    ],
    "('val', 'CT')": [
      "5",
      "6",
      "8",
      "30"
    ],
    "('val', 'T1DUAL')": [
      "5",
      "8",
      "38",
      "39"
    ],
    "('val', 'T2SPIR')": [
      "5",
      "8",
      "38",
      "39"
    ]
  },
  "train_val_patient_overlap": {
    "CT": [],
    "T1DUAL": [],
    "T2SPIR": []
  },
  "comparisons": [
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_26_0121.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p26_s1662.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/26/DICOM_anon/IMG-0009-00033.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        0.0,
        218.0,
        255.0
      ],
      "legacy_percentiles": [
        1.0,
        2.0,
        3.0,
        248.0,
        255.0
      ],
      "pixel_mae_0_255": 13.546295166015625,
      "pixel_correlation": 0.9585262432844273,
      "exact_pixels": false,
      "mask_iou": 0.9615539858728557,
      "position_error_mm": 0.6001000000001113,
      "raw_count": 107,
      "current_count": 171,
      "legacy_count": 107
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_25_0004.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p25_s1526.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/25/DICOM_anon/IMG-0008-00132.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        103.0,
        205.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        60.0,
        243.0,
        255.0
      ],
      "pixel_mae_0_255": 20.754135131835938,
      "pixel_correlation": 0.9197573541154646,
      "exact_pixels": false,
      "mask_iou": 0.7563694267515924,
      "position_error_mm": 0.0,
      "raw_count": 118,
      "current_count": 118,
      "legacy_count": 118
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_2_0138.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p2_s0604.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/2/DICOM_anon/i0011,0000b.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        0.0,
        216.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        3.0,
        241.0,
        255.0
      ],
      "pixel_mae_0_255": 12.64300537109375,
      "pixel_correlation": 0.9579951724242505,
      "exact_pixels": false,
      "mask_iou": 0.9554020100502513,
      "position_error_mm": 0.0,
      "raw_count": 81,
      "current_count": 162,
      "legacy_count": 81
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/val/images/CT_6_0033.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/val/images/CT_p6_s2728.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/6/DICOM_anon/i0065,0000b.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        0.0,
        241.0,
        255.0
      ],
      "legacy_percentiles": [
        1.0,
        2.0,
        3.0,
        255.0,
        255.0
      ],
      "pixel_mae_0_255": 15.000167846679688,
      "pixel_correlation": 0.9551779416229518,
      "exact_pixels": false,
      "mask_iou": 0.9473373653931259,
      "position_error_mm": 0.0,
      "raw_count": 82,
      "current_count": 123,
      "legacy_count": 82
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_1_0098.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p1_s0030.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/1/DICOM_anon/i0030,0000b.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        70.0,
        207.0,
        255.0
      ],
      "legacy_percentiles": [
        1.0,
        2.0,
        24.0,
        245.0,
        255.0
      ],
      "pixel_mae_0_255": 15.709365844726562,
      "pixel_correlation": 0.9505346538795987,
      "exact_pixels": false,
      "mask_iou": 0.9613787375415282,
      "position_error_mm": 0.5,
      "raw_count": 93,
      "current_count": 140,
      "legacy_count": 93
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/val/images/CT_30_0158.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/val/images/CT_p30_s2380.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00079.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        86.0,
        201.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        29.0,
        234.0,
        255.0
      ],
      "pixel_mae_0_255": 16.767532348632812,
      "pixel_correlation": 0.9393594502109649,
      "exact_pixels": false,
      "mask_iou": 0.9644585987261146,
      "position_error_mm": 2.000000165480742e-07,
      "raw_count": 197,
      "current_count": 197,
      "legacy_count": 197
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_22_0138.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p22_s0945.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/22/DICOM_anon/IMG-0005-00015.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        0.0,
        215.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        3.0,
        246.0,
        255.0
      ],
      "pixel_mae_0_255": 13.646560668945312,
      "pixel_correlation": 0.9541053018070966,
      "exact_pixels": false,
      "mask_iou": 0.9562181818181819,
      "position_error_mm": 0.40002000000004045,
      "raw_count": 101,
      "current_count": 162,
      "legacy_count": 101
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/val/images/CT_30_0014.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/val/images/CT_p30_s2524.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00223.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        91.0,
        191.65000000000146,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        3.0,
        45.0,
        236.0,
        255.0
      ],
      "pixel_mae_0_255": 21.230270385742188,
      "pixel_correlation": 0.9049599505163396,
      "exact_pixels": false,
      "mask_iou": 0.9021084337349398,
      "position_error_mm": 0.0,
      "raw_count": 197,
      "current_count": 197,
      "legacy_count": 197
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_27_0110.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p27_s1843.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/27/DICOM_anon/IMG-0010-00100.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        37.0,
        235.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        17.0,
        254.0,
        255.0
      ],
      "pixel_mae_0_255": 14.796371459960938,
      "pixel_correlation": 0.9559138253684932,
      "exact_pixels": false,
      "mask_iou": 0.9531120331950207,
      "position_error_mm": 0.0,
      "raw_count": 182,
      "current_count": 182,
      "legacy_count": 182
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/val/images/CT_5_0116.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/val/images/CT_p5_s2590.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/5/DICOM_anon/i0022,0000b.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        0.0,
        229.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        3.0,
        246.0,
        255.0
      ],
      "pixel_mae_0_255": 14.099563598632812,
      "pixel_correlation": 0.9510440980883326,
      "exact_pixels": false,
      "mask_iou": 0.9583285368942097,
      "position_error_mm": 0.7999299999999039,
      "raw_count": 95,
      "current_count": 152,
      "legacy_count": 95
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/val/images/CT_5_0142.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/val/images/CT_p5_s2573.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/5/DICOM_anon/i0005,0000b.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        0.0,
        217.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        3.0,
        248.0,
        255.0
      ],
      "pixel_mae_0_255": 13.157684326171875,
      "pixel_correlation": 0.9461979977422397,
      "exact_pixels": false,
      "mask_iou": 0.9453164556962025,
      "position_error_mm": 0.40002000000004045,
      "raw_count": 95,
      "current_count": 152,
      "legacy_count": 95
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/val/images/CT_30_0031.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/val/images/CT_p30_s2507.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/30/DICOM_anon/IMG-0002-00206.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        92.0,
        193.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        3.0,
        47.0,
        235.0,
        255.0
      ],
      "pixel_mae_0_255": 20.849884033203125,
      "pixel_correlation": 0.9118127910876037,
      "exact_pixels": false,
      "mask_iou": 0.9044585987261147,
      "position_error_mm": 0.0,
      "raw_count": 197,
      "current_count": 197,
      "legacy_count": 197
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_25_0058.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p25_s1472.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/25/DICOM_anon/IMG-0008-00078.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        113.0,
        223.65000000000146,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        69.0,
        250.0,
        255.0
      ],
      "pixel_mae_0_255": 17.996734619140625,
      "pixel_correlation": 0.9405428660595943,
      "exact_pixels": false,
      "mask_iou": 0.9701758132251406,
      "position_error_mm": 0.0,
      "raw_count": 118,
      "current_count": 118,
      "legacy_count": 118
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_10_0090.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p10_s0149.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/10/DICOM_anon/i0053,0000b.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        97.0,
        232.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        56.0,
        252.0,
        255.0
      ],
      "pixel_mae_0_255": 20.598434448242188,
      "pixel_correlation": 0.929660497053522,
      "exact_pixels": false,
      "mask_iou": 0.9445670159955875,
      "position_error_mm": 0.4000300000000152,
      "raw_count": 107,
      "current_count": 171,
      "legacy_count": 107
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_14_0033.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p14_s0279.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/14/DICOM_anon/i0073,0000b.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        0.0,
        231.0,
        255.0
      ],
      "legacy_percentiles": [
        1.0,
        2.0,
        3.0,
        255.0,
        255.0
      ],
      "pixel_mae_0_255": 12.6591796875,
      "pixel_correlation": 0.9613072363037191,
      "exact_pixels": false,
      "mask_iou": 0.8923647932131495,
      "position_error_mm": 0.5999699999999848,
      "raw_count": 93,
      "current_count": 149,
      "legacy_count": 93
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_29_0030.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p29_s2264.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/29/DICOM_anon/IMG-0012-00177.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        0.0,
        218.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        3.0,
        254.0,
        255.0
      ],
      "pixel_mae_0_255": 12.213287353515625,
      "pixel_correlation": 0.9588207172689253,
      "exact_pixels": false,
      "mask_iou": 0.8830219333874898,
      "position_error_mm": 2.842170943040401e-14,
      "raw_count": 154,
      "current_count": 154,
      "legacy_count": 154
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/val/images/CT_5_0094.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/val/images/CT_p5_s2603.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/5/DICOM_anon/i0035,0000b.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        90.0,
        219.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        32.0,
        243.0,
        255.0
      ],
      "pixel_mae_0_255": 15.323959350585938,
      "pixel_correlation": 0.9512188640358413,
      "exact_pixels": false,
      "mask_iou": 0.9615264959037644,
      "position_error_mm": 0.40002000000004045,
      "raw_count": 95,
      "current_count": 152,
      "legacy_count": 95
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/val/images/CT_6_0117.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/val/images/CT_p6_s2672.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/6/DICOM_anon/i0009,0000b.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        0.0,
        251.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        3.0,
        255.0,
        255.0
      ],
      "pixel_mae_0_255": 14.12384033203125,
      "pixel_correlation": 0.9500986451450814,
      "exact_pixels": false,
      "mask_iou": 0.9177033492822967,
      "position_error_mm": 0.0,
      "raw_count": 82,
      "current_count": 123,
      "legacy_count": 82
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_22_0011.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p22_s1024.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/22/DICOM_anon/IMG-0005-00094.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        53.0,
        211.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        3.0,
        23.0,
        250.0,
        255.0
      ],
      "pixel_mae_0_255": 17.514541625976562,
      "pixel_correlation": 0.931681508361126,
      "exact_pixels": false,
      "mask_iou": 0.875,
      "position_error_mm": 0.19994999999994434,
      "raw_count": 101,
      "current_count": 162,
      "legacy_count": 101
    },
    {
      "current": "src/pytorch_data_dir/preprocessed/train/images/CT_26_0138.png",
      "legacy_candidate": "src/pytorch_data_dir/preprocessed/train/images/CT_p26_s1652.png",
      "candidate_raw": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/26/DICOM_anon/IMG-0009-00023.dcm",
      "pair_rule": "hypothesis: nearest physical slice for current1mm index; legacy rank in same patient",
      "current_shape": [
        256,
        256
      ],
      "legacy_shape": [
        512,
        512
      ],
      "current_percentiles": [
        0.0,
        0.0,
        0.0,
        224.0,
        255.0
      ],
      "legacy_percentiles": [
        2.0,
        2.0,
        3.0,
        252.0,
        255.0
      ],
      "pixel_mae_0_255": 11.2308349609375,
      "pixel_correlation": 0.9666466735371837,
      "exact_pixels": false,
      "mask_iou": 0.9659945832079446,
      "position_error_mm": 0.3998999999998887,
      "raw_count": 107,
      "current_count": 171,
      "legacy_count": 107
    }
  ]
}
```

## M4 — Current cache identity observations

Command: `python3 -B audit/scratch/decision_cache_centering.py`. Only existing cache contents and weight bytes were read. Filename ordering below is the hypothetical static order before the mixed-family guard; it does not instantiate or bypass that guard for training. Array dimensions/self-neighbors do not prove provenance.

Backbone file SHA256: `52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be` (`/Users/navneetbavineni/STEGO/src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin`).

```json
[
  {
    "split": "train",
    "current_filename_count": 4475,
    "first": [
      "CT_10_0000.png",
      "CT_10_0001.png",
      "CT_10_0002.png"
    ],
    "last": [
      "CT_p2_s0671.png",
      "CT_p2_s0672.png",
      "CT_p2_s0673.png"
    ],
    "current_names_sha256": "31b0f284f689e02e0e7737ce66f510c92efe60f9cd4b4fd0758db50a014a99ed",
    "cache": "src/pytorch_data_dir/nns/nns_vit_base_chaos_train_five_224.npz",
    "keys": [
      "nns"
    ],
    "shape": [
      4475,
      30
    ],
    "dtype": "int64",
    "index_range": [
      0,
      4474
    ],
    "first_column_self_fraction": 1.0
  },
  {
    "split": "val",
    "current_filename_count": 1101,
    "first": [
      "CT_30_0000.png",
      "CT_30_0001.png",
      "CT_30_0002.png"
    ],
    "last": [
      "CT_p8_s2856.png",
      "CT_p8_s2857.png",
      "CT_p8_s2858.png"
    ],
    "current_names_sha256": "97fb27b98ec76a89c89fea6cf155491075b4cbd8ea322e3f459e15242ec60c6b",
    "cache": "src/pytorch_data_dir/nns/nns_vit_base_chaos_val_five_224.npz",
    "keys": [
      "nns"
    ],
    "shape": [
      1101,
      30
    ],
    "dtype": "int64",
    "index_range": [
      0,
      1100
    ],
    "first_column_self_fraction": 1.0
  }
]
```

## m1 — Centering behavior, without changing it

Source: `src/modules.py:478–481`. Paper contract is the already-recorded Eq(3) ledger in `audit/AUDIT.md:263`; no paper reread. Command: `python3 -B audit/scratch/decision_cache_centering.py`, seed11,2×3×2×2 synthetic features.

```json
[
  {
    "term": "self",
    "reference_loss": -0.20717449486255646,
    "literal_eq3_loss": -0.0750548392534256,
    "mean_affinity": 0.3521985411643982,
    "equivalence_error": 1.4901161193847656e-08
  },
  {
    "term": "neighbor",
    "reference_loss": -0.032486625015735626,
    "literal_eq3_loss": -0.00692271813750267,
    "mean_affinity": 0.12215450406074524,
    "equivalence_error": 7.450580596923828e-09
  },
  {
    "term": "negative",
    "reference_loss": 0.15386627614498138,
    "literal_eq3_loss": 0.12830236554145813,
    "mean_affinity": -0.12215450406074524,
    "equivalence_error": 0.0
  }
]
```

With `mu=mean(F)` and `Splus=max(S,0)`, current `L_ref = L_literal - mu * mean(Splus)` (floating-point residual below1e-7 here). Equivalently, current code uses an effective literal-centering shift `b - mu`. This is batch-dependent, so changing centering while retaining inherited shifts is not numerically neutral. For constant affinity0.7 and shift0.18, literal centering gives a positive0.18*Splus penalty, while restored-mean centering gives a negative0.52*Splus attraction. This is an algebraic illustration, not a measured training outcome.

## Additional provenance and corrected sample counts

Command: `python3 -B audit/scratch/decision_followup.py`. Counts reuse the same masks; T1-out uses only the18 physical matches. Cloud dates are metadata, not proof of training date. Git object checks are read-only.

```json
{
  "corrected_mask_counts": {
    "CT": {
      "0": 4806375,
      "255": 436505
    },
    "T1-in": {
      "0": 1362122,
      "63": 59452,
      "126": 9062,
      "189": 10146,
      "252": 9202
    },
    "T1-out": {
      "0": 1188695,
      "126": 6225,
      "189": 6802,
      "63": 54567,
      "252": 10399
    },
    "T2": {
      "0": 1296170,
      "63": 73093,
      "126": 9542,
      "252": 13408,
      "189": 9643
    }
  },
  "filename_order_directions": {
    "CT": {
      "descending": 20
    },
    "T1-in": {
      "ascending": 20
    },
    "T1-out": {
      "ascending": 18
    },
    "T2": {
      "ascending": 20
    }
  },
  "cloud_files": 253,
  "cloud_mtime_range": [
    "2026-04-05T17:48:36+00:00",
    "2026-05-27T17:22:57.387168+00:00"
  ],
  "epoch21_candidate": [
    "`/Users/navneetbavineni/Library/CloudStorage/GoogleDrive-navneetbavineni@gmail.com/My Drive/STEGO/checkpoints/chaos/chaos_chaos_date_Apr26_00-04-56/epoch=epoch=21-step=step=7500-miou=test/cluster/mIoU=46.8485.ckpt` | 88987475 | 2026-04-26T00:12:06.421733+00:00 | ONLINE-ONLY; UNVERIFIED"
  ],
  "runs": [
    {
      "run": "run-20260426_000459-5e6nhit6",
      "metadata": {
        "os": "macOS-26.3.1-arm64-arm-64bit-Mach-O",
        "python": "CPython 3.14.0",
        "startedAt": "2026-04-25T18:34:59.129430Z",
        "program": "/Users/navneetbavineni/STEGO/src/train_segmentation.py",
        "codePath": "src/train_segmentation.py",
        "codePathLocal": "train_segmentation.py",
        "git": {
          "remote": "https://github.com/Krin8/Stego-Chaos-Dino.git",
          "commit": "a61797f23e0dcc9222f7dedf05637ad68aeb5a1d"
        },
        "email": "navneetbavineni@gmail.com",
        "root": "../logs",
        "host": "Navneets-MacBook-Pro.local",
        "executable": "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3",
        "cpu_count": 14,
        "cpu_count_logical": 14,
        "disk": {
          "/": {
            "total": "994662584320",
            "used": "498506731520"
          }
        },
        "memory": {
          "total": "25769803776"
        },
        "apple": {},
        "writerId": "w152nkbbw30g5ebm3v0ju2puxn6vvtwl"
      },
      "config": {
        "arch": "dino",
        "model_type": "vit_small",
        "dino_patch_size": 8,
        "dim": 40,
        "extra_clusters": 5,
        "chaos_n_classes": 5,
        "use_text_prompts": "UNVERIFIED: absent",
        "max_steps": 7500,
        "batch_size": 16,
        "res": 320,
        "chaos_modality": "CT",
        "use_preprocessed_data": "UNVERIFIED: absent",
        "loader_crop_type": null,
        "crop_type": "five",
        "crop_ratio": 0.5,
        "shallow_layer_n": "UNVERIFIED: absent",
        "pytorch_data_dir": "/Users/navneetbavineni/STEGO/src/pytorch_data_dir"
      },
      "summary": {
        "test/linear/Accuracy": 90.87689971923828,
        "test/cluster/Accuracy": 93.69705963134766,
        "_step": 270,
        "test/cluster/mIoU": 46.84852981567383,
        "cd/neg_inter": 0.003647935576736927,
        "epoch": 21,
        "loss/pos_inter": 0.006055832840502262,
        "_timestamp": 1777162326.4287639,
        "loss/crf": -0.017026597633957863,
        "test/linear/mIoU": 52.8696174621582,
        "_runtime": 20245,
        "loss/linear": 0.9634168744087219,
        "loss/neg_inter": 0.007107838056981564,
        "loss/aug_alignment": -0.9990136623382568,
        "loss/pos_intra": -0.019769014790654182,
        "loss/total": -0.15111160278320312,
        "cd/pos_inter": 0.0025025259237736464,
        "_wandb": {
          "runtime": 20245
        },
        "trainer/global_step": 2499,
        "loss/cluster": -0.9995210766792297,
        "cd/pos_intra": 0.09147397428750992
      },
      "git_object_check": {
        "returncode": 128,
        "stdout": "",
        "stderr": "fatal: git cat-file: could not get object info"
      }
    },
    {
      "run": "run-20260909_044210-y3l0vxcn",
      "metadata": {
        "os": "macOS-26.6.2-arm64-arm-64bit-Mach-O",
        "python": "CPython 3.14.0",
        "startedAt": "2026-09-08T23:12:10.606181Z",
        "program": "/Users/navneetbavineni/STEGO/src/train_segmentation.py",
        "codePath": "src/train_segmentation.py",
        "codePathLocal": "train_segmentation.py",
        "git": {
          "remote": "https://github.com/Krin8/Stego-Chaos.git",
          "commit": "c859758193f15f37f2a4d5139555f7f4997ee22a"
        },
        "email": "navneetbavineni@gmail.com",
        "root": "../logs",
        "host": "qwerty-4.local",
        "executable": "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3",
        "cpu_count": 14,
        "cpu_count_logical": 14,
        "disk": {
          "/": {
            "total": "994662584320",
            "used": "965491367936"
          }
        },
        "memory": {
          "total": "25769803776"
        },
        "apple": {},
        "writerId": "5e8gxsihtljhu9n6rk1y523nu41m449g"
      },
      "config": {
        "arch": "biomedclip",
        "model_type": "vit_base",
        "dino_patch_size": 16,
        "dim": 70,
        "extra_clusters": 10,
        "chaos_n_classes": 2,
        "use_text_prompts": true,
        "max_steps": 5000,
        "batch_size": 16,
        "res": 224,
        "chaos_modality": "CT",
        "use_preprocessed_data": true,
        "loader_crop_type": "center",
        "crop_type": "five",
        "crop_ratio": 0.75,
        "shallow_layer_n": 2,
        "pytorch_data_dir": "/Users/navneetbavineni/STEGO/src/pytorch_data_dir"
      },
      "summary": {
        "loss/aug_alignment": -0.9968456625938416,
        "test/cluster/Accuracy": 92.14274597167969,
        "cd/neg_inter": -0.8450672030448914,
        "loss/neg_inter": 0.009588290937244892,
        "loss/crf": -0.014977307990193367,
        "plot_labels": {
          "size": 601619,
          "path": "media/images/plot_labels_4806_c830cc89238dcd82036e.png",
          "format": "png",
          "width": 1500,
          "height": 1200,
          "_type": "image-file",
          "sha256": "c830cc89238dcd82036ed634b7a52e699add6ca65f0360832324c43448b24b09"
        },
        "_runtime": 8594,
        "loss/text_align": 0.007158979307860136,
        "epoch": 5,
        "test/linear/Accuracy": 98.00369262695312,
        "_timestamp": 1788917704.10268,
        "loss/linear": 0.17108280956745148,
        "loss/total": -4.184839248657227,
        "_wandb": {
          "runtime": 8594
        },
        "_step": 4951,
        "conf_matrix": {
          "path": "media/images/conf_matrix_4950_244b4bc77d0d34597af2.png",
          "format": "png",
          "width": 1300,
          "height": 1000,
          "_type": "image-file",
          "sha256": "244b4bc77d0d34597af2e879c0d7244c37e13a0542df93296d24b2da21ba08f8",
          "size": 67963
        },
        "test/linear/mIoU": 87.46875,
        "cd/pos_inter": 0.9958841800689697,
        "label frequency": {
          "path": "media/images/label frequency_4950_8756699f2a1f758422a3.png",
          "format": "png",
          "width": 1000,
          "height": 400,
          "_type": "image-file",
          "sha256": "8756699f2a1f758422a33b98838096438264806a1fde97c330a07cdc93756ebb",
          "size": 57131
        },
        "loss/cluster": -1.7596685886383057,
        "loss/pos_intra": -0.35450688004493713,
        "cd/pos_intra": 0.9963012933731079,
        "loss/rec": -0.8968172669410706,
        "loss/pos_inter": -0.3413444459438324,
        "trainer/global_step": 1659,
        "test/cluster/mIoU": 30.715906143188477
      },
      "git_object_check": {
        "returncode": 0,
        "stdout": "commit",
        "stderr": ""
      }
    }
  ]
}
```

## Additional overlay and phase-geometry check

Command: `python3 -B audit/scratch/decision_overlay_check.py`; reuses selected slices. Centroids use DICOM row/column direction cosines and pixel spacing. Positive patient-X is left in the DICOM patient coordinate convention; kidney relative lateral order is corroboration, not a complete anatomical audit.

```json
{
  "overlays": [
    {
      "sequence": "T1-in",
      "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/DICOM_anon/InPhase/IMG-0034-00020.dcm",
      "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/8/T1DUAL/Ground/IMG-0034-00020.png",
      "inside_mean": 108.74484384207425,
      "outside_mean": 71.86256149409785,
      "foreground_nonzero_image_fraction": 1.0,
      "organ_centroid_patient_xyz_mm": {
        "63": [
          -117.61960007074539,
          -52.58956295164086,
          -2.9766197204589
        ],
        "126": [
          -83.12313173827931,
          -2.812321362537091,
          -2.9766197204589
        ],
        "189": [
          82.9217419785461,
          1.320036640039234,
          -2.9766197204589
        ],
        "252": [
          138.10715789712674,
          -34.23499745375389,
          -2.9766197204589
        ]
      }
    },
    {
      "sequence": "T1-out",
      "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/OutPhase/IMG-0023-00025.dcm",
      "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/Ground/IMG-0023-00026.png",
      "inside_mean": 245.1677545691906,
      "outside_mean": 93.37353215742151,
      "foreground_nonzero_image_fraction": 0.9895561357702349,
      "organ_centroid_patient_xyz_mm": {
        "126": [
          -47.45558570139971,
          3.8047433753984308,
          -71.185523986816
        ],
        "189": [
          64.56608302006347,
          22.998676793781016,
          -71.185523986816
        ]
      }
    },
    {
      "sequence": "T2",
      "image": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/DICOM_anon/IMG-0020-00019.dcm",
      "mask": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T2SPIR/Ground/IMG-0020-00019.png",
      "inside_mean": 552.0141544574125,
      "outside_mean": 95.30886190459623,
      "foreground_nonzero_image_fraction": 1.0,
      "organ_centroid_patient_xyz_mm": {
        "63": [
          -47.931930980160146,
          3.6178804732711853,
          21.81029510498
        ],
        "126": [
          -31.408703804009377,
          45.82604169846124,
          21.81029510498
        ],
        "252": [
          98.7560882273672,
          40.707280854528165,
          21.81029510498
        ]
      }
    }
  ],
  "out_phase_pixel_geometry": [
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/OutPhase/IMG-0023-00025.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/38/T1DUAL/DICOM_anon/InPhase/IMG-0023-00026.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00031.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/InPhase/IMG-0031-00032.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/10/T1DUAL/DICOM_anon/OutPhase/IMG-0046-00053.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/10/T1DUAL/DICOM_anon/InPhase/IMG-0046-00054.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T1DUAL/DICOM_anon/OutPhase/IMG-0075-00037.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/15/T1DUAL/DICOM_anon/InPhase/IMG-0075-00038.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00017.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/InPhase/IMG-0017-00018.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T1DUAL/DICOM_anon/OutPhase/IMG-0005-00001.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/34/T1DUAL/DICOM_anon/InPhase/IMG-0005-00002.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00019.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/InPhase/IMG-0031-00020.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00019.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/InPhase/IMG-0017-00020.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/DICOM_anon/OutPhase/IMG-0064-00053.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/DICOM_anon/InPhase/IMG-0064-00054.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/DICOM_anon/OutPhase/IMG-0016-00033.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/5/T1DUAL/DICOM_anon/InPhase/IMG-0016-00034.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/OutPhase/IMG-0010-00005.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/InPhase/IMG-0010-00006.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00025.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/InPhase/IMG-0017-00026.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/OutPhase/IMG-0017-00045.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/37/T1DUAL/DICOM_anon/InPhase/IMG-0017-00046.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00043.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/InPhase/IMG-0031-00044.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T1DUAL/DICOM_anon/OutPhase/IMG-0005-00031.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/22/T1DUAL/DICOM_anon/InPhase/IMG-0005-00032.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/OutPhase/IMG-0031-00015.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/32/T1DUAL/DICOM_anon/InPhase/IMG-0031-00016.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/OutPhase/IMG-0010-00013.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/2/T1DUAL/DICOM_anon/InPhase/IMG-0010-00014.dcm",
      "same_matrix_and_pixel_spacing": true
    },
    {
      "out": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/DICOM_anon/OutPhase/IMG-0064-00033.dcm",
      "in_phase": "src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/13/T1DUAL/DICOM_anon/InPhase/IMG-0064-00034.dcm",
      "same_matrix_and_pixel_spacing": true
    }
  ]
}
```

## Damaged archive: static metadata inspection

Command: `python3 -B audit/scratch/decision_pickle_keys.py`. No deserialization, tensor loading or archive repair.

```json
{
  "path": "/Users/navneetbavineni/STEGO/models/model_epoch_0720_iter_085000.pth",
  "entry": "archive/data.pkl",
  "metadata_bytes_read": 15586,
  "opcodes": 3349,
  "pickle_stop": true,
  "key_candidates": [],
  "first_strings": [
    "collections OrderedDict",
    "per_element_factor",
    "torch._utils _rebuild_tensor_v2",
    "storage",
    "torch FloatStorage",
    "94265205275920",
    "cuda:0",
    "per_element_bias",
    "94263528504240",
    "cuda:0",
    "latents.weight",
    "94265188252048",
    "cuda:0",
    "img_siren.net.0.omega_0",
    "94265167697184",
    "cuda:0",
    "img_siren.net.0.linear.weight",
    "94265167619632",
    "cuda:0",
    "img_siren.net.0.linear.bias",
    "94265167619392",
    "cuda:0",
    "img_siren.net.1.omega_0",
    "94265168750976",
    "cuda:0",
    "img_siren.net.1.linear.weight",
    "94265167619296",
    "cuda:0",
    "img_siren.net.1.linear.bias",
    "94265168748192",
    "cuda:0",
    "img_siren.net.2.omega_0",
    "94265169803664",
    "cuda:0",
    "img_siren.net.2.linear.weight"
  ],
  "note": "Static opcode inspection only; no pickle execution or tensor storage read. Archive remains unloadable by torch.load; string candidates alone do not reconstruct its full state dictionary."
}
```

</details>

## Read-only verification

Final checks: `git diff --stat` and `git diff --cached --stat` returned no changes. `git status --short` listed only the new `audit/DECISIONS.md`, `audit/scratch/decision_*.py` and `audit/scratch/overlays/`. All newly created investigation scripts parsed; all report overlay links resolve. The investigation guard rejects writes outside this report and newly created overlay images, and rejects network connections. No production files, existing audit/fix reports, datasets, caches or checkpoints were modified. No decision was applied.

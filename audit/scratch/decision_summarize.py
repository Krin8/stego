"""Put decision recommendations ahead of the retained read-only evidence."""
import decision_guard as g
import json,re
evidence=g.REPORT.read_text().replace('audit/AUDIT.md:262','audit/AUDIT.md:263')
def json_after(marker):
    return json.loads(evidence.split(marker,1)[1].split('```json\n',1)[1].split('\n```',1)[0])
geometry=json_after('## M5 — Physical-position pairing correction')
extra=json_after('## Additional provenance and corrected sample counts')
raw={}
for name in ['CT','T1-in','T1-out','T2']:
    raw[name]=json_after('### '+name+'\n')
head='''# Decisions requiring confirmation

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
| `models/model_epoch_0720_iter_085000.pth` | 502,621,184 bytes / 2026-04-02 13:58:16.173 | Safe metadata read fails: `failed finding central directory`. Classification and recoverability UNVERIFIED; an intact archive would settle it. No unsafe unpickling attempted. |
| `src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin` | 783,705,670 bytes / 2026-08-18 15:21:00.952298 | Pretrained backbone: 352 keys, visual transformer/text model, 768-wide patch embedding; no cluster_probe/linear_probe/segmentation-head keys. It cannot supply a trained STEGO head. |
| `/Users/navneetbavineni/Documents/vscode/Python/Cosmetics/SkinCareModel.safetensors` | 29,764 bytes / 2025-06-26 22:55:45.558575 | Four MLP keys with 109→64→5 shapes; neither STEGO head nor BiomedCLIP backbone. |

Metadata inspection used Torch FakeTensorMode/mmap/weights_only or the safetensors header: no tensor payload was materialized. `.bin` was inspected explicitly because the active pretrained backbone uses that extension, in addition to the requested filename suffix search (`decision_cloud_weights.py`).

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
'''
for seq,n in [('T1-in',20),('T1-out',18),('T2',20)]:
    counts=extra['corrected_mask_counts'][seq]
    head+='| '+seq+' | '+str(n)+' | '+' | '.join(f'{counts.get(str(k),0):,}' for k in [0,63,126,189,252])+' |\n'
head+='''
Source: `decision_followup.py`; per-slice paired paths and counts appear in the ledger. All sampled MRI values are exactly0/63/126/189/252; sampled CT values are0/255. MRI clipping percentages, using each slice's whole-image thresholds (lower1st percentile in both columns):

| Sequence / label | Upper99th: organ pixels clipped | Upper99.9th: organ pixels clipped |
| --- | ---: | ---: |
'''
for r in geometry:
    if r['sequence']=='CT':continue
    for k in [63,126,189,252]:
        vals=r['clipping_fraction']
        head+=f"| {r['sequence']} / {k} | {100*vals[str((k,99))]:.4f}% | {100*vals[str((k,99.9))]:.4f}% |\n"
head+='''
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
| Intensity preprocessing | Modality/phase dispatch; CT slope/intercept handling and approved HU limits; MRI percentile bounds, interpolation method for quantiles, per-slice scope, background inclusion, constant-slice policy; scale/range/dtype, clipping/rounding, RGB replication | Needed to distinguish current400HU PNGs from a future250HU contract. Record actual operations, not a generic “CHAOS preprocessing” label. |
| Spatial/model preprocessing | Input and resize sizes, interpolation/antialias settings, aspect-ratio policy, crop kind/coordinates/ratio, actual crop lineage, transform order, mask-nearest policy; mean `[0.48145466,0.4578275,0.40821073]`, std `[0.26862954,0.26130258,0.27577711]`, application count1 | Cache generation currently uses short-edge resize plus center crop224 (`precompute_knns.py:43,86`; `utils.py:191–209`). Filename `five` does not prove five crops were applied. Record actual geometry separately from config labels. |
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

The remaining evidence gaps are: actual contents of online-only candidate checkpoints; intact local `.pth` metadata; historical command arguments/dirty source/data split; derivative source lineage; complete-dataset anatomical alignment; and model-performance validation of the proposed preprocessing/centering choices. Confirming a recommendation alone does not fill these evidence gaps. After your choices, a separate implementation request can define a single data manifest and metadata-validating cache writer/reader. This investigation has not run training, regenerated caches, installed packages or changed existing project files.

## Detailed evidence ledger

<details>
<summary>Expand inventories, per-run records, sample paths, exact numerical outputs and overlays</summary>

'''
g.REPORT.write_text(head+evidence+'\n</details>\n')
print('Summary written; retained evidence bytes',len(evidence.encode()))

# Final-pipeline implementation review

Reviewed 5 October 2026. Repository: `/Users/navneetbavineni/STEGO`, branch `final-pipeline-n3`, HEAD `6afbd3d0d0d14f12920c93780d85fbcd8d0d306a`.

## Verdict

**The supplied flowchart is not an accurate description of the checkpoint it names.** It combines the legacy `src/` trainer's settings, the actual 448-resolution P1 experiment, and a later evaluator that discards part of the trained head. The repository also has a separate frozen-feature ensemble exposed through `src/eval_segmentation.py`. These must not be reported as one implementation.

**83.92% is UNVERIFIED as a recoverable result in this review.** No matching saved result was found in the searched project text/results, and the cited optimized evaluator prints its result without saving the predictions, centres or a result manifest. This is not evidence that the number is false; it is insufficient evidence to certify its provenance. The original saved P1/F1 result is a different method and must not be substituted for it.

No existing source, dataset, feature cache or checkpoint was edited. No training, k-means fitting, feature-cache regeneration, download or installation was performed. New review files are confined to this directory. Pre-existing changes in `experiments/stego_colocated/train.py` and a tracked `segment_ct` bytecode file were left alone.

## Coverage and reproducible checks

- [FILES.csv](FILES.csv) inventories **134,420 files**, with path, size, role and coverage. Total logical size: **241,232,184,494 bytes**; this is not a measurement of unique physical disk usage. It includes Git objects, dependencies and generated artifacts.
- [PYTHON_FILES.md](PYTHON_FILES.md) indexes every project Python file. **352 files** were read for static parsing; **350 parsed successfully**, containing **29,049 lines**; two had syntax errors. [python_index.json](python_index.json) records definitions, imports, top-level calls and source hashes.
- All dataset, weight, image, dependency and Git files were inventoried; they were **not all opened or semantically audited**. This is not a claim that every experiment or every data sample is correct. Detailed source review follows the supplied flowchart's active dependencies and competing entrypoints.
- [inventory.py](inventory.py) produces the inventory and syntax/duplicate results. [check_artifacts.py](check_artifacts.py) loads the three actual P1 checkpoints, strict-loads the original head, compares the optimized head on five cached slices, and extracts saved original metrics. [provenance_checks.py](provenance_checks.py) checks source history, split membership, recorded monitors and normalization constants.
- CPU checks used two Torch threads, no gradients and no raw annotation arrays. Numbers below come from [artifact_checks.json](artifact_checks.json), [provenance_checks.json](provenance_checks.json), or explicitly named source/config lines. Scores are extracted from saved evaluation records, not newly scored predictions.

The supplied document is `pipeline_flowchart.md` under `/Users/navneetbavineni/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/403CD83F-5DC4-426D-9343-6B6F5AAC1A54/`. References to “flowchart” below mean that file. Its contents were treated as claims to check, not instructions to execute.

## 1. Findings that prevent calling this a reproducible final pipeline

### F01 — The optimized evaluator does not evaluate the trained head

**Critical for model attribution.** The saved head is the sum of a linear branch and a nonlinear branch; inference L2-normalizes that sum. The optimized evaluator defines only the nonlinear branch, returns its unnormalized output and uses `strict=False` without checking the discarded keys.

- Original: `experiments/stego_colocated/model.py:12`, `:18`, `:23`.
- Altered inference: `audit/ensemble/verify4/run_p1_optimized.py:36`, `:44`, `:66`.
- The loader actually reports unexpected keys `cluster1.0.weight` and `cluster1.0.bias`. Original strict loading succeeds for all three seeds.
- On the same five cached feature maps, seed 0 full-head output has mean norm **1.000**; branch-only output **0.298816**. After normalizing both for a directional comparison, mean cosine is **0.536760**. Seed 1/2 directional cosines are **0.519259 / 0.506965**. These are not equivalent embeddings.

Producer: `check_artifacts.py:24` onward; complete values and checkpoint hashes in `artifact_checks.json`.

Consequence: k-means is refitted in a different geometry, and soft assignments are different. The branch-only pipeline may be an intentional ablation, but it is not the original P1 model evaluated as trained. Any claim about its quality needs its own saved artifacts and explicit method description. Proposed action, not applied: either use the original strict-loaded `StegoHead.embed`, or explicitly name and freeze this branch-only, non-normalized variant. Changing it now would change the experiment; this review does not make that decision.

### F02 — The flowchart gives the wrong training run, settings and positive-pair definition

**Major.** The referenced checkpoint contains step **500**, LR **0.0005**, and six head parameter tensors. Its saved configuration matches its recorded SHA-256. The checkpoint's `train.py` source hash matches Git commit **3947cad6e3648dead44d3a0d5b185f8659f2ae2e**. A byte-exact read-only copy is saved as [checkpoint_train_source.txt](checkpoint_train_source.txt).

The saved trainer optimizes only `model.parameters()` and calls the three-term correspondence objective (`checkpoint_train_source.txt:53`, `:76`; `model.py:28`). There is no learned ClusterLookup loss in that run. Clustering happens subsequently through k-means (`stages.py:30`).

| Component | Flowchart claim | Actual named P1 checkpoint / path |
|---|---|---|
| Backbone input | 224 | 448 |
| Native feature grid | 14×14, later upsampled to 28×28 | Direct 28×28 final-block features |
| Feature tap | Deepest two blocks | One final block, final LayerNorm, per-patch L2 |
| Training budget | 5,000 | 500 batches |
| Adam LR | 0.0001 | 0.0005, no scheduler |
| Intra / partner / random shifts | 0.18 / 0.12 / 0.46 | 0.794418 / 0.8869667053222656 / 0.797651 |
| Negative repetitions | 20 | 5 |
| Positive partner | Same image, augmented view | Same patient's physical-slice-order offset ±1 or ±2; co-located body-frame pairs |
| Self term | Different image crops | Spatial pairs sampled from the same cached feature map |
| Probe | Learned cluster probe in total loss | Offline K=8 k-means after training |
| Code inference | Nonlinear branch alone | Full linear + nonlinear sum, then L2 |

Evidence: `experiments/stego_colocated/P1/stage2/configs/locked.json:10`, `:17`, `:24`, `:37`, `:45`, `:62`, `:77`; `experiments/stego_colocated/loss.py:28`, `:34`, `:55`; `audit/improve/extract.py:23`, `:32`, `:40`; checkpoint inspection in `artifact_checks.json`.

The incorrect settings correspond to the separate current `src/configs/train_config.yaml:10`, `:29`, `:40`, `:52`, `:60`, `:70`. That config is not the saved P1 training configuration. `shallow_layer_n` is assigned but the extractor calls `get_intermediate_layers(..., n=1, norm=True)` regardless (`src/modules.py:175`, `:200`, `:210`).

### F03 — The default evaluation entrypoint has a preprocessing mismatch and is a different experiment

**Critical if `src/eval_segmentation.py` is used as the final evaluation.** It applies ImageNet mean/std `[.485,.456,.406] / [.229,.224,.225]` at line 79. The feature-fit contract uses BiomedCLIP `[.48145466,.4578275,.40821073] / [.26862954,.26130258,.27577711]` (`locked.json:64`; `src/chaos_preprocessing.py:17`). This changes the input distribution while retaining centres fitted under the other transform.

For windowed intensity 1.0, the expected normalized RGB is `[1.930336,2.074884,2.145897]`; this evaluator produces `[2.248908,2.428571,2.640000]`. Producer: `provenance_checks.py`, output `normalization_examples`.

It also loads **three all-20-patient frozen-feature fits**, uses 768-D backbone features and averages their liver probabilities (`src/eval_segmentation.py:35`, `:46`, `:82`, `:88`). It never loads the P1 checkpoint. This is transductive frozen-feature ensemble evaluation, not the single-seed inductive P1 pipeline. It bypasses the usual runtime stage guard by assigning the stage after import (`:15`). Its `--save-images` path writes overlays, not separate binary mask files (`:131` onward).

Proposed action, not applied: choose which entrypoint represents the final method; make its preprocessing match its centre metadata and describe its fit population accurately. Do not claim that its score validates P1.

### F04 — The 83.92% / 86.72% comparison is not sealed or attributable from the found records

**Major reporting gap.** The optimized evaluator fixes seed 0, refits k-means, derives N2, computes global validation TP/FP/FN, and only prints IoU/mIoU (`run_p1_optimized.py:62`, `:75`, `:118`, `:130`, `:158`). It does not save the fit, masks, count totals, configuration digest or result record.

The review searched inventoried `.py/.md/.txt/.log/.json/.jsonl/.csv/.yaml/.yml` files in report, results and source categories up to 40 MB for the literal rounded numbers **83.92** and **86.72**, with no matches. This does not search PDF contents, screenshots, remote logs, ignored formats or numerical values stored at other precision. Producer and exact search scope: `check_artifacts.py:55` onward.

To verify the claim, provide the original console log plus the exact code/config snapshot and predictions, or run the frozen, explicitly declared inference variant and save its masks, centres, naming decision, temperature, hashes and patient-level/global counts. No such refit or rerun was performed here.

### F05 — Current training code has drifted from checkpoint provenance

**Major reproducibility issue.** The three checkpoints' recorded package source hashes match current files except `train.py` (`artifact_checks.json`). Current `train.py:54` constructs a new ClusterLookup and adds its parameters to Adam. The current training step at line 79 still uses only the correspondence objective, and line 92 saves `model`, not this probe.

The historical matching producer uses only head parameters (`checkpoint_train_source.txt:53`). Current probe initialization consumes additional Torch randomness before training; even with unused probe gradients, it changes the random stream for the subsequent stochastic head. Therefore the current trainer cannot be described as the byte-identical producer of these checkpoints. The pre-existing uncommitted changes were preserved.

### F06 — Feature-crop geometry is confused with the N2 naming mask

**Major documentation error.** Features are extracted from a body crop obtained by thresholding the windowed image at **0.05**, choosing the largest 8-connected component, filling holes, and making a square with a 1.1 side multiplier (`audit/improve/pipeline.py:9`). For the [-150,250] window, this threshold corresponds approximately to HU > -130, with quantization at the boundary.

N2 separately uses raw HU > -500, opening with disk radius 5, largest component and hole filling (`run_p1_optimized.py:88`). It uses that body mask for naming statistics while retaining the **saved feature crop box** (`:92`). The flowchart uses the naming mask as though it were the feature crop, which would produce different features if implemented literally.

### F07 — The packaged N3 naming command is broken

**Major packaging issue; separate from the optimized P1 evaluator.** `audit/final_pipeline_n3/name_clusters.py:168` opens `metadata.update({` but closes only the dictionary at line 181, producing a SyntaxError. In addition, the wrapper passes `--images-dir` (`segment_ct.py:168`), while the child requires `--records` (`name_clusters.py:154`). Fixing the syntax alone would not fix that invocation. Static parse reproduction: `inventory_summary.json`.

`audit/ensemble/verify4/naming_logic.py:1` is also an indented fragment, not valid standalone Python. No evidence found that the optimized evaluator imports that fragment; do not attribute its syntax error to every inference route.

The existing packaged `segment_ct.py` loads frozen-feature centres and uses the original F1 threshold 0.5 (`:125`, `:138`); it is not a packaged version of the branch-only optimized P1 method.

### F08 — Mechanistic claims and validation claims exceed the evidence

**Major for a final scientific report.** Reducing dimension from 768 to 70 does not establish “90% information loss.” The head emits embeddings, not probabilities; assignment sharpness also depends on centre distances and the fitted temperature (`run_p1_optimized.py:75`, `:78`, `:137`). No saved diagnostic in this review establishes the flowchart's asserted probability distributions, causal boundary destruction or claim that sharp inputs “break” guided-filter mathematics. The implemented formula contains a positive epsilon (`:53`); a causal explanation needs controlled measurements, not channel-count arithmetic.

The later optimization report explicitly states that the four validation patients informed guided-filter sizing, morphology and naming thresholds (`audit/ensemble/verify4/VERIFY4.md:4`). Thus label-free fitting/naming is not the same as an untouched final test. Report optimized results as development-cohort results, with that qualification.

N2's `x` is **image-left to image-right**, and `y` is **top to bottom within a slice** (`run_p1_optimized.py:93`, `:105`). `y<.6` is not a test for superior/“upper abdomen” slice position. Equating image-left with anatomical patient-right requires an orientation contract; this heuristic does not use DICOM orientation vectors itself. Its transfer to differently oriented inputs is UNVERIFIED.

## 2. Actual P1 pipeline recovered from saved evidence

### Shared frozen features

Raw DICOM CT → slope/intercept HU → clip [-150,250] and map to [0,1] → uint16 derivative quantization → native orientation and native 512 frame → intensity-based square body crop described in F06 → RGB replication → bilinear antialiased resize to **448** → BiomedCLIP normalization once → frozen ViT-B/16 final normalized patch tokens, CLS excluded → per-patch L2 → **28×28×768**.

Evidence: `src/chaos_preprocessing.py:45`, `:76`; `audit/improve/pipeline.py:9`, `:23`; `audit/improve/extract.py:23`, `:32`, `:41`; `src/modules.py:157`, `:167`, `:204`; `locked.json:45`. Cache shape independently inspected: **2341×784×768**. No feature-grid upsampling from 14 to 28 occurs on this path.

Historical direct-extractor verification exists in `audit/preflight/extract_checks.json` (frozen/eval, deterministic, cache comparison). This audit reuses cached features; it did not rerun the whole DICOM-to-backbone pipeline or rehash the entire feature array.

### Actual training

16 train patients / 1,869 slices. Head: independent Dropout2d(0.1) draws into linear and nonlinear branches, add outputs; a further independent feature dropout supplies the feature signal. P1 positive partners are uniform ±1/±2 slice-order offsets within patient and series; invalid boundary draws skip only that partner term (`model.py:16`; `loss.py:28`, `:55`). This is body-frame co-location, not deformable anatomical registration.

For feature affinity A, the helper computes target-spatial mean removal with restored global mean:

`A = cosine(Fa,Fb); Acentered = A - mean_targets(A) - mean(A - mean_targets(A)) + mean(A)`.

The pair loss is `-max(cosine(Ca,Cb), 0) * (Acentered - shift)` for this configuration. Self/random reduce the sampled pair tensors; the P1 partner uses the **diagonal after centering the full spatial matrix**. Total is `0.67*self + 0.25*partner + 0.63*random`. Adam LR 0.0005, batch 16, 500 batches, no scheduler, final checkpoint only. Evidence: `reference_loss.py:51`; `loss.py:11`, `:34`, `:44`; `model.py:28`; `locked.json:10`.

### Original sealed inference

Strict-load full head → normalized 70-D sum → fit K=8 on every train patch, n_init=3, max_iter=100, tol=1e-4, Lloyd, seed-specific → N1 text naming / N2 anatomical naming / N3 agreement on train images → temperature from nearest-two squared-distance gaps → val assignments → inverse crop → guided filter **size 17**, epsilon .01 → **threshold .5** → largest 8-connected component → holes → closing disk radius 3 → save predictions and seal → score validation.

Evidence: `stages.py:10`, `:13`, `:19`, `:30`, `:38`, `:68`; `evaluate.py:8`; `audit/final_pipeline/segment_ct.py:69`. This is the route associated with the original saved P1 metrics below.

### Later optimized inference cited in the flowchart

Reuse the same 448 feature cache → seed-0 checkpoint, **nonlinear branch only, no output L2** → refit K=8 on train patches → **N2 only** → fitted-temperature soft assignment → inverse crop → guided filter **size 25**, epsilon .01 → **threshold .30** → largest component → holes → closing disk radius 3 → global validation IoU printed.

Evidence: `run_p1_optimized.py:27`, `:36`, `:47`, `:62`, `:75`, `:118`, `:132`. It has no learned-probe inference and no CRF. This variant cannot inherit the original Stage-A seal or original F1 score because its embeddings, naming path and refinement parameters differ.

## 3. What the saved P1 checkpoints actually establish

All three are 500-batch final checkpoints with config-hash matches, strict original-head load success, and deterministic original-head output on the checked cached inputs. Seed-0 checkpoint SHA-256: `1454f1bb8d2261a9a56a0fbcd6bed1d8946bf8cd7ff319d697b108ec16d28f1b`; size **8,417,040 bytes**. All checkpoint sizes/hashes are in `artifact_checks.json`.

| Seed | Final code std | Recorded cross-image statistic | Entropy | Reported collapsed | Original N1=N2=N3 cluster |
|---|---:|---:|---:|---|---:|
| 0 | 0.113530 | 0.021001 | 1.923604 | No | 3 |
| 1 | 0.114095 | 0.022080 | 1.915423 | No | 1 |
| 2 | 0.113110 | 0.028305 | 1.922532 | No | 3 |

Sources: seed-specific `logs/seed*_complete.json`, extracted by `provenance_checks.py`; naming/results by `check_artifacts.py`. All eight monitor clusters have nonzero usage. These are monitor-set statistics, not a new all-slice evaluation. The statistic called cross-image cosine is the off-diagonal dot product of **patch-averaged normalized codes**; it does not renormalize the image means (`checkpoint_train_source.txt:62`). Do not reinterpret it as cosine of unit image-level vectors. Monitor k-means is refitted at each check; its cluster IDs are not the later prediction cluster IDs (`:65`).

| Seed | Original F1, 224 patient-mean IoU ± sample SD | Original F1, 224 patient-mean Dice | Original F1, 512 patient-mean IoU | Original F1, 512 globally pooled IoU / Dice |
|---|---:|---:|---:|---:|
| 0 | 67.6563 ± 2.4606% | 80.6891% | 67.6353% | 67.7820 / 80.7977% |
| 1 | 66.7888 ± 2.3960% | 80.0694% | 66.7647% | 66.9198 / 80.1820% |
| 2 | 68.7590 ± 2.2270% | 81.4723% | 68.7590% | 68.9189 / 81.6000% |

Producer: `check_artifacts.py` reading `experiments/stego_colocated/P1/stage2/metrics/results.json`. These use validation labels for scoring; original naming is train-image based. They are **not** evaluations of the later branch-only/threshold-.30 variant. They neither verify nor refute its claimed 83.92%.

Validation patients/slices: **5:95, 6:82, 8:98, 30:197**. Train/val patient intersection is empty in `audit/naming/image_records.json`, checked by `provenance_checks.py`. Per-patient pooled IoU means sum TP/FP/FN within each patient, then average the four IoUs. The optimized script instead sums across all validation slices before taking the ratio. Those metrics weight patients differently. Its “volumetric” number is a global voxel/pixel-count ratio with no physical voxel-volume weighting (`run_p1_optimized.py:144`; no spacing weights).

## 4. File and folder roles

The complete path-level inventory is `FILES.csv`; the complete project-Python index is `PYTHON_FILES.md`. This section identifies the implementation boundaries rather than treating every historical experiment as the final method.

| Location | Role and relevance |
|---|---|
| `src/chaos_preprocessing.py`, `src/preprocess.py` | Data contract, raw-pair/derivative generation. Existing outputs consumed here; generation not run. |
| `src/modules.py`, local BiomedCLIP model/config | Shared extractor. Also contains legacy trainable heads/losses; actual saved P1 uses its separate head/loss modules. |
| `src/data.py`, `src/knn_cache.py`, `src/precompute_knns.py` | Legacy training-loader/cache route. P1 loads saved 448 features and uses adjacent-slice partners; the optimized evaluator does not build kNN. |
| `src/train_segmentation.py`, `src/configs/` | Separate Lightning STEGO training route; not the checkpoint producer named by the flowchart. |
| `src/eval_segmentation.py` | Three-seed frozen-feature ensemble; incompatible normalization and not P1 (F03). |
| `experiments/stego_colocated/model.py`, `loss.py`, `reference_loss.py` | Actual P1 head and objective, with matching checkpoint source hashes. |
| `experiments/stego_colocated/P1/stage2/` | Locked settings, final checkpoints, predictions, naming, original metrics and logs. |
| `experiments/stego_colocated/stages.py`, `evaluate.py`, `runtime.py` | Original fitting/naming/prediction/seal/score route. Runtime reuses helpers from the earlier frozen-feature package. |
| `audit/improve/` | Upstream body crops, feature extraction and native-frame geometry for the 448 cache. |
| `audit/naming/` | Image records, naming reference data and original sealed naming artifacts. |
| `audit/final_pipeline/` | Frozen-feature CLI and helpers reused by experiments. Not a learned-head deployment package. |
| `audit/final_pipeline_n3/` | Modified frozen-feature/N3 package, with broken re-naming command and duplicated helper scripts. |
| `audit/ensemble/verify4/` | Later exploratory ensemble/naming/refinement variants, including branch-only optimized P1. `run_n2_only.py:62` explicitly uses an **untrained** seed-9 head; do not attribute that route's scores to P1. |
| Other `experiments/` and `audit/` folders | Historical alternative methods, calibration, diagnostics and evaluation. Indexed individually; not evidence that a similarly named script produced the final claim. |
| `dino/`, `src/dino/`, `picie/`, `src/picie/` | Alternative/reference code; corresponding files have exact duplicates per source hashes. Not the BiomedCLIP P1 backbone. |
| `Dataset/`, `pytorch_data_dir/` | Data and derivatives; inventoried, not changed. |
| `logs/`, checkpoints, `.npy/.npz`, images, reports | Generated evidence/artifacts; relevance requires linkage to source/config, not just filenames. |
| `.git/`, environments, runtime caches | Infrastructure; metadata inventory only. |

Exact duplicate groups are enumerated in `inventory_summary.json`, including experiment runtime/model/loss copies and copied frozen-pipeline checks. Duplication is not by itself a bug, but edits in one copy do not imply another route changed. “Not used by this final path” is not equivalent to globally dead code.

## 5. Additional reproducibility gaps

- **Automated tests do not cover the claimed final route.** CI runs root-level `pytest` (`.github/workflows/tests.yml:24`); diagnostic scripts under `src/` import bare `modules` / `data` (`src/test_biomedclip.py:5`, `src/test_timm.py:16`). The previously observed GitHub failure was import collection, not evidence of numerical final-pipeline verification. This audit did not rerun CI or install its environment.
- **Local-artifact dependence:** the optimized evaluator hardcodes the user's absolute root, loads a large pre-existing feature cache, and refits at top level (`run_p1_optimized.py:12`, `:27`, `:75`). It is an analysis script rather than a standalone saved-model inference command. A clean code-only GitHub checkout without the local artifacts cannot reproduce it directly.
- **No blanket orientation guarantee:** current CHAOS records retain native orientation; N2's coordinate prior is cohort-specific unless orientation is explicitly checked for new inputs. No new orientation/generalization tests were run.
- **N2 can abstain:** when no cluster satisfies its fixed conditions, the optimized path outputs an empty probability field (`run_p1_optimized.py:118`, `:139`). This should be an explicit inference status, not mistaken for evidence that the image contains no liver.
- **Cohort performance is not external validity:** repeated choices used these four validation patients. This review establishes neither external-cohort performance nor clinical suitability.

## 6. Minimum steps before calling it final — recommendations only

1. Identify the final method unambiguously: original full P1, the branch-only optimized P1 variant, or the frozen-feature ensemble. Preserve existing results under their actual methods; do not silently relabel them.
2. Correct the flowchart using the saved configuration and checkpoint provenance above: especially 448/28×28, 500 batches, shifts, co-located positives, full training head and offline clustering.
3. Resolve the evaluation normalization mismatch and broken naming CLI if those entrypoints are part of the chosen deployment. Neither was changed here.
4. Preserve the claimed optimized run's predictions, centres, N2 choice, temperature, counts and complete provenance. Report per-patient and globally pooled native-resolution scores separately; do not mix these with 224-scale results.
5. Replace the unsupported causal explanation with measured evidence or mark it as a hypothesis. Disclose that optimized parameters were informed by the validation cohort.

## 7. Still UNVERIFIED

- Exact provenance and saved masks for the attached document's **83.92%** and **86.72%**.
- Whether nonlinear-only, non-normalized inference was deliberate; the code does it, but intent is not established.
- Full end-to-end execution of the optimized variant on all validation slices under a pinned environment; not rerun in this analysis.
- Current integrity of every byte of the large cached arrays and every saved prediction seal. Selected checkpoints and source hashes were checked, not the entire artifact store.
- The mathematical/causal claims about feature information loss, assignment overconfidence and the supposed necessity of size-25 filtering.
- Correctness of every historical branch, dataset sample, third-party dependency or binary. Inventory/static parsing must not be represented as complete semantic verification.
- Generalization beyond this repeatedly used CT cohort and orientation convention.

No corrections have been applied. This report records the discrepancies so that accepting the attached flowchart does not accidentally redefine the experiment.

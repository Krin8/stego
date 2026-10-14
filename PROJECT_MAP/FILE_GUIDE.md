# STEGO file-use guide

This guide explains the complete project structure, not just optimized_p1_pipeline. Original project files remain unchanged. It describes file roles, not a guarantee that every historical script still works.

## Start with the selected pipeline

Your selected method is audit/ensemble/verify4/run_p1_optimized.py: cached BiomedCLIP features → nonlinear-only P1 head → train K=8 fit → N2 naming → guided filtering at size 25 → threshold .30 → largest component, hole fill and closing. See optimized_p1_pipeline/README.md for the extracted subset. The original P1 trainer and frozen-feature ensemble are different routes.

## How to read this guide

Each code file below has a role and a source link. Function/call evidence is collected through AST parsing, without importing or executing the script. For obscure historical utilities, exact intended use remains UNVERIFIED where the source structure alone is insufficient. Listed calls can be inside functions; their presence does not establish that they run on every invocation.

PROJECT_MAP/ALL_FILES.csv is the path-by-path inventory of data, code and generated artifacts; this guide explains repeated generated-file families together. Git objects, installed dependencies and bytecode are infrastructure rather than research implementations.

## Every original project Python file

### audit/calibrated

- **[calibrate.py](/Users/navneetbavineni/STEGO/audit/calibrated/calibrate.py:1)** — Estimates affinity shifts using the declared feature sample for 448 shift calibration.
  Definitions: guard:14, dump:44, read:45, sha:46, batch:65, monitors:68, diversity:74, core:92. Operations: np.load:60, np.load:61, (HERE / name).write_text:44, dump:93, dump:100, dump:150, dump:193.
- **[summarize.py](/Users/navneetbavineni/STEGO/audit/calibrated/summarize.py:1)** — Summarizes the recorded diagnostic/calibration outputs for 448 shift calibration.
  Definitions: read:5, table:6. Operations: report.read_text:51, report.write_text:53, (HERE / 'summary.json').write_text:54, p.read_text:5, target.write_text:24.
### audit/colocated

- **[amended_calibration.py](/Users/navneetbavineni/STEGO/audit/colocated/amended_calibration.py:1)** — Evaluates the revised label-free attraction-target grid and admissibility gate for adjacent-slice P1 diagnostics.
  Definitions: Capture:22. Operations: dump:15, dump:72, np.load:11, (ROOT / 'audit/preflight/diagnose.py').read_text:12.
- **[check_loss.py](/Users/navneetbavineni/STEGO/audit/colocated/check_loss.py:1)** — Checks the proposed correspondence-loss behavior for adjacent-slice P1 diagnostics.
  Operations: dump:59, total.backward:41.
- **[common.py](/Users/navneetbavineni/STEGO/audit/colocated/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for adjacent-slice P1 diagnostics.
  Definitions: guard:23, read:52, dump:56, sha:60, ledger:68, feature_batch:83. Operations: np.load:76, (OUT / name).write_text:57, dump:69, Path(path).read_text:53.
- **[prepare_round.py](/Users/navneetbavineni/STEGO/audit/colocated/prepare_round.py:1)** — Writes the settings/scripts for a declared experiment round for adjacent-slice P1 diagnostics.
  Definitions: digest:6, replace:11. Operations: (OLD / 'train.py').read_text:10, (NEW / 'train.py').write_text:35, (NEW / 'check_state.py').write_text:39, (ROOT / 'audit/colocated/runner_reuse.json').write_text:56, (ROOT / 'audit/colocated/amended_calibration.json').read_text:40, (OLD / 'check_state.py').read_text:36, path.write_text:54.
- **[progress.py](/Users/navneetbavineni/STEGO/audit/colocated/progress.py:1)** — Reads experiment progress records for adjacent-slice P1 diagnostics.
  Operations: log.read_text:11, complete.read_text:18.
- **[report_amendment.py](/Users/navneetbavineni/STEGO/audit/colocated/report_amendment.py:1)** — Records the changed admissibility protocol and calibration evidence for adjacent-slice P1 diagnostics.
  Operations: (P / 'amended_summary.md').write_text:20, report.read_text:21, report.write_text:22, (P / 'amended_calibration.json').read_text:5.
- **[report_final.py](/Users/navneetbavineni/STEGO/audit/colocated/report_final.py:1)** — Summarizes completed run and evaluation records for adjacent-slice P1 diagnostics.
  Operations: (ROOT / 'audit/colocated/final_summary.md').write_text:57, (ROOT / 'audit/colocated/final_summary.json').write_text:58, report.read_text:59, report.write_text:59, p.read_text:5, (P / 'all_predictions_sealed.sha256').read_text:7.
- **[report_screen.py](/Users/navneetbavineni/STEGO/audit/colocated/report_screen.py:1)** — Summarizes the short-run screening gate for adjacent-slice P1 diagnostics.
  Operations: (ROOT / 'audit/colocated/screen_summary.md').write_text:25, report.read_text:26, report.write_text:26, p.read_text:5.
- **[verify.py](/Users/navneetbavineni/STEGO/audit/colocated/verify.py:1)** — Checks the study assertions and saved output consistency for adjacent-slice P1 diagnostics.
  Definitions: Capture:45. Operations: dump:22, dump:65, dump:98, np.load:13, (ROOT / 'audit/preflight/diagnose.py').read_text:15.
### audit/colocated/verify2

- **[builder.py](/Users/navneetbavineni/STEGO/audit/colocated/verify2/builder.py:1)** — Historical utility in adjacent-slice P1 diagnostics. Its implemented interface includes build_table. Exact experiment status is not established by this guide.
  Definitions: build_table:3. Operations: json.load:45.
- **[diagnostics.py](/Users/navneetbavineni/STEGO/audit/colocated/verify2/diagnostics.py:1)** — Historical utility in adjacent-slice P1 diagnostics. Its implemented interface includes get_soft_scores. Exact experiment status is not established by this guide.
  Definitions: get_soft_scores:18. Operations: np.load:16, KMeans(n_clusters=8, random_state=0, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:34, torch.load:52, model.load_state_dict:54, np.load:56, Path('experiments/stego_colocated/C448/stage2/predictions/seed0/fit.json').read_text:57.
- **[run_frozen.py](/Users/navneetbavineni/STEGO/audit/colocated/verify2/run_frozen.py:1)** — Historical utility in adjacent-slice P1 diagnostics. Its implemented interface includes largest, guided, score_fields_local. Exact experiment status is not established by this guide.
  Definitions: largest:27, guided:33, score_fields_local:40. Operations: np.load:74, np.load:72, np.load:79, KMeans(n_clusters=8, random_state=seed, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:97, np.load:67, pydicom.dcmread:112.
- **[run_untrained.py](/Users/navneetbavineni/STEGO/audit/colocated/verify2/run_untrained.py:1)** — Historical utility in adjacent-slice P1 diagnostics. Its implemented interface includes largest, guided, score_fields_local. Exact experiment status is not established by this guide.
  Definitions: largest:27, guided:33, score_fields_local:40. Operations: np.load:74, np.load:72, np.load:79, KMeans(n_clusters=8, random_state=seed, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:97, np.load:67, pydicom.dcmread:112.
### audit/dino

- **[common.py](/Users/navneetbavineni/STEGO/audit/dino/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for first DINO comparison.
  Definitions: guard:7, reuse:29, confusion:38, dump:39, sha:40, elapsed:45, records:46. Operations: path.read_text:30, (OUT / name).write_text:39, (OLD / 'metadata.json').read_text:46, (ROOT / 'audit/scratch/forensics/task1.py').read_text:36.
- **[extract.py](/Users/navneetbavineni/STEGO/audit/dino/extract.py:1)** — Extracts the study feature arrays for first DINO comparison.
  Definitions: forward:10. Operations: torch.load:7, model.load_state_dict:8, np.lib.format.open_memmap:18, dump:26, (OUT / 'inventory.json').read_text:5, (OUT / 'PROTOCOL.json').read_text:9.
- **[figures.py](/Users/navneetbavineni/STEGO/audit/dino/figures.py:1)** — Creates the study figures from saved arrays/results for first DINO comparison.
  Operations: np.load:6, np.load:9, fig.savefig:16, dump:17, (OUT / 'PROTOCOL.json').read_text:6.
- **[fit_score.py](/Users/navneetbavineni/STEGO/audit/dino/fit_score.py:1)** — Fits clustering and scores partitions under the study protocol for first DINO comparison.
  Operations: p.parse_args:6, np.load:8, np.lib.format.open_memmap:9, dump:12, km.fit:13, np.save:16, dump:17.
- **[pair_affinities.py](/Users/navneetbavineni/STEGO/audit/dino/pair_affinities.py:1)** — Collects patch-pair affinity values for selectivity analysis for first DINO comparison.
  Definitions: capture:19. Operations: np.load:15, dump:42, (SELECT / 'records.json').read_text:8, np.lib.format.open_memmap:16, np.lib.format.open_memmap:17, np.load:39, np.load:9.
- **[pair_score.py](/Users/navneetbavineni/STEGO/audit/dino/pair_score.py:1)** — Scores affinity discrimination/pair selectivity for first DINO comparison.
  Operations: np.load:4, np.load:8, dump:15, (OUT / 'pair_affinities.json').read_text:10, np.load:12.
- **[prepare.py](/Users/navneetbavineni/STEGO/audit/dino/prepare.py:1)** — Prepares metadata/input selection for the study for first DINO comparison.
  Operations: dump:23, dump:30, torch.load:19, torch.load:12, (OLD / 'figure_selection.json').read_text:28.
- **[report.py](/Users/navneetbavineni/STEGO/audit/dino/report.py:1)** — Builds a human-readable report from recorded results for first DINO comparison.
  Definitions: fmt:23. Operations: dump:22, (OUT / 'DINO.md').write_text:53, (OUT / 'inventory.json').read_text:2, (OUT / 'pair_selectivity.json').read_text:2, (OUT / 'D1_extract.json').read_text:46, (OUT / 'figures.json').read_text:46, (OUT / f'D1_k{k}_s{s}_scores.json').read_text:4.
- **[run_rest.py](/Users/navneetbavineni/STEGO/audit/dino/run_rest.py:1)** — Runs the remaining declared study stages for first DINO comparison.
  Operations: subprocess.run:8.
- **[validate.py](/Users/navneetbavineni/STEGO/audit/dino/validate.py:1)** — Checks the study inputs, outputs or reproducibility assertions for first DINO comparison.
  Operations: np.load:6, dump:28, (OUT / 'D1_extract.json').read_text:5, (OUT / 'pair_selectivity.json').read_text:20, np.load:12, (OUT / f'D1_k{k}_s{s}_scores.json').read_text:12.
### audit/dino2

- **[common.py](/Users/navneetbavineni/STEGO/audit/dino2/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for DINO D2/D3 comparison.
  Definitions: guard:7, reuse:29, confusion:38, dump:39, sha:40, elapsed:45, records:46. Operations: path.read_text:30, (OUT / name).write_text:39, (OLD / 'metadata.json').read_text:46, (ROOT / 'audit/scratch/forensics/task1.py').read_text:36.
- **[extract.py](/Users/navneetbavineni/STEGO/audit/dino2/extract.py:1)** — Extracts the study feature arrays for DINO D2/D3 comparison.
  Definitions: forward:25, images:33. Operations: p.parse_args:6, np.lib.format.open_memmap:50, dump:56, (OUT / 'weight_verification.json').read_text:7, (OUT / 'PROTOCOL.json').read_text:8, (folder / 'preprocessor_config.json').read_text:10, dump:39.
- **[figures.py](/Users/navneetbavineni/STEGO/audit/dino2/figures.py:1)** — Creates the study figures from saved arrays/results for DINO D2/D3 comparison.
  Operations: p.parse_args:4, np.load:8, np.load:11, fig.savefig:18, dump:19, (OUT / 'PROTOCOL.json').read_text:8.
- **[fit_score.py](/Users/navneetbavineni/STEGO/audit/dino2/fit_score.py:1)** — Fits clustering and scores partitions under the study protocol for DINO D2/D3 comparison.
  Operations: p.parse_args:6, np.load:8, np.lib.format.open_memmap:9, dump:12, km.fit:13, np.save:16, dump:17.
- **[pair_affinities.py](/Users/navneetbavineni/STEGO/audit/dino2/pair_affinities.py:1)** — Collects patch-pair affinity values for selectivity analysis for DINO D2/D3 comparison.
  Definitions: capture:19. Operations: np.load:15, dump:42, (SELECT / 'records.json').read_text:8, np.lib.format.open_memmap:16, np.lib.format.open_memmap:17, np.load:39, np.load:9.
- **[pair_score.py](/Users/navneetbavineni/STEGO/audit/dino2/pair_score.py:1)** — Scores affinity discrimination/pair selectivity for DINO D2/D3 comparison.
  Operations: np.load:4, np.load:8, dump:15, (OUT / 'pair_affinities.json').read_text:10, np.load:12.
- **[report.py](/Users/navneetbavineni/STEGO/audit/dino2/report.py:1)** — Builds a human-readable report from recorded results for DINO D2/D3 comparison.
  Definitions: fmt:24. Operations: dump:23, (OUT / 'DINO2.md').write_text:64, (OUT / 'weight_verification.json').read_text:2, (OUT / 'PROTOCOL.json').read_text:38, (OUT / 'pair_selectivity.json').read_text:2, (ROOT / 'audit/dino/summary.json').read_text:14, (OUT / f'{arm}_checks.json').read_text:43.
- **[run_all.py](/Users/navneetbavineni/STEGO/audit/dino2/run_all.py:1)** — Orchestrates the study scripts for DINO D2/D3 comparison.
  Definitions: run:5. Operations: run:15, run:22, subprocess.run:9, (H / 'weight_verification.json').read_text:11, run:13, run:20.
- **[validate.py](/Users/navneetbavineni/STEGO/audit/dino2/validate.py:1)** — Checks the study inputs, outputs or reproducibility assertions for DINO D2/D3 comparison.
  Operations: dump:31, (OUT / 'weight_verification.json').read_text:3, (OUT / 'pair_selectivity.json').read_text:9, np.load:11, np.load:18, np.load:12, (OUT / f'{arm}_checks.json').read_text:16.
- **[verify_weights.py](/Users/navneetbavineni/STEGO/audit/dino2/verify_weights.py:1)** — Computes local weight hashes and checks expected model artifacts for DINO D2/D3 comparison.
  Definitions: digest:8. Operations: (H / 'weight_verification.json').write_text:24, (folder / 'preprocessor_config.json').read_text:16, (folder / 'config.json').read_text:16.
### audit/ensemble

- **[generate_report.py](/Users/navneetbavineni/STEGO/audit/ensemble/generate_report.py:1)** — Top-level historical script in ensemble/refinement comparisons. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: json.load:12.
- **[run_ensembles.py](/Users/navneetbavineni/STEGO/audit/ensemble/run_ensembles.py:1)** — Historical utility in ensemble/refinement comparisons. Its implemented interface includes largest, guided, apply_f1, distances, Arm, embed_frozen, embed_untrained. Exact experiment status is not established by this guide.
  Definitions: largest:17, guided:23, apply_f1:30, distances:38, Arm:61, embed_frozen:69, embed_untrained:77, embed_c448:89. Operations: np.load:48, np.load:51, np.load:52, torch.load:86, m.load_state_dict:87, torch.load:98, m.load_state_dict:99.
### audit/ensemble/verify

- **[run.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify/run.py:1)** — Historical utility in ensemble/refinement comparisons. Its implemented interface includes largest, guided, apply_f1_path, postprocess_hard, embed_fn. Exact experiment status is not established by this guide.
  Definitions: largest:18, guided:24, apply_f1_path:32, postprocess_hard:40, embed_fn:61. Operations: np.load:49, np.load:50, np.load:84, (ROOT / f'audit/naming/inductive_s{s}_naming.json').read_text:82, KMeans(n_clusters=8, random_state=s, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:98, np.load:107, model.load_state_dict:71.
### audit/ensemble/verify4

- **[generate_n2_viz.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify4/generate_n2_viz.py:1)** — Historical utility in ensemble/refinement comparisons. Its implemented interface includes embedding, StegoHead, apply_postprocessing. Exact experiment status is not established by this guide.
  Definitions: embedding:33, StegoHead:39, apply_postprocessing:50. Operations: np.load:30, KMeans(n_clusters=8, random_state=s, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:76, np.load:31, pydicom.dcmread:89, plt.savefig:167.
- **[generate_p1_viz.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify4/generate_p1_viz.py:1)** — Refits/evaluates the same optimized P1 route and saves CT, ground-truth and prediction comparison PNGs.
  Definitions: embedding:33, StegoHead:39, apply_postprocessing:50. Operations: np.load:30, model.load_state_dict:68, KMeans(n_clusters=8, random_state=s, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:77, np.load:31, pydicom.dcmread:90, plt.savefig:168, torch.load:68.
- **[naming_logic.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify4/naming_logic.py:1)** — Indented code fragment; static parsing fails. Not a runnable standalone module.
  **Not runnable as written:** unexpected indent, line 1.
- **[run.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify4/run.py:1)** — Historical utility in ensemble/refinement comparisons. Its implemented interface includes largest, apply_postprocessing, iou_dice, get_train_embeddings, get_val_embeddings, run_naming, process_arm. Exact experiment status is not established by this guide.
  Definitions: largest:18, apply_postprocessing:24, iou_dice:40, get_train_embeddings:63, get_val_embeddings:89, run_naming:102, process_arm:167, evaluate_patient:230. Operations: np.load:52, np.load:55, np.load:57, pydicom.dcmread:110, np.load:178, model.load_state_dict:81, (ROOT / f'audit/naming/inductive_s{s}_naming.json').read_text:175.
- **[run_n2_only.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify4/run_n2_only.py:1)** — N2/refinement experiment with an untrained seed-9 head; not the trained P1 checkpoint.
  Definitions: embedding:30, StegoHead:36, apply_postprocessing:47. Operations: np.load:27, KMeans(n_clusters=8, random_state=s, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:74, np.load:28, pydicom.dcmread:87.
- **[run_optimized.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify4/run_optimized.py:1)** — Historical utility in ensemble/refinement comparisons. Its implemented interface includes embedding, StegoHead, get_val_embeddings, process_arm, apply_postprocessing, evaluate_patient. Exact experiment status is not established by this guide.
  Definitions: embedding:34, StegoHead:44, get_val_embeddings:55, process_arm:77, apply_postprocessing:195, evaluate_patient:213. Operations: np.load:28, np.load:29, np.load:32, KMeans(n_clusters=8, random_state=s, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:99, pydicom.dcmread:116.
- **[run_optimized2.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify4/run_optimized2.py:1)** — Historical utility in ensemble/refinement comparisons. Its implemented interface includes largest, apply_postprocessing, iou_dice, get_train_embeddings, get_val_embeddings, run_naming, process_arm. Exact experiment status is not established by this guide.
  Definitions: largest:18, apply_postprocessing:24, iou_dice:40, get_train_embeddings:63, get_val_embeddings:89, run_naming:102, process_arm:167, evaluate_patient:230. Operations: np.load:52, np.load:55, np.load:57, pydicom.dcmread:110, np.load:178, model.load_state_dict:81, (ROOT / f'audit/naming/inductive_s{s}_naming.json').read_text:175.
- **[run_optimized3.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify4/run_optimized3.py:1)** — Historical utility in ensemble/refinement comparisons. Its implemented interface includes embedding, StegoHead, get_val_embeddings, process_arm, apply_postprocessing, evaluate_patient. Exact experiment status is not established by this guide.
  Definitions: embedding:33, StegoHead:43, get_val_embeddings:54, process_arm:71, apply_postprocessing:180, evaluate_patient:196. Operations: np.load:27, np.load:28, np.load:31, KMeans(n_clusters=8, random_state=s, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:87, pydicom.dcmread:102.
- **[run_oracle.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify4/run_oracle.py:1)** — Label-assisted/oracle cluster-selection diagnostic; its scores must not be reported as label-free naming.
  Definitions: embedding:29, StegoHead:39, get_val_embeddings:50, apply_postprocessing:64. Operations: np.load:26, KMeans(n_clusters=8, random_state=s, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:93, np.load:27.
- **[run_p1_crf.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify4/run_p1_crf.py:1)** — Separate P1/CRF evaluation variant, not the selected guided-filter-only pipeline.
  Definitions: embedding:34, StegoHead:40, apply_postprocessing:51. Operations: np.load:31, model.load_state_dict:63, KMeans(n_clusters=8, random_state=s, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:72, np.load:32, pydicom.dcmread:85, torch.load:63.
- **[run_p1_optimized.py](/Users/navneetbavineni/STEGO/audit/ensemble/verify4/run_p1_optimized.py:1)** — Selected optimized pipeline: seed-0 nonlinear branch only, unnormalized head output, train K=8 fit, N2 naming, size-25 guided filter and threshold .30; prints validation metrics.
  Definitions: embedding:30, StegoHead:36, apply_postprocessing:47. Operations: np.load:27, model.load_state_dict:66, KMeans(n_clusters=8, random_state=s, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd').fit:75, np.load:28, pydicom.dcmread:88, torch.load:66.
### audit/final_pipeline

- **[build_scripts.py](/Users/navneetbavineni/STEGO/audit/final_pipeline/build_scripts.py:1)** — Creates package helper scripts for frozen-feature inference packaging.
  Operations: (OUT / 'segment_ct.py').write_text:32, (OUT / 'function_provenance.json').write_text:33, p.read_text:27.
- **[final_verify.py](/Users/navneetbavineni/STEGO/audit/final_pipeline/final_verify.py:1)** — Performs final package/output verification for frozen-feature inference packaging.
  Operations: np.load:10, dump:29, (HERE / 'function_provenance.json').read_text:3, (HERE / 'segment_ct.py').read_text:3, (HERE / 'cli_verification.json').read_text:18, (HERE / 'segment_ct.py').read_text:6, (HERE / f'{scope}_seed{seed}.json').read_text:21.
- **[package_check.py](/Users/navneetbavineni/STEGO/audit/final_pipeline/package_check.py:1)** — Checks package artifacts and their consistency for frozen-feature inference packaging.
  Operations: np.load:6, dump:14, dump:34, manifest_path.read_text:24, (OLD / 'metadata.json').read_text:3.
- **[report.py](/Users/navneetbavineni/STEGO/audit/final_pipeline/report.py:1)** — Builds a human-readable report from recorded results for frozen-feature inference packaging.
  Definitions: stats:5, fmt:8, pct:9, num:62. Operations: dump:41, (HERE / 'FINAL_PIPELINE.md').write_text:68, (HERE / f'inductive_seed{s}_results.json').read_text:3, (ROOT / f'audit/improve2/seed{s}.json').read_text:31.
- **[run_remaining.py](/Users/navneetbavineni/STEGO/audit/final_pipeline/run_remaining.py:1)** — Orchestrates unfinished package/study steps for frozen-feature inference packaging.
  Operations: subprocess.run:14, subprocess.run:12.
- **[segment_ct.py](/Users/navneetbavineni/STEGO/audit/final_pipeline/segment_ct.py:1)** — CLI and helpers for frozen-feature CT mask/overlay inference for frozen-feature inference packaging.
  Definitions: body_box:23, crop_image:34, image_tensor:41, inverse_grid:47, resize224:58, largest:62, guided:69, contract:76. Operations: p.parse_args:159, (a.output / 'run.json').write_text:203, pydicom.dcmread:100, (HERE / f'all20_seed{seed}.json').read_text:139, np.load:143, Image.fromarray(array).save:150, (HERE / 'check_reference.json').read_text:162.
- **[study.py](/Users/navneetbavineni/STEGO/audit/final_pipeline/study.py:1)** — Runs the inductive/frozen-feature fitting, mapping and evaluation study for frozen-feature inference packaging.
  Definitions: guard:10, reuse:27, confusion:37, dump:38, records:39, identity:40, fit:48, mapping:66. Operations: (HERE / name).write_text:38, np.load:53, np.lib.format.open_memmap:55, KMeans(n_clusters=8, random_state=seed, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd', copy_x=False).fit:58, np.save:63, dump:65, np.load:67.
- **[verify_cli.py](/Users/navneetbavineni/STEGO/audit/final_pipeline/verify_cli.py:1)** — Exercises the packaged command-line interface on fixed samples for frozen-feature inference packaging.
  Definitions: call:9. Operations: call:22, orderpath.write_text:28, call:29, call:30, Image.fromarray(np.zeros((512, 512), np.uint8)).save:32, call:33, dump:34.
- **[write_readme.py](/Users/navneetbavineni/STEGO/audit/final_pipeline/write_readme.py:1)** — Writes package documentation for frozen-feature inference packaging.
  Operations: (HERE / 'README.md').write_text:79.
### audit/final_pipeline_n3

- **[build_scripts.py](/Users/navneetbavineni/STEGO/audit/final_pipeline_n3/build_scripts.py:1)** — Creates package helper scripts for N3 frozen-feature packaging.
  Operations: (OUT / 'segment_ct.py').write_text:32, (OUT / 'function_provenance.json').write_text:33, p.read_text:27.
- **[final_verify.py](/Users/navneetbavineni/STEGO/audit/final_pipeline_n3/final_verify.py:1)** — Performs final package/output verification for N3 frozen-feature packaging.
  Operations: np.load:10, dump:29, (HERE / 'function_provenance.json').read_text:3, (HERE / 'segment_ct.py').read_text:3, (HERE / 'cli_verification.json').read_text:18, (HERE / 'segment_ct.py').read_text:6, (HERE / f'{scope}_seed{seed}.json').read_text:21.
- **[name_clusters.py](/Users/navneetbavineni/STEGO/audit/final_pipeline_n3/name_clusters.py:1)** — Attempts standalone N1/N2/N3 cluster naming and metadata writing. Currently has a SyntaxError and its wrapper passes a mismatched CLI argument.
  **Not runnable as written:** '(' was never closed, line 168.
- **[package_check.py](/Users/navneetbavineni/STEGO/audit/final_pipeline_n3/package_check.py:1)** — Checks package artifacts and their consistency for N3 frozen-feature packaging.
  Operations: np.load:6, dump:14, dump:34, manifest_path.read_text:24, (OLD / 'metadata.json').read_text:3.
- **[report.py](/Users/navneetbavineni/STEGO/audit/final_pipeline_n3/report.py:1)** — Builds a human-readable report from recorded results for N3 frozen-feature packaging.
  Definitions: stats:5, fmt:8, pct:9, num:62. Operations: dump:41, (HERE / 'FINAL_PIPELINE.md').write_text:68, (HERE / f'inductive_seed{s}_results.json').read_text:3, (ROOT / f'audit/improve2/seed{s}.json').read_text:31.
- **[run_remaining.py](/Users/navneetbavineni/STEGO/audit/final_pipeline_n3/run_remaining.py:1)** — Orchestrates unfinished package/study steps for N3 frozen-feature packaging.
  Operations: subprocess.run:14, subprocess.run:12.
- **[segment_ct.py](/Users/navneetbavineni/STEGO/audit/final_pipeline_n3/segment_ct.py:1)** — CLI and helpers for frozen-feature CT mask/overlay inference for N3 frozen-feature packaging.
  Definitions: body_box:23, crop_image:34, image_tensor:41, inverse_grid:47, resize224:58, largest:62, guided:69, contract:76. Operations: p.parse_args:163, (a.output / 'run.json').write_text:214, pydicom.dcmread:100, (HERE / f'all20_seed{seed}.json').read_text:139, np.load:146, subprocess.check_call:168, Image.fromarray(array).save:153.
- **[study.py](/Users/navneetbavineni/STEGO/audit/final_pipeline_n3/study.py:1)** — Runs the inductive/frozen-feature fitting, mapping and evaluation study for N3 frozen-feature packaging.
  Definitions: guard:10, reuse:27, confusion:37, dump:38, records:39, identity:40, fit:48, mapping:66. Operations: (HERE / name).write_text:38, np.load:53, np.lib.format.open_memmap:55, KMeans(n_clusters=8, random_state=seed, n_init=3, max_iter=100, tol=0.0001, algorithm='lloyd', copy_x=False).fit:58, np.save:63, dump:65, np.load:67.
- **[verify_cli.py](/Users/navneetbavineni/STEGO/audit/final_pipeline_n3/verify_cli.py:1)** — Exercises the packaged command-line interface on fixed samples for N3 frozen-feature packaging.
  Definitions: call:9. Operations: call:22, orderpath.write_text:28, call:29, call:30, Image.fromarray(np.zeros((512, 512), np.uint8)).save:32, call:33, dump:34.
- **[verify_naming.py](/Users/navneetbavineni/STEGO/audit/final_pipeline_n3/verify_naming.py:1)** — Checks N3 naming metadata/artifacts for N3 frozen-feature packaging.
  Operations: (HERE / 'naming_verification.json').write_text:25, (OLD / f'all20_seed{seed}.json').read_text:9, (HERE / f'all20_seed{seed}.json').read_text:10.
- **[write_readme.py](/Users/navneetbavineni/STEGO/audit/final_pipeline_n3/write_readme.py:1)** — Writes package documentation for N3 frozen-feature packaging.
  Operations: (HERE / 'README.md').write_text:79.
### audit/improve

- **[ceiling.py](/Users/navneetbavineni/STEGO/audit/improve/ceiling.py:1)** — Computes a label-using spatial-grid ceiling, not a segmentation method for crop/resolution/refinement study.
  Operations: np.lib.format.open_memmap:5, np.lib.format.open_memmap:6, np.load:29, dump:40, dump:41, (FINAL / 'summary.json').read_text:31.
- **[checks.py](/Users/navneetbavineni/STEGO/audit/improve/checks.py:1)** — Runs study-specific consistency checks for crop/resolution/refinement study.
  Operations: load:6, dump:23, (FORE / 'inventory.json').read_text:5, np.load:11, np.load:12.
- **[common.py](/Users/navneetbavineni/STEGO/audit/improve/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for crop/resolution/refinement study.
  Definitions: guard:8, dump:35, elapsed:36, note:37, Dimensions:46, confusion:58. Operations: (OUT / name).write_text:35, (FORE / 'common.py').read_text:40, (FORE / 'common.py').read_text:51, (FORE / 'task1.py').read_text:55.
- **[extract.py](/Users/navneetbavineni/STEGO/audit/improve/extract.py:1)** — Extracts the study feature arrays for crop/resolution/refinement study.
  Operations: p.parse_args:6, np.lib.format.open_memmap:27, dump:45, torch.hub.load:13, torch.load:14, model.load_state_dict:15, load:21.
- **[figures.py](/Users/navneetbavineni/STEGO/audit/improve/figures.py:1)** — Creates the study figures from saved arrays/results for crop/resolution/refinement study.
  Operations: np.load:6, np.load:7, fig.savefig:33, dump:34, (OUT / 'summary.json').read_text:4, (OUT / 'figure_selection.json').read_text:5, (OUT / 'B0_s0_scores.json').read_text:7.
- **[fit.py](/Users/navneetbavineni/STEGO/audit/improve/fit.py:1)** — Fits the declared clustering models and saves fit outputs for crop/resolution/refinement study.
  Operations: p.parse_args:7, np.lib.format.open_memmap:11, np.save:28, dump:29, km.fit:21.
- **[pipeline.py](/Users/navneetbavineni/STEGO/audit/improve/pipeline.py:1)** — Provides crop, resize, inverse mapping and scoring helpers for crop/resolution/refinement study.
  Definitions: metadata:8, body_box:9, crop_image:18, image_tensor:23, raw_image:27, inverse_grid:29, native_prediction:38, resize224:44. Operations: (OUT / 'metadata.json').read_text:8, np.load:64, np.load:65, np.load:68.
- **[postprocess.py](/Users/navneetbavineni/STEGO/audit/improve/postprocess.py:1)** — Applies the declared soft/guided/component refinement variants for crop/resolution/refinement study.
  Definitions: largest:14, guided:19. Operations: p.parse_args:4, np.load:8, np.load:9, (OUT / f'{parent}_s{a.seed}_scores.json').read_text:7, np.load:10, dump:61, (OUT / 'parent_selection.json').read_text:6.
- **[prepare.py](/Users/navneetbavineni/STEGO/audit/improve/prepare.py:1)** — Prepares metadata/input selection for the study for crop/resolution/refinement study.
  Operations: load:6, dump:20, (OUT / 'position_embedding_source.txt').write_text:32, dump:34, (FORE / 'inventory.json').read_text:6, (FINAL / 'verification.json').read_text:7.
- **[report.py](/Users/navneetbavineni/STEGO/audit/improve/report.py:1)** — Builds a human-readable report from recorded results for crop/resolution/refinement study.
  Definitions: fmt:22. Operations: dump:19, (OUT / 'IMPROVE.md').write_text:138, (OUT / 'ceiling.json').read_text:21, (OUT / 'figures.json').read_text:128, p.read_text:9, (OUT / 'parent_selection.json').read_text:24.
- **[score.py](/Users/navneetbavineni/STEGO/audit/improve/score.py:1)** — Computes the study metrics from previously produced inputs for crop/resolution/refinement study.
  Operations: p.parse_args:4, np.load:7, dump:24, np.load:8, (FINAL / f'k8_s{a.seed}_scores.json').read_text:19.
- **[select_parent.py](/Users/navneetbavineni/STEGO/audit/improve/select_parent.py:1)** — Selects the parent variant for exploratory refinement using recorded scores for crop/resolution/refinement study.
  Operations: dump:8, (OUT / f'{variant}_s{s}_scores.json').read_text:5.
- **[soft_weights.py](/Users/navneetbavineni/STEGO/audit/improve/soft_weights.py:1)** — Computes centre-distance-based soft cluster weights for crop/resolution/refinement study.
  Operations: p.parse_args:5, np.load:8, np.lib.format.open_memmap:9, dump:19, (OUT / f'{tag}_fit.json').read_text:7, (OUT / 'parent_selection.json').read_text:6.
- **[validate.py](/Users/navneetbavineni/STEGO/audit/improve/validate.py:1)** — Checks the study inputs, outputs or reproducibility assertions for crop/resolution/refinement study.
  Operations: dump:41, (OUT / 'summary.json').read_text:3, (FINAL / 'verification.json').read_text:34, (OUT / f'{variant}_s{seed}_scores.json').read_text:15, (OUT / 'figure_selection.json').read_text:6, (OUT / f'{variant}_s{seed}_fit.json').read_text:25, (OUT / f'B0_s0_scores.json').read_text:31.
### audit/improve2

- **[common.py](/Users/navneetbavineni/STEGO/audit/improve2/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for F1/F2 morphology study.
  Definitions: guard:6, reuse:28, dump:40, elapsed:41, metadata:42, surface:43, boundary_distances:44, boundary_summary:49. Operations: (OUT / name).write_text:40, path.read_text:29, (OLD / 'metadata.json').read_text:42, (ROOT / 'audit/scratch/forensics/task1.py').read_text:36.
- **[figures.py](/Users/navneetbavineni/STEGO/audit/improve2/figures.py:1)** — Creates the study figures from saved arrays/results for F1/F2 morphology study.
  Operations: np.load:9, fig.savefig:29, dump:31, (OUT / 'seed0.json').read_text:3, np.load:10, (OLD / 'figure_selection.json').read_text:4.
- **[report.py](/Users/navneetbavineni/STEGO/audit/improve2/report.py:1)** — Builds a human-readable report from recorded results for F1/F2 morphology study.
  Definitions: fmt:45, number:46. Operations: dump:43, (OUT / 'IMPROVE2.md').write_text:129, (OUT / f'seed{s}.json').read_text:3, np.load:4, (OUT / 'figures.json').read_text:121.
- **[run.py](/Users/navneetbavineni/STEGO/audit/improve2/run.py:1)** — Historical utility in F1/F2 morphology study. Its implemented interface includes loss_stats. Exact experiment status is not established by this guide.
  Definitions: loss_stats:18. Operations: p.parse_args:4, np.load:8, np.load:9, np.load:11, dump:81, (OLD / f'B2_448_s{a.seed}_scores.json').read_text:7, np.lib.format.open_memmap:13.
- **[validate.py](/Users/navneetbavineni/STEGO/audit/improve2/validate.py:1)** — Checks the study inputs, outputs or reproducibility assertions for F1/F2 morphology study.
  Operations: np.load:10, dump:36, np.load:21, (OUT / f'seed{s}.json').read_text:2.
### audit/naming/actual_predictions

- **[extract_visualize.py](/Users/navneetbavineni/STEGO/audit/naming/actual_predictions/extract_visualize.py:1)** — Extracts actual prediction/cluster arrays and generates mask visualizations; avoids reconstructing predictions from a PDF.
  Definitions: no_pdf:9, save_npy:23, save_png:27. Operations: np.load:19, np.load:20, (DEST / 'verification.json').write_text:87, (DEST / 'README.md').write_text:94, (BASE / 'image_records.json').read_text:18, (BASE / 'inductive_s0_naming.json').read_text:18, (BASE / 'inductive_s0_scores.json').read_text:20.
### audit/naming

- **[common.py](/Users/navneetbavineni/STEGO/audit/naming/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for label-free cluster naming.
  Definitions: guard:8, reuse:35, confusion:45, dump:46, sha:47, elapsed:52, raw_image:53, sealed_manifest:55. Operations: path.read_text:36, (OUT / name).write_text:46, p.read_text:57, (OUT / 'stageA_manifest.sha256').read_text:56, (ROOT / 'audit/scratch/forensics/task1.py').read_text:43.
- **[presentation.py](/Users/navneetbavineni/STEGO/audit/naming/presentation.py:1)** — Creates presentation-oriented naming output for label-free cluster naming.
  Operations: (p / 'report.py').read_text:4.
### audit/naming/publication

- **[audit_record.py](/Users/navneetbavineni/STEGO/audit/naming/publication/audit_record.py:1)** — Records publication output provenance for label-free cluster naming.
  Operations: np.load:8, (OUT / 'stageA_initial.log').read_text:19, (OUT / 'stageA.log').read_text:19, dump:21, (OUT / 'stageA_ledger.json').read_text:7, (OUT / 'stageB_started.json').read_text:7, (OUT / 'image_records.json').read_text:8.
- **[finalize.py](/Users/navneetbavineni/STEGO/audit/naming/publication/finalize.py:1)** — Finalizes the publication artifacts and their records for label-free cluster naming.
  Operations: dump:18, (P / 'FINAL_RESULTS.html').write_text:20, (P / 'README.md').write_text:21, (P / 'results.json').read_text:7, (P / 'reproduction_verification.json').read_text:7, (P / 'FINAL_RESULTS.md').read_text:20, (P / f'reproduction/{fit}_s{seed}_scores.json').read_text:10.
- **[report_results.py](/Users/navneetbavineni/STEGO/audit/naming/publication/report_results.py:1)** — Formats recorded results into publication tables and machine-readable exports for label-free cluster naming.
  Definitions: write_json:9, stat:10, fmt:11, pd:12, csvwrite:13, table:15, avg:46. Operations: fig.savefig:57, np.load:59, fig.savefig:67, (P / 'FINAL_RESULTS.md').write_text:134, (P / 'SUPPLEMENTARY_TABLES.md').write_text:136, (P / 'audit_verification.json').read_text:7, (P / 'stability.json').read_text:7.
- **[reproduce.py](/Users/navneetbavineni/STEGO/audit/naming/publication/reproduce.py:1)** — Checks reproduction of recorded prediction/results artifacts for label-free cluster naming.
  Operations: (BASE / 'stage_b.py').read_text:8, (HERE / 'reproduction_verification.json').write_text:26, subprocess.run:22, (BASE / f'{key}_scores.json').read_text:23, (HERE / 'reproduction' / f'{key}_scores.json').read_text:23.
- **[stability.py](/Users/navneetbavineni/STEGO/audit/naming/publication/stability.py:1)** — Computes stability comparisons from existing naming/prediction outputs for label-free cluster naming.
  Definitions: calc:9. Operations: np.load:8, dump:35, (OUT / 'image_records.json').read_text:8, np.load:14, (OUT / f'{fit}_s{s}_naming.json').read_text:14.
### audit/naming

- **[report.py](/Users/navneetbavineni/STEGO/audit/naming/report.py:1)** — Builds a human-readable report from recorded results for label-free cluster naming.
  Definitions: fmt:18, agree:19. Operations: dump:17, np.load:52, dump:67, (OUT / 'NAMING.md').write_text:71, (OUT / 'image_records.json').read_text:52, fig.savefig:63, (OUT / f'{fit}_s{s}_scores.json').read_text:3.
- **[run_b.py](/Users/navneetbavineni/STEGO/audit/naming/run_b.py:1)** — Orchestrates the Stage-B scoring/report sequence for label-free cluster naming.
  Operations: subprocess.run:8.
- **[seal.py](/Users/navneetbavineni/STEGO/audit/naming/seal.py:1)** — Creates/verifies the Stage-A artifact seal for label-free cluster naming.
  Operations: dump:11, (OUT / 'stageA_manifest.sha256').write_text:11, (OUT / 'stageA_manifest.json').read_text:7.
- **[stage_a.py](/Users/navneetbavineni/STEGO/audit/naming/stage_a.py:1)** — Fits/names clusters and seals label-free predictions before scoring for label-free cluster naming.
  Operations: dump:16, dump:21, np.load:23, np.save:40, np.save:76, np.save:86, dump:86.
- **[stage_b.py](/Users/navneetbavineni/STEGO/audit/naming/stage_b.py:1)** — Opens labels after sealing and computes prediction metrics for label-free cluster naming.
  Operations: p.parse_args:7, np.load:10, dump:65, dump:9, (OUT / f'{key}_naming.json').read_text:10, (OUT / 'image_records.json').read_text:10, np.load:12.
- **[validate.py](/Users/navneetbavineni/STEGO/audit/naming/validate.py:1)** — Checks the study inputs, outputs or reproducibility assertions for label-free cluster naming.
  Operations: np.load:4, dump:43, (OUT / 'stageA_ledger.json').read_text:3, (OUT / 'crop_keys.json').read_text:5, (OUT / 'image_records.json').read_text:5, (OUT / f'{key}_naming.json').read_text:9, (OUT / f'{key}_scores.json').read_text:9.
### audit/preflight

- **[diagnose.py](/Users/navneetbavineni/STEGO/audit/preflight/diagnose.py:1)** — Historical utility in label-free loss preflight. Its implemented interface includes guard, dump, read, sha, batch, monitors, diversity. Exact experiment status is not established by this guide.
  Definitions: guard:14, dump:44, read:45, sha:46, batch:65, monitors:68, diversity:74, core:92. Operations: np.load:60, np.load:61, (HERE / name).write_text:44, dump:93, dump:100, dump:144, dump:147.
- **[report.py](/Users/navneetbavineni/STEGO/audit/preflight/report.py:1)** — Builds a human-readable report from recorded results for label-free loss preflight.
  Definitions: put:18, table:21. Operations: dump:14, (HERE / 'PREFLIGHT.md').write_text:83, fig.savefig:94, fig.savefig:103, dump:104.
### audit/project_review_20261005

- **[check_artifacts.py](/Users/navneetbavineni/STEGO/audit/project_review_20261005/check_artifacts.py:1)** — Verifies P1 checkpoint loads and compares full versus nonlinear-only head outputs; extracts saved scores.
  Definitions: read:13, sha:14. Operations: np.load:19, (OUT / 'artifact_checks.json').write_text:63, path.read_text:16, torch.load:25, full.load_state_dict:26, reduced.load_state_dict:27, p.read_text:13.
- **[inventory.py](/Users/navneetbavineni/STEGO/audit/project_review_20261005/inventory.py:1)** — Inventories files, AST-parses project Python and identifies syntax errors and exact duplicates without executing project scripts.
  Definitions: category:6. Operations: (OUT / 'inventory_summary.json').write_text:43, (OUT / 'python_index.json').write_text:44.
- **[provenance_checks.py](/Users/navneetbavineni/STEGO/audit/project_review_20261005/provenance_checks.py:1)** — Matches checkpoint trainer hashes to Git history and checks splits, recorded monitors and normalization constants.
  Operations: torch.load:10, (OUT / 'provenance_checks.json').write_text:42, (ROOT / 'audit/naming/image_records.json').read_text:20, (ROOT / 'src/eval_segmentation.py').read_text:28, p.read_text:25, (P / 'P1/stage2/configs/locked.json').read_text:31.
### audit/scratch

- **[backbone_checks.py](/Users/navneetbavineni/STEGO/audit/scratch/backbone_checks.py:1)** — Historical utility in audit/scratch. Its implemented interface includes checked. Exact experiment status is not established by this guide.
  Definitions: checked:16. Operations: p.parse_args:13, OmegaConf.load:20.
### audit/scratch/baseline

- **[common.py](/Users/navneetbavineni/STEGO/audit/scratch/baseline/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for frozen-feature versus head baseline.
  Definitions: guard:8, dump:33, elapsed:34, note:35, Dimensions:44, confusion:56, evaluate:57. Operations: (OUT / name).write_text:33, (FORE / 'common.py').read_text:38, np.load:58, (OUT / 'samples.json').read_text:58, (FORE / 'common.py').read_text:49, (FORE / 'task1.py').read_text:53.
- **[extract.py](/Users/navneetbavineni/STEGO/audit/scratch/baseline/extract.py:1)** — Extracts the study feature arrays for frozen-feature versus head baseline.
  Definitions: patient:8. Operations: load:3, dump:12, np.lib.format.open_memmap:14, np.save:23, dump:25, (FORE / 'inventory.json').read_text:3, np.load:19.
- **[final_checks.py](/Users/navneetbavineni/STEGO/audit/scratch/baseline/final_checks.py:1)** — Checks the completed study output coverage for frozen-feature versus head baseline.
  Operations: (ROOT / 'audit/BASELINE.md').read_text:13, dump:16, (OUT / 'summary.json').read_text:3, p.read_text:5, p.read_text:14.
- **[heads.py](/Users/navneetbavineni/STEGO/audit/scratch/baseline/heads.py:1)** — Collects trained-head baseline results for frozen-feature versus head baseline.
  Operations: (FORE / 'task1.json').read_text:3, (FORE / 'inventory.json').read_text:3, np.load:6, dump:10.
- **[kmeans.py](/Users/navneetbavineni/STEGO/audit/scratch/baseline/kmeans.py:1)** — Fits and evaluates the baseline k-means variants for frozen-feature versus head baseline.
  Operations: p.parse_args:5, np.save:20, dump:23, np.load:11, km.fit:16, np.load:7, np.load:8.
- **[proxies.py](/Users/navneetbavineni/STEGO/audit/scratch/baseline/proxies.py:1)** — Measures label-free checkpoint-selection proxies and compares them with diagnostic outcomes for frozen-feature versus head baseline.
  Definitions: no_annotations:8. Operations: p.parse_args:5, torch.load:12, np.load:13, (ROOT / 'audit/run_records/fixes3_full_C/postrun_evidence.json').read_text:11, (FORE / 'inventory.json').read_text:12, load:17, dump:54.
- **[report.py](/Users/navneetbavineni/STEGO/audit/scratch/baseline/report.py:1)** — Builds a human-readable report from recorded results for frozen-feature versus head baseline.
  Operations: (ROOT / 'audit/BASELINE.md').write_text:94, dump:96, (OUT / 'heads.json').read_text:10, (OUT / 'extract.json').read_text:12, (OUT / 'samples.json').read_text:12, (OUT / 'verification.json').read_text:69, p.read_text:50.
- **[verify.py](/Users/navneetbavineni/STEGO/audit/scratch/baseline/verify.py:1)** — Checks the study assertions and saved output consistency for frozen-feature versus head baseline.
  Operations: dump:36, (OUT / 'heads.json').read_text:23, p.read_text:17, np.load:29, p.read_text:34, (FORE / 'task2.py').read_text:5.
### audit/scratch/crosspatient

- **[calibrate.py](/Users/navneetbavineni/STEGO/audit/scratch/crosspatient/calibrate.py:1)** — Estimates affinity shifts using the declared feature sample for cross-patient pair diagnostic.
  Definitions: forbid_annotations:5, capture:12, quantiles:23. Operations: p.parse_args:4, dump:38.
- **[common.py](/Users/navneetbavineni/STEGO/audit/scratch/crosspatient/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for cross-patient pair diagnostic.
  Definitions: guard:8, dump:33, elapsed:34, note:35, Dimensions:44, confusion:56, evaluate:57. Operations: (OUT / name).write_text:33, (FORE / 'common.py').read_text:38, np.load:58, (BASE / 'samples.json').read_text:58, (FORE / 'common.py').read_text:49, (FORE / 'task1.py').read_text:53.
- **[engine.py](/Users/navneetbavineni/STEGO/audit/scratch/crosspatient/engine.py:1)** — Reuses project loss/pair sampling functions for diagnostic batches for cross-patient pair diagnostic.
  Definitions: reuse_pair_function:4, initialize:7, batches:20, image_id_codes:26. Operations: load:8, (FORE / 'inventory.json').read_text:8, (OUT / f'{split}_records.json').read_text:10, np.load:14, np.load:15, (OUT / f'{split}_anchors.json').read_text:10, np.load:11.
- **[neighbours.py](/Users/navneetbavineni/STEGO/audit/scratch/crosspatient/neighbours.py:1)** — Builds/ranks the declared image-neighbor variants for cross-patient pair diagnostic.
  Definitions: ranked:14, ActualFeatureWrapper:25. Operations: load:24, (ROOT / 'src/precompute_knns.py').read_text:6, (FORE / 'inventory.json').read_text:24, np.load:35, np.save:48, dump:49, dump:61.
- **[purity.py](/Users/navneetbavineni/STEGO/audit/scratch/crosspatient/purity.py:1)** — Computes label-based pair purity using the project affinity helper for cross-patient pair diagnostic.
  Definitions: capture:22, summarize:33. Operations: p.parse_args:4, dump:50, calibration_path.read_text:5.
- **[rank_check.py](/Users/navneetbavineni/STEGO/audit/scratch/crosspatient/rank_check.py:1)** — Checks neighbor ranking/order consistency for cross-patient pair diagnostic.
  Operations: dump:13, (ROOT / 'src/precompute_knns.py').read_text:3, np.load:7.
- **[report.py](/Users/navneetbavineni/STEGO/audit/scratch/crosspatient/report.py:1)** — Builds a human-readable report from recorded results for cross-patient pair diagnostic.
  Definitions: enrich:6, comparison:7. Operations: (ROOT / 'audit/CROSSPATIENT.md').write_text:95, dump:95, (OUT / 'neighbours.json').read_text:3, (OUT / 'rank_check.json').read_text:3, (OUT / f'calibration_{s}.json').read_text:4, (OUT / f'purity_{s}.json').read_text:4.
- **[verify.py](/Users/navneetbavineni/STEGO/audit/scratch/crosspatient/verify.py:1)** — Checks the study assertions and saved output consistency for cross-patient pair diagnostic.
  Operations: (ROOT / 'audit/CROSSPATIENT.md').read_text:33, dump:34, (OUT / 'neighbours.json').read_text:3, (OUT / 'summary.json').read_text:3, np.load:9, calpath.read_text:5, (OUT / f'purity_{split}.json').read_text:5.
### audit/scratch

- **[data_checks.py](/Users/navneetbavineni/STEGO/audit/scratch/data_checks.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: OmegaConf.load:13, np.load:22.
- **[decision_cache_centering.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_cache_centering.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: (g.ROOT / 'src/modules.py').read_text:22, np.load:18.
- **[decision_cloud_weights.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_cloud_weights.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: torch.load:38.
- **[decision_families.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_families.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: pydicom.dcmread:18.
- **[decision_followup.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_followup.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: g.REPORT.read_text:8, subprocess.run:27, (folder / 'wandb-metadata.json').read_text:23, (folder / 'config.yaml').read_text:24, (folder / 'wandb-summary.json').read_text:28.
- **[decision_geometry.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_geometry.py:1)** — Historical utility in audit/scratch. Its implemented interface includes records, read_headers. Exact experiment status is not established by this guide.
  Definitions: records:7, read_headers:11. Operations: g.REPORT.read_text:6, pydicom.dcmread:18, pydicom.dcmread:13, canvas.save:45.
- **[decision_guard.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_guard.py:1)** — Historical utility in audit/scratch. Its implemented interface includes audit, append. Exact experiment status is not established by this guide.
  Definitions: audit:10, append:25. 
- **[decision_inventory.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_inventory.py:1)** — Historical utility in audit/scratch. Its implemented interface includes expired, err. Exact experiment status is not established by this guide.
  Definitions: expired:12, err:14. 
- **[decision_overlay_check.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_overlay_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: g.REPORT.read_text:5, pydicom.dcmread:10, pydicom.dcmread:21.
- **[decision_pickle_keys.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_pickle_keys.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Empty/package marker or imports/constants; no top-level function/class definition.
- **[decision_runs.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_runs.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: p.read_text:9, cfgpath.read_text:13.
- **[decision_samples.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_samples.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: pydicom.dcmread:31, canvas.save:64.
- **[decision_summarize.py](/Users/navneetbavineni/STEGO/audit/scratch/decision_summarize.py:1)** — Historical utility in audit/scratch. Its implemented interface includes json_after. Exact experiment status is not established by this guide.
  Definitions: json_after:5. Operations: g.REPORT.write_text:196, g.REPORT.read_text:4.
- **[diag_affinity.py](/Users/navneetbavineni/STEGO/audit/scratch/diag_affinity.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: p.parse_args:5, np.load:7, dump:42, (OUT / 'plan.json').read_text:6.
- **[diag_ceiling.py](/Users/navneetbavineni/STEGO/audit/scratch/diag_ceiling.py:1)** — Historical utility in audit/scratch. Its implemented interface includes scores, upsample_assignments. Exact experiment status is not established by this guide.
  Definitions: scores:19, upsample_assignments:23. Operations: p.parse_args:10, np.load:18, np.load:41, np.load:56, dump:68, (OUT / 'plan.json').read_text:11, probe.fit:44.
- **[diag_collapse.py](/Users/navneetbavineni/STEGO/audit/scratch/diag_collapse.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: torch.load:7, net.load_state_dict:9, probe.load_state_dict:10, np.load:11, dump:29, (OUT / 'plan.json').read_text:5.
- **[diag_common.py](/Users/navneetbavineni/STEGO/audit/scratch/diag_common.py:1)** — Historical utility in audit/scratch. Its implemented interface includes guard, dump, cfg, dataset, elapsed. Exact experiment status is not established by this guide.
  Definitions: guard:12, dump:30, cfg:32, dataset:35, elapsed:39. Operations: (OUT / name).write_text:31, OmegaConf.load:34.
- **[diag_extract.py](/Users/navneetbavineni/STEGO/audit/scratch/diag_extract.py:1)** — Historical utility in audit/scratch. Its implemented interface includes capture. Exact experiment status is not established by this guide.
  Definitions: capture:12. Operations: p.parse_args:6, open_memmap:19, open_memmap:20, dump:35, (OUT / 'plan.json').read_text:7, open_memmap:18.
- **[diag_plan.py](/Users/navneetbavineni/STEGO/audit/scratch/diag_plan.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: read_cache:19, dump:30, (ROOT / 'audit/DIAG.md').write_text:33.
- **[diag_report.py](/Users/navneetbavineni/STEGO/audit/scratch/diag_report.py:1)** — Historical utility in audit/scratch. Its implemented interface includes table, pct, fl, src, yn. Exact experiment status is not established by this guide.
  Definitions: table:10, pct:11, fl:12, src:13, yn:14. Operations: contact.save:73, (ROOT / 'audit/DIAG.md').write_text:79, dump:80, (OUT / 'plan.json').read_text:4, (OUT / 'collapse.json').read_text:9, (OUT / f'ceiling_b{b}_c{c}.json').read_text:7, (OUT / f'affinity_b{b}_c{c}.json').read_text:8.
- **[extract_lung.py](/Users/navneetbavineni/STEGO/audit/scratch/extract_lung.py:1)** — Historical utility in audit/scratch. Its implemented interface includes main. Exact experiment status is not established by this guide.
  Definitions: main:19. Operations: np.lib.format.open_memmap:89, pydicom.dcmread:63.
### audit/scratch/final_baseline

- **[common.py](/Users/navneetbavineni/STEGO/audit/scratch/final_baseline/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for all-patient LOPO baseline.
  Definitions: guard:8, dump:35, elapsed:36, note:37, Dimensions:46, confusion:58. Operations: (OUT / name).write_text:35, (FORE / 'common.py').read_text:40, (FORE / 'common.py').read_text:51, (FORE / 'task1.py').read_text:55.
- **[fit.py](/Users/navneetbavineni/STEGO/audit/scratch/final_baseline/fit.py:1)** — Fits the declared clustering models and saves fit outputs for all-patient LOPO baseline.
  Operations: p.parse_args:8, np.load:13, np.load:14, np.save:24, np.save:25, dump:31, (OUT / 'verification.json').read_text:11.
- **[report.py](/Users/navneetbavineni/STEGO/audit/scratch/final_baseline/report.py:1)** — Builds a human-readable report from recorded results for all-patient LOPO baseline.
  Definitions: stats:7, table:31. Operations: dump:19, report.write_text:112, (OUT / 'verification.json').read_text:3, (OUT / f'k{k}_s{s}_scores.json').read_text:5, (OUT / f'k{k}_s{s}_fit.json').read_text:6, report.read_text:22.
- **[score.py](/Users/navneetbavineni/STEGO/audit/scratch/final_baseline/score.py:1)** — Computes the study metrics from previously produced inputs for all-patient LOPO baseline.
  Definitions: stats:42. Operations: p.parse_args:5, np.load:9, load:13, np.load:14, dump:47, (OUT / 'verification.json').read_text:8, (FORE / 'inventory.json').read_text:12.
- **[validate.py](/Users/navneetbavineni/STEGO/audit/scratch/final_baseline/validate.py:1)** — Checks the study inputs, outputs or reproducibility assertions for all-patient LOPO baseline.
  Definitions: sha:4. Operations: dump:43, (OUT / 'verification.json').read_text:3, np.load:19, (OUT / f'k{k}_s{seed}_fit.json').read_text:14, (OUT / f'k{k}_s{seed}_scores.json').read_text:15.
- **[verify.py](/Users/navneetbavineni/STEGO/audit/scratch/final_baseline/verify.py:1)** — Checks the study assertions and saved output consistency for all-patient LOPO baseline.
  Definitions: sha:34. Operations: load:9, dump:47, (BASE / 'samples.json').read_text:7, (FORE / 'inventory.json').read_text:8, np.load:22, np.load:25.
### audit/scratch

- **[final_smoke.py](/Users/navneetbavineni/STEGO/audit/scratch/final_smoke.py:1)** — Historical utility in audit/scratch. Its implemented interface includes Metric. Exact experiment status is not established by this guide.
  Definitions: Metric:31. Operations: OmegaConf.load:13.
- **[fix_checks.py](/Users/navneetbavineni/STEGO/audit/scratch/fix_checks.py:1)** — Historical utility in audit/scratch. Its implemented interface includes config, metrics, modules, chaos_class, finish. Exact experiment status is not established by this guide.
  Definitions: config:17, metrics:19, modules:24, chaos_class:26, finish:31. Operations: p.parse_args:15, OmegaConf.load:18, (root / 'src/train_segmentation.py').read_text:56, loss.backward:68, (root / 'src/train_segmentation.py').read_text:77, (root / 'src/precompute_knns.py').read_text:127, (root / 'src/eval_segmentation.py').read_text:138.
- **[fixes2_cache_check.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes2_cache_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: (g.ROOT / 'src/data.py').read_text:8, (g.ROOT / 'src/precompute_knns.py').read_text:8, np.load:13, kc.read_cache:37, kc.read_cache:49.
- **[fixes2_centering_check.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes2_centering_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: (g.ROOT / 'src/modules.py').read_text:12.
- **[fixes2_guard.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes2_guard.py:1)** — Historical utility in audit/scratch. Its implemented interface includes audit, definitions, record. Exact experiment status is not established by this guide.
  Definitions: audit:11, definitions:26, record:31. Operations: (ROOT / path).read_text:27.
- **[fixes2_loader_check.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes2_loader_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: (g.ROOT / 'src/data.py').read_text:8, (g.ROOT / 'src/demo_segmentation.py').read_text:8, (g.ROOT / 'src/data.py').read_text:30, Image.fromarray(quant).save:18, (g.ROOT / 'src' / path).read_text:32.
- **[fixes2_overlays.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes2_overlays.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: (out / 'numbers.json').write_text:35, image.save:33, (out / 'numbers.json').read_text:11.
- **[fixes2_preprocess_check.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes2_preprocess_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: (g.ROOT / 'src/preprocess.py').read_text:10.
- **[fixes2_smoke.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes2_smoke.py:1)** — Historical utility in audit/scratch. Its implemented interface includes Metric. Exact experiment status is not established by this guide.
  Definitions: Metric:38. Operations: OmegaConf.load:12, F.cross_entropy(probe(code.detach()), small_label).backward:28, loss.backward:34.
- **[fixes2_split_check.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes2_split_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: load_splits:21.
- **[fixes2_unsupervised_check.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes2_unsupervised_check.py:1)** — Historical utility in audit/scratch. Its implemented interface includes NoUpdate, TinyNet, Harness, run. Exact experiment status is not established by this guide.
  Definitions: NoUpdate:16, TinyNet:21, Harness:27, run:38. Operations: OmegaConf.load:8, (g.ROOT / 'src/train_segmentation.py').read_text:12, run:47, run:53, run:59, loss.backward:36.
- **[fixes3_calibrate.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes3_calibrate.py:1)** — Historical utility in audit/scratch. Its implemented interface includes no_labels. Exact experiment status is not established by this guide.
  Definitions: no_labels:12. Operations: OmegaConf.load:18, read_cache:24, (OUT / 'calibration.json').write_text:57.
- **[fixes3_common.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes3_common.py:1)** — Historical utility in audit/scratch. Its implemented interface includes guard, record. Exact experiment status is not established by this guide.
  Definitions: guard:8, record:26. 
- **[fixes3_experiment.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes3_experiment.py:1)** — Historical utility in audit/scratch. Its implemented interface includes no_train_labels, BoundedRun, bounded_trainer. Exact experiment status is not established by this guide.
  Definitions: no_train_labels:12, BoundedRun:28, bounded_trainer:43. Operations: p.parse_args:8, OmegaConf.load:16, (OUT / f'{args.name}_result.json').write_text:70, (OUT / 'calibration.json').read_text:21, (ROOT / 'audit/scratch/diag/ceiling_b12_c0.json').read_text:62.
- **[fixes3_metric_check.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes3_metric_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: p.parse_args:4.
- **[fixes3_monitor_check.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes3_monitor_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: p.parse_args:3, (ROOT / 'src/train_segmentation.py').read_text:5, np.load:19.
- **[fixes3_results.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes3_results.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: p.read_text:6.
- **[fixes3_select.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes3_select.py:1)** — Historical utility in audit/scratch. Its implemented interface includes select. Exact experiment status is not established by this guide.
  Definitions: select:5. Operations: (OUT / 'selection.json').write_text:33, p.read_text:23.
- **[fixes3_unlabeled_check.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes3_unlabeled_check.py:1)** — Historical utility in audit/scratch. Its implemented interface includes deny, Opt, Net, Harness, run. Exact experiment status is not established by this guide.
  Definitions: deny:15, Opt:23, Net:28, Harness:31, run:39. Operations: p.parse_args:3, (ROOT / 'src/train_segmentation.py').read_text:4, run:44, OmegaConf.load:34, l.backward:37.
- **[fixes3_visual_check.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes3_visual_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: p.parse_args:3, (ROOT / 'src/train_segmentation.py').read_text:4, fig.savefig:20.
### audit/scratch/fixes4

- **[calibrate.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes4/calibrate.py:1)** — Estimates affinity shifts using the declared feature sample for D/E short-run experiment.
  Definitions: no_labels:15. Operations: OmegaConf.load:21, read_cache:27, (OUT / 'calibration.json').write_text:64.
- **[common.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes4/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for D/E short-run experiment.
  Definitions: guard:8, record:26. 
- **[evaluate.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes4/evaluate.py:1)** — Scores generated predictions or checkpoints against labels; evaluation is separate from fitting for D/E short-run experiment.
  Definitions: no_train_labels:24. Operations: (BASE / 'common.py').read_text:14, p.parse_args:20, resultpath.write_text:54, (runroot / 'result.json').read_text:23, load:30, resultpath.write_text:52, fig.savefig:49.
- **[process_check.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes4/process_check.py:1)** — Historical utility in D/E short-run experiment. Its implemented interface includes check. Exact experiment status is not established by this guide.
  Definitions: check:4. Operations: Path(output).write_text:19.
- **[report.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes4/report.py:1)** — Builds a human-readable report from recorded results for D/E short-run experiment.
  Definitions: criterion:5, pct:19, pair:20, table:21. Operations: path.read_text:60, path.write_text:67, (OUT / 'round_summary.json').write_text:68, (OUT / 'calibration.json').read_text:22, path.write_text:64, (OUT / f'{name}_evaluation.json').read_text:26, (OUT / 'experiments' / name / 'result.json').read_text:27.
- **[run.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes4/run.py:1)** — Historical utility in D/E short-run experiment. Its implemented interface includes no_train_labels, BoundedRun, bounded_trainer. Exact experiment status is not established by this guide.
  Definitions: no_train_labels:14, BoundedRun:34, bounded_trainer:54. Operations: p.parse_args:8, OmegaConf.load:17, OmegaConf.load:26, OmegaConf.save:31, (runroot / 'provenance.json').write_text:32, (runroot / 'result.json').write_text:72, (OUT / 'calibration.json').read_text:21.
- **[verify_round.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes4/verify_round.py:1)** — Top-level historical script in D/E short-run experiment. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: (OUT / 'verify_round.json').write_text:28, (folder / 'result.json').read_text:11, (folder / 'provenance.json').read_text:11, (OUT / f'{name}_process_check.json').read_text:11.
- **[verify_runner.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes4/verify_runner.py:1)** — Historical utility in D/E short-run experiment. Its implemented interface includes FakePath, FakeTrainer. Exact experiment status is not established by this guide.
  Definitions: FakePath:10, FakeTrainer:15. Operations: (ROOT / 'audit/scratch/seed_validation_experiment.py').read_text:7, (OUT / 'verify_runner.json').write_text:35, (OUT / 'run.py').read_text:9.
- **[verify_scoring.py](/Users/navneetbavineni/STEGO/audit/scratch/fixes4/verify_scoring.py:1)** — Top-level historical script in D/E short-run experiment. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: (BASE / 'common.py').read_text:9, np.load:11, (OUT / 'verify_scoring.json').write_text:15, (BASE / 'k8_c0_s0.json').read_text:11, (ROOT / 'audit/scratch/selectivity/records.json').read_text:13.
### audit/scratch

- **[followup_cache_portability.py](/Users/navneetbavineni/STEGO/audit/scratch/followup_cache_portability.py:1)** — Historical utility in audit/scratch. Its implemented interface includes dataset, identity. Exact experiment status is not established by this guide.
  Definitions: dataset:12, identity:14. Operations: kc.read_cache:33, kc.read_cache:37.
- **[followup_crf_check.py](/Users/navneetbavineni/STEGO/audit/scratch/followup_crf_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Empty/package marker or imports/constants; no top-level function/class definition.
- **[followup_geometry_check.py](/Users/navneetbavineni/STEGO/audit/scratch/followup_geometry_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Empty/package marker or imports/constants; no top-level function/class definition.
### audit/scratch/forensics

- **[backbone_check.py](/Users/navneetbavineni/STEGO/audit/scratch/forensics/backbone_check.py:1)** — Checks shared backbone identity/behavior for checkpoint failure forensics.
  Operations: dump:9, torch.load:2, torch.load:6.
- **[common.py](/Users/navneetbavineni/STEGO/audit/scratch/forensics/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for checkpoint failure forensics.
  Definitions: guard:7, dump:29, elapsed:30, inventory:31, load:36, dataset:41, scores:43, mapping_report:46. Operations: (OUT / name).write_text:29, torch.load:37, model.load_state_dict:39, (ROOT / 'audit/run_records/fixes3_full_C/postrun_evidence.json').read_text:33, (ROOT / f'audit/scratch/fixes3/fixes3_C_seed{i}_result.json').read_text:32.
- **[report.py](/Users/navneetbavineni/STEGO/audit/scratch/forensics/report.py:1)** — Builds a human-readable report from recorded results for checkpoint failure forensics.
  Definitions: put:12, table:13, pct:17, fmt:18. Operations: load:5, (ROOT / 'audit/FORENSICS.md').write_text:73, (OUT / 'summary.json').write_text:75, load:8, (OUT / n).read_text:4, load:71.
- **[task1.py](/Users/navneetbavineni/STEGO/audit/scratch/forensics/task1.py:1)** — Scores each checkpoint confusion matrix under majority, optimal-subset and diagnostic threshold mappings.
  Operations: dump:2, (ROOT / 'audit/FORENSICS.md').write_text:3, load:7, np.lib.format.open_memmap:8, dump:25, np.lib.format.open_memmap:11, np.lib.format.open_memmap:12.
- **[task2.py](/Users/navneetbavineni/STEGO/audit/scratch/forensics/task2.py:1)** — Compares checkpoint partitions and frozen-feature clustering with ARI/NMI.
  Definitions: agreement:6, compare:15. Operations: np.load:23, np.load:24, dump:39, (OUT / 'task1.json').read_text:21, np.load:21, (ROOT / 'audit/scratch/diag/plan.json').read_text:23, (OUT / 'val_order.json').read_text:23.
- **[task3.py](/Users/navneetbavineni/STEGO/audit/scratch/forensics/task3.py:1)** — Parses and plots the full-run validation trajectory.
  Operations: (ROOT / 'audit/run_records/fixes3_full_C/train.log').read_text:21, fig.savefig:31, dump:33, (root / 'validation_diagnostics.jsonl').read_text:6.
- **[task4.py](/Users/navneetbavineni/STEGO/audit/scratch/forensics/task4.py:1)** — Compares checkpoint correspondence losses on fixed train batches without training.
  Definitions: TraceImages:6, train_label_guard:9. Operations: load:14, dump:37.
- **[task5.py](/Users/navneetbavineni/STEGO/audit/scratch/forensics/task5.py:1)** — Scores saved checkpoint candidates for selection diagnostics.
  Operations: p.parse_args:4, torch.load:5, np.load:6, (ROOT / 'audit/run_records/fixes3_full_C/postrun_evidence.json').read_text:5, load:10, dump:41, (OUT / 'val_order.json').read_text:15.
### audit/scratch

- **[full_c_aux_audit.py](/Users/navneetbavineni/STEGO/audit/scratch/full_c_aux_audit.py:1)** — Historical utility in audit/scratch. Its implemented interface includes digest. Exact experiment status is not established by this guide.
  Definitions: digest:48. Operations: (OUT / 'tensorboard_validation_scalars.json').write_text:18, (OUT / 'wandb_validation_scalars.json').write_text:43, (OUT / 'auxiliary_verification.json').write_text:56, (OUT / 'postrun_evidence.json').read_text:6.
- **[full_c_evaluate_saved.py](/Users/navneetbavineni/STEGO/audit/scratch/full_c_evaluate_saved.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: (OUT / 'saved_checkpoint_evaluation.json').write_text:41, (OUT / 'postrun_evidence.json').read_text:17, torch.load:21, model.load_state_dict:23, fig.savefig:37.
- **[full_c_postrun_audit.py](/Users/navneetbavineni/STEGO/audit/scratch/full_c_postrun_audit.py:1)** — Historical utility in audit/scratch. Its implemented interface includes simple. Exact experiment status is not established by this guide.
  Definitions: simple:12. Operations: (OUT / 'train.log').read_text:29, (OUT / 'postrun_evidence.json').write_text:32, torch.load:20, (LOGDIR / 'validation_diagnostics.jsonl').read_text:11, (OUT / 'exit_code.txt').read_text:31, (ROOT / f'audit/scratch/fixes3/fixes3_C_seed{i}_result.json').read_text:31.
- **[full_c_write_summary.py](/Users/navneetbavineni/STEGO/audit/scratch/full_c_write_summary.py:1)** — Historical utility in audit/scratch. Its implemented interface includes add, metrics, mon, active, table. Exact experiment status is not established by this guide.
  Definitions: add:18, metrics:19, mon:20, active:21, table:22. Operations: (OUT / 'final_summary.txt').write_text:87, (OUT / 'postrun_evidence.json').read_text:5, (OUT / 'auxiliary_verification.json').read_text:5, (OUT / 'tensorboard_validation_scalars.json').read_text:8, (OUT / 'wandb_validation_scalars.json').read_text:8, (OUT / 'saved_checkpoint_evaluation.json').read_text:5, (OUT / 'commands.txt').read_text:31.
- **[future_baselines.py](/Users/navneetbavineni/STEGO/audit/scratch/future_baselines.py:1)** — Historical utility in audit/scratch. Its implemented interface includes collect, score. Exact experiment status is not established by this guide.
  Definitions: collect:30, score:59. Operations: p.parse_args:11, OmegaConf.load:26, loss.backward:58, opt.step:58.
- **[guard.py](/Users/navneetbavineni/STEGO/audit/scratch/guard.py:1)** — Historical utility in audit/scratch. Its implemented interface includes audit, record, definitions. Exact experiment status is not established by this guide.
  Definitions: audit:14, record:25, definitions:29. Operations: path.read_text:33.
- **[inventory.py](/Users/navneetbavineni/STEGO/audit/scratch/inventory.py:1)** — Historical utility in audit/scratch. Its implemented interface includes role. Exact experiment status is not established by this guide.
  Definitions: role:6. 
- **[log_fix.py](/Users/navneetbavineni/STEGO/audit/scratch/log_fix.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: p.read_text:12, p.write_text:15.
- **[numeric_checks.py](/Users/navneetbavineni/STEGO/audit/scratch/numeric_checks.py:1)** — Historical utility in audit/scratch. Its implemented interface includes Metric. Exact experiment status is not established by this guide.
  Definitions: Metric:10. Operations: total.backward:30, ce.backward:55, actual.mean().backward:27.
### audit/scratch/pairs

- **[codes.py](/Users/navneetbavineni/STEGO/audit/scratch/pairs/codes.py:1)** — Fits/scores k-means on learned codes and compares it with the checkpoint probe for pair purity and code-versus-probe study.
  Definitions: from_diagnostic_features:12, cached_forward:15. Operations: p.parse_args:5, load:8, torch.load:8, np.load:10, np.load:56, dump:62, (FORE / 'inventory.json').read_text:6.
- **[common.py](/Users/navneetbavineni/STEGO/audit/scratch/pairs/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for pair purity and code-versus-probe study.
  Definitions: guard:8, dump:33, elapsed:34, note:35, Dimensions:44, confusion:56, evaluate:57. Operations: (OUT / name).write_text:33, (FORE / 'common.py').read_text:38, np.load:58, (BASE / 'samples.json').read_text:58, (FORE / 'common.py').read_text:49, (FORE / 'task1.py').read_text:53.
- **[purity.py](/Users/navneetbavineni/STEGO/audit/scratch/pairs/purity.py:1)** — Computes label-based pair purity using the project affinity helper for pair purity and code-versus-probe study.
  Definitions: relations:16, spy_sample:20, hard_labels:25, capture:27, label_summary:62. Operations: load:5, np.load:10, dump:74, (OUT / 'records.json').read_text:4, (OUT / 'setup.json').read_text:4, np.load:7, np.load:8.
- **[report.py](/Users/navneetbavineni/STEGO/audit/scratch/pairs/report.py:1)** — Builds a human-readable report from recorded results for pair purity and code-versus-probe study.
  Operations: (ROOT / 'audit/PAIRS.md').write_text:75, dump:75, (OUT / 'setup.json').read_text:3, (OUT / 'purity.json').read_text:3, (OUT / f'codes_{name}.json').read_text:4.
- **[setup.py](/Users/navneetbavineni/STEGO/audit/scratch/pairs/setup.py:1)** — Prepares the study input inventory for pair purity and code-versus-probe study.
  Operations: load:4, np.load:14, dump:21, (FORE / 'inventory.json').read_text:4, dump:11, np.save:11, dump:13.
- **[verify.py](/Users/navneetbavineni/STEGO/audit/scratch/pairs/verify.py:1)** — Checks the study assertions and saved output consistency for pair purity and code-versus-probe study.
  Operations: load:16, dump:35, (OUT / 'setup.json').read_text:3, (OUT / 'purity.json').read_text:3, (OUT / f'codes_{name}.json').read_text:24, path.read_text:33.
### audit/scratch

- **[provenance.py](/Users/navneetbavineni/STEGO/audit/scratch/provenance.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: p.with_name('config.yaml').read_text:14, p.read_text:14.
- **[reference.py](/Users/navneetbavineni/STEGO/audit/scratch/reference.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Empty/package marker or imports/constants; no top-level function/class definition.
- **[run_visual_collapse_check.py](/Users/navneetbavineni/STEGO/audit/scratch/run_visual_collapse_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: torch.load:26, cluster.load_state_dict:30, linear.load_state_dict:31, OmegaConf.load:11, getattr(net, name).load_state_dict:29, (root / 'manifest.json').read_text:14.
- **[seed_support_check.py](/Users/navneetbavineni/STEGO/audit/scratch/seed_support_check.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: OmegaConf.load:6, (root / 'src/train_segmentation.py').read_text:7, OmegaConf.load:15.
- **[seed_validation_common.py](/Users/navneetbavineni/STEGO/audit/scratch/seed_validation_common.py:1)** — Historical utility in audit/scratch. Its implemented interface includes guard, record. Exact experiment status is not established by this guide.
  Definitions: guard:8, record:26. 
- **[seed_validation_experiment.py](/Users/navneetbavineni/STEGO/audit/scratch/seed_validation_experiment.py:1)** — Historical utility in audit/scratch. Its implemented interface includes no_train_labels, BoundedRun, bounded_trainer. Exact experiment status is not established by this guide.
  Definitions: no_train_labels:13, BoundedRun:30, bounded_trainer:46. Operations: p.parse_args:8, OmegaConf.load:17, OmegaConf.save:28, (OUT / f'{args.name}_result.json').write_text:73, (OUT / 'calibration.json').read_text:22, (ROOT / 'audit/scratch/diag/ceiling_b12_c0.json').read_text:65.
### audit/scratch/selectivity

- **[affinities.py](/Users/navneetbavineni/STEGO/audit/scratch/selectivity/affinities.py:1)** — Computes alternative-source patch-pair affinity arrays for affinity selectivity study.
  Definitions: no_labels:6, record_affinity:34, capture:38. Operations: load:15, np.save:29, np.load:33, dump:79, (FORE / 'inventory.json').read_text:15, path.read_text:11, np.load:17.
- **[common.py](/Users/navneetbavineni/STEGO/audit/scratch/selectivity/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for affinity selectivity study.
  Definitions: guard:8, dump:33, elapsed:34, note:35. Operations: (OUT / name).write_text:33, (FORE / 'common.py').read_text:38.
- **[extract.py](/Users/navneetbavineni/STEGO/audit/scratch/selectivity/extract.py:1)** — Extracts the study feature arrays for affinity selectivity study.
  Definitions: no_labels:5, capture:16. Operations: load:8, np.load:12, np.save:13, dump:13, np.load:23, np.load:33, np.load:37.
- **[report.py](/Users/navneetbavineni/STEGO/audit/scratch/selectivity/report.py:1)** — Builds a human-readable report from recorded results for affinity selectivity study.
  Definitions: pct:8, table:9, add:11. Operations: (ROOT / 'audit/SELECTIVITY.md').write_text:69, dump:70, (OUT / 'affinities.json').read_text:4, (OUT / 'extract.json').read_text:4, (OUT / 'verify.json').read_text:4, (OUT / f'score_{p}.json').read_text:5.
- **[score.py](/Users/navneetbavineni/STEGO/audit/scratch/selectivity/score.py:1)** — Computes the study metrics from previously produced inputs for affinity selectivity study.
  Definitions: basic:12, envelope:16, summarize:21. Operations: p.parse_args:5, np.load:9, dump:59, (OUT / 'affinities.json').read_text:6, (OUT / 'records.json').read_text:6, np.load:49, np.load:7.
- **[verify.py](/Users/navneetbavineni/STEGO/audit/scratch/selectivity/verify.py:1)** — Checks the study assertions and saved output consistency for affinity selectivity study.
  Operations: dump:33, (OUT / 'affinities.json').read_text:4, (OUT / 'extract.json').read_text:4, np.load:11, np.load:12, (OUT / f'score_{pool}.json').read_text:11, np.load:7.
### audit/scratch

- **[test_dcm.py](/Users/navneetbavineni/STEGO/audit/scratch/test_dcm.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: pydicom.dcmread:7.
- **[visual_checks.py](/Users/navneetbavineni/STEGO/audit/scratch/visual_checks.py:1)** — Top-level historical script in audit/scratch. See the operation evidence below; do not execute merely to discover its purpose.
  Operations: OmegaConf.load:16, canvas.save:47, pydicom.dcmread:45.
### audit/stego_ideas

- **[batch_shifts.py](/Users/navneetbavineni/STEGO/audit/stego_ideas/batch_shifts.py:1)** — Measures batch-wise shift calibration and code-loss admissibility for adjacent-slice and dynamic-shift ideas.
  Definitions: Capture:10. Operations: dump:30, np.load:5, (ROOT / 'audit/preflight/diagnose.py').read_text:7.
- **[common.py](/Users/navneetbavineni/STEGO/audit/stego_ideas/common.py:1)** — Supplies shared loading, metrics, path restrictions and output helpers for adjacent-slice and dynamic-shift ideas.
  Definitions: guard:9, read:31, dump:32, sha:33, reuse:38, ledger:41, feature_batch:46. Operations: np.load:43, (OUT / name).write_text:32, dump:41, Path(path).read_text:31, Path(path).read_text:39.
- **[pair_plan.py](/Users/navneetbavineni/STEGO/audit/stego_ideas/pair_plan.py:1)** — Defines same-image, neighbor and adjacent-slice comparison pools for adjacent-slice and dynamic-shift ideas.
  Definitions: make_pool:23. Operations: (ROOT / 'experiments/stego_biomedclip/preflight.py').read_text:16, np.save:20, dump:57, np.save:42.
- **[report.py](/Users/navneetbavineni/STEGO/audit/stego_ideas/report.py:1)** — Builds a human-readable report from recorded results for adjacent-slice and dynamic-shift ideas.
  Definitions: put:10, table:11, pct:15, row:50. Operations: (OUT / 'IDEAS.md').write_text:57, dump:58.
- **[score.py](/Users/navneetbavineni/STEGO/audit/stego_ideas/score.py:1)** — Computes the study metrics from previously produced inputs for adjacent-slice and dynamic-shift ideas.
  Definitions: summaries:19. Operations: dump:42, np.load:27, np.load:28.
### dino

- **[utils.py](/Users/navneetbavineni/STEGO/dino/utils.py:1)** — DINO reference utilities for augmentation, initialization, distributed training, schedules and weight loading.
  Definitions: GaussianBlur:36, Solarization:57, load_pretrained_weights:71, clip_gradients:102, cancel_gradients_last_layer:114, restart_from_checkpoint:122, cosine_scheduler:157, bool_flag:171. Operations: torch.load:131, torch.load:73, model.load_state_dict:81, torch.save:419, model.load_state_dict:97, value.load_state_dict:139, value.load_state_dict:143.
- **[vision_transformer.py](/Users/navneetbavineni/STEGO/dino/vision_transformer.py:1)** — Defines DINO ViT patch embedding, attention blocks, positional encoding and model constructors; not the BiomedCLIP implementation.
  Definitions: drop_path:25, DropPath:36, Mlp:47, Attention:66, Block:94, PatchEmbed:117, VisionTransformer:135, vit_tiny:259. 
### .

- **[download_lung_ct.py](/Users/navneetbavineni/STEGO/download_lung_ct.py:1)** — Copies pixel-bearing pydicom test DICOMs into a negatives folder; not a verified lung-only dataset downloader.
  Definitions: generate_negatives:6. Operations: pydicom.dcmread:17.
### experiments/dense_stego_biomedclip

- **[correspondence.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/correspondence.py:1)** — Creates affine two-view geometry and matched-location correspondence losses for the dense experiment.
  Definitions: affine_batch:8, views:13, transform:19, invert:20, matched_locations:22, sample:30, loss:31. 
- **[evaluate.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/evaluate.py:1)** — Scores generated predictions or checkpoints against labels; evaluation is separate from fitting for experiments/dense_stego_biomedclip.
  Definitions: confusion:12. Operations: dump:18, dump:73, np.load:20, np.load:62, np.load:47, fig.savefig:72, (HERE / 'predictions/stageA_manifest.sha256').read_text:13.
- **[integration_checks.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/integration_checks.py:1)** — Runs small checks of the integrated model/training path for experiments/dense_stego_biomedclip.
  Operations: dump:31, (HERE / 'stages.py').read_text:8.
- **[model.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/model.py:1)** — Dense refinement/upsampling correspondence model; distinct from the simple P1 70-D head.
  Definitions: Refine:5, UpsampleBlock:9, CorrespondenceModel:13. 
- **[preflight.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/preflight.py:1)** — Checks inputs, model configuration and readiness before training for experiments/dense_stego_biomedclip.
  Operations: torch.save:10, torch.load:10, b.load_state_dict:10, KMeans(n_clusters=8, n_init=3, max_iter=100, tol=0.0001, random_state=0, algorithm='lloyd').fit:13, dump:16, p.read_text:7.
- **[report.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/report.py:1)** — Builds a human-readable report from recorded results for experiments/dense_stego_biomedclip.
  Definitions: fmt:6. Operations: (HERE / 'RESULTS.md').write_text:33.
- **[run_suite.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/run_suite.py:1)** — Orchestrates experiment runs, including process-overlap checks for experiments/dense_stego_biomedclip.
  Definitions: overlap:8. Operations: (HERE / 'configs/locked.json').read_text:7, subprocess.run:9, (HERE / f'logs/{key}_process_check.json').write_text:19, subprocess.run:28, subprocess.run:23, subprocess.run:25.
- **[runtime.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/runtime.py:1)** — Supplies experiment paths, deterministic setup, input access controls and reusable inference helpers for experiments/dense_stego_biomedclip.
  Definitions: guard:9, sha:32, dump:37, read:39, reuse:40, records:49, load_image:51, seed_all:54. Operations: p.write_text:38, dump:69, Path(p).read_text:39, Path(path).read_text:41, (ROOT / 'audit/final_pipeline/segment_ct.py').read_text:46, (BASE / 'stageA_manifest.sha256').read_text:66.
- **[sanity.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/sanity.py:1)** — Runs small tensor/gradient sanity checks for experiments/dense_stego_biomedclip.
  Operations: dump:20, np.load:9, objective.backward:17.
- **[seal_predictions.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/seal_predictions.py:1)** — Hashes prediction outputs and records their completion before label-based evaluation for experiments/dense_stego_biomedclip.
  Operations: dump:15, (HERE / 'predictions/stageA_manifest.sha256').write_text:15.
- **[stages.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/stages.py:1)** — Runs fit/name/predict stages, generating centres, naming decisions and masks for experiments/dense_stego_biomedclip.
  Definitions: embedding:12, save_meta:20, score_fields:21. Operations: p.parse_args:3, torch.load:9, model.load_state_dict:9, np.load:10, dump:20, np.lib.format.open_memmap:33, KMeans(n_clusters=8, random_state=args.seed, n_init=c['kmeans_n_init'], max_iter=c['kmeans_max_iter'], tol=c['kmeans_tol'], algorithm=c['kmeans_algorithm']).fit:37.
- **[train.py](/Users/navneetbavineni/STEGO/experiments/dense_stego_biomedclip/train.py:1)** — Training entrypoint for experiments/dense_stego_biomedclip: builds the model/objective, updates weights and writes logs/checkpoints.
  Definitions: image_batch:14. Operations: p.parse_args:8, dump:40, loss.backward:28, opt.step:28, scheduler.step:28, torch.save:37, (HERE / 'configs/locked.sha256').read_text:9.
### experiments/feature_head

- **[run_ensemble.py](/Users/navneetbavineni/STEGO/experiments/feature_head/run_ensemble.py:1)** — Evaluates the feature-head ensemble experiment.
  Definitions: load_plane:12, main:17. Operations: parser.parse_args:21, np.load:27, Image.fromarray(pred.astype(np.uint8) * 255).save:116, Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save:117, (ROOT / f'audit/final_pipeline_n3/all20_seed{s}.json').read_text:28.
- **[train_cnn_head.py](/Users/navneetbavineni/STEGO/experiments/feature_head/train_cnn_head.py:1)** — Trains a small CNN on frozen features using k-means-derived pseudo-targets; separate from P1.
  Definitions: get_dice_loss:16, FeatureCNN:28, train_and_eval:42. Operations: np.load:44, np.load:45, np.load:95, Path('audit/naming/image_records.json').read_text:47, loss.backward:88, optimizer.step:89.
### experiments/self_training

- **[evaluate.py](/Users/navneetbavineni/STEGO/experiments/self_training/evaluate.py:1)** — Scores generated predictions or checkpoints against labels; evaluation is separate from fitting for experiments/self_training.
  Operations: dump:15, dump:41, np.load:17, (PACKAGE / 'all_predictions_sealed.sha256').read_text:9, np.load:26.
- **[prepare_dataset.py](/Users/navneetbavineni/STEGO/experiments/self_training/prepare_dataset.py:1)** — Builds image/pseudo-target arrays from frozen-feature k-means for self-training.
  Definitions: prepare:14. Operations: np.load:18, np.load:19, np.save:41, np.save:42, (BASE / 'image_records.json').read_text:15.
- **[runtime.py](/Users/navneetbavineni/STEGO/experiments/self_training/runtime.py:1)** — Supplies experiment paths, deterministic setup, input access controls and reusable inference helpers for experiments/self_training.
  Definitions: guard:14, sha:37, dump:42, read:44, reuse:45, records:54, load_image:56, seed_all:59. Operations: p.write_text:43, dump:74, Path(p).read_text:44, Path(path).read_text:46, (ROOT / 'audit/final_pipeline/segment_ct.py').read_text:51, (BASE / 'stageA_manifest.sha256').read_text:71.
- **[score.py](/Users/navneetbavineni/STEGO/experiments/self_training/score.py:1)** — Computes the study metrics from previously produced inputs for experiments/self_training.
  Definitions: score:5. Operations: np.load:17, np.lib.format.open_memmap:18, (BASE / 'image_records.json').read_text:14.
- **[train_fcn.py](/Users/navneetbavineni/STEGO/experiments/self_training/train_fcn.py:1)** — Trains/predicts with the small image-space segmentation network used for self-training.
  Definitions: MiniUNet:16, train:52, predict:90. Operations: np.load:54, np.load:55, torch.save:87, model.load_state_dict:94, np.lib.format.open_memmap:101, (HERE / 'predictions/prediction_manifest.json').write_text:125, torch.load:94.
- **[train_gpu_end_to_end.py](/Users/navneetbavineni/STEGO/experiments/self_training/train_gpu_end_to_end.py:1)** — Separate GPU image-space training route with augmentations and dataset loading.
  Definitions: get_training_augmentation:19, CTDataset:28, train:52. Operations: np.load:55, np.load:56, torch.save:96, loss.backward:88, optimizer.step:89.
### experiments/stego_biomedclip

- **[checks.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/checks.py:1)** — Runs study-specific consistency checks for experiments/stego_biomedclip.
  Operations: np.load:54, value.backward:60, dump:72, (HERE / 'reference/src/modules.py').read_text:13.
- **[evaluate.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/evaluate.py:1)** — Scores generated predictions or checkpoints against labels; evaluation is separate from fitting for experiments/stego_biomedclip.
  Definitions: confusion:12. Operations: dump:18, dump:67, dump:87, np.load:20, im.convert('RGB').save:91, im.convert('RGB').save:93, (HERE / 'predictions/stageA_manifest.sha256').read_text:13.
- **[geometry.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/geometry.py:1)** — Builds coordinate images and checks two-view alignment for experiments/stego_biomedclip.
  Definitions: coordinate_image:11, alignment:17. 
- **[integration_checks.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/integration_checks.py:1)** — Runs small checks of the integrated model/training path for experiments/stego_biomedclip.
  Operations: dump:27, file.read_text:26, (HERE / 'stages.py').read_text:6.
- **[model.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/model.py:1)** — Defines the experiment model/head and weighted objective for experiments/stego_biomedclip.
  Definitions: StegoHead:8, objective:28. 
- **[preflight.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/preflight.py:1)** — Checks inputs, model configuration and readiness before training for experiments/stego_biomedclip.
  Operations: np.load:16, np.save:28, dump:29, dump:32, (HERE / 'configs/locked.sha256').write_text:33, dump:36.
- **[record_trajectories.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/record_trajectories.py:1)** — Collects recorded training/collapse trajectories for experiments/stego_biomedclip.
  Definitions: rows:5. Operations: dump:30, path.read_text:6.
### experiments/stego_biomedclip/reference/src

- **[modules.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/reference/src/modules.py:1)** — Pinned upstream STEGO reference source used for implementation comparison; not the active CHAOS pipeline.
  Definitions: LambdaLayer:8, DinoFeaturizer:17, ResizeAndClassify:121, ClusterLookup:134, FeaturePyramidNet:164, DoubleConv:255, norm:275, average_norm:279. Operations: torch.load:47, self.model.load_state_dict:57, self.model.load_state_dict:62.
- **[precompute_knns.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/reference/src/precompute_knns.py:1)** — Pinned upstream STEGO reference source used for implementation comparison; not the active CHAOS pipeline.
  Definitions: get_feats:15, my_app:25. 
- **[train_segmentation.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/reference/src/train_segmentation.py:1)** — Pinned upstream STEGO reference source used for implementation comparison; not the active CHAOS pipeline.
  Definitions: get_class_labels:20, LitUnsupervisedSegmenter:53, my_app:387. Operations: trainer.fit:497, net_optim.step:228, cluster_probe_optim.step:229, linear_probe_optim.step:230.
### experiments/stego_biomedclip

- **[reference_loss.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/reference_loss.py:1)** — Implements the pinned STEGO correlation, sampling and restored-mean loss helper for experiments/stego_biomedclip.
  Definitions: norm:6, tensor_correlation:10, sample:14, super_perm:18, sample_nonzero_locations:24, ContrastiveCorrelationLoss:40. 
- **[report.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/report.py:1)** — Builds a human-readable report from recorded results for experiments/stego_biomedclip.
  Operations: (HERE / 'RESULTS.md').write_text:45, (HERE / f'logs/seed{seed}_train.jsonl').read_text:31.
- **[resume_seed1.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/resume_seed1.py:1)** — Historical seed-1 continuation runner; not a new-run entrypoint for experiments/stego_biomedclip.
  Definitions: batch:29. Operations: np.load:15, np.load:16, torch.load:17, model.load_state_dict:21, dump:62, dump:63, loss.backward:44.
- **[run_suite.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/run_suite.py:1)** — Orchestrates experiment runs, including process-overlap checks for experiments/stego_biomedclip.
  Definitions: overlap:6, run:19. Operations: subprocess.run:7, run:30, run:31, run:32, subprocess.run:22, (HERE / f'logs/seed{seed}_overlap.json').write_text:27, run:28.
- **[runtime.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/runtime.py:1)** — Supplies experiment paths, deterministic setup, input access controls and reusable inference helpers for experiments/stego_biomedclip.
  Definitions: guard:9, sha:32, dump:37, read:39, reuse:40, records:49, load_image:51, seed_all:54. Operations: p.write_text:38, dump:69, Path(p).read_text:39, Path(path).read_text:41, (ROOT / 'audit/final_pipeline/segment_ct.py').read_text:46, (BASE / 'stageA_manifest.sha256').read_text:66.
- **[seal_predictions.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/seal_predictions.py:1)** — Hashes prediction outputs and records their completion before label-based evaluation for experiments/stego_biomedclip.
  Operations: dump:15, (HERE / 'predictions/stageA_manifest.sha256').write_text:18.
- **[stages.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/stages.py:1)** — Runs fit/name/predict stages, generating centres, naming decisions and masks for experiments/stego_biomedclip.
  Definitions: embedding:12, save_meta:17, score_fields:18. Operations: p.parse_args:3, torch.load:9, model.load_state_dict:9, np.load:10, dump:17, np.lib.format.open_memmap:30, KMeans(n_clusters=8, random_state=args.seed, n_init=c['kmeans_n_init'], max_iter=c['kmeans_max_iter'], tol=c['kmeans_tol'], algorithm=c['kmeans_algorithm']).fit:34.
- **[train.py](/Users/navneetbavineni/STEGO/experiments/stego_biomedclip/train.py:1)** — Training entrypoint for experiments/stego_biomedclip: builds the model/objective, updates weights and writes logs/checkpoints.
  Definitions: batch:41. Operations: p.parse_args:11, np.load:25, np.load:26, dump:91, loss.backward:62, opt.step:62, torch.save:88.
### experiments/stego_calibrated

- **[check_state.py](/Users/navneetbavineni/STEGO/experiments/stego_calibrated/check_state.py:1)** — Checks state serialization and restoration using small updates for experiments/stego_calibrated.
  Definitions: update:11. Operations: torch.save:17, torch.load:20, dump:26, loss.backward:14, o.step:14.
- **[evaluate.py](/Users/navneetbavineni/STEGO/experiments/stego_calibrated/evaluate.py:1)** — Scores generated predictions or checkpoints against labels; evaluation is separate from fitting for experiments/stego_calibrated.
  Operations: dump:15, dump:41, np.load:17, (PACKAGE / 'all_predictions_sealed.sha256').read_text:9, np.load:26.
- **[model.py](/Users/navneetbavineni/STEGO/experiments/stego_calibrated/model.py:1)** — Defines the experiment model/head and weighted objective for experiments/stego_calibrated.
  Definitions: StegoHead:8, objective:28. 
- **[reference_loss.py](/Users/navneetbavineni/STEGO/experiments/stego_calibrated/reference_loss.py:1)** — Implements the pinned STEGO correlation, sampling and restored-mean loss helper for experiments/stego_calibrated.
  Definitions: norm:6, tensor_correlation:10, sample:14, super_perm:18, sample_nonzero_locations:24, ContrastiveCorrelationLoss:40. 
- **[run_stage2.py](/Users/navneetbavineni/STEGO/experiments/stego_calibrated/run_stage2.py:1)** — Orchestrates the post-screen stage-2 runs for experiments/stego_calibrated.
  Definitions: run:11. Operations: run:22, run:23, (HERE / 'screen_decision.json').read_text:6, run:18, subprocess.run:15, run:21.
- **[runtime.py](/Users/navneetbavineni/STEGO/experiments/stego_calibrated/runtime.py:1)** — Supplies experiment paths, deterministic setup, input access controls and reusable inference helpers for experiments/stego_calibrated.
  Definitions: guard:14, sha:37, dump:42, read:44, reuse:45, records:54, load_image:56, seed_all:59. Operations: p.write_text:43, dump:74, Path(p).read_text:44, Path(path).read_text:46, (ROOT / 'audit/final_pipeline/segment_ct.py').read_text:51, (BASE / 'stageA_manifest.sha256').read_text:71.
- **[seal_all.py](/Users/navneetbavineni/STEGO/experiments/stego_calibrated/seal_all.py:1)** — Checks and seals the selected runs together before evaluation for experiments/stego_calibrated.
  Operations: dump:22, (PACKAGE / 'all_predictions_sealed.sha256').write_text:23.
- **[seal_predictions.py](/Users/navneetbavineni/STEGO/experiments/stego_calibrated/seal_predictions.py:1)** — Hashes prediction outputs and records their completion before label-based evaluation for experiments/stego_calibrated.
  Operations: dump:15, (HERE / 'predictions/stageA_manifest.sha256').write_text:18.
- **[stages.py](/Users/navneetbavineni/STEGO/experiments/stego_calibrated/stages.py:1)** — Runs fit/name/predict stages, generating centres, naming decisions and masks for experiments/stego_calibrated.
  Definitions: embedding:13, save_meta:18, score_fields:19. Operations: p.parse_args:3, torch.load:10, model.load_state_dict:10, np.load:11, dump:18, np.lib.format.open_memmap:31, KMeans(n_clusters=8, random_state=args.seed, n_init=c['kmeans_n_init'], max_iter=c['kmeans_max_iter'], tol=c['kmeans_tol'], algorithm=c['kmeans_algorithm']).fit:35.
- **[state.py](/Users/navneetbavineni/STEGO/experiments/stego_calibrated/state.py:1)** — Captures/restores model, optimizer and random-generator state for experiments/stego_calibrated.
  Definitions: capture:6, restore:14. Operations: model.load_state_dict:16, optimizer.load_state_dict:16.
- **[train.py](/Users/navneetbavineni/STEGO/experiments/stego_calibrated/train.py:1)** — Training entrypoint for experiments/stego_calibrated: builds the model/objective, updates weights and writes logs/checkpoints.
  Definitions: overlap_check:11, main:23. Operations: p.parse_args:24, dump:27, load_splits:36, np.load:37, dump:57, torch.save:77, dump:79.
### experiments/stego_colocated

- **[check_state.py](/Users/navneetbavineni/STEGO/experiments/stego_colocated/check_state.py:1)** — Checks state serialization and restoration using small updates for experiments/stego_colocated.
  Definitions: update:12. Operations: torch.save:18, torch.load:21, dump:27, loss.backward:15, o.step:15.
- **[evaluate.py](/Users/navneetbavineni/STEGO/experiments/stego_colocated/evaluate.py:1)** — Scores generated predictions or checkpoints against labels; evaluation is separate from fitting for experiments/stego_colocated.
  Operations: dump:15, dump:41, np.load:17, (PACKAGE / 'all_predictions_sealed.sha256').read_text:9, np.load:26.
- **[loss.py](/Users/navneetbavineni/STEGO/experiments/stego_colocated/loss.py:1)** — Adds co-located adjacent-slice positives: centre the full affinity tensor, then reduce its diagonal; samples same-patient offsets ±1/±2.
  Definitions: ColocatedCorrelationLoss:6, adjacent_partners:55. 
- **[model.py](/Users/navneetbavineni/STEGO/experiments/stego_colocated/model.py:1)** — Defines the experiment model/head and weighted objective for experiments/stego_colocated.
  Definitions: StegoHead:8, objective:28. 
- **[reference_loss.py](/Users/navneetbavineni/STEGO/experiments/stego_colocated/reference_loss.py:1)** — Implements the pinned STEGO correlation, sampling and restored-mean loss helper for experiments/stego_colocated.
  Definitions: norm:6, tensor_correlation:10, sample:14, super_perm:18, sample_nonzero_locations:24, ContrastiveCorrelationLoss:40. 
- **[run_round.py](/Users/navneetbavineni/STEGO/experiments/stego_colocated/run_round.py:1)** — Orchestrates the declared screen and follow-on training stages for experiments/stego_colocated.
  Definitions: read:6, run:7. Operations: ap.parse_args:5, (P / 'screen_decision.json').write_text:28, run:40, p.read_text:6, subprocess.run:13, run:41, run:24.
- **[runtime.py](/Users/navneetbavineni/STEGO/experiments/stego_colocated/runtime.py:1)** — Supplies experiment paths, deterministic setup, input access controls and reusable inference helpers for experiments/stego_colocated.
  Definitions: guard:14, sha:37, dump:42, read:44, reuse:45, records:54, load_image:56, seed_all:59. Operations: p.write_text:43, dump:74, Path(p).read_text:44, Path(path).read_text:46, (ROOT / 'audit/final_pipeline/segment_ct.py').read_text:51, (BASE / 'stageA_manifest.sha256').read_text:71.
- **[seal_all.py](/Users/navneetbavineni/STEGO/experiments/stego_colocated/seal_all.py:1)** — Checks and seals the selected runs together before evaluation for experiments/stego_colocated.
  Operations: dump:22, (PACKAGE / 'all_predictions_sealed.sha256').write_text:23.
- **[stages.py](/Users/navneetbavineni/STEGO/experiments/stego_colocated/stages.py:1)** — Runs fit/name/predict stages, generating centres, naming decisions and masks for experiments/stego_colocated.
  Definitions: embedding:13, save_meta:18, score_fields:19. Operations: p.parse_args:3, torch.load:10, model.load_state_dict:10, np.load:11, dump:18, np.lib.format.open_memmap:31, KMeans(n_clusters=8, random_state=args.seed, n_init=c['kmeans_n_init'], max_iter=c['kmeans_max_iter'], tol=c['kmeans_tol'], algorithm=c['kmeans_algorithm']).fit:35.
- **[state.py](/Users/navneetbavineni/STEGO/experiments/stego_colocated/state.py:1)** — Captures/restores model, optimizer and random-generator state for experiments/stego_colocated.
  Definitions: capture:6, restore:14. Operations: model.load_state_dict:16, optimizer.load_state_dict:16.
- **[train.py](/Users/navneetbavineni/STEGO/experiments/stego_colocated/train.py:1)** — P1/control training loop. Current source differs from the checkpoint-matching historical producer; see the implementation review before rerunning.
  Definitions: overlap_check:13, main:25. Operations: p.parse_args:26, dump:29, load_splits:38, np.load:39, dump:72, torch.save:94, dump:96.
### experiments/stego_masked

- **[check_state.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/check_state.py:1)** — Checks state serialization and restoration using small updates for experiments/stego_masked.
  Definitions: update:11. Operations: torch.save:17, torch.load:20, dump:26, loss.backward:14, o.step:14.
- **[compute_masks.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/compute_masks.py:1)** — Builds masks used by the masked-feature experiment for experiments/stego_masked.
  Definitions: records:11, largest:15, create_masks:22. Operations: np.save:54, (BASE / 'image_records.json').read_text:12, pydicom.dcmread:34.
- **[compute_metrics.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/compute_metrics.py:1)** — Computes initial/preflight diagnostic metrics for the masked experiment for experiments/stego_masked.
  Definitions: get_step0_score:17, test_preflight_fractions:23. Operations: np.load:31.
- **[evaluate.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/evaluate.py:1)** — Scores generated predictions or checkpoints against labels; evaluation is separate from fitting for experiments/stego_masked.
  Operations: dump:15, dump:41, np.load:17, (PACKAGE / 'all_predictions_sealed.sha256').read_text:9, np.load:26.
- **[model.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/model.py:1)** — 768-D residual correction head plus an objective accepting salience/masking inputs.
  Definitions: StegoHead:8, objective:34. 
- **[reference_loss.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/reference_loss.py:1)** — Implements the pinned STEGO correlation, sampling and restored-mean loss helper for experiments/stego_masked.
  Definitions: norm:6, tensor_correlation:10, sample:14, super_perm:18, sample_nonzero_locations:24, ContrastiveCorrelationLoss:40. 
- **[run_stage2.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/run_stage2.py:1)** — Orchestrates the post-screen stage-2 runs for experiments/stego_masked.
  Definitions: run:9. Operations: run:19, run:20, run:15, subprocess.run:13, run:18.
- **[runtime.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/runtime.py:1)** — Supplies experiment paths, deterministic setup, input access controls and reusable inference helpers for experiments/stego_masked.
  Definitions: guard:14, sha:37, dump:42, read:44, reuse:45, records:54, load_image:56, seed_all:59. Operations: p.write_text:43, dump:74, Path(p).read_text:44, Path(path).read_text:46, (ROOT / 'audit/final_pipeline/segment_ct.py').read_text:51, (BASE / 'stageA_manifest.sha256').read_text:71.
- **[seal_all.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/seal_all.py:1)** — Checks and seals the selected runs together before evaluation for experiments/stego_masked.
  Operations: dump:20, (PACKAGE / 'all_predictions_sealed.sha256').write_text:21.
- **[seal_predictions.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/seal_predictions.py:1)** — Hashes prediction outputs and records their completion before label-based evaluation for experiments/stego_masked.
  Operations: dump:15, (HERE / 'predictions/stageA_manifest.sha256').write_text:18.
- **[stages.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/stages.py:1)** — Runs fit/name/predict stages, generating centres, naming decisions and masks for experiments/stego_masked.
  Definitions: embedding:13, save_meta:18, score_fields:19. Operations: p.parse_args:3, torch.load:10, model.load_state_dict:10, np.load:11, dump:18, np.lib.format.open_memmap:31, KMeans(n_clusters=8, random_state=args.seed, n_init=c['kmeans_n_init'], max_iter=c['kmeans_max_iter'], tol=c['kmeans_tol'], algorithm=c['kmeans_algorithm']).fit:35.
- **[state.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/state.py:1)** — Captures/restores model, optimizer and random-generator state for experiments/stego_masked.
  Definitions: capture:6, restore:14. Operations: model.load_state_dict:16, optimizer.load_state_dict:16.
- **[train.py](/Users/navneetbavineni/STEGO/experiments/stego_masked/train.py:1)** — Training entrypoint for experiments/stego_masked: builds the model/objective, updates weights and writes logs/checkpoints.
  Definitions: overlap_check:11, main:23. Operations: p.parse_args:24, dump:27, load_splits:36, np.load:37, np.load:38, dump:62, torch.save:84.
### experiments/stego_residual

- **[check_state.py](/Users/navneetbavineni/STEGO/experiments/stego_residual/check_state.py:1)** — Checks state serialization and restoration using small updates for experiments/stego_residual.
  Definitions: update:11. Operations: torch.save:17, torch.load:20, dump:26, loss.backward:14, o.step:14.
- **[evaluate.py](/Users/navneetbavineni/STEGO/experiments/stego_residual/evaluate.py:1)** — Scores generated predictions or checkpoints against labels; evaluation is separate from fitting for experiments/stego_residual.
  Operations: dump:15, dump:41, np.load:17, (PACKAGE / 'all_predictions_sealed.sha256').read_text:9, np.load:26.
- **[model.py](/Users/navneetbavineni/STEGO/experiments/stego_residual/model.py:1)** — 768-D residual correction head: adds a learned correction to frozen features and normalizes for inference.
  Definitions: StegoHead:8, objective:34. 
- **[reference_loss.py](/Users/navneetbavineni/STEGO/experiments/stego_residual/reference_loss.py:1)** — Implements the pinned STEGO correlation, sampling and restored-mean loss helper for experiments/stego_residual.
  Definitions: norm:6, tensor_correlation:10, sample:14, super_perm:18, sample_nonzero_locations:24, ContrastiveCorrelationLoss:40. 
- **[run_stage2.py](/Users/navneetbavineni/STEGO/experiments/stego_residual/run_stage2.py:1)** — Orchestrates the post-screen stage-2 runs for experiments/stego_residual.
  Definitions: run:9. Operations: run:18, run:19, subprocess.run:13, run:17.
- **[runtime.py](/Users/navneetbavineni/STEGO/experiments/stego_residual/runtime.py:1)** — Supplies experiment paths, deterministic setup, input access controls and reusable inference helpers for experiments/stego_residual.
  Definitions: guard:14, sha:37, dump:42, read:44, reuse:45, records:54, load_image:56, seed_all:59. Operations: p.write_text:43, dump:74, Path(p).read_text:44, Path(path).read_text:46, (ROOT / 'audit/final_pipeline/segment_ct.py').read_text:51, (BASE / 'stageA_manifest.sha256').read_text:71.
- **[seal_all.py](/Users/navneetbavineni/STEGO/experiments/stego_residual/seal_all.py:1)** — Checks and seals the selected runs together before evaluation for experiments/stego_residual.
  Operations: dump:20, (PACKAGE / 'all_predictions_sealed.sha256').write_text:21.
- **[seal_predictions.py](/Users/navneetbavineni/STEGO/experiments/stego_residual/seal_predictions.py:1)** — Hashes prediction outputs and records their completion before label-based evaluation for experiments/stego_residual.
  Operations: dump:15, (HERE / 'predictions/stageA_manifest.sha256').write_text:18.
- **[stages.py](/Users/navneetbavineni/STEGO/experiments/stego_residual/stages.py:1)** — Runs fit/name/predict stages, generating centres, naming decisions and masks for experiments/stego_residual.
  Definitions: embedding:13, save_meta:18, score_fields:19. Operations: p.parse_args:3, torch.load:10, model.load_state_dict:10, np.load:11, dump:18, np.lib.format.open_memmap:31, KMeans(n_clusters=8, random_state=args.seed, n_init=c['kmeans_n_init'], max_iter=c['kmeans_max_iter'], tol=c['kmeans_tol'], algorithm=c['kmeans_algorithm']).fit:35.
- **[state.py](/Users/navneetbavineni/STEGO/experiments/stego_residual/state.py:1)** — Captures/restores model, optimizer and random-generator state for experiments/stego_residual.
  Definitions: capture:6, restore:14. Operations: model.load_state_dict:16, optimizer.load_state_dict:16.
- **[train.py](/Users/navneetbavineni/STEGO/experiments/stego_residual/train.py:1)** — Training entrypoint for experiments/stego_residual: builds the model/objective, updates weights and writes logs/checkpoints.
  Definitions: overlap_check:11, main:23. Operations: p.parse_args:24, dump:27, load_splits:36, np.load:37, dump:57, torch.save:77, dump:79.
### .

- **[patch.py](/Users/navneetbavineni/STEGO/patch.py:1)** — Rewrites the end of run_optimized.py to print evaluation summaries. Historical mutation script; not an inference dependency.
  Empty/package marker or imports/constants; no top-level function/class definition.
- **[patch2.py](/Users/navneetbavineni/STEGO/patch2.py:1)** — Derives run_optimized2.py from another evaluation script, changing the compared arms and refinement parameters. Historical mutation script.
  Empty/package marker or imports/constants; no top-level function/class definition.
### picie

- **[backbone.py](/Users/navneetbavineni/STEGO/picie/backbone.py:1)** — Defines ResNet variants for the PiCIE/reference feature pyramid.
  Definitions: conv3x3:23, conv1x1:29, BasicBlock:34, Bottleneck:74, ResNet:123, _resnet:227, resnet18:236, resnet34:247. Operations: model.load_state_dict:232.
- **[fpn.py](/Users/navneetbavineni/STEGO/picie/fpn.py:1)** — Defines the PiCIE panoptic feature-pyramid decoder.
  Definitions: PanopticFPN:7, FPNDecoder:19. 
### src

- **[__init__.py](/Users/navneetbavineni/STEGO/src/__init__.py:1)** — Marks src as a Python package; currently empty.
  Empty/package marker or imports/constants; no top-level function/class definition.
- **[chaos.py](/Users/navneetbavineni/STEGO/src/chaos.py:1)** — Older conversion script: per-slice min/max to RGB PNG, rank-based mask pairing and its own patient split. Not the approved v2 CT preprocessing. Has top-level data-writing behavior and an install fallback.
  Definitions: dicom_to_png:49, save_zero_mask:58, process_patient_slices:62. Operations: pydicom.dcmread:50, Image.fromarray(np.zeros(shape, dtype=np.uint8)).save:59, img.save:76.
- **[chaos_preprocessing.py](/Users/navneetbavineni/STEGO/src/chaos_preprocessing.py:1)** — Defines approved intensity normalization, DICOM/PNG reading, image-mask association, native derivative generation and image transforms.
  Definitions: canonical_json:21, digest:25, file_sha256:29, preprocessing_contract:37, normalize_pixels:45, dicom_modality:63, read_image:76, pair_series:86. Operations: pydicom.dcmread:79, pydicom.dcmread:102, load_splits:133, pydicom.dcmread:71, Image.fromarray(array).save:184, load_splits:189.
- **[chaos_splits.py](/Users/navneetbavineni/STEGO/src/chaos_splits.py:1)** — Loads the frozen patient split and rejects duplicate or overlapping train/validation patients.
  Definitions: validate_splits:8, load_splits:21, patient_split:25. Operations: Path(path).read_text:22.
- **[collapse_monitor.py](/Users/navneetbavineni/STEGO/src/collapse_monitor.py:1)** — Computes code diversity/cluster usage and implements threshold-based collapse stopping.
  Definitions: CollapseMonitor:6, CollapseStopper:42. 
- **[count_empty_labels.py](/Users/navneetbavineni/STEGO/src/count_empty_labels.py:1)** — Diagnostic counting masks with no foreground.
  Empty/package marker or imports/constants; no top-level function/class definition.
- **[crf.py](/Users/navneetbavineni/STEGO/src/crf.py:1)** — Optional dense-CRF segmentation post-processing; not the optimized P1 guided-filter stage.
  Definitions: dense_crf:9. 
- **[crop_datasets.py](/Users/navneetbavineni/STEGO/src/crop_datasets.py:1)** — Creates cropped image datasets for the earlier STEGO data workflow.
  Definitions: _random_crops:14, RandomCropComputer:60, my_app:125. Operations: Image.fromarray(img_arr).save:116, Image.fromarray(label_arr).save:117.
- **[data.py](/Users/navneetbavineni/STEGO/src/data.py:1)** — Loads CHAOS images and masks and constructs paired/neighbor batches; also contains materialized and external-negative dataset wrappers.
  Definitions: bit_get:23, create_pascal_label_colormap:27, create_chaos_colormap:37, CHAOS:92, MaterializedDataset:245, ContrastiveSegDataset:273, NegativeImageDataset:424. Operations: load_splits:122, manifest_path.read_text:188, pydicom.dcmread:449, read_cache:347.
- **[demo_segmentation.py](/Users/navneetbavineni/STEGO/src/demo_segmentation.py:1)** — Earlier DICOM inference demonstration with checkpoint loading, cluster-to-class assignment and segmentation display/output.
  Definitions: get_device:30, clean_segmentation:36, DicomImageFolder:53, build_assignment_lut:87, get_cluster_assignments:121, apply_cluster_mapping:196, my_app:209. Operations: plt.savefig:276.
### src/dino

- **[utils.py](/Users/navneetbavineni/STEGO/src/dino/utils.py:1)** — DINO reference utilities for augmentation, initialization, distributed training, schedules and weight loading.
  Definitions: GaussianBlur:36, Solarization:57, load_pretrained_weights:71, clip_gradients:102, cancel_gradients_last_layer:114, restart_from_checkpoint:122, cosine_scheduler:157, bool_flag:171. Operations: torch.load:131, torch.load:73, model.load_state_dict:81, torch.save:419, model.load_state_dict:97, value.load_state_dict:139, value.load_state_dict:143.
- **[vision_transformer.py](/Users/navneetbavineni/STEGO/src/dino/vision_transformer.py:1)** — Defines DINO ViT patch embedding, attention blocks, positional encoding and model constructors; not the BiomedCLIP implementation.
  Definitions: drop_path:25, DropPath:36, Mlp:47, Attention:66, Block:94, PatchEmbed:117, VisionTransformer:135, vit_tiny:259. 
### src

- **[download_datasets.py](/Users/navneetbavineni/STEGO/src/download_datasets.py:1)** — Dataset download helper for the earlier workflow; may access the network and create files.
  Definitions: my_app:9. 
- **[download_models.py](/Users/navneetbavineni/STEGO/src/download_models.py:1)** — Downloads original STEGO/reference model artifacts; not needed to evaluate cached optimized P1.
  Empty/package marker or imports/constants; no top-level function/class definition.
- **[eval_segmentation.py](/Users/navneetbavineni/STEGO/src/eval_segmentation.py:1)** — Current implementation evaluates a three-seed frozen-feature ensemble using all20 centres. It does not load P1. Review flags ImageNet normalization instead of BiomedCLIP normalization.
  Definitions: load_plane:17, main:22. Operations: parser.parse_args:26, np.load:30, (ROOT / 'audit/naming/image_records.json').read_text:31, np.load:35, (ROOT / f'audit/final_pipeline_n3/all20_seed{s}.json').read_text:36, Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save:132.
- **[knn_cache.py](/Users/navneetbavineni/STEGO/src/knn_cache.py:1)** — Builds cache identity, validates metadata and indices, and handles cache naming/writing and explicit archival.
  Definitions: git_commit:17, expected_identity:25, cache_path:49, array_digest:53, validate_indices:57, cache_metadata:69, read_cache:75, prepare_target:93. Operations: np.load:76.
- **[modules.py](/Users/navneetbavineni/STEGO/src/modules.py:1)** — Defines DINO/BiomedCLIP feature extractors, projection heads, ClusterLookup and correspondence/CRF losses. Shared model components; not a complete training command.
  Definitions: LambdaLayer:8, DinoFeaturizer:17, BiomedCLIPFeaturizer:122, ResizeAndClassify:249, ClusterLookup:263, FeaturePyramidNet:312, DoubleConv:403, norm:423. Operations: torch.load:48, self.model.load_state_dict:58, self.model.load_state_dict:63, json.load:141.
### src/picie

- **[backbone.py](/Users/navneetbavineni/STEGO/src/picie/backbone.py:1)** — Defines ResNet variants for the PiCIE/reference feature pyramid.
  Definitions: conv3x3:23, conv1x1:29, BasicBlock:34, Bottleneck:74, ResNet:123, _resnet:227, resnet18:236, resnet34:247. Operations: model.load_state_dict:232.
- **[fpn.py](/Users/navneetbavineni/STEGO/src/picie/fpn.py:1)** — Defines the PiCIE panoptic feature-pyramid decoder.
  Definitions: PanopticFPN:7, FPNDecoder:19. 
### src

- **[plot_dino_correspondence.py](/Users/navneetbavineni/STEGO/src/plot_dino_correspondence.py:1)** — Plots feature correspondence/similarity maps for inspecting spatial relationships.
  Definitions: plot_heatmap:19, get_heatmaps:39, my_app:62. Operations: animation.ArtistAnimation(fig, frames, blit=True).save:246.
- **[plot_potsdam.py](/Users/navneetbavineni/STEGO/src/plot_potsdam.py:1)** — Visualization script for the original Potsdam dataset workflow, not CHAOS P1.
  Definitions: my_app:12. Operations: Image.fromarray(reshaped_img.cpu().numpy()).save:94, Image.fromarray(reshaped_preds).save:95.
- **[plot_pr_curves.py](/Users/navneetbavineni/STEGO/src/plot_pr_curves.py:1)** — Feature-affinity/recalibration diagnostic and precision-recall plotting experiment.
  Definitions: super_perm:22, prep_fd_coord:28, prep_fd:34, prep_fd_2:40, plot_auc_raw:47, CRFModule:53, LitRecalibrator:70, my_app:227. Operations: trainer.fit:290.
- **[precompute_knns.py](/Users/navneetbavineni/STEGO/src/precompute_knns.py:1)** — Extracts pooled image features, finds nearest-neighbor images and writes a validated neighbor cache.
  Definitions: get_feats:17, my_app:27. Operations: write_cache:107.
- **[preprocess.py](/Users/navneetbavineni/STEGO/src/preprocess.py:1)** — CLI plans or writes the new CHAOS derivatives through chaos_preprocessing; also retains older preprocessing functions. Dry-run is default; --write generates outputs.
  Definitions: load_dicom_series_as_sitk:29, load_mask_stack_as_sitk:50, resample_volume:79, clip_and_normalize:109, resize_image:122, resize_mask:126, process_patient:134, process_chaos_dataset:186. Operations: parser.parse_args:280, parse_args:288.
- **[show_rgb.py](/Users/navneetbavineni/STEGO/src/show_rgb.py:1)** — Image-display diagnostic for RGB conversion/preprocessing.
  Operations: plt.savefig:43.
- **[test_biomedclip.py](/Users/navneetbavineni/STEGO/src/test_biomedclip.py:1)** — Standalone BiomedCLIP loading/shape smoke check; despite its test_ name, not the complete unit-test suite.
  Definitions: main:7. Operations: OmegaConf.load:10.
- **[test_timm.py](/Users/navneetbavineni/STEGO/src/test_timm.py:1)** — RGB-pixel self/neighbor correspondence visualization; does not extract backbone features. Its filename is misleading.
  Definitions: load_example_images:18, compute_feature_similarity:58, visualize_self_correspondence:72, visualize_knn_correspondence:114, main:171. Operations: fig.savefig:279, fig_knn.savefig:280.
### src/tests

- **[conftest.py](/Users/navneetbavineni/STEGO/src/tests/conftest.py:1)** — Creates synthetic DICOM fixtures and helpers for the unit tests.
  Definitions: write_dicom:15, dicom_factory:50, read_dicom:64. 
- **[test_data.py](/Users/navneetbavineni/STEGO/src/tests/test_data.py:1)** — Unit tests for data. Tests are checks, not training/inference entrypoints; this guide does not assert they currently pass.
  Definitions: _train_root:23, _add_ct_patient:29, _add_mr_patient:46, _simple_transforms:60, test_bit_get_and_colormaps:64, test_load_mask_maps_ct_and_mri_pixels_and_preserves_l_mode:82, test_load_dicom_as_pil_applies_fixed_abdominal_window_and_metadata_defaults:93, test_chaos_rejects_bad_modality_and_image_set:120. Operations: Image.fromarray(raw, mode='L').save:85, pydicom.dcmread:112, Image.fromarray(raw, mode='L').save:167, Image.fromarray(np.full((16, 16), 255, dtype=np.uint8), mode='L').save:191, Image.fromarray(np.full((16, 16), 63, dtype=np.uint8), mode='L').save:192, (image_dir / 'not_png.txt').write_text:193, Image.fromarray(np.full((16, 16), 1, dtype=np.uint8), mode='L').save:206.
- **[test_modules.py](/Users/navneetbavineni/STEGO/src/tests/test_modules.py:1)** — Unit tests for modules. Tests are checks, not training/inference entrypoints; this guide does not assert they currently pass.
  Definitions: test_norm_normalizes_rows_and_zero_rows_are_finite:10, test_average_norm_matches_hand_computed_value:18, test_tensor_correlation_matches_explicit_reference:24, test_sample_corner_coordinate_returns_corner_pixel_with_align_corners:39, test_super_perm_is_in_range_and_has_no_fixed_points:52, test_sample_nonzero_locations_stays_in_nonzero_quadrant_and_handles_empty:61, test_lambda_layer_applies_lambda:75, test_cluster_lookup_init_from_copies_truncates_and_zeroes_tail:80. 
- **[test_preprocess.py](/Users/navneetbavineni/STEGO/src/tests/test_preprocess.py:1)** — Unit tests for preprocess. Tests are checks, not training/inference entrypoints; this guide does not assert they currently pass.
  Definitions: _root:8, _ct_patient:15, _mr_patient:29, test_preprocess_ct_slice_windowed_uint8_and_constant_below_window:41, test_preprocess_mri_slice_constant_branch_and_n4_fallback:58, test_collect_samples_aligns_ct_by_index_and_mr_by_stem:83, test_process_full_dataset_writes_split_outputs_and_skips_corrupt_slice:107. Operations: (root / 'CT' / '.DS_Store').write_text:99, Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode='L').save:26, Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode='L').save:38.
- **[test_train_segmentation.py](/Users/navneetbavineni/STEGO/src/tests/test_train_segmentation.py:1)** — Unit tests for train segmentation. Tests are checks, not training/inference entrypoints; this guide does not assert they currently pass.
  Definitions: test_get_class_labels_for_chaos_and_rejects_unknown_dataset:9, test_train_import_is_hydra_guarded_and_add_spatial_coords_preserves_code:17. 
- **[test_utils.py](/Users/navneetbavineni/STEGO/src/tests/test_utils.py:1)** — Unit tests for utils. Tests are checks, not training/inference entrypoints; this guide does not assert they currently pass.
  Definitions: test_one_hot_feats_has_expected_shape_dtype_and_values:15, test_resize_preserves_batch_channels_and_resizes_spatial_dimensions:27, test_shuffle_returns_a_permutation_of_rows:33, test_normalize_unnorm_round_trip_and_unnormalize_clones_input:41, test_prep_for_plot_resize_and_rescale_modes:51, test_to_target_tensor_converts_l_image_to_int64_chw:72, test_prep_args_normalizes_supported_cli_forms_and_rejects_bare_args:80, test_get_transform_crops_image_and_label_to_requested_resolution:95. 
### src

- **[train_crf.py](/Users/navneetbavineni/STEGO/src/train_crf.py:1)** — Separate CRF experiment/parameter-optimization route; not P1 head training.
  Definitions: norm:21, prep:24, entropy:30, my_app:36. Operations: plt.savefig:74, loss.backward:129, optim.step:130.
- **[train_segmentation.py](/Users/navneetbavineni/STEGO/src/train_segmentation.py:1)** — Earlier Lightning training program: builds loaders and backbone/head, computes losses, manages probes, validation, optimizers and checkpoints. Not the saved P1 experiment producer.
  Definitions: get_class_labels:33, LitUnsupervisedSegmenter:40, my_app:591. Operations: trainer.fit:724, net_optim.step:303, cluster_probe_optim.step:304, linear_probe_optim.step:305, schedulers.step:314, sched.step:312, fig.savefig:476.
- **[unlabeled_training.py](/Users/navneetbavineni/STEGO/src/unlabeled_training.py:1)** — Wraps datasets to expose image inputs without labels in label-free training.
  Definitions: ImagesOnlyDataset:6. 
- **[utils.py](/Users/navneetbavineni/STEGO/src/utils.py:1)** — Shared metrics and utilities: Hungarian/many-to-one cluster mapping, transformations, normalization helpers, plotting, model loading and batch collation.
  Definitions: prep_for_plot:23, add_plot:35, shuffle:68, add_hparams_fixed:72, resize:82, one_hot_feats:86, load_model:90, UnNormalize:149. Operations: plt.savefig:37, torch.load:97, model.load_state_dict:100, torch.load:108, model.load_state_dict:111, image.save:54, torch.load:122.
- **[validation_visuals.py](/Users/navneetbavineni/STEGO/src/validation_visuals.py:1)** — Selects fixed validation slices and plots comparable images, masks and cluster outputs.
  Definitions: fixed_validation_indices:8, plot_fixed_slices:17. 
- **[vis_corr.py](/Users/navneetbavineni/STEGO/src/vis_corr.py:1)** — Interactive/figure-oriented correspondence inspection: load example images, build a featurizer, compute patch similarities and plot them.
  Definitions: norm:44, tensor_correlation:49, sample:54, build_featurizer:61, SimpleNamespace:73, make_cfg:78, get_transform:89, find_sample_indices:118. Operations: OmegaConf.load:81, fig.savefig:669, fig2.savefig:703.

## Configuration and documentation files

- `src/configs/train_config.yaml`: settings for the legacy Lightning training entrypoint; not the saved P1 settings.
- `src/configs/demo_config.yaml`: image/checkpoint/output settings for the earlier demo; contains environment-specific paths.
- `src/configs/eval_config.yaml`: older evaluation settings; the current rewritten src/eval_segmentation.py uses argparse/hardcoded artifact paths instead.
- `src/configs/plot_config.yml`: settings for the original plotting workflow.
- `src/configs/chaos_splits_v2.json`: fixed patient assignments, shared by approved preprocessing and loaders.
- Experiment `configs/locked.json`: exact declared settings for that experiment/stage. Do not substitute another folder’s config.
- Matching `.sha256` files: fingerprints for verifying configuration/artifact identity; no learned parameters.
- Local model `open_clip_config.json` / `preprocessor_config.json` / `config.json`: architecture and/or input normalization settings for their associated backbone weights.
- `src/README.md`: original upstream STEGO usage documentation; not a reliable command guide for every later CHAOS branch.
- Root `README.md`: navigation to the organized view.
- Root handoff Markdown file: project history, decisions and continuation context.
- `src/STEGO_Colab_Demo.ipynb`: upstream notebook demonstration. Not audited as the optimized P1 entrypoint.
- `src/stego.pdf`: method-reference paper, not executable code.
- `.github/workflows/tests.yml`: GitHub automated test job; installs its declared environment and invokes pytest.
- `.gitignore`: patterns controlling which files Git ignores; does not remove existing files.

## Research reports

- [AUDIT.md](/Users/navneetbavineni/STEGO/audit/AUDIT.md) — Original method/backbone correctness findings.
- [BASELINE.md](/Users/navneetbavineni/STEGO/audit/BASELINE.md) — Frozen k-means versus learned-head comparison and leave-one-patient-out mapping.
- [CROSSPATIENT.md](/Users/navneetbavineni/STEGO/audit/CROSSPATIENT.md) — Same-patient versus cross-patient positive-neighbor comparison.
- [DECISIONS.md](/Users/navneetbavineni/STEGO/audit/DECISIONS.md) — Evidence and unresolved decisions about preprocessing, provenance and caches.
- [DIAG.md](/Users/navneetbavineni/STEGO/audit/DIAG.md) — Feature ceiling, affinity calibration and collapse diagnosis.
- [FINAL_BASELINE.md](/Users/navneetbavineni/STEGO/audit/FINAL_BASELINE.md) — Frozen-feature baseline across the full labeled CT cohort.
- [FIXES.md](/Users/navneetbavineni/STEGO/audit/FIXES.md) — Implementation-change ledger for this numbered fixes round, with recorded checks and remaining issues.
- [FIXES2.md](/Users/navneetbavineni/STEGO/audit/FIXES2.md) — Implementation-change ledger for this numbered fixes round, with recorded checks and remaining issues.
- [FIXES3.md](/Users/navneetbavineni/STEGO/audit/FIXES3.md) — Implementation-change ledger for this numbered fixes round, with recorded checks and remaining issues.
- [FIXES4.md](/Users/navneetbavineni/STEGO/audit/FIXES4.md) — Implementation-change ledger for this numbered fixes round, with recorded checks and remaining issues.
- [FIXES5.md](/Users/navneetbavineni/STEGO/audit/FIXES5.md) — Implementation-change ledger for this numbered fixes round, with recorded checks and remaining issues.
- [FIXES6.md](/Users/navneetbavineni/STEGO/audit/FIXES6.md) — Implementation-change ledger for this numbered fixes round, with recorded checks and remaining issues.
- [FIXES7.md](/Users/navneetbavineni/STEGO/audit/FIXES7.md) — Implementation-change ledger for this numbered fixes round, with recorded checks and remaining issues.
- [FIXES8.md](/Users/navneetbavineni/STEGO/audit/FIXES8.md) — Implementation-change ledger for this numbered fixes round, with recorded checks and remaining issues.
- [FORENSICS.md](/Users/navneetbavineni/STEGO/audit/FORENSICS.md) — Cluster mapping versus information, checkpoint trajectories and objective/outcome diagnosis.
- [PAIRS.md](/Users/navneetbavineni/STEGO/audit/PAIRS.md) — Pair-label purity and whether learned codes or probe assignment lose liver specificity.
- [SELECTIVITY.md](/Users/navneetbavineni/STEGO/audit/SELECTIVITY.md) — How well feature affinity distinguishes liver-target from background-target pairs.
- [WRITEUP.md](/Users/navneetbavineni/STEGO/audit/WRITEUP.md) — Research narrative/results write-up; claims still need their underlying evidence.

Other study-local reports (IMPROVE.md, IMPROVE2.md, NAMING.md, DINO.md, DINO2.md, PREFLIGHT.md, IDEAS.md and VERIFY*.md) explain the experiments in their own directories. A report is not a model, and a quoted score must be tied to that report’s exact fit/mapping/evaluation protocol.

## Generated files: what every repeated family is for

- Raw `.dcm` files: CT/MRI pixel data and acquisition metadata. In the selected optimized pipeline, train CT DICOMs provide HU/body statistics for N2 naming.
- Dataset `images/*.png`: derivative image inputs. Dataset `labels/`, `Ground/` or `masks/` PNGs: reference annotations, not predictions. Names and parent directories distinguish these uses.
- `*_features.npy`, `dense_features.npy`, `fit_sample.npy`: cached vectors or fit populations, whose shape and source metadata specify backbone/head, grid, channels and sample order.
- `*centers.npy`: fitted k-means cluster centres; these are learned without gradient descent.
- `*neighbors*.npy` / kNN cache arrays: image-neighbor indices used for pair construction, not organ masks.
- `raw_clusters.npy`: low-resolution cluster IDs; `raw_clusters_native.npy`: cluster IDs mapped into the native image frame.
- `N1_hard.npy`, `N2_hard.npy`, `N3_hard.npy`: binary predictions before F1 under each naming rule. Corresponding `*_F1.npy` files contain post-processed predictions.
- `scoring_masks_*.npy`: ground-truth annotation arrays for scoring; not output from the model.
- `ct_native.npy`: native-frame intensity images saved alongside predictions.
- `train_targets.npy` / pseudo-label arrays: derived training targets in self-training experiments. They are not automatically the same as human annotations; inspect the producing script.
- `.pt/.pth/.ckpt/.safetensors`: model tensors and sometimes optimizer/RNG state. The format alone does not identify the architecture or training status; consult state keys and metadata.
- `image_records.json` / `metadata.json`: ordered image identifiers, file paths, patient membership, crop geometry or experiment-specific metadata.
- `fit.json`: clustering settings, fit population, temperature, centre/checkpoint digests and fit status.
- `naming.json`: cluster-to-organ decisions and supporting text/anatomical scores.
- `prediction_manifest.json`, `stageA_manifest.json`, `all_predictions_sealed.json`: output identities and hashes used to freeze predictions before label-based scoring.
- `results.json`, `*_scores.json`, confusion arrays: recorded scoring outputs; need the associated scale/mapping/patient aggregation definition.
- `*_access.json`, ledger files: recorded file-access/label-use history; these do not independently certify every future rerun.
- `*_train.jsonl`, `*_complete.json`, timing and monitor records: training loss, progress, collapse diagnostics and completion information.
- W&B event files, config, summary, output.log and media: experiment tracking data and visualizations for a particular run.
- PNG/PDF plots outside dataset directories: visual evidence or reports. They are not substitutes for original prediction arrays.
- Root log0/1/2 and *_check.txt: historical outputs; exact producer must be established rather than guessed from the seed-like filename.
- `__pycache__/*.pyc`: generated Python bytecode, not source code.
- `.venv-crf/`: installed Python environment/dependencies, not project-specific pipeline code.
- `.git/`: commits, references and version-control internals; do not manually reorganize these.
- `.DS_Store`: macOS folder-view metadata; no segmentation role.
- Empty `data/` or other unused directories: no behavior merely because the folder exists.

## Newly organized views

- `PROJECT_MAP/01_pipeline` through `10_original_tree`: navigation links only; no new model implementation. Editing through a link changes its original target.
- `PROJECT_MAP/ALL_FILES.csv`: original-file to organized-view mapping.
- `PROJECT_MAP/LINKS.csv`: each descriptive link and its original target.
- `PROJECT_MAP/VERIFICATION.json`: content-preservation and link-verification results for organization, not model accuracy.
- `optimized_p1_pipeline/`: selected-method copies plus shared-input links. FILE_MANIFEST.csv identifies each copy’s original source; VERIFICATION.json checks copying, not inference accuracy. Its provenance files describe the original training and are not interchangeable with its optimized evaluator.

Coverage: 355 original project Python files individually listed; copied code under optimized_p1_pipeline refers back to those originals. 81 historical utilities have intentionally limited role descriptions with source-operation evidence rather than unsupported claims. No source was imported, training started, or data changed to generate this guide.

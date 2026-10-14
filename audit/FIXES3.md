# Audit fixes 3

Branch audit-fixes-3; snapshot23a2415 from audit-fixes-2. Existing audit work was snapshotted before branching. No derivative/cache regeneration or edits to existing data/checkpoints. Each task gets its own commit.

The user chose reuse of the DIAG frozen-feature linear reference; new runs must not fit a supervised probe. Any final candidate choice uses only label-free collapse monitors, never validation IoU/Dice or ground-truth overlays.

## Task 1 — APPLIED

Changed src/utils.py: UnsupervisedMetrics.compute uses per-cluster majority mapping when extra_clusters>0, with ties/unused clusters mapped to background; aggregates a2×2 real-class histogram. K2 retains Hungarian. Adds liver_IoU/liver_Dice; src/eval_segmentation.py persists metrics and the exact cluster_to_class map, and uses real-class confusion labels. src/configs/train_config.yaml sets extra_clusters6 (K8 total). No training/objective label path is introduced.

Reproduce/verify: `python3 -B audit/scratch/fixes3_metric_check.py [--fixed]`. Before: perfect8-cluster confusion scored25% and map[0,0,0,0,1,0,0,0]. After:100% mIoU/IoU/Dice and map[0,0,0,0,1,1,1,1]; K2 Hungarian and non-perfect real-class denominators pass. Mapping uses validation labels only for evaluation: it is an oracle report, not deployable unsupervised naming. Old K2 checkpoints retain their own saved config; new K8 probe parameters cannot be loaded into a K2 model unchanged.

## Task 2 — APPLIED

Added src/collapse_monitor.py and validation hooks in src/train_segmentation.py; config switch collapse_early_stop defaultsFalse. Every validation logs native-grid L2-code population std (mean across channels), exact all-cross-image mean cosine, full-resolution raw-cluster usage fractions and natural-log entropy. Batch-independent accumulation; no labels enter these statistics. New validation_diagnostics.jsonl records optimizer steps, batches, monitors, evaluation metrics and the mapping. Sanity checks log monitors but do not count toward stopping.

Thresholds: cosine>0.99 OR std<0.01 for3 consecutive real validation checks; a healthy check resets the streak. Compared with collapsed cosine0.99991/std0.001105 in DIAG. `fixes3_monitor_check.py --fixed` measured an independent20-train frozen-feature reference cosine0.476688/std0.023266 (768-D; not directly dimension-comparable with70-D code std), and random70-D healthy code std0.119416. Before check reproduced missing hooks/switch. After: constant codes stop on third check, healthy resets, and split-batch/all-batch results agree. These are diagnostic thresholds, not validated clinical or segmentation-quality criteria.

## Task 3 — APPLIED

Added src/validation_visuals.py and targeted validation logging changes. Four native-order middle slices are selected once per validation from four distinct patients using metadata only. Selection is deterministic and independent of labels, batch size, RNG and predictions. Only those four outputs are retained for plots. Rows: input CT, ground-truth red overlay, raw K8 colors with ID legend, many-to-one mapped liver overlay. PNG paths and fixed sample IDs are recorded in validation_diagnostics.jsonl; sanity validation can omit the panel if it does not traverse all four selected samples.

Before `fixes3_visual_check.py`: reproduced random batch selection. The first rendering check aborted on the macOS interactive backend; the CPU test environment now explicitly uses Agg. After rerunning `--fixed`: four distinct patient selections remain stable and the synthetic K8 panel renders. Labels enter only the displayed ground truth and evaluation majority mapping. Per-run real overlays are produced in task5.

## Task 4 — VERIFIED; recommendation only

`python3 -B audit/scratch/fixes3_calibrate.py` uses actual project preprocessing, frozen final-block extractor, sample() and restored-mean ContrastiveCorrelationLoss.helper. It chooses the same seeded200 train images across16 patients without labels; uses55 existing-cache edges retained inside the sample. A runtime audit guard rejects annotation/validation-image reads; none occurred. No feature/kNN cache was generated. Output: audit/scratch/fixes3/calibration.json.

Config mapping: positive-intra25th percentile -> pos_intra_shift; kNN-positive25th -> pos_inter_shift; random90th/85th -> neg_inter_shift. This is the user's heuristic, NOT from the paper. Reproduced DIAG rounded values(0.4584,0.4525,0.7792) within0.00005 from fresh train-image forwards. A uses unchanged(0.18,0.12,0.46); B uses exact measured(0.4584466218948364,0.45247218012809753,0.7791634202003479); C uses the same positives and negative0.7459585666656494. Before: random attraction68.3058%; after applying thresholds to the same measured affinities:10%/15%, with75% intra and kNN attraction. Loss weights unchanged. Deterministic eval-mode calibration omits training dropout; short experiments test actual training with its existing dropout. No validation scores were read to choose these thresholds.

## Task 5 — preparation / predeclared comparison

Before fixes3_unlabeled_check.py: training always fit diagnostic CE and opened train masks. After --fixed: ImagesOnlyDataset returns identical approved image transforms with annotation-free compatibility placeholders; actual training_step has zero linear-probe gradients/objective contribution, and changing labels leaves the objective/head gradients unchanged. The approved baseline defaults train_linear_probeFalse. Training image/cache files remain untouched and feature-identity source files are unchanged.

All A/B/C runs use seed0, K8, batch16, all1869 train slices, all472 validation slices, val_check_interval50 (checks at batch50/100 inside each117-batch epoch; a terminal validation is added if the budget ends between checks), existing dropout and unmodified loss weights. cfg.max_steps remains5000 ONLY to preserve the baseline cosine-LR schedule; the runner's Trainer max_steps1500 plus an independent callback hard-cap500 batches prevent any full run. Three optimizer steps still count per batch; the zero-gradient linear step preserves accounting and changes no linear parameter. Early-stop enabled. New checkpoints/logs use unique audit/scratch/fixes3/experiments names. Existing DIAG final-block linear reference60.8896% IoU is reused (100-val-slice scope), not fitted on new runs.

Predeclared LABEL-FREE selection: eligible runs must reach500 batches without collapse early-stop, finish at cosine<=0.99 and std>=0.01, and use at least two clusters at>=1% each. Among eligible runs choose lowest final cross-image cosine, then highest cluster entropy if cosine ties within1e-4. This selects a non-collapse candidate, NOT the best anatomical segmentation. If none qualifies, no winner is claimed. Validation labels, scores, mappings and overlays are excluded from this selection. Validation labels are used for majority mapping, IoU/Dice and displayed GT overlays only; the untrained current linear head is not a supervised reference.

Experiment harness note: first A launch ended before training because the write guard did not handle TensorBoard byte-string paths. Fixed only the scratch guard with os.fsdecode; existing failed-launch output retained. The retry uses a fresh experiment name. This was a harness error, not a model result.

Task5 interim result: A reached217 batches/651 optimizer steps and stopped after three consecutive unhealthy validations. Final std0.00453197, cosine0.99855174, usage100% cluster4, entropy0; majority map all-background, liver IoU/Dice0. New checkpoint/panel saved; no training-mask reads and linear parameters unchanged. B/C continue within their independent500-batch caps. Full timing is recorded per result; CPU experiments take longer than the earlier diagnostic scripts, within the explicitly authorized batch budget.

The synthetic panel initially required an explicit Agg test backend. Neither that correction nor the byte-path guard correction changed model training behavior. New run scripts retain all failed-launch artifacts; only the successful A retry is compared with B/C.

## Task 5 — completed experiments

Producer: `audit/scratch/fixes3_experiment.py`; hard budget500 batches, actual project entrypoint and validation hooks. The first A attempt failed before training; A below is the successful fresh-name retry.

| Run | Batches / optimizer steps | Early stop | Code std | Cross-image cosine | Entropy (nats) | Cluster liver IoU / Dice % | Reused frozen linear IoU % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 217 / 651 | True | 0.00453197 | 0.99855174 | -0.000000 | 0.0000 / 0.0000 | 60.8896 |
| B | 500 / 1500 | False | 0.11218597 | 0.06630412 | 1.942499 | 0.0000 / 0.0000 | 60.8896 |
| C | 500 / 1500 | False | 0.11230376 | 0.04633728 | 1.833085 | 43.7490 / 60.8686 | 60.8896 |

### A — fixes3_A_seed0_retry1
- Usage fractions (IDs0–7): [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
- Evaluation-only majority map: [0, 0, 0, 0, 0, 0, 0, 0]
- Overlay: [fixes3_A_seed0_retry1_final.png](/Users/navneetbavineni/STEGO/audit/scratch/fixes3/fixes3_A_seed0_retry1_final.png)
- New checkpoint: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_A_seed0_retry1/checkpoints/last.ckpt`
- Summary: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/fixes3_A_seed0_retry1_result.json`
- Runtime: 651.83s; no existing checkpoint/data/cache mutation.

### B — fixes3_B_seed0
- Usage fractions (IDs0–7): [0.08493792528266603, 0.21366413107218524, 0.11100561616330855, 0.03127030986520667, 0.05278964654585351, 0.1527198836367174, 0.2034564603781131, 0.1501560270559495]
- Evaluation-only majority map: [0, 0, 0, 0, 0, 0, 0, 0]
- Overlay: [fixes3_B_seed0_final.png](/Users/navneetbavineni/STEGO/audit/scratch/fixes3/fixes3_B_seed0_final.png)
- New checkpoint: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_B_seed0/checkpoints/last.ckpt`
- Summary: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/fixes3_B_seed0_result.json`
- Runtime: 1227.23s; no existing checkpoint/data/cache mutation.

### C — fixes3_C_seed0
- Usage fractions (IDs0–7): [0.05702685023294275, 0.156837508242174, 0.07101756055971117, 0.0834827508863715, 0.00011522998367779316, 0.19363809728737894, 0.2180472617741482, 0.21983474103359565]
- Evaluation-only majority map: [0, 0, 1, 0, 0, 0, 0, 0]
- Overlay: [fixes3_C_seed0_final.png](/Users/navneetbavineni/STEGO/audit/scratch/fixes3/fixes3_C_seed0_final.png)
- New checkpoint: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_C_seed0/checkpoints/last.ckpt`
- Summary: `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/fixes3_C_seed0_result.json`
- Runtime: 1227.04s; no existing checkpoint/data/cache mutation.

All runs use final normalized frozen features and restored-mean centering; only correspondence shifts differ. Loss weights, seed, data, batch size, learning-rate schedule and initial model/probe setup are shared. No validation metric controls checkpoint selection: the scratch runner saves only the terminal checkpoint. Labels appear only in evaluation mapping/metrics and GT overlays. The60.8896% linear reference comes from the previously authorized DIAG supervised fit on100 validation slices, and is NOT a per-run supervised probe nor directly the same472-slice evaluation scope. Current logger linear metrics are from an untrained, unchanged head and must not be interpreted as that reference.

Task5 before/after: A empirically reproduces collapse with the current shifts; calibrated B/C are verified through the reported completed batches using actual training, monitor hooks and full validation. This is not evidence of convergence or final quality.

Changed-file anchors: task1 `src/utils.py:272,302`, `src/eval_segmentation.py` metrics_and_mapping output, `src/configs/train_config.yaml` extra_clusters; task2 `src/collapse_monitor.py:6,42`, `src/train_segmentation.py:326,443,460`; task3 `src/validation_visuals.py:8,17`; task5 `src/unlabeled_training.py:6`, `src/train_segmentation.py:281,635`, `src/configs/train_config.yaml:107`.

## Task 6 — VERIFIED: label-free selection and limits

`python3 -B audit/scratch/fixes3_select.py --self-test` confirms that swapping validation scores cannot change selection and that early-stopped/single-cluster runs are rejected. The actual selector reads only budget completion, early-stop, normalized-code std, cross-image cosine, raw usage and entropy. Both B and C qualify. C is chosen solely because final cosine0.0463373 is below B0.0663041; entropy was not needed to break a tie. This is the predeclared non-collapse choice, not a validation-label-selected anatomy winner. Producer/inputs: audit/scratch/fixes3_select.py and audit/scratch/fixes3/selection.json.

Label-dependent comparisons, reported separately: majority cluster-to-class mapping on the472 validation slices, liver IoU/Dice, and GT/prediction overlays. The reused DIAG60.8896% linear reference is a prior supervised fit evaluated on100 slices; no new supervised fitting was performed. All training-mask reads are forbidden by the experiment guard and the image-only loader. The current untrained linear head remains unchanged and its automatic logger metrics are not the reported supervised reference.

B is non-collapsed but none of its8 clusters is majority liver, so all map to background and liver IoU/Dice are0. C's class map is[0,0,1,0,0,0,0,0]; its IoU43.7490% and Dice60.8686% are evaluation-only oracle metrics. The fixed overlays visibly retain coarse regions and spill into other anatomy; C is not established as an accurate organ segmentation. A is collapsed and stopped. B/C complete500 batches each. Only4 validation patients are available; full-run stability, boundary accuracy, other seeds, GPU execution and generalization remain UNVERIFIED. No full5000-batch run was executed.

## Full-run command — NOT RUN

The default config retains control A's shifts. Use the explicit candidate overrides below; the command starts a new experiment rather than resuming or touching the existing checkpoints. Device selection follows the project's existing CUDA-if-available behavior.

```sh
cd /Users/navneetbavineni/STEGO/src
WANDB_MODE=offline python3 -B train_segmentation.py experiment_name=fixes3_full_C max_steps=5000 extra_clusters=6 train_linear_probe=False collapse_early_stop=True pos_intra_shift=0.4584466218948364 pos_inter_shift=0.45247218012809753 neg_inter_shift=0.7459585666656494
```

## Commit and verification record

| Task | Status | Commit | Verification |
| --- | --- | --- | --- |
| 1 | APPLIED | 0b98854 | Perfect K8 partition100%; K2 Hungarian retained; IoU/Dice/map recorded |
| 2 | APPLIED | 97763df | Healthy/constant/batch-invariance tests; A stops on third bad check |
| 3 | APPLIED | ef2fa66 | Same4 patient slices across all real experiment panels |
| 4 | VERIFIED | 9d7b7c1 | Fresh train-only extraction reproduces shifts; no mask/val reads |
| 5 | COMPLETED | d914160 | A217; B/C500 batches; unchanged linear parameters; fresh checkpoints only |
| 6 | VERIFIED | commit containing fixes3_select.py | Selection invariant to validation-score changes; C selected without labels |

Final regression: metric, monitor and image-only tests PASS; changed source/scripts parse; git diff --check clean. git diff23a2415 for feature-identity source files (modules.py,data.py,precompute_knns.py,chaos_preprocessing.py) and existing dataset/checkpoint directories shows no changes. New experiment checkpoints/logs remain in the new audit/scratch/fixes3/experiments directories; compact result JSONs, stdout and final panels are committed. Headless font-cache and synthetic test-render artifacts remain untracked scratch files.

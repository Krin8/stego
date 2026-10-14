# Audit fixes

Snapshot: `8dc806c` (`snapshot before audit fixes`), preserved on `main`.
Working branch: `audit-fixes`.
No training, cache regeneration, package installation, dataset or checkpoint edits are authorized or performed. Snapshotting added existing untracked files to Git without changing their contents.

Checks use CPU, a 110-second alarm and the audit's write/network guard. Source definitions are isolated where full imports create unrelated files. Before/after checks exercise the same script and use `--expect-fixed` after each change. No claims of CUDA execution or complete training-entrypoint execution are made.

## Findings

ID | Status | Commit | Files and lines changed (in that commit) | Verification | Remaining work
--- | --- | --- | --- | --- | ---

C1 | APPLIED | `6e2009b` | src/utils.py:258–263 | Before LUT [1,0,0,0]; after [0,0,1,0], perfect metrics retained; fix_checks.py C1 --expect-fixed PASS. | Unmatched display policy remains background; M1 handles real-class averaging.

M1 | APPLIED | `4c4e3ae` | src/configs/train_config.yaml:31, src/utils.py:289 | 80/10/10 mIoU: 62.96296% before, 94.44444% after; extra_clusters=0; PASS. | New CT baseline has 2 clusters; old extra-cluster probe checkpoints need their saved architecture/config.

C2 | APPLIED | `9d720bd` | src/configs/train_config.yaml:63 | Ramp raw flip error reproduces at 15; active weighted error changes 7.5 to 0; PASS. | Disabled only, as requested. Re-enabling augmentation still exposes the coordinate bug and rotation-padding issue.

C3 | APPLIED | `1df6611` | src/modules.py:551, src/modules.py:554–556, src/train_segmentation.py:159, src/train_segmentation.py:175, src/train_segmentation.py:616 | External construction true to false; observed shifts [0,1,2] to nonzero [1,2]; singleton rejected; tiny CPU head/loss/backward finite; PASS. | Batch size must exceed 1, including the final batch. No external data touched. Existing inactive NegativeImageDataset class retained.

M3 | APPLIED | `db77cbd` | src/configs/train_config.yaml:37–38, src/configs/train_config.yaml:65, src/configs/train_config.yaml:76, src/configs/train_config.yaml:80, src/modules.py:271, src/modules.py:275, src/modules.py:301, src/modules.py:306 | Real constructor dimension branch/head 512 to 70; probe loss -1.34657 to -1; Gaussian init and all baseline flags PASS. | BREAKING: 512-D head/probe checkpoints incompatible with new 70-D baseline. Removed orthogonal reset too, to retain Gaussian reset semantics. Extra loss code retained but disabled; unused entropy/diversity calculations remain.

M9 | APPLIED | `19bc7ce` | src/configs/train_config.yaml:55, src/configs/train_config.yaml:61 | Bottom-right bug reproduces at [1.5,0.5]; salience disabled and sample grid 25 to 11; PASS. | Inactive salience implementation still has coordinate defects; re-enabling requires a separate fix.

M2 | APPLIED | `4e7236a` | src/modules.py:167, src/modules.py:169–170, src/modules.py:195, src/modules.py:204, src/modules.py:209, src/modules.py:211 | Strict keys matched; full pretrained model frozen/eval false to true; normalized-final-token error 116.155 to 0; repeated eval feature error 0; 224/320 grids PASS. | Heads stay trainable; dropout is intentionally stochastic in training mode. Added full_model.eval in forward to survive parent.train(). n/shallow_layer_n no longer select the baseline layer. CUDA runtime UNVERIFIED; old cache identity is invalid.

M4 | PARTIAL | `b5dc7d0` | src/data.py:553–556, src/data.py:580–583, src/precompute_knns.py:67 | Before rank6, metadata accepted, invalid indices clamped, train-mode cache model; after rank7, explicit rejection/assertions, eval mode; PASS. | BLOCKED: provide the verified ordered sample/preprocessing manifest for the future metadata schema/writer. Current caches lack metadata and will be rejected. Writer still emits no metadata and skips existing paths; do NOT regenerate yet. No NPZ was changed.

M7 | APPLIED | `b68c747` | src/eval_segmentation.py:136, src/eval_segmentation.py:139–140 | Exact evaluator unary error .813666 to 0 vs cosine-times-2 log probabilities; CPU routing spy now forwards label.device for both CRF outputs; PASS. | CUDA execution remains UNVERIFIED. Explicitly requested device fix verified by routing behavior, not a GPU run; no real CRF dataset evaluation.

M8 | APPLIED | `5784be9` | src/train_segmentation.py:653 | Actual Trainer expression: requested 7 head updates previously allowed 3 batches, now 21 optimizer steps / 7 batches; 3 optimizer calls verified; PASS. | global_step still counts optimizer updates, 3 per batch. hist_freq=100 before-batch modulus fires every 100 batches (300 updates); event-file reset modulus2000 after batch fires every 2000 batches (6000 updates). Checkpoint every_n_train_steps=100 uses optimizer-step units, checked at batch end (300-step common multiples). Plot/log x-coordinates use optimizer steps; scalar_log_freq=10 is passed unchanged to Lightning. val_freq=50 is batches, not global_step; reset_probe_steps/checkpoint_freq remain unused until m4. No schedule/LR changes beyond requested budget conversion.

M10 | APPLIED | `7f830cf` | src/demo_segmentation.py:225, src/train_segmentation.py:276–277 | Empty-mask CE NaN to differentiable zero with zero gradient; missing demo mapping now raises; valid mapping unchanged; PASS. | Demo requires a validated mapping. Morphology retained; no inference was run on a trained checkpoint.

M5 | PARTIAL | `636eb87` | src/data.py:393–396, src/data.py:430 | Mock filesystem mixed CT families previously accepted, now ValueError; raw MRI previously called CT window, now NotImplementedError before loading; PASS. | BLOCKED: Which filename family and source-to-slice manifest are authoritative? What CT window and MRI normalization/label orientation should be used? Current mixed preprocessed splits intentionally fail. Only protective stubs applied; no dataset changes.

M6 | PARTIAL | `afcc495` | src/data.py:426–427 | Synthetic .nii.gz sample previously reached DICOM preprocessing, now raises explicit NIfTI NotImplementedError before dispatch; PASS. | BLOCKED: Where are paired NIfTI files, and what orientation, affine/slice ordering and mask encoding are authoritative? Five-crop/native-volume reconstruction remains unimplemented; only requested dispatch stub added. Directories containing only NIfTI are still not discovered by the DICOM collector.

m2 | APPLIED | `3b6dcc5` | src/test_timm.py:2, src/test_timm.py:227, src/test_timm.py:270, src/vis_corr.py:110–111, src/vis_corr.py:419, src/vis_corr.py:434, src/vis_corr.py:437, src/vis_corr.py:440, src/vis_corr.py:443, src/vis_corr.py:451, src/vis_corr.py:457, src/vis_corr.py:463, src/vis_corr.py:500, src/vis_corr.py:505, src/vis_corr.py:511, src/vis_corr.py:676, src/vis_corr.py:686, src/vis_corr.py:694 | Initial isolated test hit an unrelated utils import; harness supplied that unused import, then diagnostic CLIP mean/std and sampled-pair/RGB labels PASS. | No figures regenerated. Legacy variable/function names retained; figures are illustrative sampled pairs, not actual cached-kNN evidence.

m3 | APPLIED | `019abd1` | src/utils.py:29, src/utils.py:31 | Absolute channel error .0665453 to 5.96e-8; constant image NaN to finite; PASS. | Use rescale=False for absolute preprocessing comparisons; optional resize preserved.

m4 | APPLIED | `812c6c5` | src/configs/train_config.yaml:17, src/configs/train_config.yaml:34, src/configs/train_config.yaml:47, src/configs/train_config.yaml:49, src/configs/train_config.yaml:52, src/configs/train_config.yaml:71, src/configs/train_config.yaml:83, src/configs/train_config.yaml:87, src/test_biomedclip.py:3–4, src/test_biomedclip.py:10, src/test_biomedclip.py:14 | Eight unsupported defaults removed; nonexistent Python config import removed; actual YAML load returns dim70; PASS. | No duplicate files deleted. Historical configs may retain these no-op keys; default Hydra overrides for removed keys will now be rejected. Full legacy smoke entrypoint not run because optional imports write files.

m5 | APPLIED | `37f6268` | src/utils.py:206, src/vis_corr.py:112 | Training and diagnostic image resize now BILINEAR; label resize remains NEAREST; PASS. | Declared choice: Paper A bilinear, as requested, rather than model-card bicubic. This changes feature/preprocessing identity and requires future metadata-aware cache regeneration. Upstream metric-filter bug was not restored.

m1 | BLOCKED | `dabec28` | No production edits | No source edit; restored-global-mean centering retained. | QUESTION: Should correspondence use literal Paper A Eq(3) row-mean centering, or the pinned reference implementation that restores the global mean?

m6 | BLOCKED | `39a0ba0` | No production edits | No source edit; historical provenance cannot be reconstructed. | QUESTION: Which exact checkpoint path, training command, resolved config, dirty-source snapshot and source-to-split manifest produced the reported results?

## New observations

No new production defects were changed outside the audited findings. Two deliberate limitations are recorded in the corresponding rows: the cache writer cannot yet produce a verified replacement, and disabled experimental paths remain recoverable with their existing defects.

## Final CPU verification

- Config: **PASS**, dimension70, extra_clusters0, text disabled.
- Real image forward: **PASS**, `CT_28_0110.png` → `(1,3,224,224)` → pretrained features `(1,768,14,14)` → codes `(1,70,14,14)`, finite FP32 on CPU.
- Correspondence loss: **PASS**, configured loss −0.3474243879 on a synthetic two-item batch formed from the real feature map and its reflection; no optimizer step.
- Synthetic metric update/compute/mapping: **PASS**, mIoU/accuracy100% and expected swapped-label lookup.
- All16 lightweight finding checks: **PASS** after all edits. M2's separate real-checkpoint test also passed strict key matching, frozen/eval state, exact normalized-final-token equality, deterministic repeated eval and224/320 shapes.
- Changed Python files parse. No real training entrypoint, trained-head checkpoint round-trip or CUDA execution was tested. Full dataset startup is deliberately blocked by M4/M5 guards. This smoke test reads one sample directly; it does not bypass the guards for training.

Exact repeatable CPU smoke command:

```sh
cd /Users/navneetbavineni/STEGO
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -B audit/scratch/final_smoke.py
```

## Remaining decisions and blocked work

- **M4:** Which verified ordered sample IDs and preprocessing manifest should define the replacement cache? After this is established, a metadata-producing writer plus identity validation is still required. The applied protection only rejects missing metadata; it does not certify arbitrary metadata supplied manually. Do not add a dummy metadata field. Existing cache files are unchanged.
- **M5:** Which filename family/source-to-slice manifest is authoritative, and what CT window, MRI normalization and image/mask orientation/label convention should be used? The existing mixed processed split now raises an explicit error.
- **M6:** Where are the paired NIfTI files, and what orientation, affine, slice ordering and label encoding are authoritative? Is the five-crop/native-volume workflow required for these inputs? Only the dispatch protection was applied.
- **m1:** Choose literal Paper A Eq(3) versus the pinned reference's restored-global-mean variant. Code remains unchanged pending that choice.
- **m6:** Supply the exact checkpoint, command, resolved config, dirty-source snapshot and split manifest for historical results. No historical metric has been re-created or certified here.

## Breaking changes and interval units

- Default head/probes are70-D, CT cluster count2. Old512-D or extra-cluster probes are incompatible with this new baseline configuration. Loading an old saved config does not make its results comparable: feature layer/norm and resize behavior also changed.
- Pretrained backbone is now entirely frozen; final normalized tokens replace raw block11. The old text experiment cannot update its pretrained visual projection if re-enabled.
- Unversioned caches, mixed preprocessing families, raw MRI and explicit NIfTI dispatch fail instead of silently proceeding. Missing demo mappings fail explicitly. Single-item negative batches fail an assertion.
- No-op config defaults were removed; legacy explicit overrides may be rejected by Hydra. Augmentation and salience bugs remain in their disabled experimental code paths.
- `cfg.max_steps` now denotes head/batch updates; Trainer limit is3×that number because each batch performs3 optimizer steps. With the current5000 default, the limit is15000 optimizer updates.
- Explicit `global_step` uses still count optimizer updates: histogram frequency100 is tested at batch boundaries, so it fires at common multiples of100 and3 (every100 batches/300 updates); event-file reset modulus2000 fires every2000 batches/6000 updates. Plot/log x-coordinates likewise use optimizer updates. Checkpoint interval100 remains expressed in optimizer-step units and is checked at batch end, so save opportunities are common multiples of100 and3.
- `val_freq=50` is a batch interval; the AML branch uses250 batches. `scalar_log_freq=10` is passed unchanged to Lightning; its effective scalar-flush cadence was not independently executed, so do not infer10 head updates from the value. The explicit logging coordinates are verified optimizer-step units. Removed `checkpoint_freq` and `reset_probe_steps` had no effect.

## Next commands — NOT RUN

**Do not run cache generation or training on this branch yet.** M4/M5 require a verified data manifest and metadata-aware writer/reader; M6 applies if NIfTI is the intended source. Current `precompute_knns.py` skips existing files and writes only `nns`, so merely invoking it cannot produce a usable verified replacement. No truthful currently-working cache-regeneration command exists until that code/contract is completed. Do not delete existing caches or bypass the guards to force progress.

After those blockers are resolved, use these existing entrypoints in order. The first command becomes a regeneration command only once the metadata-aware replacement policy is implemented; these are not claims that the present guarded workflow can run:

```sh
cd /Users/navneetbavineni/STEGO/src
AUDIT_PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3

# 1. Generate verified replacement neighbors after M4/M5 are resolved.
"$AUDIT_PY" -B precompute_knns.py

# 2. Train a new baseline after cache validation succeeds.
"$AUDIT_PY" -B train_segmentation.py

# 3. Paste the absolute path of the NEW 70-D baseline checkpoint when read waits.
read -r audit_checkpoint
"$AUDIT_PY" -B eval_segmentation.py pytorch_data_dir=/Users/navneetbavineni/STEGO/src/pytorch_data_dir "model_paths=[\"${audit_checkpoint}\"]"
```

If a new canonical data root is chosen for M5, that path must replace the current data root consistently for all three entrypoints; it cannot be guessed now. Historical checkpoint re-evaluation is a separate m6 task and must use its exact saved architecture/preprocessing contract.

## Repository verification

Snapshot remains `8dc806c`. One separate `fix(ID)` commit exists for each of19 findings, including documentation-only commits for the two blocked decisions. Incidental end-of-file newline changes were removed and folded into their corresponding commits; no unrelated formatting change remains. Final documentation/smoke script is a separate documentation commit.

`git log --oneline 8dc806c..HEAD` and `git diff 8dc806c --stat` were run. The diff contains only the audited source/config files and audit helpers/report. `audit/AUDIT.md`, datasets, cached neighbors and checkpoints were not changed after the snapshot. Runtime write guards remained enabled for checks.

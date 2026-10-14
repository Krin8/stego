# Handoff: label-free CT liver segmentation on CHAOS with frozen BiomedCLIP (STEGO adaptation)

Owner: Navneet Bavineni. Prepared 3 October 2026. Compiled from the conversation record and the agent reports the owner pasted. The compiler did not run any code. Numbers are as reported in the evidence files (section 11); verify before quoting them.

## 0. Status at a glance

- **Goal:** unsupervised liver segmentation in abdominal CT (CHAOS CT) with a frozen BiomedCLIP ViT-B/16 backbone, needing no ground-truth masks at inference.
- **Best pipeline today (no network training of our own):** frozen BiomedCLIP patch tokens at 448 input (28×28 grid) → K-means K=8 → label-free liver-cluster naming (rules N1/N2/N3) → fixed F1 refinement. Inductive result on 4 held-out patients: IoU 77.96 ± 2.07, Dice 87.61 ± 1.31 (pooled pixels).
- **STEGO-style trained head:** never beat clustering the frozen features. The latest sealed experiment (448 input, three seeds, locked configuration) collapsed in all three seeds, first detected at batch 150. Validation labels were never opened in it. As far as known, its Stage B (scoring) has not been run.
- **Do not** change the sealed configuration, open validation masks to select anything, or claim that STEGO cannot work with BiomedCLIP (section 9).
- **Do first:** answer the questions in section 14, then follow the priorities in section 10.

## 1. Goal, scope and research questions

- Task: liver vs background on axial CT slices of CHAOS. Out of scope: MRI (T1-DUAL, T2-SPIR), multi-organ segmentation, and NIfTI (the owner will not use NIfTI; the data is DICOM and PNG).
- Questions: (1) Does a STEGO-style correspondence head on frozen BiomedCLIP features improve on clustering those features directly? (2) What is the best label-free pipeline reachable from the frozen features? (3) How much do the results, and the word 'unsupervised', depend on cluster naming and on label-based development choices?
- Reference method: STEGO (Hamilton et al., ICLR 2022, arXiv 2203.08414). Frozen backbone; a small head trained with a correspondence loss on self, kNN and random-image pairs; K equal to the class count; Hungarian matching with labels for evaluation only; a linear probe trained with labels as a separate diagnostic. The paper says labels are not used to tune parameters and gives a heuristic (mean kNN similarity about 0.3, mean random similarity about 0.0).

## 2. Environment and locations

- Machine: Mac, CPU only. Python 3.14 at /Library/Frameworks/Python.framework/Versions/3.14/bin/python3. About 1 training batch per second on CPU at 224 input. MPS exists but has not been used. Interactive zsh does not accept # comments in pasted blocks.
- Project root: /Users/navneetbavineni/STEGO. Folders: src/ (code), audit/ (evidence and scratch scripts), experiments/stego\_biomedclip/ (latest sealed run: logs/, predictions/, configs/locked.json), logs/, checkpoints/.
- BiomedCLIP weights: src/BiomedCLIP-PubMedBERT\_256-vit\_base\_patch16\_224/open\_clip\_pytorch\_model.bin, loaded through open\_clip. The pipeline uses the image tower; the text tower is used only by naming rule N1. Its SHA-256 has not been recorded here; record it.
- DINO weights tested (audit/dino2/weights/): facebook/dino-vitb16, commit f205d5d8e640a89a2b8ef0369670dfc37cc07fc2, converted safetensors SHA-256 2c465a30aa590f9b3fbe36feb962a1eb35635da809643f42857cf64ee450007c (source pytorch\_model.bin SHA-256 a064e36c67289caaa5c949c0b3f7f31a0fcbcba5721f5fa12419933ec1f4fe6e); facebook/dinov2-base, commit f9e44c814b77203eaa57a6bdbbd535f21ede1415, SHA-256 d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841.
- Git: branches audit-fixes → audit-fixes-2 → -3 → -4. Snapshot commit 9f13b0c4be7fbe4b7d17fd6c3d9de81e09ecf374 on audit-fixes-4, tag chaos-v2-final. .git reached 4.2 GB because large .npy and .ckpt files were committed; untracking them was recommended, so check git status. Later work (improve2, final\_pipeline, dino, dino2, naming, experiments/) may be uncommitted.
- pydensecrf is not installed; the CRF import is lazy and evaluation uses run\_crf=False.
- Running train\_segmentation.py with its default config collapses (old shifts 0.18/0.12/0.46, collapse\_early\_stop off). Never launch it without explicit overrides.

## 3. Data and preprocessing contract

- 20 labeled CT patients: 16 train (1,869 slices) and 4 validation (472 slices; patients 5, 6, 8, 30 with 95, 82, 98, 197 slices). The split is frozen in a versioned manifest and no patient is in both splits. The validation patients have informed many development choices, so they are not an untouched test set.
- Derivatives come from raw DICOM (src/preprocess.py; folder src/pytorch\_data\_dir/preprocessed\_v2/ with manifest.json): CT window \[-150, 250\] HU scaled to \[0, 1\], stored as 16-bit PNG (round(65535·x)), native 512×512, no resampling. Masks are 0/255. Files are named CT\_\<patient>\_\<slice>.png under images/ and labels/. Image-to-mask pairing is established before any sorting. Legacy CT\_p files and out-phase MRI are excluded.
- Model input: resize to 224 or 448, replicate to 3 channels, apply BiomedCLIP mean/std exactly once (the loader divides uint16 by 65535). Liver is about 7.6% of validation pixels.
- Neighbour caches (src/pytorch\_data\_dir/nns/nns\_chaos\_v2\_\*) carry identity metadata (preprocessing contract, weights digest, layer, k, ordered sample IDs). Loaders reject caches without it.

## 4. History and what each stage established

| Stage | What was done | Outcome |
| --- | --- | --- |
| 1. Audit (audit/AUDIT.md) | Read-only agent audit against the STEGO paper, reference repo and BiomedCLIP | Found: lookup could map liver to background (C1); mIoU counted a fictitious class (M1); flip-axis bug in the augmentation loss (C2); external negatives from another dataset (C3); extra loss terms so it was not the plain baseline (M3); inconsistent CT/MRI preprocessing (M5); frozen/eval and token-normalization issues (M2); caches without provenance (M4); step-count unit bug (M8). Earlier reported numbers (46.85 DINO mIoU; 30.7 cluster / 87.5 linear BiomedCLIP) are void. |
| 2. Fix rounds (FIXES.md to FIXES3.md, DECISIONS.md) | Corrections applied one commit each; data rebuilt from raw DICOM; split frozen; cache manifests; collapse monitors; fixed-slice logging; configurable seed; lazy CRF | Correct wiring established; learning not yet |
| 3. First v2 training (original shifts) | chaos\_v2\_ct | Head collapsed. DIAG.md: frozen features carry liver information (K=8 k-means about 64% IoU; supervised linear probe about 61%); K=2 fails; 68% of random cross-image pairs counted as attracting at the original shifts, so constant codes scored as well as learned ones |
| 4. Recalibrated shifts (FIXES3.md, FORENSICS.md) | K=8 (6 extra clusters), many-to-one mapping; shifts A (old), B (10%), C (15% random-pair attraction) | A collapsed at batch 217; B and C did not. C seed 0: 43.75% liver IoU; seeds 1 and 2: 0% strict majority (31.9–32.9% optimal mapping); full 5,000-batch run best 44.40% at batch 600, final 0% majority / 29.81% optimal. The correspondence loss kept improving while liver precision fell, so loss is not a selection signal |
| 5. Baseline comparison (BASELINE.md) | Frozen K=8 k-means vs trained heads, same 472 slices | k-means 64.14% vs at most 44.40%. No label-free checkpoint-selection proxy worked |
| 6. Why the head fails (PAIRS.md, CROSSPATIENT.md, SELECTIVITY.md, FIXES4.md) | Pair-purity, neighbour provenance, affinity selectivity, pre-declared round D/E | Attracting positives mostly mix liver with background (73–79% of liver→background pairs attract at the original shifts); 99.7% of cached neighbours were same-patient but excluding them changed nothing; affinity is selective at stricter thresholds (AUC 0.94, liver-target precision about 83% at 5% attraction) yet D/E (3 seeds each) reached only 40.2% and 37.3% mean 4-patient LOPO IoU and failed the declared criterion; k-means on the head's 70-d codes scored 28–35% |
| 7. Reliable baseline (FINAL\_BASELINE.md) | Frozen K=8 k-means, leave-one-patient-out (LOPO) over 20 patients | 62.87 ± 2.60 IoU |
| 8. Improvement (IMPROVE.md, IMPROVE2.md) | 448 input; label-free post-processing | 69.78 hard; 79.27 with F1 (exploratory) |
| 9. Inductive check and packaging (final\_pipeline/) | Fit on 16 patients, score 4 held-out; segment\_ct.py | 77.96 ± 2.07 (F1) |
| 10. Backbone alternatives (DINO.md, DINO2.md) | DINO v1 B/8, B/16, DINOv2 B/14, L/14 | All clearly worse than BiomedCLIP |
| 11. Label-free naming (naming/NAMING.md) | Rules N1, N2, N3 | Same cluster as the label-assisted mapping in all fits |
| 12. Sealed STEGO at 448 (experiments/stego\_biomedclip) | 3 seeds, locked configuration | Collapse in 3/3 seeds |

## 5. Current pipeline specification (the frozen pipeline)

1. CT slice → window \[-150, 250\] HU → body mask (native HU > -500, opening disk 5, largest 8-connected component, hole filling) → square body crop with margin → resize to 448 → BiomedCLIP normalization.
2. Frozen BiomedCLIP ViT-B/16, final-block patch tokens (28×28 grid, CLS dropped), L2-normalized, uncentered. Eval mode, no gradients.
3. K-means K=8 on body patches, seeds 0/1/2. Two fits: inductive (16 train patients, 1,869 slices) and transductive (all 20, 2,341 slices). Fits are label-free.
4. Liver-cluster naming (once per fit, label-free):
   - N1: for each cluster take the 200 body patches nearest its centre; crop 112×112 native-windowed pixels around each; embed with the BiomedCLIP image tower; average the normalized vectors; compare with text prompts 'an axial abdominal CT image showing the {organ}' (liver, spleen, kidney, stomach, pancreas, gallbladder, aorta, vertebra, muscle, subcutaneous fat, bowel, lung, heart, rib, air); z-score each prompt across the 8 clusters; liver is the cluster with the highest liver score.
   - N2: anatomical prior. Candidates have area-weighted centroid x < 0.5 (image-left) and y < 0.6 (anterior), with mean HU in \[40, 180\]; liver is the largest candidate (abstain if none).
   - N3: accept only if N1 and N2 agree, otherwise abstain. No abstentions occurred in the 6 saved fits.
5. F1 refinement: soft score of the named liver cluster (saved per-fit temperature) → inverse crop → guided filter (radius 8, eps 0.01) → threshold ≥ 0.5 → largest 2D connected component (8-connected) → hole fill (4-connected) → closing with disk 3. F2 adds a 3-slice median across neighbouring slices (optional, off by default).
6. Packaged code: audit/final\_pipeline/segment\_ct.py (with a --check mode), study.py, README.md. The saved mapping in that script was fitted with labels from all 20 patients; it still needs updating to use N1–N3 if deployment must be label-free.

## 6. Results ledger

Metrics: IoU and Dice are pooled over a patient's slices, then mean ± SD across patients (the patient is the unit). 'Mean-slice' averages slice-level scores instead. Scores at 224 are primary; native 512 differs by under 0.2 points everywhere. LOPO: clusters mapped to liver using the other patients' labels. Inductive: fit and mapping on 16 train patients, scored on the 4 held-out. Transductive: all 20 patients' images clustered together.

| Pipeline | Protocol | IoU % | Dice % | Status |
| --- | --- | --: | --: | --- |
| Frozen K=8, 224, hard | 20-patient LOPO | 62.87 ± 2.60 | 77.17 ± 1.96 | locked baseline |
| Frozen K=16, 224 | 20-patient LOPO | 58.20 ± 3.41 | 73.52 ± 2.73 | worse |
| 448 input (28×28), hard (B2) | 20-patient LOPO | 69.78 ± 2.96 | 82.17 ± 2.09 | pre-declared |
| 336 input (21×21) | 20-patient LOPO | 68.96 ± 2.82 | 81.60 ± 1.98 | pre-declared |
| 448 + soft + guided + largest component (B5) | 20-patient LOPO | 76.50 ± 2.14 | 86.67 ± 1.37 | exploratory (parent chosen with labels) |
| F1 | 20-patient LOPO | 79.27 ± 2.85 | 88.41 ± 1.77 | exploratory |
| F2 | 20-patient LOPO | 79.39 ± 2.83 | 88.48 ± 1.76 | exploratory |
| F1, hard-only counterpart | inductive, 4 held-out | 70.83 ± 1.92 | 82.91 ± 1.31 |  |
| F1 | inductive, 4 held-out, pooled | 77.96 ± 2.07 | 87.61 ± 1.31 | headline |
| F1 | inductive, mean-slice | 66.00 ± 7.82 | 75.25 ± 8.13 | slices with fewer than 2,000 liver pixels (30 slices) score 0.18% pooled |
| F1 with label-free naming | transductive, 20 patients | 78.81–80.20 | 88.12–88.99 | by seed; inductive identical to F1 above |
| DINO v1 B/8, 224, K=8 | 20-patient LOPO | 46.04 ± 4.87 | 62.90 ± 4.75 | worse |
| DINO v1 B/16, 448, K=8 | 20-patient LOPO | 41.35 ± 6.23 (majority 31.61) | 58.24 ± 6.50 | worse |
| DINOv2 B/14, 392, K=8 | 20-patient LOPO | 24.17 ± 4.22 (majority 0) | 38.62 ± 5.37 | worse |
| DINOv2 L/14, 224, K=8 | 20-patient LOPO | 29.61 ± 4.27 | 45.48 ± 4.98 | worse |
| STEGO head (224, 14×14), C runs | 4 val patients | 29.8–44.4 | 46–61 | below frozen features |
| STEGO head, D / E | 4 val patients, 3 seeds | 40.2 / 37.3 mean |  | failed pre-declared criterion |
| K=2 with one-to-one Hungarian | LOPO | BiomedCLIP about 17; DINO B/8 about 0; B/16 15.7; DINOv2 0.7 |  | STEGO's own naming protocol fails here |

## 7. Verified findings and untested hypotheses

- **Verified:** frozen BiomedCLIP features carry liver information; at the original shifts the loss rewards constant codes; recalibrated shifts avoid collapse but do not beat the frozen features at 224; majority mapping is biased against a rare class (all-background prediction scores mIoU 46.18% with liver IoU 0); neighbour provenance and pair purity as in the table above; the affinity separates liver pairs at selective thresholds.
- **Untested hypotheses for the head's failure:** a background-dominated loss; compression 768 → 70 dimensions; the cluster-probe path (its input is detached in the repo, so it cannot repair the head); missing STEGO ingredients (five-crop, CRF, K equal to the class count); whether 448/28×28 changes anything with calibrated shifts. Effective rank and PCA spectrum of the frozen features have not been measured.
- Blockiness comes from the 28×28 patch grid; F1 hides much of it. A head trained on that grid inherits it. A label-free feature upsampler is a literature lead, not checked against this setup.

## 8. Sealed STEGO@448 experiment

- Frozen BiomedCLIP (0 trainable backbone parameters) → STEGO-style head (698,252 trainable parameters) → three original correspondence components, alignment disabled → post-training K-means K=8 → N1/N2/N3 → F1.
- Adam, learning rate 0.0005, 5,000 batches per seed, no scheduler, pinned loss weights and shifts (values are in experiments/stego\_biomedclip/configs/locked.json; the compiler has not seen them), checkpoint selection by a fixed training-image correspondence test, validation labels blocked until all seeds produced predictions.
- Results: all three seeds collapsed, first detection at batch 150.

| Seed | Final code std | Final cross-image cosine | Bad checks | Status |
| --- | --: | --: | --: | --- |
| 0 | 0.0000556 | 0.9999999 | 100 | Collapsed |
| 1 | 0.0000124 | 1.0000000 | 46 | Collapsed |
| 2 | 0.0000581 | 0.9999998 | 100 | Collapsed |

- Completion records report labels\_opened=false and validation\_images\_used=0. Seed 1 was interrupted at batch 2,650 and resumed from its saved head at 2,700 with the same configuration; optimizer and RNG state were not serialized, so that segment is not bit-identical. The meaning of the 'bad checks' counts (100/46/100) is unexplained; check whether the counter is capped or restarted at the resume.
- Collapse monitors (code std, cross-image cosine, cluster usage and entropy) exist in the training code; the stop rule used earlier is cosine above 0.99 or code std below 0.01 for 3 consecutive checks. Frozen final-block feature cosine across images is about 0.513, so the collapse is made by the head and the loss, not by the backbone.

## 9. Label-use ledger and wording

- **Label-free:** K-means fitting; the body mask; N1, N2, N3 (N1 uses BiomedCLIP's language pretraining, N2 uses hand-written anatomical rules); the F1 operations; the STEGO head training in the sealed experiment.
- **Label-informed (development history):** K=8, 448 resolution, the CT window (partly from masks), the F1 steps and the parent variant were chosen using label-based scores, mostly on the same 4 validation patients; cluster-to-liver mapping in the LOPO and inductive reference numbers; checkpoint and seed comparisons in earlier runs.
- **Backbone:** BiomedCLIP was pretrained on biomedical image-caption pairs, so it is not self-supervised; DINO is, but scored far lower.
- **Use:** 'unsupervised feature clustering with label-free cluster naming; K, resolution, window and post-processing were fixed during development on labeled patients'. Avoid: 'strictly unsupervised', 'BiomedCLIP is self-supervised', 'K=8 is supervised'.
- **Do not claim:** that STEGO cannot work with BiomedCLIP (only this configuration collapses, and recalibrated shifts at 224 did not beat the frozen features); that 77.96 is an untouched-test result; that F1 or B5 results are non-exploratory.

## 10. Open items, in priority order

1. **Resolve section 14.** Read locked.json; check git status; confirm whether Stage B was run.
2. **Pre-flight diagnostic (read-only, no training):** pinned shift values with file:line; attraction rates for same-image, kNN and random pairs on 200 train slices at 448; the three-term loss for constant codes, random codes and cluster codes; frozen-feature diversity (mean cosine, per-channel std, PCA spectrum, effective rank for the final block and blocks 9 and 11); head output diversity at initialization for seeds 0 to 2; compare with the shifts used earlier. Aim: show whether the loss structure predicts collapse.
3. **Stage B for the sealed run:** only if the owner wants the record complete; report it as a collapsed run, never as a method result.
4. **One bounded retraining round at 448, only if pre-declared:** label-free shift calibration on train images (20%/20%/10% attraction, or the STEGO heuristic: kNN mean about 0.3, random mean about 0.0; pick one beforehand); 500 batches, 3 seeds, collapse early-stop on, same Stage A/B pipeline. Success only if the head's mean inductive F1 IoU exceeds 77.96 by at least one patient SD (about 2 points); otherwise stop. Expectation, a guess: it will not pass.
5. **Update segment\_ct.py** to the N1–N3 naming (copy to a new folder, do not overwrite), then re-run its --check.
6. **Independent cohort test:** the same pipeline, fixed, on a CT set nothing was selected on (for example the official CHAOS test set if its submission is open).
7. **Optional:** the few-label naming curve (name clusters with 1, 2, 5, 10 patients); a naming-stability test over random patient subsets; a label-free feature upsampler to reduce blockiness; DINO arms with different settings only if the owner insists.
8. **Write-up:** results table from section 6, limitations from section 9, and the STEGO analysis from sections 4 and 7.
9. **Housekeeping:** untrack large artifacts, add an audit/README.md evidence index, back up audit/, logs/, checkpoints/ and both weights folders.

## 11. Evidence index (under /Users/navneetbavineni/STEGO/audit/ unless stated)

- AUDIT.md (2.9 MB; read the final report at the top, skip Appendix A), FIXES.md, FIXES2.md, FIXES3.md, FIXES4.md (fix and experiment logs), DECISIONS.md (preprocessing, cache, provenance decisions).
- DIAG.md (frozen-feature ceiling, affinity calibration, collapse), FORENSICS.md (seeds and checkpoints of the C runs), BASELINE.md (identical-sample comparison), PAIRS.md, CROSSPATIENT.md, SELECTIVITY.md (pair diagnostics), FINAL\_BASELINE.md (20-patient baseline).
- improve/IMPROVE.md (grid ceiling and variants B0–B6), improve2/IMPROVE2.md (F1, F2), final\_pipeline/FINAL\_PIPELINE.md, README.md, segment\_ct.py, study.py, dino/DINO.md, dino2/DINO2.md, naming/NAMING.md and EXECUTION\_NOTES.md (Stage A manifest SHA-256 ceb1f5e236980247444416469c03c86339348d4e73e51c2208de39f2d430416b).
- run\_records/ (per-run commands, configs, commits), scratch/ (producing scripts), and /Users/navneetbavineni/STEGO/experiments/stego\_biomedclip/ (sealed run).

## 12. Working protocol for agents (what worked)

- Read-only first. State in every prompt what may not change: project source, datasets, caches, existing checkpoints. New outputs go in a new audit subfolder.
- One commit per fix on a new branch, with reproduce-before and verify-after for every fix. Log in a FIXESn.md file.
- Pre-declare everything before looking at labels: parameters, arms, the decision rule, and the success criterion. Run only declared arms, report all of them, and do not tune after results.
- Stage A / Stage B: Stage A reads no masks and writes a manifest with hashes; Stage B opens labels only to score. A bug found in Stage B means rerunning both stages and saying so.
- Every claim needs file:line or a producing script. Mark anything unchecked as UNVERIFIED. Check for other training processes before each run and never overlap runs.
- Downloads only for named repositories, safetensors preferred; never trust remote code or torch.hub; load .bin files only with weights\_only=True.
- Cap long jobs: time one batch, and use a fixed 200-slice subset first if a job projects past about 90 minutes.

## 13. Known pitfalls and past mistakes in earlier guidance

- Strict-majority mapping returns 0% for a rare class when no cluster is more than half liver; always also report the IoU-optimal mapping and name which one is used. All-background gives mIoU 46.18%.
- The PNGs are 16-bit; casting to uint8 without scaling makes a CT look like noise.
- Checkpoint selection by correspondence loss picks collapsed models. Select by collapse monitors.
- Do not compare 62.87 (20-patient LOPO, 224, hard) with 77.96 (4 patients, 448, F1) as if only one thing changed.
- Earlier guidance errors: the inherited shifts were said to be an ablation (they were the cause of the collapse); same-patient neighbours were guessed to be the problem (they were not); selective shifts were expected to help the head (they did not).

## 14. Questions the receiving agent must resolve first

1. What are the pinned shifts, loss weights, batch size and dropout in experiments/stego\_biomedclip/configs/locked.json?
2. Has Stage B been run for the sealed run, and where are the sealed predictions?
3. What does git status show: which of improve2, final\_pipeline, dino, dino2, naming and experiments are uncommitted or untracked?
4. What is the SHA-256 of the BiomedCLIP weights file?
5. What do the bad-check counts 100/46/100 mean?
6. Does the owner want a retraining round (item 4 in section 10) or the write-up first?

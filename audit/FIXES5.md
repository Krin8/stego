# FIXES5: bounded 448 calibrated STEGO round

Branch: codex/audit-fixes-5. No original tracked source was dirty; pre-existing untracked artifacts remain untouched. No training process found in the initial authorized process inspection.

Primary naming rule: N3 agreement; report N1/N2 as secondary. Primary scoring: 224, patient-pooled pixels then four-patient mean, then mean across seeds. Both candidates and all seeds are predeclared; no label-based selection.

E excluded as requested. The stated numerical rationale is reversed: 0.018163 < 0.019855; exclusion is retained without adding a candidate.

Task 1: complete (commit 599789b).

## Task 1 — APPLIED / VERIFIED

Producer: `audit/calibrated/calibrate.py core`; fixed preflight seed/sample, actual imported loss, dropout off. Freshly recomputed quantiles exactly equal the corresponding preflight quantiles; every loss re-evaluated with the same coordinates/permutations. No labels read.

| Candidate | Shifts intra/kNN/random | Attraction % | Constant total | Random total | K8 total | Admissible |
|---|---|---|---|---|---|---|
| C448 | [0.4247578978538513, 0.42549750208854675, 0.7603381872177124] | [75.0, 75.0, 15.0] | -0.00512864 | -0.00425668 | -0.08423423 | False |
| D448 | [0.7944176197052002, 0.785980761051178, 0.7976505756378174] | [20.0, 20.0, 10.0] | 0.35617134 | 0.01564344 | 0.00037086 | True |

Mean restored-mean affinity: kNN 0.57956362; random 0.53827548. Unweighted components in `audit/calibrated/objective.json`. Admissibility fixed: constant > random AND K8 < random.

## Task 2 — APPLIED / VERIFIED

New code lives only in `experiments/stego_calibrated/`. Head and reference loss are byte-for-byte copies of the sealed implementation. The standalone runtime redirects all writes to the new experiment, blocks labels until Stage B, blocks network, and uses two CPU threads. The split is checked against `src/configs/chaos_splits_v2.json`; old feature and neighbor arrays are read-only.

`state.py` captures model, Adam moments/step, torch RNG, Python RNG, NumPy global RNG, data generator RNG, permutation/cursor, model mode and bad-check counter. `check_state.py` reproduces the original deficiency (missing optimizer/RNG in the sealed checkpoint schema by inspection) and verifies exact next-update tensor/loss/RNG equality after a fresh disk save/load in the new format; `audit/calibrated/state_verification.json` is the result. Bit identity is verified on the same CPU/runtime/thread count, not across hardware or library versions. Production CLI deliberately has no resume option and refuses existing run paths; interrupted runs require a fresh unique attempt from seed initialization, retaining failed artifacts.

Every run takes a package-wide nonblocking file lock and records a full process snapshot before training. It aborts on any other detected Python training job, never kills a process, records PID/config hashes, checks timing after 10 batches, and aborts above 90 minutes. Every 50 batches it records std/cosine and K=8 k-means usage on the fixed 16 train-monitor code grids (seed 104729, n_init=3, max_iter=100); this post-hoc monitor fit never enters the objective. Usage cluster IDs are newly fitted at each check and are not semantically comparable across checks; usage alone is not a health test. Healthy requires cosine <0.99 AND std >0.01.

### Bad checks (two sentences)
The counter counts consecutive monitor checks satisfying cosine >0.99 OR code std <0.01, resets to zero at a healthy check, and marks collapse at three; monitors occur every 50 training batches (`experiments/stego_biomedclip/train.py:66–79`). It is not capped: seeds 0/2 reached 100 checks, whereas `resume_seed1.py:32` reset `bad=0`, so the final 46 count covers the continuation from batch 2700 rather than the complete trajectory.

### Duplicate seed-1 final records (two sentences)
The existing log contains interleaved resumed sequences ending at batch 5000 with 47 and 46 bad checks (consistent with restarts from 2650 and 2700), and the resumed writer unconditionally appends to the same `seed1_train.jsonl` and overwrites the same last/best paths (`resume_seed1.py:34`, :60–63); this accounts for the two final std values 1.2421762e-5 and 1.2384027e-5. The saved resume metadata confirms the 2700 continuation without restored optimizer/RNG, but historical PID attribution and which concurrent writer last supplied each checkpoint are UNVERIFIED because a durable process history is absent; no sealed log was edited and the current process snapshot contains no training process.

Stage 1/2 configs are sealed before training; only D448 is admissible. No checkpoint selection: stage 2 uses final.pt only. All validation labels remain closed.

Task 3 progress: D448 seed 0 completed 150 batches; screen_pass=True; final std=0.11372209, cosine=0.02431929. Evidence: `experiments/stego_calibrated/D448/stage1/logs/seed0_complete.json`. Seed 1 launched only after seed 0 exited with code 0 and a fresh overlap check passed.

<!-- GENERATED RESULTS -->
## Task 3 — Stage 1 screen

C448 was not run because it failed admissibility. D448 uses the same frozen head/loss implementation and seeded initialization; only the calibrated shifts differ from the sealed baseline. Production runs are sequential, use a package-wide exclusive lock, and have individual process snapshots with no other training job.

| Candidate | Seed | Batch | Code std | Cosine | Usage C0…C7 (%) | Healthy |
| --- | --- | --- | --- | --- | --- | --- |
| D448 | 0 | 0 | 0.04131734 | 0.86023968 | 17.07, 10.24, 23.13, 5.72, 14.05, 5.66, 9.59, 14.54 | True |
| D448 | 0 | 50 | 0.11165997 | 0.03253835 | 6.42, 27.94, 17.94, 4.94, 14.41, 7.34, 8.73, 12.28 | True |
| D448 | 0 | 100 | 0.11276515 | 0.02706319 | 6.37, 15.34, 14.28, 15.84, 13.54, 6.39, 12.97, 15.27 | True |
| D448 | 0 | 150 | 0.11372209 | 0.02431929 | 6.23, 18.70, 13.68, 14.57, 13.46, 7.05, 20.34, 5.96 | True |
| D448 | 1 | 0 | 0.04575163 | 0.82948053 | 16.21, 28.40, 4.50, 5.68, 18.10, 5.96, 4.93, 16.24 | True |
| D448 | 1 | 50 | 0.11280610 | 0.02809830 | 7.02, 19.00, 12.10, 6.44, 26.82, 14.65, 5.16, 8.82 | True |
| D448 | 1 | 100 | 0.11369539 | 0.02513426 | 12.36, 15.09, 15.00, 6.35, 13.46, 15.13, 16.32, 6.30 | True |
| D448 | 1 | 150 | 0.11434014 | 0.02186409 | 6.24, 14.29, 13.93, 16.17, 18.36, 12.40, 5.84, 12.78 | True |
| D448 | 2 | 0 | 0.05203552 | 0.78031081 | 5.96, 10.28, 15.43, 21.72, 5.58, 6.29, 19.06, 15.69 | True |
| D448 | 2 | 50 | 0.11202686 | 0.03284588 | 6.51, 27.57, 18.15, 11.79, 9.41, 14.34, 5.12, 7.11 | True |
| D448 | 2 | 100 | 0.11315526 | 0.02749867 | 13.72, 15.73, 16.01, 14.36, 6.25, 6.39, 14.72, 12.83 | True |
| D448 | 2 | 150 | 0.11353802 | 0.02641558 | 6.23, 13.98, 14.33, 11.89, 16.22, 5.94, 13.13, 18.27 | True |

| Candidate | Seed | Budget | First 10 seconds | Projected minutes | Actual minutes |
| --- | --- | --- | --- | --- | --- |
| D448 | 0 | 150 | 10.078 | 2.520 | 2.562 |
| D448 | 1 | 150 | 10.681 | 2.670 | 2.596 |
| D448 | 2 | 150 | 10.143 | 2.536 | 2.602 |

Stage 1 gate: **PASS**. Every seed must be healthy at all three checks and have lower cosine at batch 150 than at initialization; the gate is applied without labels.

## Task 4 — Stage 2

All runs start from scratch and use the final batch-500 state; no checkpoint selection.

| Candidate | Seed | Batch | Code std | Cosine | Usage C0…C7 (%) | Healthy |
| --- | --- | --- | --- | --- | --- | --- |
| D448 | 0 | 50 | 0.11165997 | 0.03253835 | 6.42, 27.94, 17.94, 4.94, 14.41, 7.34, 8.73, 12.28 | True |
| D448 | 0 | 100 | 0.11276515 | 0.02706319 | 6.37, 15.34, 14.28, 15.84, 13.54, 6.39, 12.97, 15.27 | True |
| D448 | 0 | 150 | 0.11372209 | 0.02431929 | 6.23, 18.70, 13.68, 14.57, 13.46, 7.05, 20.34, 5.96 | True |
| D448 | 0 | 200 | 0.11426850 | 0.02192504 | 14.10, 13.85, 18.65, 14.57, 5.92, 6.17, 9.49, 17.24 | True |
| D448 | 0 | 250 | 0.11460431 | 0.01923727 | 6.10, 19.67, 14.18, 7.89, 8.31, 25.57, 5.58, 12.70 | True |
| D448 | 0 | 300 | 0.11460385 | 0.01860624 | 6.08, 19.08, 14.72, 22.32, 13.30, 13.70, 5.56, 5.24 | True |
| D448 | 0 | 350 | 0.11466427 | 0.01638865 | 6.06, 19.47, 8.74, 13.39, 12.21, 5.46, 26.28, 8.39 | True |
| D448 | 0 | 400 | 0.11411825 | 0.02585649 | 12.88, 11.51, 14.71, 23.98, 5.43, 6.03, 4.56, 20.90 | True |
| D448 | 0 | 450 | 0.11423106 | 0.02278706 | 12.70, 11.48, 6.03, 20.45, 8.61, 5.38, 27.10, 8.25 | True |
| D448 | 0 | 500 | 0.11464351 | 0.01490266 | 6.02, 11.89, 14.89, 20.53, 11.23, 20.12, 5.26, 10.05 | True |
| D448 | 1 | 50 | 0.11280610 | 0.02809830 | 7.02, 19.00, 12.10, 6.44, 26.82, 14.65, 5.16, 8.82 | True |
| D448 | 1 | 100 | 0.11369539 | 0.02513426 | 12.36, 15.09, 15.00, 6.35, 13.46, 15.13, 16.32, 6.30 | True |
| D448 | 1 | 150 | 0.11434014 | 0.02186409 | 6.24, 14.29, 13.93, 16.17, 18.36, 12.40, 5.84, 12.78 | True |
| D448 | 1 | 200 | 0.11436898 | 0.02922396 | 13.94, 13.40, 6.15, 14.02, 19.52, 5.60, 6.03, 21.32 | True |
| D448 | 1 | 250 | 0.11487088 | 0.02175558 | 13.61, 12.97, 10.16, 19.95, 17.37, 5.53, 14.32, 6.10 | True |
| D448 | 1 | 300 | 0.11489567 | 0.02007494 | 6.06, 5.38, 22.85, 20.58, 14.11, 14.18, 4.51, 12.33 | True |
| D448 | 1 | 350 | 0.11512240 | 0.01720147 | 14.01, 12.89, 19.73, 18.61, 5.48, 6.06, 8.62, 14.60 | True |
| D448 | 1 | 400 | 0.11494533 | 0.02060077 | 14.01, 12.51, 6.06, 19.91, 19.32, 14.79, 4.37, 9.03 | True |
| D448 | 1 | 450 | 0.11491673 | 0.01911089 | 13.46, 20.63, 5.30, 14.88, 23.72, 6.03, 11.69, 4.29 | True |
| D448 | 1 | 500 | 0.11474062 | 0.02075766 | 6.02, 11.55, 20.92, 13.15, 8.55, 5.19, 27.26, 7.37 | True |
| D448 | 2 | 50 | 0.11202686 | 0.03284588 | 6.51, 27.57, 18.15, 11.79, 9.41, 14.34, 5.12, 7.11 | True |
| D448 | 2 | 100 | 0.11315526 | 0.02749867 | 13.72, 15.73, 16.01, 14.36, 6.25, 6.39, 14.72, 12.83 | True |
| D448 | 2 | 150 | 0.11353802 | 0.02641558 | 6.23, 13.98, 14.33, 11.89, 16.22, 5.94, 13.13, 18.27 | True |
| D448 | 2 | 200 | 0.11398855 | 0.01986131 | 11.32, 19.53, 13.27, 22.01, 14.19, 5.72, 6.16, 7.80 | True |
| D448 | 2 | 250 | 0.11370048 | 0.02488751 | 11.77, 12.82, 20.03, 13.11, 16.25, 6.11, 14.31, 5.60 | True |
| D448 | 2 | 300 | 0.11367923 | 0.02606644 | 6.07, 20.04, 9.45, 12.64, 22.58, 14.45, 9.34, 5.44 | True |
| D448 | 2 | 350 | 0.11392336 | 0.02107289 | 13.79, 20.20, 5.47, 14.75, 6.05, 18.96, 8.47, 12.31 | True |
| D448 | 2 | 400 | 0.11374189 | 0.02318036 | 13.82, 11.69, 6.02, 20.03, 14.48, 3.65, 21.05, 9.26 | True |
| D448 | 2 | 450 | 0.11393449 | 0.01842825 | 6.03, 12.12, 7.94, 8.47, 19.92, 13.31, 26.87, 5.35 | True |
| D448 | 2 | 500 | 0.11379173 | 0.01913199 | 13.34, 20.16, 12.13, 8.50, 6.03, 23.32, 7.56, 8.97 | True |

| Seed | First 10 seconds | Projected minutes | Actual minutes |
| --- | --- | --- | --- |
| 0 | 9.650 | 8.042 | 8.371 |
| 1 | 9.757 | 8.131 | 8.416 |
| 2 | 10.635 | 8.862 | 8.451 |

### Stage B — all rules and scales

Labels are used only for validation scoring, after all three selected runs were globally sealed. N1 uses pretrained image/text similarities; N2 uses a fixed anatomical prior; N3 accepts their agreement and scores abstention as an empty mask. Patient-pooled IoU sums TP/FP/FN over all slices of that patient; mean-slice averages individual slice scores (empty union scores zero, inherited unchanged). Primary: N3, F1, 224, four-patient mean of pooled-pixel scores; SD is sample SD across patients.

| Candidate | Seed | Rule | Named cluster | Prediction | Scale | Aggregation | IoU % mean ± SD | Dice % mean ± SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D448 | 0 | N1 | 1 | hard | 224 | pooled | 70.6668 ± 2.0532 | 82.7999 ± 1.4007 |
| D448 | 0 | N1 | 1 | hard | 224 | mean_slice | 61.8036 ± 6.2449 | 73.7281 ± 6.3048 |
| D448 | 0 | N1 | 1 | hard | 512 | pooled | 70.7488 ± 1.9665 | 82.8573 ± 1.3400 |
| D448 | 0 | N1 | 1 | hard | 512 | mean_slice | 61.9064 ± 6.1982 | 73.8193 ± 6.2593 |
| D448 | 0 | N1 | 1 | F1 | 224 | pooled | 66.2786 ± 2.4487 | 79.7004 ± 1.7692 |
| D448 | 0 | N1 | 1 | F1 | 224 | mean_slice | 54.2659 ± 7.7836 | 66.1320 ± 8.4981 |
| D448 | 0 | N1 | 1 | F1 | 512 | pooled | 66.2573 ± 2.4237 | 79.6854 ± 1.7509 |
| D448 | 0 | N1 | 1 | F1 | 512 | mean_slice | 54.2517 ± 7.7327 | 66.1268 ± 8.4485 |
| D448 | 0 | N2 | 1 | hard | 224 | pooled | 70.6668 ± 2.0532 | 82.7999 ± 1.4007 |
| D448 | 0 | N2 | 1 | hard | 224 | mean_slice | 61.8036 ± 6.2449 | 73.7281 ± 6.3048 |
| D448 | 0 | N2 | 1 | hard | 512 | pooled | 70.7488 ± 1.9665 | 82.8573 ± 1.3400 |
| D448 | 0 | N2 | 1 | hard | 512 | mean_slice | 61.9064 ± 6.1982 | 73.8193 ± 6.2593 |
| D448 | 0 | N2 | 1 | F1 | 224 | pooled | 66.2786 ± 2.4487 | 79.7004 ± 1.7692 |
| D448 | 0 | N2 | 1 | F1 | 224 | mean_slice | 54.2659 ± 7.7836 | 66.1320 ± 8.4981 |
| D448 | 0 | N2 | 1 | F1 | 512 | pooled | 66.2573 ± 2.4237 | 79.6854 ± 1.7509 |
| D448 | 0 | N2 | 1 | F1 | 512 | mean_slice | 54.2517 ± 7.7327 | 66.1268 ± 8.4485 |
| D448 | 0 | N3 | 1 | hard | 224 | pooled | 70.6668 ± 2.0532 | 82.7999 ± 1.4007 |
| D448 | 0 | N3 | 1 | hard | 224 | mean_slice | 61.8036 ± 6.2449 | 73.7281 ± 6.3048 |
| D448 | 0 | N3 | 1 | hard | 512 | pooled | 70.7488 ± 1.9665 | 82.8573 ± 1.3400 |
| D448 | 0 | N3 | 1 | hard | 512 | mean_slice | 61.9064 ± 6.1982 | 73.8193 ± 6.2593 |
| D448 | 0 | N3 | 1 | F1 | 224 | pooled | 66.2786 ± 2.4487 | 79.7004 ± 1.7692 |
| D448 | 0 | N3 | 1 | F1 | 224 | mean_slice | 54.2659 ± 7.7836 | 66.1320 ± 8.4981 |
| D448 | 0 | N3 | 1 | F1 | 512 | pooled | 66.2573 ± 2.4237 | 79.6854 ± 1.7509 |
| D448 | 0 | N3 | 1 | F1 | 512 | mean_slice | 54.2517 ± 7.7327 | 66.1268 ± 8.4485 |
| D448 | 1 | N1 | 3 | hard | 224 | pooled | 70.5800 ± 1.9195 | 82.7419 ± 1.3109 |
| D448 | 1 | N1 | 3 | hard | 224 | mean_slice | 61.6991 ± 6.1550 | 73.6235 ± 6.2992 |
| D448 | 1 | N1 | 3 | hard | 512 | pooled | 70.6635 ± 1.8289 | 82.8003 ± 1.2476 |
| D448 | 1 | N1 | 3 | hard | 512 | mean_slice | 61.7996 ± 6.1073 | 73.7113 ± 6.2525 |
| D448 | 1 | N1 | 3 | F1 | 224 | pooled | 66.1114 ± 2.4071 | 79.5799 ± 1.7428 |
| D448 | 1 | N1 | 3 | F1 | 224 | mean_slice | 54.0835 ± 7.7602 | 65.9914 ± 8.4961 |
| D448 | 1 | N1 | 3 | F1 | 512 | pooled | 66.0850 ± 2.3882 | 79.5611 ± 1.7290 |
| D448 | 1 | N1 | 3 | F1 | 512 | mean_slice | 54.0605 ± 7.7180 | 65.9773 ± 8.4582 |
| D448 | 1 | N2 | 3 | hard | 224 | pooled | 70.5800 ± 1.9195 | 82.7419 ± 1.3109 |
| D448 | 1 | N2 | 3 | hard | 224 | mean_slice | 61.6991 ± 6.1550 | 73.6235 ± 6.2992 |
| D448 | 1 | N2 | 3 | hard | 512 | pooled | 70.6635 ± 1.8289 | 82.8003 ± 1.2476 |
| D448 | 1 | N2 | 3 | hard | 512 | mean_slice | 61.7996 ± 6.1073 | 73.7113 ± 6.2525 |
| D448 | 1 | N2 | 3 | F1 | 224 | pooled | 66.1114 ± 2.4071 | 79.5799 ± 1.7428 |
| D448 | 1 | N2 | 3 | F1 | 224 | mean_slice | 54.0835 ± 7.7602 | 65.9914 ± 8.4961 |
| D448 | 1 | N2 | 3 | F1 | 512 | pooled | 66.0850 ± 2.3882 | 79.5611 ± 1.7290 |
| D448 | 1 | N2 | 3 | F1 | 512 | mean_slice | 54.0605 ± 7.7180 | 65.9773 ± 8.4582 |
| D448 | 1 | N3 | 3 | hard | 224 | pooled | 70.5800 ± 1.9195 | 82.7419 ± 1.3109 |
| D448 | 1 | N3 | 3 | hard | 224 | mean_slice | 61.6991 ± 6.1550 | 73.6235 ± 6.2992 |
| D448 | 1 | N3 | 3 | hard | 512 | pooled | 70.6635 ± 1.8289 | 82.8003 ± 1.2476 |
| D448 | 1 | N3 | 3 | hard | 512 | mean_slice | 61.7996 ± 6.1073 | 73.7113 ± 6.2525 |
| D448 | 1 | N3 | 3 | F1 | 224 | pooled | 66.1114 ± 2.4071 | 79.5799 ± 1.7428 |
| D448 | 1 | N3 | 3 | F1 | 224 | mean_slice | 54.0835 ± 7.7602 | 65.9914 ± 8.4961 |
| D448 | 1 | N3 | 3 | F1 | 512 | pooled | 66.0850 ± 2.3882 | 79.5611 ± 1.7290 |
| D448 | 1 | N3 | 3 | F1 | 512 | mean_slice | 54.0605 ± 7.7180 | 65.9773 ± 8.4582 |
| D448 | 2 | N1 | 2 | hard | 224 | pooled | 70.6156 ± 2.0248 | 82.7651 ± 1.3814 |
| D448 | 2 | N1 | 2 | hard | 224 | mean_slice | 61.7984 ± 6.2408 | 73.7286 ± 6.3211 |
| D448 | 2 | N1 | 2 | hard | 512 | pooled | 70.6995 ± 1.9404 | 82.8238 ± 1.3223 |
| D448 | 2 | N1 | 2 | hard | 512 | mean_slice | 61.9042 ± 6.1985 | 73.8211 ± 6.2776 |
| D448 | 2 | N1 | 2 | F1 | 224 | pooled | 65.6416 ± 2.3894 | 79.2386 ± 1.7420 |
| D448 | 2 | N1 | 2 | F1 | 224 | mean_slice | 53.6903 ± 7.7088 | 65.6805 ± 8.4739 |
| D448 | 2 | N1 | 2 | F1 | 512 | pooled | 65.6215 ± 2.3684 | 79.2242 ± 1.7260 |
| D448 | 2 | N1 | 2 | F1 | 512 | mean_slice | 53.6796 ± 7.6687 | 65.6757 ± 8.4346 |
| D448 | 2 | N2 | 2 | hard | 224 | pooled | 70.6156 ± 2.0248 | 82.7651 ± 1.3814 |
| D448 | 2 | N2 | 2 | hard | 224 | mean_slice | 61.7984 ± 6.2408 | 73.7286 ± 6.3211 |
| D448 | 2 | N2 | 2 | hard | 512 | pooled | 70.6995 ± 1.9404 | 82.8238 ± 1.3223 |
| D448 | 2 | N2 | 2 | hard | 512 | mean_slice | 61.9042 ± 6.1985 | 73.8211 ± 6.2776 |
| D448 | 2 | N2 | 2 | F1 | 224 | pooled | 65.6416 ± 2.3894 | 79.2386 ± 1.7420 |
| D448 | 2 | N2 | 2 | F1 | 224 | mean_slice | 53.6903 ± 7.7088 | 65.6805 ± 8.4739 |
| D448 | 2 | N2 | 2 | F1 | 512 | pooled | 65.6215 ± 2.3684 | 79.2242 ± 1.7260 |
| D448 | 2 | N2 | 2 | F1 | 512 | mean_slice | 53.6796 ± 7.6687 | 65.6757 ± 8.4346 |
| D448 | 2 | N3 | 2 | hard | 224 | pooled | 70.6156 ± 2.0248 | 82.7651 ± 1.3814 |
| D448 | 2 | N3 | 2 | hard | 224 | mean_slice | 61.7984 ± 6.2408 | 73.7286 ± 6.3211 |
| D448 | 2 | N3 | 2 | hard | 512 | pooled | 70.6995 ± 1.9404 | 82.8238 ± 1.3223 |
| D448 | 2 | N3 | 2 | hard | 512 | mean_slice | 61.9042 ± 6.1985 | 73.8211 ± 6.2776 |
| D448 | 2 | N3 | 2 | F1 | 224 | pooled | 65.6416 ± 2.3894 | 79.2386 ± 1.7420 |
| D448 | 2 | N3 | 2 | F1 | 224 | mean_slice | 53.6903 ± 7.7088 | 65.6805 ± 8.4739 |
| D448 | 2 | N3 | 2 | F1 | 512 | pooled | 65.6215 ± 2.3684 | 79.2242 ± 1.7260 |
| D448 | 2 | N3 | 2 | F1 | 512 | mean_slice | 53.6796 ± 7.6687 | 65.6757 ± 8.4346 |

### Per-patient Stage B

| Seed | Rule | Prediction | Scale | Aggregation | Patient | Slices | IoU % | Dice % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | N1 | hard | 224 | pooled | 5 | 95 | 73.5090 | 84.7322 |
| 0 | N1 | hard | 224 | pooled | 6 | 82 | 70.5385 | 82.7244 |
| 0 | N1 | hard | 224 | pooled | 8 | 98 | 68.6507 | 81.4117 |
| 0 | N1 | hard | 224 | pooled | 30 | 197 | 69.9689 | 82.3314 |
| 0 | N1 | hard | 224 | mean_slice | 5 | 95 | 68.3796 | 80.2827 |
| 0 | N1 | hard | 224 | mean_slice | 6 | 82 | 65.8996 | 77.9624 |
| 0 | N1 | hard | 224 | mean_slice | 8 | 98 | 56.5924 | 68.6147 |
| 0 | N1 | hard | 224 | mean_slice | 30 | 197 | 56.3428 | 68.0526 |
| 0 | N1 | hard | 512 | pooled | 5 | 95 | 73.4952 | 84.7231 |
| 0 | N1 | hard | 512 | pooled | 6 | 82 | 70.5494 | 82.7320 |
| 0 | N1 | hard | 512 | pooled | 8 | 98 | 68.8529 | 81.5537 |
| 0 | N1 | hard | 512 | pooled | 30 | 197 | 70.0975 | 82.4204 |
| 0 | N1 | hard | 512 | mean_slice | 5 | 95 | 68.4746 | 80.3673 |
| 0 | N1 | hard | 512 | mean_slice | 6 | 82 | 65.9198 | 77.9754 |
| 0 | N1 | hard | 512 | mean_slice | 8 | 98 | 56.7014 | 68.6911 |
| 0 | N1 | hard | 512 | mean_slice | 30 | 197 | 56.5295 | 68.2435 |
| 0 | N1 | F1 | 224 | pooled | 5 | 95 | 66.7466 | 80.0576 |
| 0 | N1 | F1 | 224 | pooled | 6 | 82 | 69.3152 | 81.8771 |
| 0 | N1 | F1 | 224 | pooled | 8 | 98 | 63.4292 | 77.6229 |
| 0 | N1 | F1 | 224 | pooled | 30 | 197 | 65.6233 | 79.2440 |
| 0 | N1 | F1 | 224 | mean_slice | 5 | 95 | 58.6015 | 71.8658 |
| 0 | N1 | F1 | 224 | mean_slice | 6 | 82 | 63.0197 | 74.9281 |
| 0 | N1 | F1 | 224 | mean_slice | 8 | 98 | 48.2953 | 59.5157 |
| 0 | N1 | F1 | 224 | mean_slice | 30 | 197 | 47.1470 | 58.2182 |
| 0 | N1 | F1 | 512 | pooled | 5 | 95 | 66.6927 | 80.0187 |
| 0 | N1 | F1 | 512 | pooled | 6 | 82 | 69.2862 | 81.8569 |
| 0 | N1 | F1 | 512 | pooled | 8 | 98 | 63.4624 | 77.6477 |
| 0 | N1 | F1 | 512 | pooled | 30 | 197 | 65.5880 | 79.2183 |
| 0 | N1 | F1 | 512 | mean_slice | 5 | 95 | 58.5429 | 71.8193 |
| 0 | N1 | F1 | 512 | mean_slice | 6 | 82 | 62.9570 | 74.8762 |
| 0 | N1 | F1 | 512 | mean_slice | 8 | 98 | 48.3587 | 59.5736 |
| 0 | N1 | F1 | 512 | mean_slice | 30 | 197 | 47.1483 | 58.2382 |
| 0 | N2 | hard | 224 | pooled | 5 | 95 | 73.5090 | 84.7322 |
| 0 | N2 | hard | 224 | pooled | 6 | 82 | 70.5385 | 82.7244 |
| 0 | N2 | hard | 224 | pooled | 8 | 98 | 68.6507 | 81.4117 |
| 0 | N2 | hard | 224 | pooled | 30 | 197 | 69.9689 | 82.3314 |
| 0 | N2 | hard | 224 | mean_slice | 5 | 95 | 68.3796 | 80.2827 |
| 0 | N2 | hard | 224 | mean_slice | 6 | 82 | 65.8996 | 77.9624 |
| 0 | N2 | hard | 224 | mean_slice | 8 | 98 | 56.5924 | 68.6147 |
| 0 | N2 | hard | 224 | mean_slice | 30 | 197 | 56.3428 | 68.0526 |
| 0 | N2 | hard | 512 | pooled | 5 | 95 | 73.4952 | 84.7231 |
| 0 | N2 | hard | 512 | pooled | 6 | 82 | 70.5494 | 82.7320 |
| 0 | N2 | hard | 512 | pooled | 8 | 98 | 68.8529 | 81.5537 |
| 0 | N2 | hard | 512 | pooled | 30 | 197 | 70.0975 | 82.4204 |
| 0 | N2 | hard | 512 | mean_slice | 5 | 95 | 68.4746 | 80.3673 |
| 0 | N2 | hard | 512 | mean_slice | 6 | 82 | 65.9198 | 77.9754 |
| 0 | N2 | hard | 512 | mean_slice | 8 | 98 | 56.7014 | 68.6911 |
| 0 | N2 | hard | 512 | mean_slice | 30 | 197 | 56.5295 | 68.2435 |
| 0 | N2 | F1 | 224 | pooled | 5 | 95 | 66.7466 | 80.0576 |
| 0 | N2 | F1 | 224 | pooled | 6 | 82 | 69.3152 | 81.8771 |
| 0 | N2 | F1 | 224 | pooled | 8 | 98 | 63.4292 | 77.6229 |
| 0 | N2 | F1 | 224 | pooled | 30 | 197 | 65.6233 | 79.2440 |
| 0 | N2 | F1 | 224 | mean_slice | 5 | 95 | 58.6015 | 71.8658 |
| 0 | N2 | F1 | 224 | mean_slice | 6 | 82 | 63.0197 | 74.9281 |
| 0 | N2 | F1 | 224 | mean_slice | 8 | 98 | 48.2953 | 59.5157 |
| 0 | N2 | F1 | 224 | mean_slice | 30 | 197 | 47.1470 | 58.2182 |
| 0 | N2 | F1 | 512 | pooled | 5 | 95 | 66.6927 | 80.0187 |
| 0 | N2 | F1 | 512 | pooled | 6 | 82 | 69.2862 | 81.8569 |
| 0 | N2 | F1 | 512 | pooled | 8 | 98 | 63.4624 | 77.6477 |
| 0 | N2 | F1 | 512 | pooled | 30 | 197 | 65.5880 | 79.2183 |
| 0 | N2 | F1 | 512 | mean_slice | 5 | 95 | 58.5429 | 71.8193 |
| 0 | N2 | F1 | 512 | mean_slice | 6 | 82 | 62.9570 | 74.8762 |
| 0 | N2 | F1 | 512 | mean_slice | 8 | 98 | 48.3587 | 59.5736 |
| 0 | N2 | F1 | 512 | mean_slice | 30 | 197 | 47.1483 | 58.2382 |
| 0 | N3 | hard | 224 | pooled | 5 | 95 | 73.5090 | 84.7322 |
| 0 | N3 | hard | 224 | pooled | 6 | 82 | 70.5385 | 82.7244 |
| 0 | N3 | hard | 224 | pooled | 8 | 98 | 68.6507 | 81.4117 |
| 0 | N3 | hard | 224 | pooled | 30 | 197 | 69.9689 | 82.3314 |
| 0 | N3 | hard | 224 | mean_slice | 5 | 95 | 68.3796 | 80.2827 |
| 0 | N3 | hard | 224 | mean_slice | 6 | 82 | 65.8996 | 77.9624 |
| 0 | N3 | hard | 224 | mean_slice | 8 | 98 | 56.5924 | 68.6147 |
| 0 | N3 | hard | 224 | mean_slice | 30 | 197 | 56.3428 | 68.0526 |
| 0 | N3 | hard | 512 | pooled | 5 | 95 | 73.4952 | 84.7231 |
| 0 | N3 | hard | 512 | pooled | 6 | 82 | 70.5494 | 82.7320 |
| 0 | N3 | hard | 512 | pooled | 8 | 98 | 68.8529 | 81.5537 |
| 0 | N3 | hard | 512 | pooled | 30 | 197 | 70.0975 | 82.4204 |
| 0 | N3 | hard | 512 | mean_slice | 5 | 95 | 68.4746 | 80.3673 |
| 0 | N3 | hard | 512 | mean_slice | 6 | 82 | 65.9198 | 77.9754 |
| 0 | N3 | hard | 512 | mean_slice | 8 | 98 | 56.7014 | 68.6911 |
| 0 | N3 | hard | 512 | mean_slice | 30 | 197 | 56.5295 | 68.2435 |
| 0 | N3 | F1 | 224 | pooled | 5 | 95 | 66.7466 | 80.0576 |
| 0 | N3 | F1 | 224 | pooled | 6 | 82 | 69.3152 | 81.8771 |
| 0 | N3 | F1 | 224 | pooled | 8 | 98 | 63.4292 | 77.6229 |
| 0 | N3 | F1 | 224 | pooled | 30 | 197 | 65.6233 | 79.2440 |
| 0 | N3 | F1 | 224 | mean_slice | 5 | 95 | 58.6015 | 71.8658 |
| 0 | N3 | F1 | 224 | mean_slice | 6 | 82 | 63.0197 | 74.9281 |
| 0 | N3 | F1 | 224 | mean_slice | 8 | 98 | 48.2953 | 59.5157 |
| 0 | N3 | F1 | 224 | mean_slice | 30 | 197 | 47.1470 | 58.2182 |
| 0 | N3 | F1 | 512 | pooled | 5 | 95 | 66.6927 | 80.0187 |
| 0 | N3 | F1 | 512 | pooled | 6 | 82 | 69.2862 | 81.8569 |
| 0 | N3 | F1 | 512 | pooled | 8 | 98 | 63.4624 | 77.6477 |
| 0 | N3 | F1 | 512 | pooled | 30 | 197 | 65.5880 | 79.2183 |
| 0 | N3 | F1 | 512 | mean_slice | 5 | 95 | 58.5429 | 71.8193 |
| 0 | N3 | F1 | 512 | mean_slice | 6 | 82 | 62.9570 | 74.8762 |
| 0 | N3 | F1 | 512 | mean_slice | 8 | 98 | 48.3587 | 59.5736 |
| 0 | N3 | F1 | 512 | mean_slice | 30 | 197 | 47.1483 | 58.2382 |
| 1 | N1 | hard | 224 | pooled | 5 | 95 | 73.2574 | 84.5648 |
| 1 | N1 | hard | 224 | pooled | 6 | 82 | 70.3588 | 82.6007 |
| 1 | N1 | hard | 224 | pooled | 8 | 98 | 68.7118 | 81.4547 |
| 1 | N1 | hard | 224 | pooled | 30 | 197 | 69.9920 | 82.3474 |
| 1 | N1 | hard | 224 | mean_slice | 5 | 95 | 68.1629 | 80.1262 |
| 1 | N1 | hard | 224 | mean_slice | 6 | 82 | 65.7560 | 77.9042 |
| 1 | N1 | hard | 224 | mean_slice | 8 | 98 | 56.6277 | 68.6127 |
| 1 | N1 | hard | 224 | mean_slice | 30 | 197 | 56.2498 | 67.8510 |
| 1 | N1 | hard | 512 | pooled | 5 | 95 | 73.2379 | 84.5518 |
| 1 | N1 | hard | 512 | pooled | 6 | 82 | 70.3724 | 82.6101 |
| 1 | N1 | hard | 512 | pooled | 8 | 98 | 68.9238 | 81.6034 |
| 1 | N1 | hard | 512 | pooled | 30 | 197 | 70.1200 | 82.4359 |
| 1 | N1 | hard | 512 | mean_slice | 5 | 95 | 68.2529 | 80.2071 |
| 1 | N1 | hard | 512 | mean_slice | 6 | 82 | 65.7761 | 77.9128 |
| 1 | N1 | hard | 512 | mean_slice | 8 | 98 | 56.7376 | 68.6917 |
| 1 | N1 | hard | 512 | mean_slice | 30 | 197 | 56.4317 | 68.0337 |
| 1 | N1 | F1 | 224 | pooled | 5 | 95 | 66.4399 | 79.8365 |
| 1 | N1 | F1 | 224 | pooled | 6 | 82 | 69.1123 | 81.7354 |
| 1 | N1 | F1 | 224 | pooled | 8 | 98 | 63.2792 | 77.5104 |
| 1 | N1 | F1 | 224 | pooled | 30 | 197 | 65.6143 | 79.2375 |
| 1 | N1 | F1 | 224 | mean_slice | 5 | 95 | 58.2852 | 71.5791 |
| 1 | N1 | F1 | 224 | mean_slice | 6 | 82 | 62.9044 | 74.9102 |
| 1 | N1 | F1 | 224 | mean_slice | 8 | 98 | 48.0240 | 59.2684 |
| 1 | N1 | F1 | 224 | mean_slice | 30 | 197 | 47.1204 | 58.2078 |
| 1 | N1 | F1 | 512 | pooled | 5 | 95 | 66.3855 | 79.7972 |
| 1 | N1 | F1 | 512 | pooled | 6 | 82 | 69.0841 | 81.7156 |
| 1 | N1 | F1 | 512 | pooled | 8 | 98 | 63.2989 | 77.5252 |
| 1 | N1 | F1 | 512 | pooled | 30 | 197 | 65.5716 | 79.2063 |
| 1 | N1 | F1 | 512 | mean_slice | 5 | 95 | 58.2290 | 71.5415 |
| 1 | N1 | F1 | 512 | mean_slice | 6 | 82 | 62.8394 | 74.8554 |
| 1 | N1 | F1 | 512 | mean_slice | 8 | 98 | 48.0544 | 59.2830 |
| 1 | N1 | F1 | 512 | mean_slice | 30 | 197 | 47.1192 | 58.2295 |
| 1 | N2 | hard | 224 | pooled | 5 | 95 | 73.2574 | 84.5648 |
| 1 | N2 | hard | 224 | pooled | 6 | 82 | 70.3588 | 82.6007 |
| 1 | N2 | hard | 224 | pooled | 8 | 98 | 68.7118 | 81.4547 |
| 1 | N2 | hard | 224 | pooled | 30 | 197 | 69.9920 | 82.3474 |
| 1 | N2 | hard | 224 | mean_slice | 5 | 95 | 68.1629 | 80.1262 |
| 1 | N2 | hard | 224 | mean_slice | 6 | 82 | 65.7560 | 77.9042 |
| 1 | N2 | hard | 224 | mean_slice | 8 | 98 | 56.6277 | 68.6127 |
| 1 | N2 | hard | 224 | mean_slice | 30 | 197 | 56.2498 | 67.8510 |
| 1 | N2 | hard | 512 | pooled | 5 | 95 | 73.2379 | 84.5518 |
| 1 | N2 | hard | 512 | pooled | 6 | 82 | 70.3724 | 82.6101 |
| 1 | N2 | hard | 512 | pooled | 8 | 98 | 68.9238 | 81.6034 |
| 1 | N2 | hard | 512 | pooled | 30 | 197 | 70.1200 | 82.4359 |
| 1 | N2 | hard | 512 | mean_slice | 5 | 95 | 68.2529 | 80.2071 |
| 1 | N2 | hard | 512 | mean_slice | 6 | 82 | 65.7761 | 77.9128 |
| 1 | N2 | hard | 512 | mean_slice | 8 | 98 | 56.7376 | 68.6917 |
| 1 | N2 | hard | 512 | mean_slice | 30 | 197 | 56.4317 | 68.0337 |
| 1 | N2 | F1 | 224 | pooled | 5 | 95 | 66.4399 | 79.8365 |
| 1 | N2 | F1 | 224 | pooled | 6 | 82 | 69.1123 | 81.7354 |
| 1 | N2 | F1 | 224 | pooled | 8 | 98 | 63.2792 | 77.5104 |
| 1 | N2 | F1 | 224 | pooled | 30 | 197 | 65.6143 | 79.2375 |
| 1 | N2 | F1 | 224 | mean_slice | 5 | 95 | 58.2852 | 71.5791 |
| 1 | N2 | F1 | 224 | mean_slice | 6 | 82 | 62.9044 | 74.9102 |
| 1 | N2 | F1 | 224 | mean_slice | 8 | 98 | 48.0240 | 59.2684 |
| 1 | N2 | F1 | 224 | mean_slice | 30 | 197 | 47.1204 | 58.2078 |
| 1 | N2 | F1 | 512 | pooled | 5 | 95 | 66.3855 | 79.7972 |
| 1 | N2 | F1 | 512 | pooled | 6 | 82 | 69.0841 | 81.7156 |
| 1 | N2 | F1 | 512 | pooled | 8 | 98 | 63.2989 | 77.5252 |
| 1 | N2 | F1 | 512 | pooled | 30 | 197 | 65.5716 | 79.2063 |
| 1 | N2 | F1 | 512 | mean_slice | 5 | 95 | 58.2290 | 71.5415 |
| 1 | N2 | F1 | 512 | mean_slice | 6 | 82 | 62.8394 | 74.8554 |
| 1 | N2 | F1 | 512 | mean_slice | 8 | 98 | 48.0544 | 59.2830 |
| 1 | N2 | F1 | 512 | mean_slice | 30 | 197 | 47.1192 | 58.2295 |
| 1 | N3 | hard | 224 | pooled | 5 | 95 | 73.2574 | 84.5648 |
| 1 | N3 | hard | 224 | pooled | 6 | 82 | 70.3588 | 82.6007 |
| 1 | N3 | hard | 224 | pooled | 8 | 98 | 68.7118 | 81.4547 |
| 1 | N3 | hard | 224 | pooled | 30 | 197 | 69.9920 | 82.3474 |
| 1 | N3 | hard | 224 | mean_slice | 5 | 95 | 68.1629 | 80.1262 |
| 1 | N3 | hard | 224 | mean_slice | 6 | 82 | 65.7560 | 77.9042 |
| 1 | N3 | hard | 224 | mean_slice | 8 | 98 | 56.6277 | 68.6127 |
| 1 | N3 | hard | 224 | mean_slice | 30 | 197 | 56.2498 | 67.8510 |
| 1 | N3 | hard | 512 | pooled | 5 | 95 | 73.2379 | 84.5518 |
| 1 | N3 | hard | 512 | pooled | 6 | 82 | 70.3724 | 82.6101 |
| 1 | N3 | hard | 512 | pooled | 8 | 98 | 68.9238 | 81.6034 |
| 1 | N3 | hard | 512 | pooled | 30 | 197 | 70.1200 | 82.4359 |
| 1 | N3 | hard | 512 | mean_slice | 5 | 95 | 68.2529 | 80.2071 |
| 1 | N3 | hard | 512 | mean_slice | 6 | 82 | 65.7761 | 77.9128 |
| 1 | N3 | hard | 512 | mean_slice | 8 | 98 | 56.7376 | 68.6917 |
| 1 | N3 | hard | 512 | mean_slice | 30 | 197 | 56.4317 | 68.0337 |
| 1 | N3 | F1 | 224 | pooled | 5 | 95 | 66.4399 | 79.8365 |
| 1 | N3 | F1 | 224 | pooled | 6 | 82 | 69.1123 | 81.7354 |
| 1 | N3 | F1 | 224 | pooled | 8 | 98 | 63.2792 | 77.5104 |
| 1 | N3 | F1 | 224 | pooled | 30 | 197 | 65.6143 | 79.2375 |
| 1 | N3 | F1 | 224 | mean_slice | 5 | 95 | 58.2852 | 71.5791 |
| 1 | N3 | F1 | 224 | mean_slice | 6 | 82 | 62.9044 | 74.9102 |
| 1 | N3 | F1 | 224 | mean_slice | 8 | 98 | 48.0240 | 59.2684 |
| 1 | N3 | F1 | 224 | mean_slice | 30 | 197 | 47.1204 | 58.2078 |
| 1 | N3 | F1 | 512 | pooled | 5 | 95 | 66.3855 | 79.7972 |
| 1 | N3 | F1 | 512 | pooled | 6 | 82 | 69.0841 | 81.7156 |
| 1 | N3 | F1 | 512 | pooled | 8 | 98 | 63.2989 | 77.5252 |
| 1 | N3 | F1 | 512 | pooled | 30 | 197 | 65.5716 | 79.2063 |
| 1 | N3 | F1 | 512 | mean_slice | 5 | 95 | 58.2290 | 71.5415 |
| 1 | N3 | F1 | 512 | mean_slice | 6 | 82 | 62.8394 | 74.8554 |
| 1 | N3 | F1 | 512 | mean_slice | 8 | 98 | 48.0544 | 59.2830 |
| 1 | N3 | F1 | 512 | mean_slice | 30 | 197 | 47.1192 | 58.2295 |
| 2 | N1 | hard | 224 | pooled | 5 | 95 | 73.4521 | 84.6944 |
| 2 | N1 | hard | 224 | pooled | 6 | 82 | 70.3834 | 82.6177 |
| 2 | N1 | hard | 224 | pooled | 8 | 98 | 68.6777 | 81.4307 |
| 2 | N1 | hard | 224 | pooled | 30 | 197 | 69.9491 | 82.3177 |
| 2 | N1 | hard | 224 | mean_slice | 5 | 95 | 68.4374 | 80.3243 |
| 2 | N1 | hard | 224 | mean_slice | 6 | 82 | 65.8077 | 77.9483 |
| 2 | N1 | hard | 224 | mean_slice | 8 | 98 | 56.5079 | 68.5102 |
| 2 | N1 | hard | 224 | mean_slice | 30 | 197 | 56.4409 | 68.1316 |
| 2 | N1 | hard | 512 | pooled | 5 | 95 | 73.4410 | 84.6870 |
| 2 | N1 | hard | 512 | pooled | 6 | 82 | 70.3950 | 82.6257 |
| 2 | N1 | hard | 512 | pooled | 8 | 98 | 68.8819 | 81.5740 |
| 2 | N1 | hard | 512 | pooled | 30 | 197 | 70.0801 | 82.4084 |
| 2 | N1 | hard | 512 | mean_slice | 5 | 95 | 68.5462 | 80.4184 |
| 2 | N1 | hard | 512 | mean_slice | 6 | 82 | 65.8243 | 77.9547 |
| 2 | N1 | hard | 512 | mean_slice | 8 | 98 | 56.6195 | 68.5895 |
| 2 | N1 | hard | 512 | mean_slice | 30 | 197 | 56.6267 | 68.3216 |
| 2 | N1 | F1 | 224 | pooled | 5 | 95 | 66.1308 | 79.6129 |
| 2 | N1 | F1 | 224 | pooled | 6 | 82 | 68.5390 | 81.3331 |
| 2 | N1 | F1 | 224 | pooled | 8 | 98 | 62.7736 | 77.1299 |
| 2 | N1 | F1 | 224 | pooled | 30 | 197 | 65.1232 | 78.8783 |
| 2 | N1 | F1 | 224 | mean_slice | 5 | 95 | 58.0374 | 71.4163 |
| 2 | N1 | F1 | 224 | mean_slice | 6 | 82 | 62.3316 | 74.4478 |
| 2 | N1 | F1 | 224 | mean_slice | 8 | 98 | 47.6170 | 58.9208 |
| 2 | N1 | F1 | 224 | mean_slice | 30 | 197 | 46.7751 | 57.9370 |
| 2 | N1 | F1 | 512 | pooled | 5 | 95 | 66.0669 | 79.5666 |
| 2 | N1 | F1 | 512 | pooled | 6 | 82 | 68.5281 | 81.3254 |
| 2 | N1 | F1 | 512 | pooled | 8 | 98 | 62.8134 | 77.1600 |
| 2 | N1 | F1 | 512 | pooled | 30 | 197 | 65.0775 | 78.8448 |
| 2 | N1 | F1 | 512 | mean_slice | 5 | 95 | 57.9651 | 71.3591 |
| 2 | N1 | F1 | 512 | mean_slice | 6 | 82 | 62.3016 | 74.4220 |
| 2 | N1 | F1 | 512 | mean_slice | 8 | 98 | 47.6835 | 58.9706 |
| 2 | N1 | F1 | 512 | mean_slice | 30 | 197 | 46.7681 | 57.9510 |
| 2 | N2 | hard | 224 | pooled | 5 | 95 | 73.4521 | 84.6944 |
| 2 | N2 | hard | 224 | pooled | 6 | 82 | 70.3834 | 82.6177 |
| 2 | N2 | hard | 224 | pooled | 8 | 98 | 68.6777 | 81.4307 |
| 2 | N2 | hard | 224 | pooled | 30 | 197 | 69.9491 | 82.3177 |
| 2 | N2 | hard | 224 | mean_slice | 5 | 95 | 68.4374 | 80.3243 |
| 2 | N2 | hard | 224 | mean_slice | 6 | 82 | 65.8077 | 77.9483 |
| 2 | N2 | hard | 224 | mean_slice | 8 | 98 | 56.5079 | 68.5102 |
| 2 | N2 | hard | 224 | mean_slice | 30 | 197 | 56.4409 | 68.1316 |
| 2 | N2 | hard | 512 | pooled | 5 | 95 | 73.4410 | 84.6870 |
| 2 | N2 | hard | 512 | pooled | 6 | 82 | 70.3950 | 82.6257 |
| 2 | N2 | hard | 512 | pooled | 8 | 98 | 68.8819 | 81.5740 |
| 2 | N2 | hard | 512 | pooled | 30 | 197 | 70.0801 | 82.4084 |
| 2 | N2 | hard | 512 | mean_slice | 5 | 95 | 68.5462 | 80.4184 |
| 2 | N2 | hard | 512 | mean_slice | 6 | 82 | 65.8243 | 77.9547 |
| 2 | N2 | hard | 512 | mean_slice | 8 | 98 | 56.6195 | 68.5895 |
| 2 | N2 | hard | 512 | mean_slice | 30 | 197 | 56.6267 | 68.3216 |
| 2 | N2 | F1 | 224 | pooled | 5 | 95 | 66.1308 | 79.6129 |
| 2 | N2 | F1 | 224 | pooled | 6 | 82 | 68.5390 | 81.3331 |
| 2 | N2 | F1 | 224 | pooled | 8 | 98 | 62.7736 | 77.1299 |
| 2 | N2 | F1 | 224 | pooled | 30 | 197 | 65.1232 | 78.8783 |
| 2 | N2 | F1 | 224 | mean_slice | 5 | 95 | 58.0374 | 71.4163 |
| 2 | N2 | F1 | 224 | mean_slice | 6 | 82 | 62.3316 | 74.4478 |
| 2 | N2 | F1 | 224 | mean_slice | 8 | 98 | 47.6170 | 58.9208 |
| 2 | N2 | F1 | 224 | mean_slice | 30 | 197 | 46.7751 | 57.9370 |
| 2 | N2 | F1 | 512 | pooled | 5 | 95 | 66.0669 | 79.5666 |
| 2 | N2 | F1 | 512 | pooled | 6 | 82 | 68.5281 | 81.3254 |
| 2 | N2 | F1 | 512 | pooled | 8 | 98 | 62.8134 | 77.1600 |
| 2 | N2 | F1 | 512 | pooled | 30 | 197 | 65.0775 | 78.8448 |
| 2 | N2 | F1 | 512 | mean_slice | 5 | 95 | 57.9651 | 71.3591 |
| 2 | N2 | F1 | 512 | mean_slice | 6 | 82 | 62.3016 | 74.4220 |
| 2 | N2 | F1 | 512 | mean_slice | 8 | 98 | 47.6835 | 58.9706 |
| 2 | N2 | F1 | 512 | mean_slice | 30 | 197 | 46.7681 | 57.9510 |
| 2 | N3 | hard | 224 | pooled | 5 | 95 | 73.4521 | 84.6944 |
| 2 | N3 | hard | 224 | pooled | 6 | 82 | 70.3834 | 82.6177 |
| 2 | N3 | hard | 224 | pooled | 8 | 98 | 68.6777 | 81.4307 |
| 2 | N3 | hard | 224 | pooled | 30 | 197 | 69.9491 | 82.3177 |
| 2 | N3 | hard | 224 | mean_slice | 5 | 95 | 68.4374 | 80.3243 |
| 2 | N3 | hard | 224 | mean_slice | 6 | 82 | 65.8077 | 77.9483 |
| 2 | N3 | hard | 224 | mean_slice | 8 | 98 | 56.5079 | 68.5102 |
| 2 | N3 | hard | 224 | mean_slice | 30 | 197 | 56.4409 | 68.1316 |
| 2 | N3 | hard | 512 | pooled | 5 | 95 | 73.4410 | 84.6870 |
| 2 | N3 | hard | 512 | pooled | 6 | 82 | 70.3950 | 82.6257 |
| 2 | N3 | hard | 512 | pooled | 8 | 98 | 68.8819 | 81.5740 |
| 2 | N3 | hard | 512 | pooled | 30 | 197 | 70.0801 | 82.4084 |
| 2 | N3 | hard | 512 | mean_slice | 5 | 95 | 68.5462 | 80.4184 |
| 2 | N3 | hard | 512 | mean_slice | 6 | 82 | 65.8243 | 77.9547 |
| 2 | N3 | hard | 512 | mean_slice | 8 | 98 | 56.6195 | 68.5895 |
| 2 | N3 | hard | 512 | mean_slice | 30 | 197 | 56.6267 | 68.3216 |
| 2 | N3 | F1 | 224 | pooled | 5 | 95 | 66.1308 | 79.6129 |
| 2 | N3 | F1 | 224 | pooled | 6 | 82 | 68.5390 | 81.3331 |
| 2 | N3 | F1 | 224 | pooled | 8 | 98 | 62.7736 | 77.1299 |
| 2 | N3 | F1 | 224 | pooled | 30 | 197 | 65.1232 | 78.8783 |
| 2 | N3 | F1 | 224 | mean_slice | 5 | 95 | 58.0374 | 71.4163 |
| 2 | N3 | F1 | 224 | mean_slice | 6 | 82 | 62.3316 | 74.4478 |
| 2 | N3 | F1 | 224 | mean_slice | 8 | 98 | 47.6170 | 58.9208 |
| 2 | N3 | F1 | 224 | mean_slice | 30 | 197 | 46.7751 | 57.9370 |
| 2 | N3 | F1 | 512 | pooled | 5 | 95 | 66.0669 | 79.5666 |
| 2 | N3 | F1 | 512 | pooled | 6 | 82 | 68.5281 | 81.3254 |
| 2 | N3 | F1 | 512 | pooled | 8 | 98 | 62.8134 | 77.1600 |
| 2 | N3 | F1 | 512 | pooled | 30 | 197 | 65.0775 | 78.8448 |
| 2 | N3 | F1 | 512 | mean_slice | 5 | 95 | 57.9651 | 71.3591 |
| 2 | N3 | F1 | 512 | mean_slice | 6 | 82 | 62.3016 | 74.4220 |
| 2 | N3 | F1 | 512 | mean_slice | 8 | 98 | 47.6835 | 58.9706 |
| 2 | N3 | F1 | 512 | mean_slice | 30 | 197 | 46.7681 | 57.9510 |

### Criterion outcome

**FAILURE** — N3/F1/224 seed IoUs: [66.27859037561032, 66.11141056225584, 65.6416188812218]; mean 66.0105%, worst seed 65.6416%.

Success: the mean over the three seeds of the F1 pooled-pixel IoU is at least 80.03 AND no seed is below 77.96. Partial: mean above 77.96 but not success. Anything else: failure.

Global prediction seal SHA256: `74e185d3fb27533d2e043a46744c4fb3e6cf47ddaa6521716df2651a93375e5e`. All N1/N2 secondary rows are reported; no candidate, seed, naming rule or checkpoint was selected using scores.

### Limitations and UNVERIFIED

The four validation patients informed earlier choices of K, resolution, window and refinement; this is not an untouched test. Historical seed-1 process/checkpoint lineage remains UNVERIFIED; current serialization equality is verified only on the same CPU/runtime, not across platforms. Label-free cluster usage is a newly fitted fixed-monitor K=8 diagnostic, so IDs do not track semantic identities across checks. No full-length run, external cohort, or improved anatomical specificity is established by passing the stability gate. No installs or downloads; sealed experiment, old caches/checkpoints and datasets were not modified.

## Commit record

Task 1 calibration: `599789b` (`fix(task1): calibrate two label-free 448 shift candidates`). Task 2 safety/state serialization: `fdae63b` (`fix(task2): isolate bounded runs and verify complete state serialization`). Task 3 Stage 1: `2734c4e` (`fix(task3): complete all-seed D448 stability screen`). Task 4 Stage 2 and Stage B: final commit recorded after this report update.

Producing scripts: `audit/calibrated/calibrate.py`, `experiments/stego_calibrated/check_state.py`, `train.py`, `stages.py`, `seal_all.py`, `evaluate.py`, and `audit/calibrated/summarize.py`. The new stage source preserves inherited naming/F1 and changes only run-root plumbing, final-checkpoint loading and gating.

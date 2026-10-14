# Label-free preflight: sealed 448 / 28×28 STEGO head

All numerical diagnostics are produced by `audit/preflight/diagnose.py` (`core` and `extract`); this report and the derived tables/figures are produced by `audit/preflight/report.py`. No gradients, optimizer steps, label/mask files, or validation images were used. CPU only, `torch.set_num_threads(2)`. Existing experiment sources, caches and checkpoints were read-only. New intermediate arrays and reports are confined to `audit/preflight/`.


## 1. Pinned configuration and implementation

| Parameter | Exact value | Evidence (repository-relative file:line) |
| --- | --- | --- |
| Shifts: intra / kNN / random cross-image | 0.18 / 0.12 / 0.46 | experiments/stego_biomedclip/configs/locked.json:24 |
| Loss weights: intra / kNN / random | 0.67 / 0.25 / 0.63; overall 1.0 | experiments/stego_biomedclip/configs/locked.json:27 |
| LR / optimizer / schedule | 0.0005 / Adam / none | experiments/stego_biomedclip/configs/locked.json:12; experiments/stego_biomedclip/train.py:33 |
| Batch / budget | 16 / 5000 batches | experiments/stego_biomedclip/configs/locked.json:10 |
| Dropout | Dropout2d(p=0.1), independent calls for both branches and feature signal | experiments/stego_biomedclip/model.py:11; :18; :21 |
| Head | Conv1×1(768→70) + [Conv1×1(768→768), ReLU, Conv1×1(768→70)]; biases enabled | experiments/stego_biomedclip/model.py:12 |
| Initialization | PyTorch Conv2d defaults: kaiming_uniform_(a=sqrt(5)); bias uniform ±1/sqrt(fan_in). All three convolutions have fan_in=768. No manual head initialization. | experiments/stego_biomedclip/model.py:9; /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch/nn/modules/conv.py:182 |
| Centering | Pointwise target mean removed, global mean restored; normalized feature correlation | experiments/stego_biomedclip/reference_loss.py:54 |
| Clamping | Code cosine clamped below at 0; stabalize=False (no 0.8 cap) | experiments/stego_biomedclip/reference_loss.py:61; configs/locked.json:20 |
| Spatial sampling / negatives | 11×11 uniform grid-sampled coordinates; 5 non-self batch permutations | experiments/stego_biomedclip/configs/locked.json:17; reference_loss.py:91; :108 |
| kNN selection | Uniform rank among first 7 of stored 30, train-only | experiments/stego_biomedclip/train.py:26; :57 |
| Other losses | Geometric augmentation alignment 0; actual objective sums only the three correspondence terms | experiments/stego_biomedclip/configs/locked.json:31; model.py:28 |
| Monitor rule | Every 50 batches; cosine >0.99 OR std <0.01 for 3 consecutive checks. Reports collapse; does not early-stop. | experiments/stego_biomedclip/train.py:66; :77; configs/locked.json:44 |

Pinned configuration SHA256: `defd959e1e09586cbc4eb7163b21347643d71153eb9286baea577915cb1968d2`. Imported head/loss/training source hashes match the sealed preflight hashes (`verification.json`); backbone weight SHA256 is `52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be`. The sealed training used 4 CPU threads; these diagnostics use the requested 2. No configuration was changed.


## 2. Sampling and loss comparison

Fixed RNG seed 20261003; 200 uniformly sampled slices without replacement from 1869 train slices, covering all 16 train patients. Sample IDs and patient counts: `sample.json`; neighbor IDs: `pair_plan.json`. Frozen final-block 448 tokens are read only at training indices from `audit/improve/B2_448_features.npy`. A fresh replay of all 200 training images reproduces the cached final tokens with maximum absolute error 0.0; body crop boxes agree: True.

The exact imported `ContrastiveCorrelationLoss.forward/helper` samples and centers the affinities. For constant code and shift zero, its negative loss recovers the affinity without duplicating the centering formula. Same-image includes the helper’s diagonal/self pairs. kNN neighbors come from the full train-only cache, selected uniformly among its first 7 ranks; random negatives use the actual 5 non-self batch permutations. Batch size is 16, with one final batch of 8; centering is recomputed in each helper call exactly as in the loss. Counts are pair draws, not independent patients: [2928200, 2928200, 14641000] for intra/kNN/random, respectively.

Feature dropout is disabled for these normalized-token comparisons, matching the fixed train monitor’s evaluation mode. Constant codes are the same nonzero 70-D vector everywhere; random codes are independent normalized Gaussian 70-D fields; cluster codes are one-hot K=8 assignments padded to 70-D from the saved train-only seed-0 448 fit (`audit/final_pipeline/inductive_seed0_centers.npy`, digest in `pair_plan.json`). No fit or label mapping is performed. The loss bilinearly samples all codes, including the one-hot grids, before its code normalization. Identical sampling/permutations are replayed for every code type and shift group; the loss weights remain pinned for all comparisons.


| Shift group | Intra / kNN / random shifts | Attract intra % | Attract kNN % | Attract random % | Constant total | Random total | K8-code total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pinned | 0.18000000 / 0.12000000 / 0.46000000 | 96.613 | 98.175 | 64.390 | -0.434704 | -0.026546 | -0.178473 |
| C | 0.45844662 / 0.45247218 / 0.74595857 | 69.460 | 70.571 | 17.020 | 0.015127 | -0.003050 | -0.079080 |
| D | 0.79650706 / 0.78205675 / 0.77916342 | 19.699 | 20.578 | 12.422 | 0.344943 | 0.015121 | -0.001826 |
| E | 0.86347353 / 0.84708124 / 0.82346648 | 10.229 | 11.075 | 6.927 | 0.433978 | 0.019855 | 0.018163 |

| Shift group | Code | Intra loss (unweighted) | kNN loss (unweighted) | Random loss (unweighted) | Weighted total |
| --- | --- | --- | --- | --- | --- |
| Pinned | constant | -0.40373080 | -0.45956402 | -0.07827558 | -0.43470426 |
| Pinned | random | -0.02790068 | -0.02198452 | -0.00374072 | -0.02654625 |
| Pinned | cluster | -0.15176692 | -0.15873368 | -0.05889741 | -0.17847263 |
| C | constant | -0.12528393 | -0.12709153 | 0.20768325 | 0.01512733 |
| C | random | -0.01164117 | -0.00607614 | 0.00995064 | -0.00304972 |
| C | cluster | -0.08533676 | -0.08175264 | -0.00232727 | -0.07907997 |
| D | constant | 0.21277683 | 0.20249335 | 0.24088814 | 0.34494334 |
| D | random | 0.00809941 | 0.00969407 | 0.01154045 | 0.01512061 |
| D | cluster | -0.00468428 | -0.00544019 | 0.00424152 | -0.00182636 |
| E | constant | 0.27974338 | 0.26751791 | 0.28519123 | 0.43397801 |
| E | random | 0.01200983 | 0.01280541 | 0.01366163 | 0.01985476 |
| E | cluster | 0.01129219 | 0.00961567 | 0.01300583 | 0.01816336 |

Lower is better. At the pinned shifts, constant codes beat K8 codes by 0.25623163 loss units. This is a direct comparison of candidate representations, not a proof that constant codes are the unique/global optimum. C, D and E remove this constant-over-K8 advantage on the same sample; E nevertheless gives K8 codes a positive objective. These measurements cannot establish semantic quality or guarantee stable optimization.

Historical shift sources: `audit/CROSSPATIENT.md:73` (C), `audit/FIXES4.md:20` (D/E). Their earlier attraction targets came from another resolution/sample; they are not assumed to transfer. Here the random-attraction rates for D/E are about 12.4%/6.9%, not the historical 10%/5%.


Label-free quantiles on this exact sample (diagnostic thresholds only; not applied):

| Desired attraction | Intra shift | kNN shift | Random shift |
| --- | --- | --- | --- |
| 25% | 0.75934887 | 0.75145727 | 0.69136167 |
| 20% | 0.79441762 | 0.78598076 | 0.72524232 |
| 15% | 0.82899964 | 0.81972188 | 0.76033819 |
| 10% | 0.86519808 | 0.85495961 | 0.79765058 |
| 5% | 0.91204143 | 0.89958489 | 0.84248185 |

These quantiles are heuristics, not paper-prescribed settings. The measured C/D/E comparisons support reduced random attraction and selective positives as candidates for removing constant-code preference; no universal attraction fraction prevents collapse.


## 3. Feature diversity

All sources use the same 200 train slices and all 784 patch vectors per image. Blocks are 1-indexed (9, 11, final=12), each passed through the model’s shared final LayerNorm, with CLS excluded, then patch L2. The project extractor is imported; its existing `get_intermediate_layers` call is intercepted only to request indices [8,10,11] and returns the final tensor back to the unchanged extractor (`src/modules.py:210`). Preprocessing functions are AST-imported unchanged from `audit/final_pipeline/segment_ct.py:23`, :33 and :41 to avoid that file’s global side effects. No mask is used for the body crop.

“Patch-pair cosine” is the exact mean across every pair of patches in different images: mean dot product of unnormalized image means of unit patch vectors. “Normalized-GAP cosine” L2-normalizes those image means first and measures image-level similarity; these are different quantities. Channel std uses all 156,800 tokens with sample variance. PCA uses the full centered 768×768 covariance (for statistical analysis only); effective rank = exp(entropy of all normalized nonnegative eigenvalues).

| Source | Patch-pair cosine | Normalized-GAP cosine | Channel std mean [min,max] | PC1 % | PC1–10 % | PC1–50 % | Effective rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Final (12) | 0.500660 | 0.929153 | 0.022490 [0.012149, 0.178834] | 29.77 | 74.15 | 89.74 | 25.460 |
| Block 9 | 0.292994 | 0.914351 | 0.028084 [0.017490, 0.148311] | 21.46 | 59.11 | 79.84 | 57.499 |
| Block 11 | 0.428253 | 0.950987 | 0.024458 [0.013276, 0.179594] | 29.13 | 73.20 | 88.92 | 27.185 |

Every per-channel std and all 768 explained-variance ratios are in `diversity_final.json`, `diversity_block9.json`, and `diversity_block11.json`; spectrum figure: `pca_spectrum.png`. Full extraction checks: `extract_checks.json`; identical-call max error 0.0, trainable backbone parameters 0, extraction seconds 23.882. The final-token diversity is nonzero; near-unit head cosine is not already present in the patch-pair backbone representation.


## 4. Initialization and observed trajectories

Initialization uses the sealed monitor’s exact 16 fixed train images (one per patient), head construction after torch.manual_seed(seed), eval mode, no gradients. Code std and cross-image cosine match `experiments/stego_biomedclip/train.py:72–76`: L2-normalize 70-D codes; average per-image/per-channel spatial std; average dot products of image-mean unit patch vectors across different images. No learned state is loaded. Monitor IDs are in `initialization.json`.

| Fresh head seed | Code std | Cross-image patch-pair cosine |
| --- | --- | --- |
| 0 | 0.04131734 | 0.86023968 |
| 1 | 0.04575163 | 0.82948053 |
| 2 | 0.05203552 | 0.78031081 |

Existing logs only, with every monitor record preserved below. `trajectories.json` records the source SHA256, byte offsets and record ordinals; no final checkpoint was re-run. Seed 1 contains 7343 JSON records and 146 monitor records, including duplicate batches and interleaved resumed streams. The resume record explicitly says optimizer and RNG state were not restored (`experiments/stego_biomedclip/logs/seed1_resume.json:5`). Therefore a unique uninterrupted seed-1 trajectory after the first resume is UNVERIFIED; the source log alone cannot attribute duplicate rows to a unique process/checkpoint. Both final batch-5000 observations are retained. This does not affect the seed-1 collapse already observed before the first resume.

### Seed 0 — every recorded monitor

| Log record | Batch | Code std | Cross-image cosine | Bad checks | Collapse flag |
| --- | --- | --- | --- | --- | --- |
| 50 | 50 | 0.001912285 | 0.999661148 | 1 | False |
| 100 | 100 | 0.001257731 | 0.999854505 | 2 | False |
| 150 | 150 | 0.000920761 | 0.999922991 | 3 | True |
| 200 | 200 | 0.000726985 | 0.999952197 | 4 | True |
| 250 | 250 | 0.000603054 | 0.999967217 | 5 | True |
| 300 | 300 | 0.000518740 | 0.999975741 | 6 | True |
| 350 | 350 | 0.000456642 | 0.999981165 | 7 | True |
| 400 | 400 | 0.000410414 | 0.999984860 | 8 | True |
| 450 | 450 | 0.000373881 | 0.999987543 | 9 | True |
| 500 | 500 | 0.000344340 | 0.999989450 | 10 | True |
| 550 | 550 | 0.000319502 | 0.999990821 | 11 | True |
| 600 | 600 | 0.000298838 | 0.999992013 | 12 | True |
| 650 | 650 | 0.000281041 | 0.999992967 | 13 | True |
| 700 | 700 | 0.000265732 | 0.999993682 | 14 | True |
| 750 | 750 | 0.000252425 | 0.999994278 | 15 | True |
| 800 | 800 | 0.000240225 | 0.999994874 | 16 | True |
| 850 | 850 | 0.000229657 | 0.999995291 | 17 | True |
| 900 | 900 | 0.000220101 | 0.999995649 | 18 | True |
| 950 | 950 | 0.000211491 | 0.999996066 | 19 | True |
| 1000 | 1000 | 0.000203613 | 0.999996305 | 20 | True |
| 1050 | 1050 | 0.000196219 | 0.999996543 | 21 | True |
| 1100 | 1100 | 0.000189518 | 0.999996781 | 22 | True |
| 1150 | 1150 | 0.000183478 | 0.999996960 | 23 | True |
| 1200 | 1200 | 0.000177726 | 0.999997199 | 24 | True |
| 1250 | 1250 | 0.000172406 | 0.999997318 | 25 | True |
| 1300 | 1300 | 0.000167554 | 0.999997437 | 26 | True |
| 1350 | 1350 | 0.000162838 | 0.999997735 | 27 | True |
| 1400 | 1400 | 0.000158381 | 0.999997735 | 28 | True |
| 1450 | 1450 | 0.000154394 | 0.999997973 | 29 | True |
| 1500 | 1500 | 0.000150502 | 0.999997973 | 30 | True |
| 1550 | 1550 | 0.000146791 | 0.999998093 | 31 | True |
| 1600 | 1600 | 0.000143295 | 0.999998212 | 32 | True |
| 1650 | 1650 | 0.000140042 | 0.999998212 | 33 | True |
| 1700 | 1700 | 0.000136882 | 0.999998331 | 34 | True |
| 1750 | 1750 | 0.000133924 | 0.999998450 | 35 | True |
| 1800 | 1800 | 0.000131131 | 0.999998450 | 36 | True |
| 1850 | 1850 | 0.000128474 | 0.999998450 | 37 | True |
| 1900 | 1900 | 0.000125859 | 0.999998629 | 38 | True |
| 1950 | 1950 | 0.000123396 | 0.999998689 | 39 | True |
| 2000 | 2000 | 0.000121017 | 0.999998748 | 40 | True |
| 2050 | 2050 | 0.000118714 | 0.999998748 | 41 | True |
| 2100 | 2100 | 0.000116538 | 0.999998868 | 42 | True |
| 2150 | 2150 | 0.000114520 | 0.999998868 | 43 | True |
| 2200 | 2200 | 0.000112458 | 0.999998987 | 44 | True |
| 2250 | 2250 | 0.000110486 | 0.999998987 | 45 | True |
| 2300 | 2300 | 0.000108596 | 0.999998987 | 46 | True |
| 2350 | 2350 | 0.000106795 | 0.999998987 | 47 | True |
| 2400 | 2400 | 0.000105028 | 0.999998987 | 48 | True |
| 2450 | 2450 | 0.000103289 | 0.999999106 | 49 | True |
| 2500 | 2500 | 0.000101629 | 0.999999106 | 50 | True |
| 2550 | 2550 | 0.000100032 | 0.999999106 | 51 | True |
| 2600 | 2600 | 0.000098496 | 0.999999106 | 52 | True |
| 2650 | 2650 | 0.000097001 | 0.999999106 | 53 | True |
| 2700 | 2700 | 0.000095566 | 0.999999106 | 54 | True |
| 2750 | 2750 | 0.000094107 | 0.999999106 | 55 | True |
| 2800 | 2800 | 0.000092723 | 0.999999225 | 56 | True |
| 2850 | 2850 | 0.000091434 | 0.999999225 | 57 | True |
| 2900 | 2900 | 0.000090174 | 0.999999225 | 58 | True |
| 2950 | 2950 | 0.000088843 | 0.999999225 | 59 | True |
| 3000 | 3000 | 0.000087604 | 0.999999285 | 60 | True |
| 3050 | 3050 | 0.000086379 | 0.999999285 | 61 | True |
| 3100 | 3100 | 0.000085224 | 0.999999225 | 62 | True |
| 3150 | 3150 | 0.000084079 | 0.999999285 | 63 | True |
| 3200 | 3200 | 0.000082992 | 0.999999225 | 64 | True |
| 3250 | 3250 | 0.000081915 | 0.999999344 | 65 | True |
| 3300 | 3300 | 0.000080875 | 0.999999344 | 66 | True |
| 3350 | 3350 | 0.000079846 | 0.999999344 | 67 | True |
| 3400 | 3400 | 0.000078814 | 0.999999464 | 68 | True |
| 3450 | 3450 | 0.000077830 | 0.999999404 | 69 | True |
| 3500 | 3500 | 0.000076904 | 0.999999464 | 70 | True |
| 3550 | 3550 | 0.000075932 | 0.999999464 | 71 | True |
| 3600 | 3600 | 0.000074998 | 0.999999464 | 72 | True |
| 3650 | 3650 | 0.000074145 | 0.999999464 | 73 | True |
| 3700 | 3700 | 0.000073299 | 0.999999464 | 74 | True |
| 3750 | 3750 | 0.000072433 | 0.999999464 | 75 | True |
| 3800 | 3800 | 0.000071562 | 0.999999464 | 76 | True |
| 3850 | 3850 | 0.000070792 | 0.999999642 | 77 | True |
| 3900 | 3900 | 0.000069982 | 0.999999642 | 78 | True |
| 3950 | 3950 | 0.000069149 | 0.999999642 | 79 | True |
| 4000 | 4000 | 0.000068415 | 0.999999702 | 80 | True |
| 4050 | 4050 | 0.000067611 | 0.999999642 | 81 | True |
| 4100 | 4100 | 0.000066926 | 0.999999702 | 82 | True |
| 4150 | 4150 | 0.000066178 | 0.999999702 | 83 | True |
| 4200 | 4200 | 0.000065432 | 0.999999702 | 84 | True |
| 4250 | 4250 | 0.000064761 | 0.999999702 | 85 | True |
| 4300 | 4300 | 0.000064066 | 0.999999702 | 86 | True |
| 4350 | 4350 | 0.000063392 | 0.999999762 | 87 | True |
| 4400 | 4400 | 0.000062716 | 0.999999762 | 88 | True |
| 4450 | 4450 | 0.000062072 | 0.999999762 | 89 | True |
| 4500 | 4500 | 0.000061419 | 0.999999762 | 90 | True |
| 4550 | 4550 | 0.000060833 | 0.999999762 | 91 | True |
| 4600 | 4600 | 0.000060225 | 0.999999762 | 92 | True |
| 4650 | 4650 | 0.000059631 | 0.999999762 | 93 | True |
| 4700 | 4700 | 0.000059021 | 0.999999762 | 94 | True |
| 4750 | 4750 | 0.000058507 | 0.999999762 | 95 | True |
| 4800 | 4800 | 0.000057870 | 0.999999762 | 96 | True |
| 4850 | 4850 | 0.000057288 | 0.999999821 | 97 | True |
| 4900 | 4900 | 0.000056703 | 0.999999881 | 98 | True |
| 4950 | 4950 | 0.000056174 | 0.999999762 | 99 | True |
| 5000 | 5000 | 0.000055623 | 0.999999881 | 100 | True |

### Seed 1 — every recorded monitor

| Log record | Batch | Code std | Cross-image cosine | Bad checks | Collapse flag |
| --- | --- | --- | --- | --- | --- |
| 50 | 50 | 0.001971224 | 0.999636471 | 1 | False |
| 100 | 100 | 0.001303139 | 0.999839246 | 2 | False |
| 150 | 150 | 0.000937950 | 0.999917746 | 3 | True |
| 200 | 200 | 0.000725370 | 0.999951065 | 4 | True |
| 250 | 250 | 0.000601896 | 0.999966443 | 5 | True |
| 300 | 300 | 0.000519530 | 0.999975085 | 6 | True |
| 350 | 350 | 0.000461658 | 0.999980390 | 7 | True |
| 400 | 400 | 0.000417631 | 0.999984026 | 8 | True |
| 450 | 450 | 0.000382194 | 0.999986589 | 9 | True |
| 500 | 500 | 0.000353865 | 0.999988556 | 10 | True |
| 550 | 550 | 0.000329078 | 0.999990106 | 11 | True |
| 600 | 600 | 0.000307488 | 0.999991357 | 12 | True |
| 650 | 650 | 0.000288525 | 0.999992371 | 13 | True |
| 700 | 700 | 0.000270758 | 0.999993324 | 14 | True |
| 750 | 750 | 0.000254073 | 0.999994159 | 15 | True |
| 800 | 800 | 0.000239039 | 0.999994814 | 16 | True |
| 850 | 850 | 0.000225102 | 0.999995410 | 17 | True |
| 900 | 900 | 0.000212510 | 0.999995947 | 18 | True |
| 950 | 950 | 0.000201375 | 0.999996305 | 19 | True |
| 1000 | 1000 | 0.000191344 | 0.999996722 | 20 | True |
| 1050 | 1050 | 0.000182422 | 0.999996960 | 21 | True |
| 1100 | 1100 | 0.000174446 | 0.999997258 | 22 | True |
| 1150 | 1150 | 0.000167047 | 0.999997437 | 23 | True |
| 1200 | 1200 | 0.000160372 | 0.999997735 | 24 | True |
| 1250 | 1250 | 0.000154329 | 0.999997854 | 25 | True |
| 1300 | 1300 | 0.000148564 | 0.999997973 | 26 | True |
| 1350 | 1350 | 0.000143534 | 0.999998152 | 27 | True |
| 1400 | 1400 | 0.000138928 | 0.999998212 | 28 | True |
| 1450 | 1450 | 0.000134605 | 0.999998331 | 29 | True |
| 1500 | 1500 | 0.000130671 | 0.999998450 | 30 | True |
| 1550 | 1550 | 0.000126918 | 0.999998510 | 31 | True |
| 1600 | 1600 | 0.000123479 | 0.999998689 | 32 | True |
| 1650 | 1650 | 0.000120163 | 0.999998748 | 33 | True |
| 1700 | 1700 | 0.000117213 | 0.999998748 | 34 | True |
| 1750 | 1750 | 0.000114277 | 0.999998868 | 35 | True |
| 1800 | 1800 | 0.000111632 | 0.999998987 | 36 | True |
| 1850 | 1850 | 0.000109147 | 0.999998987 | 37 | True |
| 1900 | 1900 | 0.000106758 | 0.999998987 | 38 | True |
| 1950 | 1950 | 0.000104460 | 0.999998987 | 39 | True |
| 2000 | 2000 | 0.000102361 | 0.999998987 | 40 | True |
| 2050 | 2050 | 0.000100278 | 0.999999106 | 41 | True |
| 2100 | 2100 | 0.000098351 | 0.999999166 | 42 | True |
| 2150 | 2150 | 0.000096417 | 0.999999106 | 43 | True |
| 2200 | 2200 | 0.000094657 | 0.999999106 | 44 | True |
| 2250 | 2250 | 0.000092993 | 0.999999166 | 45 | True |
| 2300 | 2300 | 0.000091358 | 0.999999225 | 46 | True |
| 2350 | 2350 | 0.000089781 | 0.999999225 | 47 | True |
| 2400 | 2400 | 0.000088291 | 0.999999225 | 48 | True |
| 2450 | 2450 | 0.000086842 | 0.999999225 | 49 | True |
| 2500 | 2500 | 0.000085393 | 0.999999225 | 50 | True |
| 2550 | 2550 | 0.000084070 | 0.999999225 | 51 | True |
| 2600 | 2600 | 0.000082755 | 0.999999344 | 52 | True |
| 2650 | 2650 | 0.000081554 | 0.999999285 | 53 | True |
| 2743 | 2700 | 0.000071033 | 0.999999642 | 1 | False |
| 2807 | 2750 | 0.000062725 | 0.999999762 | 2 | False |
| 2879 | 2750 | 0.000062748 | 0.999999762 | 1 | False |
| 2907 | 2800 | 0.000057008 | 0.999999821 | 3 | True |
| 2978 | 2800 | 0.000056820 | 0.999999881 | 2 | False |
| 3007 | 2850 | 0.000052041 | 0.999999881 | 4 | True |
| 3078 | 2850 | 0.000052169 | 0.999999881 | 3 | True |
| 3108 | 2900 | 0.000048152 | 0.999999821 | 5 | True |
| 3178 | 2900 | 0.000048046 | 0.999999881 | 4 | True |
| 3208 | 2950 | 0.000045256 | 0.999999881 | 6 | True |
| 3278 | 2950 | 0.000044837 | 0.999999940 | 5 | True |
| 3307 | 3000 | 0.000042240 | 0.999999940 | 7 | True |
| 3378 | 3000 | 0.000042130 | 0.999999940 | 6 | True |
| 3407 | 3050 | 0.000039903 | 0.999999940 | 8 | True |
| 3478 | 3050 | 0.000039675 | 0.999999940 | 7 | True |
| 3507 | 3100 | 0.000037759 | 0.999999940 | 9 | True |
| 3579 | 3100 | 0.000037551 | 1.000000000 | 8 | True |
| 3607 | 3150 | 0.000035765 | 1.000000000 | 10 | True |
| 3678 | 3150 | 0.000035689 | 1.000000000 | 9 | True |
| 3707 | 3200 | 0.000034090 | 1.000000000 | 11 | True |
| 3778 | 3200 | 0.000033958 | 1.000000000 | 10 | True |
| 3807 | 3250 | 0.000032588 | 1.000000000 | 12 | True |
| 3878 | 3250 | 0.000032481 | 1.000000000 | 11 | True |
| 3907 | 3300 | 0.000031353 | 1.000000000 | 13 | True |
| 3978 | 3300 | 0.000031121 | 1.000000000 | 12 | True |
| 4007 | 3350 | 0.000029955 | 1.000000000 | 14 | True |
| 4078 | 3350 | 0.000029830 | 1.000000000 | 13 | True |
| 4107 | 3400 | 0.000028807 | 1.000000000 | 15 | True |
| 4178 | 3400 | 0.000028572 | 1.000000000 | 14 | True |
| 4207 | 3450 | 0.000027708 | 1.000000000 | 16 | True |
| 4279 | 3450 | 0.000027565 | 1.000000000 | 15 | True |
| 4307 | 3500 | 0.000026658 | 1.000000000 | 17 | True |
| 4379 | 3500 | 0.000026593 | 1.000000000 | 16 | True |
| 4407 | 3550 | 0.000025772 | 1.000000000 | 18 | True |
| 4479 | 3550 | 0.000025595 | 1.000000000 | 17 | True |
| 4507 | 3600 | 0.000024888 | 1.000000000 | 19 | True |
| 4579 | 3600 | 0.000024746 | 1.000000000 | 18 | True |
| 4607 | 3650 | 0.000024037 | 1.000000000 | 20 | True |
| 4679 | 3650 | 0.000023921 | 1.000000000 | 19 | True |
| 4707 | 3700 | 0.000023268 | 1.000000000 | 21 | True |
| 4778 | 3700 | 0.000023159 | 1.000000000 | 20 | True |
| 4807 | 3750 | 0.000022615 | 1.000000000 | 22 | True |
| 4878 | 3750 | 0.000022435 | 1.000000000 | 21 | True |
| 4907 | 3800 | 0.000021778 | 1.000000000 | 23 | True |
| 4978 | 3800 | 0.000021816 | 1.000000000 | 22 | True |
| 5008 | 3850 | 0.000021247 | 1.000000000 | 24 | True |
| 5078 | 3850 | 0.000021094 | 1.000000000 | 23 | True |
| 5107 | 3900 | 0.000020584 | 1.000000000 | 25 | True |
| 5178 | 3900 | 0.000020534 | 1.000000000 | 24 | True |
| 5207 | 3950 | 0.000020015 | 1.000000000 | 26 | True |
| 5278 | 3950 | 0.000019968 | 1.000000000 | 25 | True |
| 5307 | 4000 | 0.000019467 | 1.000000000 | 27 | True |
| 5378 | 4000 | 0.000019443 | 1.000000000 | 26 | True |
| 5407 | 4050 | 0.000019026 | 1.000000000 | 28 | True |
| 5478 | 4050 | 0.000018934 | 1.000000000 | 27 | True |
| 5507 | 4100 | 0.000018564 | 1.000000000 | 29 | True |
| 5578 | 4100 | 0.000018459 | 1.000000000 | 28 | True |
| 5607 | 4150 | 0.000018036 | 1.000000000 | 30 | True |
| 5678 | 4150 | 0.000017968 | 1.000000000 | 29 | True |
| 5707 | 4200 | 0.000017510 | 1.000000000 | 31 | True |
| 5778 | 4200 | 0.000017492 | 1.000000000 | 30 | True |
| 5807 | 4250 | 0.000017190 | 1.000000000 | 32 | True |
| 5878 | 4250 | 0.000017066 | 1.000000000 | 31 | True |
| 5907 | 4300 | 0.000016748 | 1.000000000 | 33 | True |
| 5978 | 4300 | 0.000016686 | 1.000000000 | 32 | True |
| 6007 | 4350 | 0.000016300 | 1.000000000 | 34 | True |
| 6078 | 4350 | 0.000016286 | 1.000000000 | 33 | True |
| 6107 | 4400 | 0.000015983 | 1.000000000 | 35 | True |
| 6178 | 4400 | 0.000015879 | 1.000000000 | 34 | True |
| 6207 | 4450 | 0.000015617 | 1.000000000 | 36 | True |
| 6278 | 4450 | 0.000015593 | 1.000000000 | 35 | True |
| 6307 | 4500 | 0.000015265 | 1.000000000 | 37 | True |
| 6378 | 4500 | 0.000015217 | 1.000000000 | 36 | True |
| 6407 | 4550 | 0.000014956 | 1.000000000 | 38 | True |
| 6478 | 4550 | 0.000014848 | 1.000000000 | 37 | True |
| 6507 | 4600 | 0.000014547 | 1.000000000 | 39 | True |
| 6579 | 4600 | 0.000014576 | 1.000000000 | 38 | True |
| 6607 | 4650 | 0.000014310 | 1.000000000 | 40 | True |
| 6678 | 4650 | 0.000014256 | 1.000000000 | 39 | True |
| 6707 | 4700 | 0.000014030 | 1.000000000 | 41 | True |
| 6778 | 4700 | 0.000013927 | 1.000000000 | 40 | True |
| 6807 | 4750 | 0.000013683 | 1.000000000 | 42 | True |
| 6878 | 4750 | 0.000013654 | 1.000000000 | 41 | True |
| 6907 | 4800 | 0.000013446 | 1.000000000 | 43 | True |
| 6978 | 4800 | 0.000013383 | 1.000000000 | 42 | True |
| 7007 | 4850 | 0.000013166 | 1.000000000 | 44 | True |
| 7078 | 4850 | 0.000013208 | 1.000000000 | 43 | True |
| 7107 | 4900 | 0.000012978 | 1.000000000 | 45 | True |
| 7179 | 4900 | 0.000012883 | 1.000000000 | 44 | True |
| 7207 | 4950 | 0.000012649 | 1.000000000 | 46 | True |
| 7279 | 4950 | 0.000012646 | 1.000000000 | 45 | True |
| 7307 | 5000 | 0.000012422 | 1.000000000 | 47 | True |
| 7343 | 5000 | 0.000012384 | 1.000000000 | 46 | True |

Seed 0 and seed 1 both first meet the bad-check condition at batch 50 and first set the three-consecutive-check collapse flag at batch 150 (`trajectories.json`, original log rows 50/100/150). Their initialized heads were spatially variable before optimization; the logged training rapidly removed this variation. `trajectories.png` plots all recorded checks; seed-1 duplicates are deliberately retained as separate points.


## 5. Evidence, limits and reproduction

Commands (working directory: repository root):
```sh
python3 -B audit/preflight/diagnose.py core
python3 -B audit/preflight/diagnose.py extract
python3 -B audit/preflight/report.py
```

Access ledgers: `core_access.json`, `extract_access.json`, `report_access.json`. The runtime rejects label/mask paths, validation-image paths, network access, and writes outside this output directory. The shared cached feature file contains train and validation images; only explicit train indices were materialized. The image-only metadata includes all sample descriptors; no annotation or validation image was opened. No training, experiment edits, cache regeneration, or installs occurred.

UNVERIFIED: the full stochastic training objective with feature dropout enabled was not estimated here; comparisons use normalized frozen tokens/dropout-off as requested and as used in the fixed monitor. Initialization diagnostics do not isolate the contribution of bias versus each branch. No labels were opened, so anatomical selectivity or segmentation performance of any layer/shift is unverified. No stable alternative training or guarantee follows from these loss comparisons. A unique seed-1 continuation after resume requires process-specific logs/checkpoint lineage and restored optimizer/RNG evidence; current combined logs are insufficient. Confidence intervals over patient resampling were not computed; pair draws are highly dependent.

Prediction: the pinned loss strongly favors constant codes over the tested clustered codes and predicts collapse, consistent with the already-recorded seed-0/1 trajectories. C/D/E-like reductions in attraction remove that measured preference here, but no attraction percentage alone guarantees non-collapse or liver segmentation.


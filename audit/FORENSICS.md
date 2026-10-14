# CHAOS liver-cluster forensics


## Checkpoints found

| Checkpoint | Found | Bytes | Exact path |
| --- | --- | --- | --- |
| C seed 0 (500) | True | 792434993 | `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_C_seed0/checkpoints/last.ckpt` |
| C seed 1 (500) | True | 792435185 | `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_C_seed1/checkpoints/last.ckpt` |
| C seed 2 (500) | True | 792435185 | `/Users/navneetbavineni/STEGO/audit/scratch/fixes3/experiments/fixes3_C_seed2/checkpoints/last.ckpt` |
| Full best (600) | True | 792437477 | `/Users/navneetbavineni/STEGO/checkpoints/chaos/chaos_fixes3_full_C_date_Sep30_19-47-58/checkpoint-epoch=05-step=1800-mIoU=test/cluster/mIoU=69.3704.ckpt` |
| Full final (5000) | True | 792455565 | `/Users/navneetbavineni/STEGO/checkpoints/chaos/chaos_fixes3_full_C_date_Sep30_19-47-58/checkpoint-epoch=42-step=15000-mIoU=test/cluster/mIoU=46.1792.ckpt` |


All five found and strictly loaded. Task1 evaluates all472 validation slices, all224×224 pixels (23,683,072 total;1,809,782 liver). Training/source/data/cache/checkpoints unchanged; only new diagnostic scratch artifacts and this report written. Model, head, probe, transforms, correspondence helper and eval metric are imported from project source. CPU only,570s hard timeout per producer script. Seed20260930. Provenance: `audit/scratch/forensics/task1.py`, `task1.json`, `inventory.json`.


## 1. Mapping versus information

All scores here are percent. Best single cluster means maximum single-cluster liver IoU, not maximum recall. Current mapping requires precision>50%; ties go to background. IoU-optimal subset enumerates all256 assignments. Alternatives and thresholds use validation labels: optimistic diagnostic oracle scores, NOT newly learned unsupervised segmentation results or deployable mappings. No threshold is chosen for deployment.

| Checkpoint | Majority IoU | IoU-optimal IoU | Optimal clusters | Best single cluster | Single precision / recall |
| --- | --- | --- | --- | --- | --- |
| C seed 0 (500) | 43.748989 | 43.748989 | [2] | [2] | 63.182362 / 58.718288 |
| C seed 1 (500) | 0.000000 | 32.852791 | [4] | [4] | 44.761092 / 55.254777 |
| C seed 2 (500) | 0.000000 | 31.947812 | [0, 2] | [0] | 36.569611 / 69.438032 |
| Full best (600) | 44.400492 | 44.400492 | [2] | [2] | 54.506920 / 70.541866 |
| Full final (5000) | 0.000000 | 29.806593 | [2, 5] | [2] | 34.619100 / 57.511126 |


### C seed 0 (500) — all8 clusters

| ID | Pixel fraction (%) | Liver pixels | Background pixels | Liver precision (%) | Liver recall (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 5.702685 | 0 | 1350571 | 0.000000 | 0.000000 |
| 1 | 15.683751 | 21938 | 3692456 | 0.590621 | 1.212190 |
| 2 | 7.101756 | 1062673 | 619241 | 63.182362 | 58.718288 |
| 3 | 8.348275 | 51326 | 1925802 | 2.595988 | 2.836032 |
| 4 | 0.011523 | 0 | 2729 | 0.000000 | 0.000000 |
| 5 | 19.363810 | 218220 | 4367725 | 4.758452 | 12.057806 |
| 6 | 21.804726 | 455625 | 4708404 | 8.823053 | 25.175684 |
| 7 | 21.983474 | 0 | 5206362 | 0.000000 | 0.000000 |

Clusters >50% liver: [2]. Active clusters>=1%: 7.


### C seed 1 (500) — all8 clusters

| ID | Pixel fraction (%) | Liver pixels | Background pixels | Liver precision (%) | Liver recall (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 17.454142 | 389371 | 3744306 | 9.419483 | 21.514801 |
| 1 | 12.732001 | 367505 | 2647824 | 12.187891 | 20.306589 |
| 2 | 17.825876 | 0 | 4221715 | 0.000000 | 0.000000 |
| 3 | 8.076195 | 528 | 1912163 | 0.027605 | 0.029175 |
| 4 | 9.433164 | 999991 | 1234072 | 44.761092 | 55.254777 |
| 5 | 7.338659 | 51446 | 1686574 | 2.960035 | 2.842663 |
| 6 | 21.199750 | 930 | 5019822 | 0.018523 | 0.051387 |
| 7 | 5.940213 | 11 | 1406814 | 0.000782 | 0.000608 |

Clusters >50% liver: []. Active clusters>=1%: 8.


### C seed 2 (500) — all8 clusters

| ID | Pixel fraction (%) | Liver pixels | Background pixels | Liver precision (%) | Liver recall (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 14.509929 | 1256677 | 2179720 | 36.569611 | 69.438032 |
| 1 | 23.188736 | 8727 | 5483078 | 0.158910 | 0.482213 |
| 2 | 6.872698 | 407650 | 1220016 | 25.045065 | 22.524812 |
| 3 | 8.549288 | 2293 | 2022441 | 0.113249 | 0.126700 |
| 4 | 9.978912 | 84833 | 2278480 | 3.589580 | 4.687471 |
| 5 | 12.409062 | 48832 | 2890015 | 1.661604 | 2.698226 |
| 6 | 4.025090 | 0 | 953265 | 0.000000 | 0.000000 |
| 7 | 20.466285 | 770 | 4846275 | 0.015886 | 0.042547 |

Clusters >50% liver: []. Active clusters>=1%: 8.


### Full best (600) — all8 clusters

| ID | Pixel fraction (%) | Liver pixels | Background pixels | Liver precision (%) | Liver recall (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 5.952163 | 2 | 1409653 | 0.000142 | 0.000111 |
| 1 | 17.965794 | 12862 | 4241990 | 0.302290 | 0.710693 |
| 2 | 9.889709 | 1276654 | 1065533 | 54.506920 | 70.541866 |
| 3 | 4.158481 | 7134 | 977722 | 0.724370 | 0.394191 |
| 4 | 0.237997 | 6 | 56359 | 0.010645 | 0.000332 |
| 5 | 15.429730 | 202559 | 3451675 | 5.543132 | 11.192453 |
| 6 | 22.915971 | 310565 | 5116641 | 5.722374 | 17.160354 |
| 7 | 23.450155 | 0 | 5553717 | 0.000000 | 0.000000 |

Clusters >50% liver: [2]. Active clusters>=1%: 7.


### Full final (5000) — all8 clusters

| ID | Pixel fraction (%) | Liver pixels | Background pixels | Liver precision (%) | Liver recall (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 4.863529 | 20 | 1151813 | 0.001736 | 0.001105 |
| 1 | 16.408201 | 1302 | 3884664 | 0.033505 | 0.071942 |
| 2 | 12.694755 | 1040826 | 1965682 | 34.619100 | 57.511126 |
| 3 | 11.785621 | 8051 | 2783146 | 0.288443 | 0.444860 |
| 4 | 18.841660 | 254 | 4462030 | 0.005692 | 0.014035 |
| 5 | 10.151365 | 617155 | 1787000 | 25.670350 | 34.101069 |
| 6 | 10.843302 | 142152 | 2425875 | 5.535456 | 7.854648 |
| 7 | 14.411568 | 22 | 3413080 | 0.000645 | 0.001216 |

Clusters >50% liver: []. Active clusters>=1%: 8.


### Mapping sensitivity (IoU / Dice %, and liver-assigned IDs)

| Checkpoint | Rule | Liver IDs | IoU | Dice | Recall |
| --- | --- | --- | --- | --- | --- |
| C seed 0 (500) | majority >50% | [2] | 43.748989 | 60.868586 | 58.718288 |
| C seed 0 (500) | optimal subset | [2] | 43.748989 | 60.868586 | 58.718288 |
| C seed 0 (500) | precision >10% | [2] | 43.748989 | 60.868586 | 58.718288 |
| C seed 0 (500) | precision >20% | [2] | 43.748989 | 60.868586 | 58.718288 |
| C seed 0 (500) | precision >30% | [2] | 43.748989 | 60.868586 | 58.718288 |
| C seed 0 (500) | precision >40% | [2] | 43.748989 | 60.868586 | 58.718288 |
| C seed 1 (500) | majority >50% | [] | 0.000000 | 0.000000 | 0.000000 |
| C seed 1 (500) | optimal subset | [4] | 32.852791 | 49.457435 | 55.254777 |
| C seed 1 (500) | precision >10% | [1, 4] | 24.026236 | 38.743796 | 75.561366 |
| C seed 1 (500) | precision >20% | [4] | 32.852791 | 49.457435 | 55.254777 |
| C seed 1 (500) | precision >30% | [4] | 32.852791 | 49.457435 | 55.254777 |
| C seed 1 (500) | precision >40% | [4] | 32.852791 | 49.457435 | 55.254777 |
| C seed 2 (500) | majority >50% | [] | 0.000000 | 0.000000 | 0.000000 |
| C seed 2 (500) | optimal subset | [0, 2] | 31.947812 | 48.424921 | 91.962844 |
| C seed 2 (500) | precision >10% | [0, 2] | 31.947812 | 48.424921 | 91.962844 |
| C seed 2 (500) | precision >20% | [0, 2] | 31.947812 | 48.424921 | 91.962844 |
| C seed 2 (500) | precision >30% | [0] | 31.499596 | 47.908278 | 69.438032 |
| C seed 2 (500) | precision >40% | [] | 0.000000 | 0.000000 | 0.000000 |
| Full best (600) | majority >50% | [2] | 44.400492 | 61.496317 | 70.541866 |
| Full best (600) | optimal subset | [2] | 44.400492 | 61.496317 | 70.541866 |
| Full best (600) | precision >10% | [2] | 44.400492 | 61.496317 | 70.541866 |
| Full best (600) | precision >20% | [2] | 44.400492 | 61.496317 | 70.541866 |
| Full best (600) | precision >30% | [2] | 44.400492 | 61.496317 | 70.541866 |
| Full best (600) | precision >40% | [2] | 44.400492 | 61.496317 | 70.541866 |
| Full final (5000) | majority >50% | [] | 0.000000 | 0.000000 | 0.000000 |
| Full final (5000) | optimal subset | [2, 5] | 29.806593 | 45.924621 | 91.612194 |
| Full final (5000) | precision >10% | [2, 5] | 29.806593 | 45.924621 | 91.612194 |
| Full final (5000) | precision >20% | [2, 5] | 29.806593 | 45.924621 | 91.612194 |
| Full final (5000) | precision >30% | [2] | 27.568161 | 43.221068 | 57.511126 |
| Full final (5000) | precision >40% | [] | 0.000000 | 0.000000 | 0.000000 |

Interpretation: seed1 has a44.761092%-pure cluster containing55.254777% of liver; strict majority rejects it. Seed2 single cluster0 captures69.438032% of liver at36.569611% precision. Final single cluster2 captures57.511126% at34.619100% precision. Final optimal union[2,5] captures91.612194% of liver but includes3,752,682 background pixels; precision30.642844%, IoU29.806593%. Therefore zero is a threshold/mapping discontinuity, but the assignments also have genuine loss of liver specificity: oracle IoU remains below seed0/best600. The failure is not absence of all liver information and not just ID permutation. Source:task1.py/task1.json.


## 2. Label-free partition comparisons

ARI and arithmetic-mean-normalized NMI use all23,683,072 corresponding pixels. They are invariant to cluster-ID permutations. Contingency-based formulas were checked against sklearn on a synthetic example. Pixel observations are highly correlated; these are descriptive agreement values, not significance tests. Source:task2.py/task2.json.

| Pair | ARI | NMI |
| --- | --- | --- |
| seed0 vs seed1 | 0.290543566 | 0.456452130 |
| seed0 vs seed2 | 0.266097854 | 0.456933650 |
| seed0 vs best600 | 0.758601208 | 0.757441274 |
| seed0 vs final | 0.320225905 | 0.492978551 |
| seed1 vs seed2 | 0.533997029 | 0.607311454 |
| seed1 vs best600 | 0.297494953 | 0.456549536 |
| seed1 vs final | 0.449579523 | 0.605697121 |
| seed2 vs best600 | 0.259047710 | 0.440001035 |
| seed2 vs final | 0.508029132 | 0.641620846 |
| best600 vs final | 0.330908100 | 0.496771391 |

All150 visual-trunk parameter/buffer tensors are exactly equal across the five checkpoints (backbone_check.py/backbone_check.json). Thus the frozen encoder is shared; learned head/probe partitions differ. These assignment comparisons do not by themselves establish equality/inequality of the latent70-D code geometry independent of the probe.


### Agreement with frozen-feature K=8 k-means

Reproduced both DIAG preprocessing cases rather than silently choosing one: normalized final-block LayerNorm tokens; either no train-mean subtraction or subtract mean over39,200 train patches, then per-patch L2. KMeans fixed seed20260930,n_init3,max_iter100,tol1e-4; fit ONLY the original200 train-slice features. All472 validation feature vectors came from the actual current extractor in Task1. KMeans14×14 labels are nearest-neighbour enlarged to224×224, matching DIAG. Loss helper restored-global-mean affinity centering is a pairwise operation used in Task4, not a substitute for this vector preprocessing. No labels enter fitting/agreement.

| Checkpoint | Active>=1% | No mean: ARI / NMI | Train mean subtracted: ARI / NMI |
| --- | --- | --- | --- |
| C seed 0 (500) | 7 | 0.193800282 / 0.340141719 | 0.185948728 / 0.335050810 |
| C seed 1 (500) | 8 | 0.218348343 / 0.365112325 | 0.221461355 / 0.362083428 |
| C seed 2 (500) | 8 | 0.195389675 / 0.346974113 | 0.180199551 / 0.339104091 |
| Full best (600) | 7 | 0.222949218 / 0.368144426 | 0.205673949 / 0.362469806 |
| Full final (5000) | 8 | 0.182526507 / 0.353968487 | 0.185770123 / 0.348290765 |

On the original100 DIAG val slices, reproduced K8 majority IoUs exactly:64.158320% without mean subtraction;65.509625% with subtraction. Fresh features versus stored DIAG features are not bit-identical: max absolute difference0.002151489. The cause of this small feature discrepancy is UNVERIFIED; the scored KMeans assignments reproduce the recorded DIAG scores. No claim that these100-slice scores are scores on all472 slices is made.


## 3. Full-run trajectory

Producer:task3.py; data:task3.json. Parsed85 validation records directly from JSONL and independently from the W&B offline binary; scalar agreement checked. train.log provides terminal max_steps=15000 and debug excerpts, not a complete85-row metric table. W&B scanner requires a writable file handle, so only a new scratch copy was scanned; the original binary was never opened for write by these diagnostics.

First logged absence:batch50 before liver mapping forms. Mapping first appears at100, remains through568, first disappears after being present at635 and never reappears in subsequent75 validations. Thus the LOGGED transition lies in(568,635]. Saved checkpoint600 still has a majority-liver cluster, so saved-state evidence narrows the disappearance to(600,635]; the exact intervening batch and whether transient reappearance occurred between checks are UNVERIFIED.

Mechanism: cluster2 itself does not vanish. From saved600 to700, its liver recall rises70.541866% ->74.972068%, but background pixels rise1,065,533 ->1,625,656 while liver pixels rise1,276,654 ->1,356,831. Its precision crosses54.506920% ->45.493275%; majority mapping abruptly switches it to background. Thus the zero score reflects increasing background contamination crossing a hard threshold, not a disappearing raw cluster. Source:task5.py/task5_0_10.json.

Sampling schedule:1869 train slices / batch16 =>117 batches per epoch. val_freq50 is measured in batches, restarting each epoch; validations at epoch offsets50 and100. Gaps alternate50 and67.42 complete epochs yield84 checks, and the final partial86-batch epoch adds one check at4964:85 total. M8 converted cfg.max_steps5000 to Trainer max_steps15000 because3 optimizers step per batch. It did not change val_freq into optimizer-step units. Evidence:src/train_segmentation.py:692-706 and audit/FIXES.md:32,86-87; expected batch schedule asserted against all85 records in task3.py.

![Full-run trajectory](/Users/navneetbavineni/STEGO/audit/scratch/forensics/full_run_trajectory.png)

| Batch | Optimizer steps | Liver IoU % | Dice % | mIoU % | Code std | Cosine | Entropy | Liver IDs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 150 | 0.000000 | 0.000000 | 46.179166 | 0.105835590 | 0.142615050 | 1.603166851 | [] |
| 100 | 300 | 30.644438 | 46.912730 | 62.146622 | 0.110021644 | 0.072296040 | 1.573104806 | [2] |
| 167 | 501 | 31.822446 | 48.280770 | 62.881547 | 0.110912840 | 0.053837690 | 1.638282165 | [2] |
| 217 | 651 | 30.026048 | 46.184668 | 62.021506 | 0.111518395 | 0.043769754 | 1.619142350 | [2] |
| 284 | 852 | 25.215554 | 40.275434 | 59.469664 | 0.111824884 | 0.040413424 | 1.639009507 | [2] |
| 334 | 1002 | 35.827714 | 52.754647 | 65.059018 | 0.111923703 | 0.040755906 | 1.720029624 | [2] |
| 401 | 1203 | 38.490489 | 55.585748 | 66.399759 | 0.111351787 | 0.054682582 | 1.770048380 | [2] |
| 451 | 1353 | 43.827960 | 60.944980 | 68.987989 | 0.112192525 | 0.043042526 | 1.800333661 | [2] |
| 518 | 1554 | 45.918703 | 62.937379 | 69.795221 | 0.112195377 | 0.049464670 | 1.818243077 | [2] |
| 568 | 1704 | 45.430696 | 62.477452 | 69.370395 | 0.112123982 | 0.056844192 | 1.822581160 | [2] |
| 635 | 1905 | 0.000000 | 0.000000 | 46.179166 | 0.111305297 | 0.081312308 | 1.855060260 | [] |
| 685 | 2055 | 0.000000 | 0.000000 | 46.179166 | 0.112675865 | 0.061704238 | 1.861109145 | [] |
| 752 | 2256 | 0.000000 | 0.000000 | 46.179166 | 0.113030167 | 0.057934736 | 1.911945680 | [] |
| 802 | 2406 | 0.000000 | 0.000000 | 46.179166 | 0.113759040 | 0.047381346 | 1.953413448 | [] |
| 869 | 2607 | 0.000000 | 0.000000 | 46.179166 | 0.110398115 | 0.103682866 | 1.888218285 | [] |
| 919 | 2757 | 0.000000 | 0.000000 | 46.179166 | 0.110167242 | 0.109959767 | 1.896569887 | [] |
| 986 | 2958 | 0.000000 | 0.000000 | 46.179166 | 0.112654967 | 0.071130723 | 1.930996822 | [] |
| 1036 | 3108 | 0.000000 | 0.000000 | 46.179166 | 0.111268194 | 0.093221410 | 1.906302746 | [] |
| 1103 | 3309 | 0.000000 | 0.000000 | 46.179166 | 0.113187886 | 0.064198722 | 1.920674141 | [] |
| 1153 | 3459 | 0.000000 | 0.000000 | 46.179166 | 0.112026342 | 0.082616249 | 1.904577911 | [] |
| 1220 | 3660 | 0.000000 | 0.000000 | 46.179166 | 0.111855853 | 0.085825311 | 1.903782220 | [] |
| 1270 | 3810 | 0.000000 | 0.000000 | 46.179166 | 0.110674766 | 0.105173265 | 1.900029650 | [] |
| 1337 | 4011 | 0.000000 | 0.000000 | 46.179166 | 0.112348212 | 0.077801072 | 1.908689257 | [] |
| 1387 | 4161 | 0.000000 | 0.000000 | 46.179166 | 0.111986605 | 0.085381608 | 1.913437828 | [] |
| 1454 | 4362 | 0.000000 | 0.000000 | 46.179166 | 0.111249470 | 0.097255254 | 1.893935406 | [] |
| 1504 | 4512 | 0.000000 | 0.000000 | 46.179166 | 0.112798671 | 0.071955181 | 1.884130699 | [] |
| 1571 | 4713 | 0.000000 | 0.000000 | 46.179166 | 0.112260197 | 0.081835248 | 1.897935151 | [] |
| 1621 | 4863 | 0.000000 | 0.000000 | 46.179166 | 0.111159378 | 0.099988868 | 1.903056110 | [] |
| 1688 | 5064 | 0.000000 | 0.000000 | 46.179166 | 0.111542524 | 0.094036502 | 1.898439806 | [] |
| 1738 | 5214 | 0.000000 | 0.000000 | 46.179166 | 0.111905057 | 0.088470315 | 1.898761502 | [] |
| 1805 | 5415 | 0.000000 | 0.000000 | 46.179166 | 0.112468171 | 0.080009379 | 1.902110656 | [] |
| 1855 | 5565 | 0.000000 | 0.000000 | 46.179166 | 0.111855314 | 0.090751516 | 1.899998113 | [] |
| 1922 | 5766 | 0.000000 | 0.000000 | 46.179166 | 0.111930701 | 0.088575206 | 1.900767187 | [] |
| 1972 | 5916 | 0.000000 | 0.000000 | 46.179166 | 0.111757188 | 0.092863631 | 1.897946367 | [] |
| 2039 | 6117 | 0.000000 | 0.000000 | 46.179166 | 0.111324063 | 0.099179884 | 1.897342160 | [] |
| 2089 | 6267 | 0.000000 | 0.000000 | 46.179166 | 0.110844369 | 0.106600429 | 1.910767503 | [] |
| 2156 | 6468 | 0.000000 | 0.000000 | 46.179166 | 0.112150226 | 0.085929960 | 1.917766328 | [] |
| 2206 | 6618 | 0.000000 | 0.000000 | 46.179166 | 0.112848906 | 0.076029279 | 1.906816326 | [] |
| 2273 | 6819 | 0.000000 | 0.000000 | 46.179166 | 0.111384471 | 0.099180993 | 1.908600379 | [] |
| 2323 | 6969 | 0.000000 | 0.000000 | 46.179166 | 0.112372127 | 0.082976311 | 1.910442690 | [] |
| 2390 | 7170 | 0.000000 | 0.000000 | 46.179166 | 0.113522014 | 0.064650897 | 1.913225823 | [] |
| 2440 | 7320 | 0.000000 | 0.000000 | 46.179166 | 0.111831587 | 0.091949641 | 1.912247140 | [] |
| 2507 | 7521 | 0.000000 | 0.000000 | 46.179166 | 0.112700059 | 0.078242353 | 1.928595297 | [] |
| 2557 | 7671 | 0.000000 | 0.000000 | 46.179166 | 0.111871805 | 0.092133390 | 1.932103104 | [] |
| 2624 | 7872 | 0.000000 | 0.000000 | 46.179166 | 0.112256660 | 0.086330112 | 1.926559928 | [] |
| 2674 | 8022 | 0.000000 | 0.000000 | 46.179166 | 0.112714065 | 0.079023436 | 1.935705296 | [] |
| 2741 | 8223 | 0.000000 | 0.000000 | 46.179166 | 0.113005721 | 0.074751785 | 1.932779133 | [] |
| 2791 | 8373 | 0.000000 | 0.000000 | 46.179166 | 0.112551739 | 0.081365404 | 1.944598940 | [] |
| 2858 | 8574 | 0.000000 | 0.000000 | 46.179166 | 0.113209519 | 0.070847542 | 1.944299874 | [] |
| 2908 | 8724 | 0.000000 | 0.000000 | 46.179166 | 0.112768271 | 0.077989613 | 1.951765925 | [] |
| 2975 | 8925 | 0.000000 | 0.000000 | 46.179166 | 0.111953025 | 0.091194560 | 1.948838132 | [] |
| 3025 | 9075 | 0.000000 | 0.000000 | 46.179166 | 0.111855753 | 0.093407634 | 1.949631714 | [] |
| 3092 | 9276 | 0.000000 | 0.000000 | 46.179166 | 0.112300950 | 0.085467174 | 1.958367766 | [] |
| 3142 | 9426 | 0.000000 | 0.000000 | 46.179166 | 0.112295332 | 0.086154581 | 1.968816103 | [] |
| 3209 | 9627 | 0.000000 | 0.000000 | 46.179166 | 0.111815275 | 0.094173043 | 1.976741891 | [] |
| 3259 | 9777 | 0.000000 | 0.000000 | 46.179166 | 0.112713419 | 0.079750881 | 1.975920268 | [] |
| 3326 | 9978 | 0.000000 | 0.000000 | 46.179166 | 0.112014085 | 0.091278050 | 1.985748318 | [] |
| 3376 | 10128 | 0.000000 | 0.000000 | 46.179166 | 0.113000915 | 0.075566383 | 1.990603191 | [] |
| 3443 | 10329 | 0.000000 | 0.000000 | 46.179166 | 0.111724788 | 0.095836344 | 1.990721190 | [] |
| 3493 | 10479 | 0.000000 | 0.000000 | 46.179166 | 0.111879320 | 0.093450845 | 1.996077699 | [] |
| 3560 | 10680 | 0.000000 | 0.000000 | 46.179166 | 0.112616476 | 0.082073796 | 2.002164141 | [] |
| 3610 | 10830 | 0.000000 | 0.000000 | 46.179166 | 0.112685401 | 0.080660080 | 2.005480320 | [] |
| 3677 | 11031 | 0.000000 | 0.000000 | 46.179166 | 0.112823085 | 0.078246193 | 2.007054654 | [] |
| 3727 | 11181 | 0.000000 | 0.000000 | 46.179166 | 0.112297591 | 0.086679430 | 2.007927038 | [] |
| 3794 | 11382 | 0.000000 | 0.000000 | 46.179166 | 0.112797573 | 0.078643587 | 2.011695426 | [] |
| 3844 | 11532 | 0.000000 | 0.000000 | 46.179166 | 0.112477637 | 0.084219576 | 2.015440836 | [] |
| 3911 | 11733 | 0.000000 | 0.000000 | 46.179166 | 0.112450852 | 0.084729734 | 2.014581457 | [] |
| 3961 | 11883 | 0.000000 | 0.000000 | 46.179166 | 0.113091055 | 0.074453484 | 2.014030313 | [] |
| 4028 | 12084 | 0.000000 | 0.000000 | 46.179166 | 0.112493357 | 0.084167432 | 2.015483227 | [] |
| 4078 | 12234 | 0.000000 | 0.000000 | 46.179166 | 0.112704655 | 0.080256286 | 2.016071007 | [] |
| 4145 | 12435 | 0.000000 | 0.000000 | 46.179166 | 0.112546995 | 0.083294602 | 2.016992286 | [] |
| 4195 | 12585 | 0.000000 | 0.000000 | 46.179166 | 0.112886724 | 0.077539525 | 2.019299553 | [] |
| 4262 | 12786 | 0.000000 | 0.000000 | 46.179166 | 0.112382491 | 0.085984777 | 2.019993787 | [] |
| 4312 | 12936 | 0.000000 | 0.000000 | 46.179166 | 0.112905836 | 0.077334234 | 2.019456472 | [] |
| 4379 | 13137 | 0.000000 | 0.000000 | 46.179166 | 0.112782414 | 0.079420756 | 2.020665063 | [] |
| 4429 | 13287 | 0.000000 | 0.000000 | 46.179166 | 0.112641678 | 0.081642528 | 2.021636209 | [] |
| 4496 | 13488 | 0.000000 | 0.000000 | 46.179166 | 0.112560196 | 0.082894997 | 2.020922630 | [] |
| 4546 | 13638 | 0.000000 | 0.000000 | 46.179166 | 0.112667960 | 0.081185695 | 2.021911547 | [] |
| 4613 | 13839 | 0.000000 | 0.000000 | 46.179166 | 0.112606232 | 0.082298684 | 2.022435109 | [] |
| 4663 | 13989 | 0.000000 | 0.000000 | 46.179166 | 0.112727757 | 0.080377966 | 2.022181080 | [] |
| 4730 | 14190 | 0.000000 | 0.000000 | 46.179166 | 0.112636168 | 0.081826220 | 2.023066886 | [] |
| 4780 | 14340 | 0.000000 | 0.000000 | 46.179166 | 0.112716126 | 0.080595493 | 2.023180088 | [] |
| 4847 | 14541 | 0.000000 | 0.000000 | 46.179166 | 0.112539441 | 0.083384124 | 2.024361442 | [] |
| 4897 | 14691 | 0.000000 | 0.000000 | 46.179166 | 0.112559471 | 0.083061706 | 2.024315331 | [] |
| 4964 | 14892 | 0.000000 | 0.000000 | 46.179166 | 0.112626985 | 0.081972335 | 2.024332253 | [] |


## 4. Correspondence objective versus liver outcome

Producer:task4.py/task4.json.16 fixed train batches×16 images, fixed shuffled indices, identical recorded top7 cache-neighbour draws and seeds for all random sampling. No train annotations opened (guard enforced). Same actual extractor/head forward and ContrastiveCorrelationLoss.forward/helper with restored-mean centering; dropout OFF/eval; no backward or optimizer. Negative pairs use the actual16 random nonzero shifts per batch. This diagnostic isolates the correspondence objective, not loss/total including the detached-code cluster-probe loss. Since probe input is detached (src/train_segmentation.py:275-298), baseline head updates are driven by correspondence, but the experiment is observational rather than a causal ablation.

| Checkpoint | pos_intra | pos_inter | neg_inter | Weighted correspondence |
| --- | --- | --- | --- | --- |
| Full best (600) | -0.151331959 | -0.148326471 | 0.010963553 | -0.131566992 |
| Full final (5000) | -0.154178016 | -0.150276813 | 0.007604524 | -0.136077624 |

Weights unchanged:0.67 intra+0.25 inter+0.63 negative, correspondence_weight1. Final minus best600 =-0.004510632. All16 paired batches have lower correspondence loss at final. The final objective therefore rewards a state with worse cluster-level liver specificity (best600 majority/optimal44.400492%; final majority0%, optimal29.806593%). This demonstrates misalignment on this controlled diagnostic; it does not identify a unique causal hyperparameter or prove training-time stochastic loss was lower at every update.


Mean feature-affinity-weighted code similarity is explicitly E[fd*cd], with fd from the actual centered helper and cd actual code cosine. Also report E[fd*max(cd,0)] because zero_clamp=True in training. These are unnormalized averages, not ratios by the affinity sum.

| Checkpoint | Pairs | E[fd*cd] | E[fd*max(cd,0)] | E[cd] | E[fd] | E[(fd-shift)*max(cd,0)] |
| --- | --- | --- | --- | --- | --- | --- |
| Full best (600) | knn | 0.392964194 | 0.402476532 | 0.533961087 | 0.600507345 | 0.148326471 |
| Full best (600) | random | 0.036286276 | 0.127272550 | -0.016033155 | 0.545364598 | -0.010963553 |
| Full final (5000) | knn | 0.412304295 | 0.417581018 | 0.574958630 | 0.600507345 | 0.150276813 |
| Full final (5000) | random | 0.015877986 | 0.082586115 | -0.015410529 | 0.545364598 | -0.007604524 |

| Fixed batch | Best600 correspondence | Final correspondence | Final−best |
| --- | --- | --- | --- |
| 0 | -0.131776310 | -0.136303702 | -0.004527393 |
| 1 | -0.130108494 | -0.134957886 | -0.004849392 |
| 2 | -0.126339310 | -0.130822086 | -0.004482776 |
| 3 | -0.135299200 | -0.139790653 | -0.004491453 |
| 4 | -0.133904516 | -0.138195775 | -0.004291259 |
| 5 | -0.132701344 | -0.137737574 | -0.005036230 |
| 6 | -0.135490041 | -0.140730873 | -0.005240831 |
| 7 | -0.131772711 | -0.136234020 | -0.004461309 |
| 8 | -0.124146058 | -0.129094666 | -0.004948609 |
| 9 | -0.130219668 | -0.133855906 | -0.003636238 |
| 10 | -0.133570513 | -0.137239842 | -0.003669330 |
| 11 | -0.131135371 | -0.135160701 | -0.004025331 |
| 12 | -0.133061573 | -0.137201431 | -0.004139858 |
| 13 | -0.130264976 | -0.135272476 | -0.005007499 |
| 14 | -0.131680758 | -0.135797845 | -0.004117086 |
| 15 | -0.133601028 | -0.138846545 | -0.005245517 |


## 5. Every saved full-run checkpoint

Producer:task5.py --start N --stop N+10 for N=0,10,20,30,40. All50 on-disk checkpoints evaluated on all472 val slices. Actual imported extractor/head/probe code is used; saved frozen-trunk tensors are checked exactly against seed0. For speed, the trunk output is supplied from Task1 scratch diagnostic activations after live-forward feature AND code equality on16 images per checkpoint. Maximum reuse discrepancy across all50: 0.0 features, 0.0 codes. No pipeline cache is generated/updated. Full per-cluster confusion matrices and paths are in task5_*.json.

Best freshly evaluated majority-IoU checkpoint: batch600, IoU44.400492%, Dice61.496317%; `/Users/navneetbavineni/STEGO/checkpoints/chaos/chaos_fixes3_full_C_date_Sep30_19-47-58/checkpoint-epoch=05-step=1800-mIoU=test/cluster/mIoU=69.3704.ckpt`.

Best freshly evaluated IoU-optimal checkpoint: batch600, IoU44.400492%, Dice61.496317%; `/Users/navneetbavineni/STEGO/checkpoints/chaos/chaos_fixes3_full_C_date_Sep30_19-47-58/checkpoint-epoch=05-step=1800-mIoU=test/cluster/mIoU=69.3704.ckpt`.

Both criteria select batch600, agreeing with the recorded cluster-mIoU-selected checkpoint. Recorded callback score69.370392% was from earlier validation568, not the saved600 weights. Best logged unsaved validation518 reached45.918703% IoU /62.937379% Dice; no checkpoint518 exists. New exhaustive checkpoint evaluation confirms best among the50 SAVED candidates, not among all5000 training states. Evaluation/selection uses val labels and is optimistic; no held-out test claim.

| Batch | Stored mIoU % (stale) | Score from validation batch | Majority IoU % | Majority Dice % | Optimal IoU % | Optimal Dice % | Optimal IDs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 46.1792 | 100 | 30.644484 | 46.912786 | 30.644484 | 46.912786 | [2] |
| 200 | 62.8815 | 167 | 29.864555 | 45.993389 | 29.864555 | 45.993389 | [2] |
| 300 | 59.4697 | 284 | 37.117633 | 54.139839 | 37.117633 | 54.139839 | [2] |
| 400 | 65.059 | 334 | 38.734015 | 55.839247 | 38.734015 | 55.839247 | [2] |
| 500 | 68.988 | 451 | 43.757652 | 60.876971 | 43.757652 | 60.876971 | [2] |
| 600 | 69.3704 | 568 | 44.400492 | 61.496317 | 44.400492 | 61.496317 | [2] |
| 700 | 46.1792 | 685 | 0.000000 | 0.000000 | 39.495139 | 56.625828 | [2] |
| 800 | 46.1792 | 752 | 0.000000 | 0.000000 | 35.368219 | 52.254834 | [2] |
| 900 | 46.1792 | 869 | 0.000000 | 0.000000 | 36.074661 | 53.021864 | [2] |
| 1000 | 46.1792 | 986 | 0.000000 | 0.000000 | 35.606134 | 52.514046 | [2] |
| 1100 | 46.1792 | 1036 | 0.000000 | 0.000000 | 33.755587 | 50.473536 | [2] |
| 1200 | 46.1792 | 1153 | 0.000000 | 0.000000 | 34.220063 | 50.990980 | [2] |
| 1300 | 46.1792 | 1270 | 0.000000 | 0.000000 | 32.978557 | 49.599812 | [2] |
| 1400 | 46.1792 | 1387 | 0.000000 | 0.000000 | 32.634040 | 49.209148 | [2] |
| 1500 | 46.1792 | 1454 | 0.000000 | 0.000000 | 30.720426 | 47.001722 | [2] |
| 1600 | 46.1792 | 1571 | 0.000000 | 0.000000 | 31.496917 | 47.905179 | [2] |
| 1700 | 46.1792 | 1688 | 0.000000 | 0.000000 | 30.680698 | 46.955210 | [2] |
| 1800 | 46.1792 | 1738 | 0.000000 | 0.000000 | 30.998396 | 47.326375 | [2] |
| 1900 | 46.1792 | 1855 | 0.000000 | 0.000000 | 30.422378 | 46.652084 | [2] |
| 2000 | 46.1792 | 1972 | 0.000000 | 0.000000 | 30.194226 | 46.383357 | [2] |
| 2100 | 46.1792 | 2089 | 0.000000 | 0.000000 | 30.075938 | 46.243662 | [2] |
| 2200 | 46.1792 | 2156 | 0.000000 | 0.000000 | 30.121914 | 46.297988 | [2] |
| 2300 | 46.1792 | 2273 | 0.000000 | 0.000000 | 29.584558 | 45.660623 | [2] |
| 2400 | 46.1792 | 2390 | 0.000000 | 0.000000 | 29.483126 | 45.539720 | [2] |
| 2500 | 46.1792 | 2440 | 0.000000 | 0.000000 | 29.349964 | 45.380707 | [2] |
| 2600 | 46.1792 | 2557 | 0.000000 | 0.000000 | 29.233541 | 45.241414 | [2] |
| 2700 | 46.1792 | 2674 | 0.000000 | 0.000000 | 29.284850 | 45.302833 | [2] |
| 2800 | 46.1792 | 2791 | 0.000000 | 0.000000 | 29.559788 | 45.631115 | [2] |
| 2900 | 46.1792 | 2858 | 0.000000 | 0.000000 | 29.259385 | 45.272357 | [2] |
| 3000 | 46.1792 | 2975 | 0.000000 | 0.000000 | 29.341470 | 45.370553 | [2] |
| 3100 | 46.1792 | 3092 | 0.000000 | 0.000000 | 29.255553 | 45.267769 | [2] |
| 3200 | 46.1792 | 3142 | 0.000000 | 0.000000 | 29.510897 | 45.572841 | [2] |
| 3300 | 46.1792 | 3259 | 0.000000 | 0.000000 | 29.438026 | 45.485900 | [2] |
| 3400 | 46.1792 | 3376 | 0.000000 | 0.000000 | 29.343896 | 45.373453 | [2] |
| 3500 | 46.1792 | 3493 | 0.000000 | 0.000000 | 29.496872 | 45.556115 | [2] |
| 3600 | 46.1792 | 3560 | 0.000000 | 0.000000 | 29.239554 | 45.248615 | [2] |
| 3700 | 46.1792 | 3677 | 0.000000 | 0.000000 | 29.161915 | 45.155595 | [2] |
| 3800 | 46.1792 | 3794 | 0.000000 | 0.000000 | 29.025646 | 44.992056 | [2] |
| 3900 | 46.1792 | 3844 | 0.000000 | 0.000000 | 28.855939 | 44.787907 | [2, 5] |
| 4000 | 46.1792 | 3961 | 0.000000 | 0.000000 | 28.982784 | 44.940546 | [2, 5] |
| 4100 | 46.1792 | 4078 | 0.000000 | 0.000000 | 29.125206 | 45.111573 | [2, 5] |
| 4200 | 46.1792 | 4195 | 0.000000 | 0.000000 | 29.232011 | 45.239583 | [2, 5] |
| 4300 | 46.1792 | 4262 | 0.000000 | 0.000000 | 29.308283 | 45.330867 | [2, 5] |
| 4400 | 46.1792 | 4379 | 0.000000 | 0.000000 | 29.171485 | 45.167067 | [2, 5] |
| 4500 | 46.1792 | 4496 | 0.000000 | 0.000000 | 29.443875 | 45.492883 | [2, 5] |
| 4600 | 46.1792 | 4546 | 0.000000 | 0.000000 | 29.503234 | 45.563702 | [2, 5] |
| 4700 | 46.1792 | 4663 | 0.000000 | 0.000000 | 29.546485 | 45.615263 | [2, 5] |
| 4800 | 46.1792 | 4780 | 0.000000 | 0.000000 | 29.664889 | 45.756241 | [2, 5] |
| 4900 | 46.1792 | 4897 | 0.000000 | 0.000000 | 29.721240 | 45.823244 | [2, 5] |
| 5000 | 46.1792 | 4964 | 0.000000 | 0.000000 | 29.806593 | 45.924621 | [2, 5] |


## Verdict

1. Liver information is present in the failing cluster assignments but diluted by background. Majority mapping hides it completely because no cluster is>50% liver. Oracle subsets recover29.81–32.85% IoU, below43.75–44.40% for seed0/best600: mapping threshold AND poorer organ specificity contribute.

2. Seeds learn different partitions, not the same partition with renamed IDs: permutation-invariant ARIs seed0–seed1=0.291 and seed0–seed2=0.266. Shared frozen visual weights are identical. Attribution between learned code geometry and probe partition changes remains UNVERIFIED without further label-free latent-space analysis.

3. A liver-mapped cluster survives at saved600, disappears by logged635, and never returns at subsequent logged checks. Representation variance/diversity stays healthy; liver-specific clustering degrades into mixed tissue groups. Exact within-interval timing and transient changes are UNVERIFIED.

4. The correspondence objective is lower at final in all16 fixed train batches while liver-specific clustering is worse. This is evidence that minimizing the current correspondence objective does not preserve liver structure. It is not proof that any single shift, weight, resolution, dropout or cluster initialization caused the failure.


## UNVERIFIED and bounds

- All labels are validation/diagnosis only, except no train labels are opened at all in Task4. Optimal/threshold mappings and checkpoint selection use the same472 slices and are optimistic. Patient-level uncertainty and generalization beyond4 validation patients were not tested.
- ARI/NMI describe pixel partitions, not a full latent-code equivalence test. High background prevalence influences global partition agreement.
- Dropout-on objective, gradient directions, causal ablations, head/probe-specific attribution, and alternative losses/thresholds as actual learned solutions were not tested. No training performed.
- DIAG old/fresh feature max-absolute discrepancy0.0021514892578125 remains unexplained; both DIAG100 K8 oracle scores reproduce exactly.
- No intermediate weights exist between the saved100-batch checkpoints; best unsaved518 cannot be evaluated as a saved artifact. No retraining filled gaps.


## Runtime and audit provenance

Completed producer runtimes: Task1 95.937551250s; Task2 7.573356292s; Task3 3.057226750s; Task4 38.411236584s; backbone check 5.055767625s. Task5 chunks: 85.751373959s, 88.635870542s, 85.776239417s, 84.744551333s, 81.957526417s. All below570s.

Two diagnostic implementation errors were corrected only in scratch: initial Task1 JSON encoding of NumPy IDs; initial W&B scanner requested writable access to original and was blocked before open by the read-only guard. Successful retries are task1_verified.log and task3_verified.log. No source/data/cache/checkpoint edits occurred. All scripts/results under `audit/scratch/forensics/`; this report is generated by report.py. The 150-tensor equality test and source file paths are in backbone_check.json; raw integer confusion matrices make every score independently recomputable.

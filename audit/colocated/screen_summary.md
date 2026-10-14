
### Amended Task 2: completed screen

Producers: `experiments/stego_colocated/train.py` and `run_round.py screen`; raw per-batch logs, initial monitors, per-run timing, overlap snapshots, complete.json and final checkpoints are in each arm’s `stage1/`. Monitor std is the mean per-channel spatial std of L2-normalized codes; cosine averages off-diagonal dot products of per-image mean normalized patch codes. Cluster usage comes from fixed train-only K8 monitor clustering, not the Stage-A fit. All numbers below are label-free.

| Arm | Seed | Batch | Code std | Cross-image cosine | Usage C0–C7 (%) | Healthy |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| C448 | 0 | 0 | 0.04131734 | 0.86023968 | 17.07, 10.24, 23.13, 5.72, 14.05, 5.66, 9.59, 14.54 | True |
| C448 | 0 | 50 | 0.11165995 | 0.03253857 | 6.42, 4.93, 17.94, 27.95, 7.34, 8.74, 14.40, 12.28 | True |
| C448 | 0 | 100 | 0.11276505 | 0.02706367 | 6.37, 15.34, 14.28, 15.84, 13.54, 6.39, 12.97, 15.27 | True |
| C448 | 0 | 150 | 0.11372218 | 0.02431655 | 6.23, 18.70, 13.68, 14.57, 13.46, 7.05, 20.34, 5.96 | True |
| C448 | 1 | 0 | 0.04575163 | 0.82948053 | 16.21, 28.40, 4.50, 5.68, 18.10, 5.96, 4.93, 16.24 | True |
| C448 | 1 | 50 | 0.11280607 | 0.02809900 | 7.02, 19.00, 12.10, 6.44, 26.82, 14.65, 5.16, 8.82 | True |
| C448 | 1 | 100 | 0.11369527 | 0.02513381 | 12.36, 15.09, 15.00, 6.35, 13.46, 15.13, 16.32, 6.30 | True |
| C448 | 1 | 150 | 0.11433984 | 0.02185925 | 6.24, 14.29, 13.93, 16.17, 18.36, 12.40, 5.84, 12.78 | True |
| C448 | 2 | 0 | 0.05203552 | 0.78031081 | 5.96, 10.28, 15.43, 21.72, 5.58, 6.29, 19.06, 15.69 | True |
| C448 | 2 | 50 | 0.11202689 | 0.03284555 | 6.51, 27.57, 18.15, 11.81, 9.41, 14.34, 5.12, 7.10 | True |
| C448 | 2 | 100 | 0.11315519 | 0.02750064 | 13.72, 15.73, 16.01, 14.36, 6.25, 6.39, 14.72, 12.83 | True |
| C448 | 2 | 150 | 0.11353809 | 0.02641634 | 6.23, 13.98, 14.33, 11.89, 16.22, 5.94, 13.13, 18.29 | True |
| P1 | 0 | 0 | 0.04131734 | 0.86023968 | 17.07, 10.24, 23.13, 5.72, 14.05, 5.66, 9.59, 14.54 | True |
| P1 | 0 | 50 | 0.11093386 | 0.03492450 | 6.42, 14.16, 17.91, 27.96, 12.25, 8.78, 7.28, 5.25 | True |
| P1 | 0 | 100 | 0.11244816 | 0.02438963 | 12.99, 15.08, 15.01, 14.03, 13.68, 6.45, 16.41, 6.35 | True |
| P1 | 0 | 150 | 0.11350754 | 0.02288040 | 6.23, 13.67, 14.29, 13.19, 5.98, 19.10, 15.79, 11.74 | True |
| P1 | 1 | 0 | 0.04575163 | 0.82948053 | 16.21, 28.40, 4.50, 5.68, 18.10, 5.96, 4.93, 16.24 | True |
| P1 | 1 | 50 | 0.11229265 | 0.02827542 | 12.03, 26.99, 6.45, 5.33, 8.81, 14.41, 18.87, 7.13 | True |
| P1 | 1 | 100 | 0.11366793 | 0.02266041 | 12.25, 16.38, 6.35, 14.49, 11.22, 15.22, 8.52, 15.57 | True |
| P1 | 1 | 150 | 0.11412589 | 0.02552684 | 6.24, 16.53, 18.82, 14.17, 12.70, 13.69, 5.91, 11.95 | True |
| P1 | 2 | 0 | 0.05203552 | 0.78031081 | 5.96, 10.28, 15.43, 21.72, 5.58, 6.29, 19.06, 15.69 | True |
| P1 | 2 | 50 | 0.11153311 | 0.03286690 | 6.52, 18.17, 27.68, 7.07, 5.37, 9.41, 13.89, 11.89 | True |
| P1 | 2 | 100 | 0.11251824 | 0.03222046 | 6.43, 9.05, 26.27, 19.89, 12.52, 14.05, 6.30, 5.49 | True |
| P1 | 2 | 150 | 0.11317246 | 0.02894565 | 11.62, 14.10, 18.35, 12.29, 14.31, 5.99, 6.25, 17.09 | True |

| Arm | Seed | First 10 batches (s) | Projected 150 (s) | Actual run (s) |
| --- | ---: | ---: | ---: | ---: |
| C448 | 0 | 10.63 | 159.41 | 158.55 |
| C448 | 1 | 10.23 | 153.40 | 155.72 |
| C448 | 2 | 11.17 | 167.62 | 156.83 |
| P1 | 0 | 10.28 | 154.13 | 155.94 |
| P1 | 1 | 10.31 | 154.59 | 156.35 |
| P1 | 2 | 10.19 | 152.80 | 157.92 |

| Arm | Seed 0 screen | Seed 1 screen | Seed 2 screen | Advances |
| --- | --- | --- | --- | --- |
| C448 | True | True | True | True |
| P1 | True | True | True | True |

Each seed must be healthy at 50, 100 and 150 AND have batch-150 cosine below its own initialization. Arm decisions are the conjunction of all three seeds. State serialization passed an exact synthetic next-update replay; actual runs started fresh and did not resume. Overlap checks ran before each seed and the same package lock prevented overlapping trainers. Validation labels remain closed.

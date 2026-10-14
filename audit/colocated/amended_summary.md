
### Amended Task 1 results

Producer: `audit/colocated/amended_calibration.py` → `amended_calibration.json`; source/sample protocol hashes and annotation-access ledger are retained. The original 20% shift and losses reproduced within 1e-10. Partner values are diagnostic only. Full means are weighted by anchor count. No segmentation scores or annotation files were read.

| Target | Shift | Actual attraction % | Partner constant / random / K8 | Full constant / random / K8 | Full K8 advantage | Batches passed | Admissible |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| 20% | 1.041011333465576 | 19.999157 | 0.059268037 / 0.002877948 / 0.053947838 | 0.319384499 / 0.013892496 / 0.014990836 | -0.001098340 | 3/13 | False |
| 40% | 0.977739393711090 | 39.998313 | -0.004003962 / -0.000144805 / -0.006047329 | 0.303566499 / 0.013136808 / -0.000007955 | 0.013144763 | 13/13 | True |
| 60% | 0.939586400985718 | 59.997470 | -0.042156989 / -0.001967525 / -0.042224428 | 0.294028245 / 0.012681128 / -0.009052230 | 0.021733358 | 13/13 | True |
| 80% | 0.906445264816284 | 79.996627 | -0.075298159 / -0.003550809 / -0.073649227 | 0.285742956 / 0.012285307 / -0.016908430 | 0.029193736 | 13/13 | True |
| 90% | 0.886966705322266 | 89.998313 | -0.094776734 / -0.004481377 / -0.092119020 | 0.280873313 / 0.012052665 / -0.021525878 | 0.033578542 | 13/13 | True |

| Batch (slices) | 20% | 40% | 60% | 80% | 90% |
| --- | --- | --- | --- | --- | --- |
| 1 (16) | PASS | PASS | PASS | PASS | PASS |
| 2 (16) | FAIL | PASS | PASS | PASS | PASS |
| 3 (16) | FAIL | PASS | PASS | PASS | PASS |
| 4 (16) | FAIL | PASS | PASS | PASS | PASS |
| 5 (16) | FAIL | PASS | PASS | PASS | PASS |
| 6 (16) | FAIL | PASS | PASS | PASS | PASS |
| 7 (16) | FAIL | PASS | PASS | PASS | PASS |
| 8 (16) | FAIL | PASS | PASS | PASS | PASS |
| 9 (16) | PASS | PASS | PASS | PASS | PASS |
| 10 (16) | PASS | PASS | PASS | PASS | PASS |
| 11 (16) | FAIL | PASS | PASS | PASS | PASS |
| 12 (16) | FAIL | PASS | PASS | PASS | PASS |
| 13 (8) | FAIL | PASS | PASS | PASS | PASS |

**Selected P1 target: 90%; shift 0.8869667053222656. Selection used only label-free full losses.** Self/random stay 0.794418/0.797651; C448 partner stays 0.785981. This passes the amended gate; the original partner-only failure remains recorded above.

Preparation reuses the prior bounded runner in a new output package, with only P1 partner dispatch, candidate identity and requested config metadata changed. The copied Stage-A entry gate is generalized from D448 to its configured arm. No previous experiment is modified. `runner_reuse.json` records source copies. Optimizer/RNG serialization is verified by `experiments/stego_colocated/check_state.py` on P1 synthetic tensors: exact next-update equality for model, Adam and RNG state, without dataset reads. Production runs cannot resume; interrupted runs require a fresh attempt.

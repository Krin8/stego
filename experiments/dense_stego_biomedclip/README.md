# Dense frozen-BiomedCLIP experiment

Implementation is isolated from the existing baseline. **Training has not started.** `configs/proposed.json` remains unlocked pending the user's budget/decoder choice and resolution of suspended existing training.

Start with ARCHITECTURE.md, TRAINING_PROTOCOL.md and LABEL_USAGE.md. RESULTS.md distinguishes completed sanity checks from results that do not yet exist.

Code:
- runtime.py: offline CPU runtime, source-function reuse, label/write guards.
- model.py: projection armA and learned dense decoder armB.
- correspondence.py: known-geometric positives, spatially separated negatives, bidirectional InfoNCE.
- train.py: train16 only, frozen backbone, objective-based checkpoint selection.
- stages.py: train-only k-means/naming, validation inference, actual arrays and exact replay.
- seal_predictions.py: seal all six prediction sets before labels.
- evaluate.py: scoring, diagnostics and figures from saved arrays.
- report.py: all configurations, matched baseline comparison and seed stability.
- run_suite.py: sequential controller with overlap checks and stop-on-failure behavior.

Completed checks:
- sanity.py: actual train-image two-view forward/backward, zero backbone gradients, no optimizer steps; baseline feature equality.
- integration_checks.py: synthetic geometric/loss tests, exact inheritedF1 parity, dense output capacity, deliberate forbidden-label-open rejection.
- preflight.py: parse checks, synthetic checkpoint replay and installed k-means/distance parity.

No notebook or command needs to modify original source, baseline artifacts, datasets, caches or checkpoints. The synthetic checkpoint in logs/synthetic_checkpoint.pt is an untrained serialization test, not an experimental result.

Pre-scoring implementation repairs: the reused baseline distance formula's hard-coded768 was generalized to the centre feature dimension; late AST-loaded helpers were explicitly rebound in test/evaluation code. Initial failure logs are retained. No labels or optimizer updates were involved. Detailed results of the passing reruns are in logs/.

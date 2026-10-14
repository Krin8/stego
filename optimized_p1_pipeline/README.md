# Optimized P1 pipeline — extracted files

Selected method: **seed 0, nonlinear head branch only, no head-output L2, K=8, N2 naming, guided size 25 / epsilon 0.01, threshold 0.30, largest component, hole filling, closing disk radius 3**. This collection preserves the existing method exactly; it does not fix or endorse its deviations from the original trained head.

## Contents

- `01_evaluation/`: actual optimized evaluator and its matching visualization script.
- `02_support/`: exact runtime, geometry, preprocessing and split dependencies. Their original relative hierarchy is retained within this section to disambiguate same-named files.
- `03_checkpoint/seed0_final.pt`: byte-identical copy of the checkpoint used by the evaluator.
- `04_metadata/`: feature provenance, sample order/crop geometry, split records and checkpoint training settings.
- `05_shared_inputs/`: links to the exact feature/scoring arrays, CT image files and train DICOM files required by the evaluator. No entire dataset, unrelated modality or other experiment is included. Large arrays are not duplicated.
- `06_provenance/`: original head/loss and hash-matched historical training source. These establish where the checkpoint came from; they are not the optimized inference implementation and must not be run using their descriptive copied filenames.

`FILE_MANIFEST.csv` gives every included file's role and original location. Copied-file SHA-256 values are verified in `VERIFICATION.json`. Linked data are existence/target checked, not fully rehashed.

## Important execution boundary

**This is a curated extraction, not a portable standalone package.** Code contents were not changed. The evaluator hardcodes `/Users/navneetbavineni/STEGO` and imports support files from their original locations. Runtime modules also derive paths from their own locations. Running the copied evaluator on this machine still uses those original support paths and refits k-means. The visualization script still writes to its original output path. No evaluator, refit or visualization generation was run during extraction.

Original files have therefore been retained. Do not delete or relocate the original project based on this collection. A truly standalone relocation requires explicit path changes and verification; symlinks alone do not preserve `Path(__file__).resolve()` behavior. Dataset/cache links also require their original targets.

## What is deliberately excluded

Other seeds, other experiment arms, the frozen-feature ensemble evaluator, broken N3 re-naming command, original whole-project audit reports, unrelated diagnostics, Python environments, Git history, downloads, and duplicate exploratory variants.

The optimized evaluator reads cached frozen features; it does not call the backbone. Therefore the backbone weight file, full feature-extraction research stack and training runners are not required to execute this cached-input evaluator and are not included. Rebuilding features or retraining from raw CT is outside this extraction.

## Result attribution

The checkpoint was trained for **500 batches at LR 0.0005**, with 448 input / a direct 28×28 grid. The optimized script discards `cluster1` through a non-strict load and uses only `cluster2`. The original full-head F1 results do not establish this optimized variant's score. No saved 83.92% result, optimized centres or prediction arrays were located by the preceding review; do not treat this extraction as verification of that number.

The settings have not been corrected or tuned, and no code content has been modified.

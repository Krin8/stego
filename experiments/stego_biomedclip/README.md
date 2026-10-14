# Frozen BiomedCLIP + actual STEGO correspondence experiment

All new files live here. Existing src, baseline, naming, F1, datasets, caches and checkpoints are not edited. Pinned reference sources are local under reference/; runtime is offline, CPU only, no installation required.

The user fixed seeds0/1/2,5000 batches each, original three losses, alignment off. See FIDELITY.md for exact equations/config and every adaptation; LABEL_USAGE.md for the label boundary. The70-D learned grid stays28×28.

Order (from project root):

```sh
python3 -B experiments/stego_biomedclip/checks.py
python3 -B experiments/stego_biomedclip/integration_checks.py
python3 -B experiments/stego_biomedclip/preflight.py
python3 -B experiments/stego_biomedclip/run_suite.py
```

Preflight and suite refuse to overwrite existing sealed config/runs. Do not rerun a completed suite in the same directory. Before EACH seed the suite checks process overlap; any failure stops the sequence. Logs go under logs/, selected/final head weights under checkpoints/seedN/. Predictions/seedN/ stores float32 normalized projected features, raw28-grid and native512 cluster assignments, CTs, and each N1/N2/N3 hard/F1 binary array. Native cluster ID8 denotes outside-crop background, not a ninth fitted cluster. Metadata records per-slice image geometry and checkpoint hashes.

metrics/results.json holds per-patient pooled and mean-slice IoU/Dice at224 and512, tiny-liver diagnostics and seed mask/representation stability. metrics/comparison.csv contains matched baseline and STEGO rows for every seed/rule; metrics/across_seeds.json uses sample SD across3 seed patient means. Primary visualization is predeclared seed0/N3, with all other seeds/rules also saved; no winning seed is chosen.

No metrics or prediction figures are valid until the suite finishes. Current execution state is recorded under logs/. Numerical parity checks do not establish semantic transfer or segmentation quality.

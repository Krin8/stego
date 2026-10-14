# Locked refinement protocol

Only F1 and F2; neither selected by score. Reuse IMPROVE B2-448 K8 seeds0/1/2 hard grids, soft weights, temperature and inherited other19-patient mappings. No forward pass, k-means refitting, model loading, or parameter tuning.

F1: cached soft assignment → original guided filter (native radius8, epsilon0.01) → >=0.5 → largest8-connected 2D component → per-slice binary hole fill (4-connected background) → binary closing with native disk x²+y²<=9. Closing uses three pixels of zero padding, then crops back to512. F2: F1 → binary median along physically sorted same-patient slices, window(3,1,1), nearest-endpoint replication. No cross-patient mixing or physical spacing resampling.

References: B2-448 hard and original B5-2D (largest component of HARD assignments, as in IMPROVE). F1's largest-component stage instead follows guided soft predictions, as explicitly requested.

Primary224 scoring uses the same nearest resize from native512; native512 supplementary. All20 patients/2341 slices, seeds0/1/2. Average seeds within patient, then mean/sample SD/min/max across patients. Both inherited strict-majority and optimal-subset mappings are checked; for this saved parent their cluster subsets are identical across rules and resolutions. No remapping after refinement. F1/F2 exploratory due to historical label-based parent selection.

Diagnostics: native pixels. Empty-GT slices: total/mean FP pixels per slice and count of slices with any FP, separately by seed/patient and seed-average (counts averaged over seeds explicitly labelled). Multi-component GT:8-connectivity, no area cutoff; compare captured GT immediately before and after each largest-component step, recording removed true-positive pixels and previously hit components entirely lost. Tiny annotation islands are not filtered from this analysis.

Boundary distances: symmetric 2D surface-pixel Euclidean distances, in native pixels, pooled over all slices within each patient; surface = foreground minus4-neighbour erosion. HD95 =95th percentile of pooled bidirectional distances; mean surface distance =their arithmetic mean. No3D/mm interpretation. Both-empty slices add no boundary points and are counted; one-empty slices contribute infinity for each unmatched surface point (reported explicitly, never silently dropped). Also retain finite-distance-only summaries to distinguish boundary errors from missed/extra objects; these are conditional diagnostics.

Figures: same8 indices, plus3 worst B5-2D native per-slice IoUs using fixed seed0 across all2341 slices (ascending IoU, index tie-break). Both-empty union has undefined IoU and is excluded from ranking, counted explicitly. Selection uses labels. Seed0 fixed for all11 rows; no best-seed selection. PNG >=1600px wide and PDF. Red truth, green prediction.

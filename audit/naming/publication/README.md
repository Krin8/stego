# Final results package

Start with FINAL_RESULTS.html or FINAL_RESULTS.md. Full per-configuration/per-patient tables are in SUPPLEMENTARY_TABLES.md and the CSVs. Figure PNGs and vector PDFs are supplied. Sources and environment versions are recorded in provenance.json. No training or fitting was performed.

Execution order used: audit_record.py; reproduce.py (six sequential StageB evaluations); stability.py (label-free mask comparison); report_results.py; finalize.py. The reproduction script deliberately refuses to overwrite an existing output. Do not delete or overwrite original StageA files to repeat the experiment. report_results.py and finalize.py only regenerate this publication folder's reporting outputs.

The historical leakage audit is evidence-based, not an independent OS-level attestation. See FINAL_RESULTS.md for NOT VERIFIED items.

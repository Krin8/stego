# Execution notes

D2 first loaded and passed its forward checks, but serializing the loader diagnostics failed because the installed transformers returned empty sets. This was a logging failure, not a weight-load failure. The audit-only JSON writer now converts sets to sorted lists; the identical model/configuration was rerun successfully. Original evidence: D2_extract.log and D2_logging_retry.json. Successful evidence: D2_extract_verified.log and D2_checks.json. No weights, preprocessing values or model parameters were changed.

Both safetensors and the D2 original binary matched the exact supplied SHA256 digests before loading. The original binary was hashed only. All model construction used local installed transformers classes, local_files_only=True, trust_remote_code=False, use_safetensors=True; no torch.hub or package installation. The read/write guard rejects network connections and writes outside audit/dino2, including any write into its pre-existing weights directory.

# Verification notes

The first direct CLI test failed offline because redirecting XDG_CACHE_HOME also redirected the default Hugging Face lookup away from the existing local cache. The package now preserves the existing HF_HOME before redirecting writable plotting caches into this package. No network requests, installs, pretrained-weight changes or pipeline-parameter changes were needed.

Original failure evidence: `verify_cli.log`, `cli_test/check_seed0.stderr`. Successful rerun: `verify_cli_retry.log`, `cli_test_verified/`, `cli_verification.json`. All three seeds pass the five-slice check. `final_verification.json` records exact function provenance, artifact digests, optional F2 equality, and no tracked source diff.

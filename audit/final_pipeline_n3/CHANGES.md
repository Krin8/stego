# Changes for Label-Free Naming Package (N3)

## Task 1: Label-Free Naming
- Evaluated N3 naming on the 3 saved K=8 (uncentered) model fits in the package.
- Generated `metadata.json` mapping for each fit containing `liver_cluster`, `n1_cluster`, `n2_cluster`, `abstention_status`, margins, and hashes.
- Verified lack of label/mask access during naming via file-access ledger.

## Task 2: Package Changes
- Modified `segment_ct.py` to read `liver_cluster` dynamically from the new `metadata.json` instead of a static mapping.
- Removed all legacy mapping logic and mapping files from the package.
- Added a `--name-clusters` argument to regenerate `metadata.json` for custom fits using purely label-free naming.
- Updated `README.md` to document the label-free guarantee and metadata structure.

## Task 3: Verification
- Same-cluster check: Verified that the cluster selected by N3 matches the label-derived optimal cluster exactly for all seeds.
- Pipeline equivalence: Confirmed pixel-by-pixel equivalence of the final predictions between the old and new packages on the validation set.

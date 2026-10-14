# STEGO project navigation

This is a descriptive, role-based view of the existing project. The entries are **relative symbolic links**, not duplicate datasets or renamed implementations. All original filenames and contents remain unchanged.

## Start here

- [Pipeline routes and review](01_pipeline/)
- [Core source, configuration and tests](02_core_source/)
- [Training experiments](03_experiments/)
- [Diagnostics and comparisons](04_diagnostics/)
- [Reports and decisions](05_reports/)
- [Results and logs](06_results_and_logs/)
- [Datasets and weights](07_data_and_weights/)
- [Historical maintenance scripts](08_maintenance/)
- [Infrastructure](09_infrastructure/)
- [Complete original tree](10_original_tree/)

[ALL_FILES.csv](ALL_FILES.csv) maps every inventoried original file to this view. [LINKS.csv](LINKS.csv) gives the exact original target for every descriptive alias. [VERIFICATION.json](VERIFICATION.json) records link and content-preservation checks.

## Keep the pipeline variants distinct

The saved P1 experiment, branch-only optimized P1 evaluator, and frozen-feature ensemble are separate methods. Organization does not resolve the implementation discrepancies in the [review](01_pipeline/06_implementation_review/REVIEW.md).

## How to use the links

Browse and inspect through these folders. Execute scripts using their **original paths and documented working directories**: some scripts derive artifact paths from `__file__` or the working directory, so executing a renamed link is not an equivalent command. Editing a linked file edits its original target. Deleting a link itself does not delete its target. Do not recursively copy this view into the project or upload it with links dereferenced: data and checkpoints are intentionally not copied.

The complete original-tree view includes Git metadata and the local environment for completeness; neither belongs in a code-only artifact upload. New project files added after this inventory need a refreshed inventory. No source refactor, data movement, cleanup, training or installation was performed.

Read the [file-by-file use-case guide](FILE_GUIDE.md) for explanations of the source files, configurations, reports and generated artifacts.

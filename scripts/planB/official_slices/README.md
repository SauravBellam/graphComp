# Official Plan B slices (fixed Oct 12, 2026; never edited afterwards)

`dev_slice.txt` (24) and `holdout_slice.txt` (24) are the slices of record for every Plan B run, on Kaggle and on the VM.
The VM runner (`scripts/planB/vm/dev_runs.py`) and the Kaggle notebooks read them from here by default.

How they were drawn: `scripts/planB/make_slices.py`, seed 20261012, size 24, stratified over repo, cell (d x v),
gold_has_async and gold_is_entry_point, eligible = A2-usable and valid. Validity = the CPU B1 run on Kaggle
(`planB_results/b1/kaggle_subprocess_cpu/`: gold and empty passes x 2, import check), with one correction: the 34 fastapi
tasks whose gold patch failed **only** because the sandbox lacked `inline_snapshot` / `dirty_equals` (pytest exit 2, no
other missing module; listed in `provisional_tasks.json`) count as provisionally valid, since every Plan B run now
installs those packages. 77 tasks were eligible (fastapi 44, rich 33); requests and httpx are excluded (their tests import
the installed copy, not the task's code).

Validity is still measured in each environment (VM: `dev_runs.py` step 1; Kaggle: notebook 01). A slice task that is not
valid in an environment keeps its place in the slice and is excluded from that environment's resolution rates; the slices
themselves are not redrawn. `profile.csv` is the profile with this draw's `sub_valid` and `sub_reason` columns;
`slices_meta.json` holds the rule, counts and balance tables.

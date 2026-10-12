# Non-usable A2 tasks: 38 of 129

Plan A loose end 1.4 (`HANDOFF_planBC.md`). Produced Oct 11, 2026 on the laptop by
`make_nonusable_table.py` in this folder, run from `planA_results/vm-a2/results/`. It reads `a2_v3/a2_tasks.csv` (125 rows) and
`a2_v3/a2_run_status.csv` (129 rows). The four requests `no_tests` tasks are only in the second file,
which is why the handoff's one-liner over `a2_tasks.csv` alone prints 34. Neither input was modified.

| File | Contents |
| --- | --- |
| `a2_nonusable.csv` | one row per non-usable task: run class (pre-fix), post-fix run class, expected and confirmed fail-to-pass counts, reason, source file |
| `make_nonusable_table.py` | the exact code that produced the CSV |
| `make_nonusable_table.out` | its printed counts |

Input sha256: `a2_tasks.csv` 360ac14e…4365, `a2_run_status.csv` 0f7c9644…1118.

## Counts

| Class | Tasks |
| --- | --- |
| clean run, no fail-to-pass test expected | 20 |
| clean run, expected fail-to-pass not confirmed (fastapi_14851, rich_3006, rich_4079) | 3 |
| pre-fix pytest collection error | 9 |
| pre-fix import error | 2 |
| requests, no collectable tests | 4 |
| **total** | **38** |

Usable: 91 (fastapi 45, rich 39, requests 7), matching the paper.

## Correction to the paper's Limitations paragraph

The paper lists rich_4006 as killed and counts 22 clean runs without a confirmed fail-to-pass test.
In `a2_v3`, rich_4006 was recovered (README section 6) and is usable. The 38 are 23 clean runs,
9 collection errors, 2 import errors and 4 requests tasks without collectable tests. The 23rd clean
run is httpx_3672 (encode/httpx). Fix the paper, not the table.

## Table

| instance_id | repo | run_class | post_run_class | f2p expected | f2p confirmed | reason | source |
|---|---|---|---|---|---|---|---|
| fastapi_13207 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| fastapi_13713 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| fastapi_13786 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| fastapi_14077 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| fastapi_14371 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| fastapi_14448 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| fastapi_14458 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| fastapi_14479 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| fastapi_14487 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| fastapi_14492 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| fastapi_14791 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| fastapi_14851 | fastapi/fastapi | clean | clean | 3 | 0 | clean run, expected fail-to-pass not confirmed | a2_tasks.csv |
| fastapi_15023 | fastapi/fastapi | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| httpx_3672 | encode/httpx | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| requests_7309 | psf/requests | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| rich_3006 | Textualize/rich | clean | clean | 1 | 0 | clean run, expected fail-to-pass not confirmed | a2_tasks.csv |
| rich_3043 | Textualize/rich | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| rich_3067 | Textualize/rich | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| rich_3105 | Textualize/rich | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| rich_3454 | Textualize/rich | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| rich_3942 | Textualize/rich | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| rich_4070 | Textualize/rich | clean | clean | 0 | 0 | clean run, no fail-to-pass test expected | a2_tasks.csv |
| rich_4079 | Textualize/rich | clean | clean | 1 | 0 | clean run, expected fail-to-pass not confirmed | a2_tasks.csv |
| fastapi_14186 | fastapi/fastapi | collection_error | clean | 0 | 0 | pre-fix pytest collection error | a2_tasks.csv |
| fastapi_14262 | fastapi/fastapi | collection_error | clean | 8 | 0 | pre-fix pytest collection error | a2_tasks.csv |
| fastapi_14605 | fastapi/fastapi | collection_error | clean | 0 | 0 | pre-fix pytest collection error | a2_tasks.csv |
| fastapi_14609 | fastapi/fastapi | collection_error | clean | 1 | 0 | pre-fix pytest collection error | a2_tasks.csv |
| fastapi_14978 | fastapi/fastapi | collection_error | clean | 22 | 0 | pre-fix pytest collection error | a2_tasks.csv |
| fastapi_15030 | fastapi/fastapi | collection_error | clean | 25 | 0 | pre-fix pytest collection error | a2_tasks.csv |
| fastapi_15745 | fastapi/fastapi | collection_error | clean | 31 | 0 | pre-fix pytest collection error | a2_tasks.csv |
| fastapi_15785 | fastapi/fastapi | collection_error | clean | 5 | 0 | pre-fix pytest collection error | a2_tasks.csv |
| requests_6757 | psf/requests | collection_error | clean | 3 | 0 | pre-fix pytest collection error | a2_tasks.csv |
| fastapi_15661 | fastapi/fastapi | import_error | clean | 14 | 0 | pre-fix import error | a2_tasks.csv |
| rich_3930 | Textualize/rich | import_error | clean | 12 | 0 | pre-fix import error | a2_tasks.csv |
| requests_6589 | psf/requests | no_tests | no_tests | - | - | no collectable tests | a2_run_status.csv only |
| requests_6592 | psf/requests | no_tests | no_tests | - | - | no collectable tests | a2_run_status.csv only |
| requests_6629 | psf/requests | no_tests | no_tests | - | - | no collectable tests | a2_run_status.csv only |
| requests_6644 | psf/requests | no_tests | no_tests | - | - | no collectable tests | a2_run_status.csv only |

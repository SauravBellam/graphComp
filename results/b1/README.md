# B1: task profile and slices

`profile.csv`: one row per task (129). Built Oct 12 from Plan A outputs and the B0 Docker gold pass.

| Column | Source and rule |
| --- | --- |
| repo, cell, multi_file, n_anchors | `vm-a2/results/a6_v3/task_profile.csv` (cell = d x v) |
| a2_usable | `a2_v3/a2_tasks.csv` `usable` (91 True). The 4 requests tasks absent from a2_tasks (6589, 6592, 6629, 6644: no collectable tests) are False |
| gold_has_async | any gold symbol in `a1_v2/a1_gold_symbols.jsonl` with `kind=async_function` (22 tasks; 21 fastapi, 1 httpx) |
| gold_has_async_absent | same, restricted to `present_in_graph=False` (21 tasks); the handoff's alternative rule, kept for comparison |
| gold_is_entry_point | `a2_tasks.csv` `gold_executed_is_entry` > 0 (40 tasks; **32 of the 72 A2-usable tasks with executed in-graph gold**, the paper's figure). Blank where A2 has no row |
| entry_hops | `a2_tasks.csv` `gold_min_hops_from_entry_directed` (0 to 7), `none` where undefined |
| a2_gold_passes_post | `a2_tasks.csv` `harness_all_expected_pass_post`: gold patch makes all expected tests pass in A2's own venvs (repo code tested, network on) |
| b0_docker_* | `planB_results/b0/b0_gold/gold_patch_resolution.csv`, swegemma 0.2.11 Phase 2 in the public Docker sandbox. **Environment caveat (B0.md section 3.3): that sandbox tests released requests/fastapi wheels, not the repo, so these flags are environment artifacts for those repos.** valid = empty fails and gold passes (39 tasks: 38 rich, 1 fastapi) |

Slices: NOT YET WRITTEN. A2-usable AND b0_docker_valid = 31 tasks, all rich; two disjoint
three-repo-stratified 24-task slices are not possible on that flag. The gold-resolving definition for
the slices is pending a decision (see the session notes / B0.md section 3.3).

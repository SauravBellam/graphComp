# Local recount, Oct 11, 2026 (not the VM rerun)

Produced on the analysis machine from files already in this repository, with the fixed scripts in
`scripts/` (commit after "Fix Plan A scripts"). The competition archive is not here, so:

- **Provided graphs were rebuilt** from `vm-a2/results/a6/graphs_v2/` (nodes with `in_provided_graph`
  or kind `provided_only`; edges of type `calls_static`). They lack the containment pseudo-edges of
  the real files, which `analyze.py` strips before edge comparison but which the reachability BFS
  can traverse. Entry-distance and reachability numbers are therefore approximate (if anything,
  slightly pessimistic); edge recall/precision are unaffected.
- **Gold symbols are the Oct 10 A1 output** (`<locals>` ids), flattened on the fly by `analyze.py`.
- **Test outcomes, run classes and discriminativeness are exact**: they come from the real traces in
  `vm-a2/results/a2_runs/`.

| Folder | Command |
| --- | --- |
| `a2_v2/` | `a2/analyze.py --data <stand-in archive> --runs vm-a2/results/a2_runs --a1 <a1_tasks.csv + a1_gold_symbols.jsonl> --out a2_v2` |
| `a8/` | `a8_metrics.py --data <stand-in> --a1 <same> --a2-analysis a2_v2 --runs vm-a2/results/a2_runs --out a8` |

`ops/rerun_fixes.sh` reproduces both on the VM against the real archive, into `<results>/a2_v2` and `<results>/a8`.

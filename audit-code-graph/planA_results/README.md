# Plan A results: run of 2026-10-10

Raw outputs of every Plan A step, copied from the two GCP VMs, plus the scripts that produced
them. Nothing here has been interpreted or edited; result files are exactly as the scripts wrote them.

**Status: Plan A complete; Oct 11 follow-ups in section 6.** All steps ran and are downloaded: A1, A2 (+ rerun of 4 tasks), A2
analysis, A3 (vector statistics + encoder identification for all 3 candidate models), A4, A5 (modes
a0/a/b/d) and A6. A5 mode (c) was not run: it requires an encoder identified by A3, and
`identified_encoder` is `null` in both A3 summaries.

---

## 1. Folder layout

| Path | Contents |
|---|---|
| `scripts/` | All code used (see section 2). `scripts/planA/` = pipeline, `scripts/tests/` = synthetic test suite, `scripts/ops/` = VM setup and launch scripts, `scripts/REVIEW.md` = code review notes |
| `vm-main/results/a1/` | A1 outputs. Canonical files are the un-suffixed ones; `*_shardKof8*` are the 8 shard inputs to the merge. **In the GitHub copy, `a1_async_audit.csv` (145 MB, over GitHub's 100 MB limit) is stored as `a1_async_audit.csv.gz`; `gunzip -k` restores the original byte-for-byte.** |
| `vm-main/results/a3/` | A3 **vector statistics only** (`--stats-only`): `a3_vectors_stats.csv`, `a3_summary.json` |
| `vm-main/results/a4/` | A4 outputs |
| `vm-main/results/a5/` | A5 outputs (modes a0, a, b, d) |
| `vm-main/results/logs/`, `run_report.json`, `results_main.log` | logs and data manifest for the vm-main stages |
| `vm-a2/results/a2_runs/<task>/` | A2 per-task runs, 129 folders (current; 3 rich tasks are from the rerun, see section 4) |
| `vm-a2/results/a2_runs_killed/` | the **first** A2 attempt of rich_3480, rich_3772, rich_4006, rich_4070, kept as a record (pre-fix run killed, no pre trace) |
| `vm-a2/results/a2/` | A2 analysis (current, includes the rerun traces) |
| `vm-a2/results/a2_before_rerun/` | A2 analysis as it was before the rerun, kept as a record |
| `vm-a2/results/a6/` | A6 outputs (current); `graphs_v2/` holds 127 graphs |
| `vm-a2/results/a3/` | A3 full run for **embeddinggemma-300m** and **all-MiniLM-L6-v2**: `a3_per_model.csv`, `a3_summary.json`, `a3_vectors_stats.csv`. Its `a3_summary.json` also lists embeddinggemma-2 with a load error (missing `pillow`); that model was run separately, next row |
| `vm-a2/results/a3_gemma2/` | A3 full run for **embeddinggemma-2** only (same files) |
| `vm-a2/results/*.log` | `a2_full.log` (main A2 run), `a2_rerun4.log` (rerun), `a2_analyze.log`, `a6.log`, `a3.log`, `a3_gemma2.log` |
| `provenance/env_vm-*.txt` | host specs, OS, Python, numpy/scipy, sha256 of the code on each VM at download time and of `tasks.jsonl` |
| `provenance/scripts_sha256.txt` | sha256 of every file in `scripts/` (the final code) |
| `provenance/a3env_freeze_vm-a2.txt` | Python version and every package version in the A3 environment (torch 2.14.1+cpu, torchvision 0.29.1+cpu, transformers 5.19.0, sentence-transformers 6.1.0, pillow 12.3.0) |

A1 output also exists on vm-a2 (byte-identical copy used as A2-analysis/A6 input), and the A6
output from before the rerun (`a6_before_rerun/`, 630 MB) was left on vm-a2, not downloaded.

---

## 2. The scripts and what each one tests

All in `scripts/planA/` unless noted. Every step reads the competition archive
(`tasks.jsonl`, `graphs/`, `embeddings/`, `snapshots/`).

### Shared code
| Script | Role |
|---|---|
| `common.py` | Loaders for tasks, graphs, embeddings and snapshot `.tgz` files; a copy of the harness's 4-tier name resolver (`resolve_node_name`: exact → suffix → case-insensitive → substring); unified-diff parser; mapping of gold-patch lines to functions with `ast`; path→module rules (drops `src/` and `docs_src/` as the provided graph does); identifier extraction from issue text; BM25; BFS distances |
| `stats.py` | Wilson intervals (task-level rates), task-cluster bootstrap (pooled ratios), exact McNemar (paired comparisons) |
| `run_all.py` | Orchestrator: runs the steps below in dependency order, in parallel where independent, and writes `run_report.json` with a data manifest |

### Experiments
| Step | Script | What it tests | Main outputs |
|---|---|---|---|
| **A1** static census | `a1_census.py` | Is every function/class in the real source a node of the provided graph? Parses every `.py` file of each snapshot and checks presence by kind (sync function, **async function**, class, nested), source vs test files, per repo. For each missing def, what the agent's tools would return if it queried that name (self / no match / wrong node, which resolver tier, whether the wrong node is a test). Per task: gold-patch symbols and whether they are nodes; issue identifiers and how they resolve; graph distance `d` from issue-mentioned nodes to gold nodes (strict/any anchors × directed/undirected); lexical visibility `v` of the gold file (BM25 rank of the issue text over graph node text, and over the real snapshot source: `v_snap_*`) | `a1_summary.json`, `a1_tasks.csv`, `a1_graphs.csv`, `a1_async_audit.csv`, `a1_gold_symbols.jsonl`, `a1_identifiers.jsonl` |
| **A2** dynamic oracle: runs | `a2/run_task.py` + `a2/trace_plugin.py` | Execute each task's target tests and record the **real** call graph. Per task: unpack snapshot, fresh venv, editable install + test dependencies, apply `test_patch`, run pytest **pre-fix** under the tracer, apply the gold `patch`, run pytest **post-fix** under the tracer. The tracer (`sys.setprofile` + `threading.setprofile`) records every repo function executed per test and every caller→callee edge (direct, and via library frames) | per task folder: `status.json`, `trace_pre.json`, `trace_post.json`, `pytest_pre.log`, `pytest_post.log`, junit `.xml`, `deps.json`, `env_freeze.txt`, `setup.log` |
| **A2** analysis | `a2/analyze.py` | Compare the provided `calls` graph with the traced calls: **recall** (share of real call edges the graph holds; "representable" = between existing nodes, "agent-facing" = including edges to absent async/nested/sync defs), cause of every miss, **observed precision** (static edges whose endpoints ran that were actually seen), whether each gold function is in the graph and was executed by a fail-to-pass test, static reachability from the failing test to the gold function. Also run validity per task: run class, patch-apply strategy, fail-to-pass check against `tasks.jsonl`, `usable` flag | `a2_summary.json`, `a2_tasks.csv`, `a2_run_status.csv`, `a2_gold.jsonl`, `a2_dyn_edges.jsonl`, `a2_static_unconfirmed.jsonl` |
| **A3** encoder audit | `a3_encoder.py` | Which public model produced the provided 256-d embeddings? Statistics of the vectors (norm, dominance of one direction, PC1 share). Then for EmbeddingGemma-300m, EmbeddingGemma-2 and all-MiniLM-L6-v2 (control): re-embed sampled nodes (text, name, id, name+text; with/without documented prompts), cosine to the provided vector and rank correlation; "identified" if mean cosine ≥ 0.99 | `a3_vectors_stats.csv`, `a3_summary.json` (+ `a3_per_model.csv` from the full run) |
| **A4** similarity probe | `a4_probe.py` | What does embedding similarity track? Cosine of call-adjacent pairs, same-module pairs, name-overlap pairs, gold co-change pairs, containment pairs vs random pairs (AUROC); composition of each node's 10 nearest neighbours | `a4_relations.csv`, `a4_knn.csv`, `a4_summary.json` |
| **A5** retrieval modes | `a5_retrieval.py` | How often does each retrieval mode surface the gold function/file for the issue text? (a0) resolved identifier anchors only; (a) harness mode = anchors + `search_similar_code` cosine ranking; (b) BM25 over node text; (c) free-text embedding (needs A3's encoder); (d) harness + 1-hop call neighbours. Hit@10 (gold node), Hit@5 (gold file), MRR, paired McNemar between modes | `a5_tasks.csv`, `a5_summary.json` |
| **A6** release artefacts | `a6_release.py` | Builds a corrected typed graph per commit from the source (all modules, classes, functions **including async**; edges `contains`, `imports`, `inherits`, `calls_static`, `calls_dynamic` from A2 traces, `tests`) in the harness's JSON format, and the per-task profile for Plan B (d × v cell, gold coverage, A2 recall/precision, fail-to-pass, reachability) | `graphs_v2/*.json`, `task_profile.csv`, `a6_summary.json` |
| image | `a2/Dockerfile.audit` | Docker image for A2's docker backend (not used in this run; A2 used venvs) | |

### Test suite and operations
| Script | Role |
|---|---|
| `tests/make_fixture.py` | Builds a small synthetic archive (an async def, a nested function, `docs_src/` and `src/` files, a spurious and a missing edge, a new method added by a gold patch) |
| `tests/test_pipeline.py` | Runs the whole pipeline on that fixture and checks unit behaviour (resolver tiers, diff parsing, stats, pytest-run classification) and output shapes |
| `ops/vm_setup.sh` | VM preparation, run once as root per VM: user + sudo + SSH key; system packages (git, patch, python3-venv, build-essential, numpy, scipy, rsync, pigz, tmux); `~/a3env` venv for A3 and A5 mode (c) (sentence-transformers, CPU torch + torchvision, numpy, scipy, pillow); prints the manual Hugging Face token steps (gated EmbeddingGemma licence + read token saved to `~/.hf_token`) |
| `ops/run_main.sh` | The exact launcher used on vm-main for A1 + A4 + A5 + A3-stats |

---

## 3. Commands that produced each result

| Result | Machine | Command (from the pipeline folder) |
|---|---|---|
| A1, A4, A5 | vm-main | `run_all.py --data archive --out results --cache cache --stages a1,a4,a5 --a1-shards 8 --jobs 8` |
| A3 stats | vm-main | `a3_encoder.py --data archive --out results/a3 --stats-only --graphs 10` |
| A2 runs | vm-a2 | `a2/run_task.py --data archive --out results/a2_runs --cache cache --backend venv --jobs 16 --keep` |
| A2 rerun (3 rich tasks + rich_4006) | vm-a2 | same + `--jobs 4 --test-timeout 60 --task-ids rich_3480 rich_3772 rich_4006 rich_4070` |
| A2 analysis | vm-a2 | `a2/analyze.py --data archive --runs results/a2_runs --a1 results/a1 --out results/a2` |
| A6 | vm-a2 | `a6_release.py --data archive --a1 results/a1 --a2-runs results/a2_runs --a2-analysis results/a2 --out results/a6` |
| A3 full (300m + MiniLM) | vm-a2 | `a3_encoder.py --data archive --out results/a3 --graphs 10` (in `~/a3env`, `HF_TOKEN` set; default `--sample 200`) |
| A3 full (embeddinggemma-2) | vm-a2 | `a3_encoder.py --data archive --out results/a3_gemma2 --graphs 10 --models google/embeddinggemma-2` (same env, after installing `pillow` and `torchvision`) |

## 4. Run facts to know before reading the numbers

- **Environment:** Debian 13, Python 3.13.5 on both VMs (same interpreter version as the harness).
  A2 used per-task venvs with network access (PyPI + `archive/wheels` as find-links), no Docker.
- **Code versions:** the scripts in `scripts/` are the final versions. Changes made during the run,
  all in the code review and its follow-ups: snapshot-based `v_snap_*` and source/test omission
  split (a1_census.py); A6 cell on `v_snap_file_rank_src` (a6_release.py); worker-pool pickling fix
  (run_task.py, before the main A2 run); `--test-timeout` option and `killed` run class
  (run_task.py, before the rerun only). The main A2 run used run_task.py sha256 `b142383d…`, the
  rerun `ad55f049…`.
- **First A2 attempt of 4 rich tasks:** the pre-fix pytest runs of rich_3480, rich_3772, rich_4006,
  rich_4070 did not finish (10–17 min, memory up to ~36 GB in two of them) and ended with SIGKILL
  (exit −9); no pre-fix trace. Kept in `a2_runs_killed/`.
- **Rerun of those 4:** with a 60 s per-test pytest-timeout (the harness and all other 125 tasks
  use none). rich_3480, rich_3772, rich_4070 completed with pre-fix traces (1, 1 and ≥1 tests hit
  the timeout). rich_4006 did not respond to the timeout, was killed again after ~7.5 min and has
  **no pre-fix trace**; its run class is `killed`.
- **Run classes** in the current `a2_run_status.csv`: clean 113, collection_error 9, no_tests 4,
  import_error 2, killed 1.
- **Timeouts:** the harness kills any command at 300 s; A2's whole-run cap was 1800 s.
- **A3 models:** embeddinggemma-2 failed to load twice before running: first `pillow` was missing,
  then `torchvision` (its processor is multimodal). Both were installed into `~/a3env` and the model
  was run on its own; the logs of the two failed attempts were overwritten by the successful run.
  `ops/vm_setup.sh` now installs both. All three models used the same 10 graphs and 200 sampled nodes
  per graph. embeddinggemma-2 took ~2.5 h on CPU (00:58–03:30 UTC), the other two ~25 min together (00:22–00:46 UTC).
- **A3 earlier stats-only run:** `vm-main/results/a3/` is the vector-statistics-only run from
  vm-main; `vm-a2/results/a3*/a3_vectors_stats.csv` are the same statistics recomputed by the full runs.

## 5. Not run / not downloaded

- **A5 mode (c) free-text retrieval:** not run, because no encoder was identified by A3.
- Left on vm-a2: `~/cache/a2/<task>/` (63 GB of kept workspaces + venvs; venvs only work at that
  path on that VM), `~/a6cache/`, `~/results/a6_before_rerun/`, the A1 copy, the archive copy.

---

## 6. Oct 11 follow-up runs (vm-a2)

Two analysis passes over the same archive and traces, after the code fixes in `ANALYSIS_2026-10-11.md`
and `FIXES_2026-10-11.md` and the remaining gaps. Earlier outputs are untouched; each pass writes new
folders. Only rich_4006 ran tests again. Use the **latest** folder of each step.

| Folder (under `vm-a2/results/`) | Step | Produced by |
|---|---|---|
| `a1_v2/` | A1 with the provided graph's flattened ids for nested defs, class-dropped counts. `a1_async_audit.csv` is stored as `.csv.gz` in the GitHub copy (158 MB) | `ops/rerun_fixes.sh` |
| `a7/` | A7 id reconciliation (new): provided nodes vs source defs | `ops/rerun_fixes.sh` |
| `a2_v2/` | A2 analysis: nested mapping, entry-point reachability, fail-to-pass statuses | `ops/rerun_fixes.sh` |
| `a5_v2/`, `a6_v2/` | A5 and A6 on `a1_v2` / `a2_v2` | `ops/rerun_fixes.sh` |
| `a2_runs/rich_4006/` | rich_4006 rerun with per-test checkpoints and hung-test deselection (`--timeout 300 --test-timeout 60 --deselect-hung`); the previous attempt is in `a2_runs_killed2/` | `ops/run_gaps.sh` |
| **`a2_v3/`** | A2 analysis, latest: adds `pre_skipped`, the diff-aware truly-added expected-test set (`strict_truly_added_tasks`), post-trace edges for the 11 tasks without a usable pre trace (`usable_plus_post_edge_tasks`, `post_edge_tasks_only`; gold-incident edges dropped), and the recovered rich_4006 | `ops/run_gaps.sh` |
| **`a5_v3/`** | A5, latest: `overall_all_tasks` / `per_repo_all_tasks` (all tasks, no gold node = miss) next to the earlier denominator; mode **e_bm25_snapshot** (BM25 over the real snapshot source, scored against every gold symbol) | `ops/run_gaps.sh` |
| **`a8/`** | A8 edge audit (new): `a8_edges.csv` (every unconfirmed, missed and captured edge, classified by how the callee appears in the caller's source, with callee-name multiplicity), `a8_sample.csv` (80 non-test edges for hand review, `human_label` / `notes` columns empty), `a8_summary.json` | `ops/run_gaps.sh` |
| **`a6_v3/`** | A6, latest: contains-edge fix and new task-profile columns | `ops/run_gaps.sh` |
| `*.log`, `scripts_sha256_v2.txt`, `scripts_sha256_v3.txt` | step logs and checksums of the code each pass used | both |

Commands (vm-a2, from the scripts folder):
`bash ops/rerun_fixes.sh ~/graphComp/archive ~/results ~/cache_v2 8`, then
`bash ops/run_gaps.sh ~/graphComp/archive ~/results ~/cache_v2`.

Run facts:
- **rich_4006 recovered:** the pre-fix run hung on 4 test cases in `tests/test_cells.py` (escape-character
  inputs) that ignore pytest-timeout. Each was deselected after the 300 s run limit and recorded as
  failed in the trace (`hung_tests` in `status.json` and `trace_pre.json`); the 5th attempt ran clean.
  rich_4006 is now usable. Run classes in `a2_v3`: clean 114, collection_error 9, no_tests 4,
  import_error 2. Usable tasks: 91 of 129.
- **Fail-to-pass:** strict (every expected test confirmed) 55 tasks under both the harness rule and the
  truly-added rule; the truly-added rule keeps 379 of the expected tests; no `pre_skipped` cases occurred.
- **A8 populations:** 40,624 unconfirmed, 18,232 missed and 11,200 captured distinct edges (graph, src, dst).
- Code for both passes is in `scripts/` (final versions; tests: 14 pass on vm-a2, Python 3.13.5).

## 7. Non-usable tasks (38 of 129; HANDOFF_planBC.md section 1.4)

Table, counts and the exact script: **`vm-a2/results/a2_nonusable/`** (`README.md`, `a2_nonusable.csv`,
`make_nonusable_table.py`). Derived from `a2_v3/a2_tasks.csv` plus `a2_v3/a2_run_status.csv` (the 4
requests `no_tests` tasks are only in the latter). 38 = 23 clean runs (20 with no expected fail-to-pass
test, 3 with expected tests none confirmed), 9 collection errors, 2 import errors, 4 requests tasks
without collectable tests. Usable 91 (fastapi 45, rich 39, requests 7).

**Paper correction:** the Limitations paragraph's "rich_4006 (killed)" and "22 clean runs" are out of
date; rich_4006 is usable in `a2_v3`, and the 23rd clean run is httpx_3672.

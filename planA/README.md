# Plan A pipeline: execution-grounded audit of the competition graph and embeddings

Implements the babystep chain of `HANDOFF.md` (Plan A, section A.4) as runnable, shardable
scripts. All CPU. Written and tested without the competition data (it cannot be redistributed
and the cloud sandbox cannot reach Kaggle), against a synthetic archive that mimics the documented
formats, so **every data-dependent assumption is tagged `VERIFY-ON-DATA` in the code and listed
below**. The first run on the real archive must check that list before any number is reported.

## Layout

| File | Step | What it produces |
| --- | --- | --- |
| `common.py` | shared | task/graph/embedding/snapshot loaders, harness-identical 4-tier `resolve_node_name`, unified-diff parser, gold-symbol mapping via `ast`, identifier extraction, BM25, path-to-module rules |
| `a1_census.py` | A1 + B1 | per-graph census (`a1_graphs.csv`), async-omission audit (`a1_async_audit.csv`), per-task gold coverage, identifier resolution tiers, distance `d`, visibility `v` (`a1_tasks.csv`, `a1_gold_symbols.jsonl`, `a1_identifiers.jsonl`, `a1_summary.json`) |
| `a2/trace_plugin.py` | A2 | pytest plugin: `sys.setprofile` + `threading.setprofile` tracer, per-test executed sets, direct and via-library edges, junit outcomes |
| `a2/run_task.py` | A2 | per-task runner: snapshot -> venv (or Docker) -> `pip install -e .` -> test deps best effort -> `test_patch` -> pytest **pre** -> gold `patch` -> pytest **post** |
| `a2/analyze.py` | A2 | static `calls` vs dynamic oracle: recall, observed precision, gold-incident metrics, fail-to-pass gold reachability (`a2_tasks.csv`, `a2_summary.json`, `a2_edges_sample.jsonl`) |
| `a2/Dockerfile.audit` | A2 | python:3.13-slim + git + pytest, with network (2 vCPU / 4 GiB like the harness) |
| `a3_encoder.py` | A3 | encoder identification (EmbeddingGemma v1/v2 at 256-d, prompts, name vs text) + vector statistics |
| `a4_probe.py` | A4 | what cosine tracks: call adjacency, same module, name overlap, gold co-change vs random; kNN composition |
| `a5_retrieval.py` | A5 | gold Hit@10 / file Hit@5 for harness mode, anchors only, BM25, free text (if A3 identified an encoder), harness + 1-hop |
| `a6_release.py` | A6 | typed graph v2 per commit (`contains`, `imports`, `inherits`, `calls_static`, `calls_dynamic` with test provenance, `tests`) in the harness's node_link format; per-task profile for Plan B |
| `run_all.py` | all | orchestrator: A1 shards, A2 shards and A3 in parallel; then A4 and A5; then A2 analysis; then A6 |
| `../tests/make_fixture.py`, `../tests/test_pipeline.py` | test | synthetic archive + end-to-end test (`pytest tests/`) |

## Running on the real archive

```bash
pip install -r planA/requirements.txt          # + sentence-transformers torch for A3
ARCHIVE=/Users/sauravbellam/Desktop/Projects/graphComp/archive   # local path given by the user

# everything, on one machine (A2 uses venvs; use --a2-backend docker if Docker is available)
python planA/run_all.py --data $ARCHIVE --out results --jobs 4 --a1-shards 2 --a2-shards 3 --a2-jobs 2

# quick first look (A1 only, verifies finding 5.1 across all 127 graphs):
python planA/a1_census.py --data $ARCHIVE --out results/a1

# multi-machine A2: give each VM a disjoint shard, copy results/a2_runs/* back, then analyze
python planA/a2/run_task.py --data $ARCHIVE --out results/a2_runs --cache /scratch/cache --backend docker --shard 0/4 --jobs 7
python planA/a2/analyze.py --data $ARCHIVE --runs results/a2_runs --a1 results/a1 --out results/a2

# A3 needs huggingface.co and, for v1, HF_TOKEN with the EmbeddingGemma licence accepted
HF_TOKEN=... python planA/a3_encoder.py --data $ARCHIVE --out results/a3
```

Order of value (HANDOFF section 7): A1 first (standalone finding on async omission), A3 and A4 in
parallel, A2 as the long pole, then A5, A6.

What runs in parallel: A1 shards, A2 shards, A3 are independent; A4 and A5 only need A1; A2
analysis needs A2 + A1; A6 needs A1 and (optionally) A2. `run_all.py` encodes exactly this DAG.

## Where each claim's evidence lives

Every pooled number comes with raw numerators and denominators per task, plus an interval from
`stats.py`: Wilson for task-level yes/no rates, task-cluster bootstrap for pooled edge ratios, and
exact McNemar for paired comparisons on the same tasks.

| Question | File and field |
| --- | --- |
| Are async defs missing? (finding 5.1) | `a1/a1_summary.json` `omission_direct_files`, `omission_per_repo`; modules made only of async defs: `omission_inferred_module_files` |
| What do the tools return for a missing def? | `a1/a1_async_audit.csv` `q_name_*`, `q_qual_*` (self / no_match / wrong_node, tier, wrong node is a test, same name); totals in `a1_summary.json` `misresolution` |
| ...for the gold function of each task? | `a1/a1_gold_symbols.jsonl` `agent_query_*`; `a1_tasks.csv` `n_gold_query_wrong_node` |
| How many tasks have async gold functions? | `a1_summary.json` `task_rates.has_gold_async`, per repo |
| Did each A2 run actually work? | `a2/a2_run_status.csv` `run_class`, missing modules, patch strategies, fail-to-pass check against `tasks.jsonl`; `a2_summary.json` `coverage` |
| Which runs count? | `usable` = both runs clean and at least one expected fail-to-pass test confirmed; summaries report `usable_tasks` and `all_analyzed_tasks` |
| How much of the real call graph does the provided graph hold, and why is the rest missing? | `a2_summary.json` `usable_tasks.<all|nontest|gold>_<direct|any>`: `recall_representable`, `recall_agent_facing`, `miss_share_by_cause` (missed between existing nodes vs absent async / nested / sync endpoint); every edge in `a2_dyn_edges.jsonl` |
| Is the provided graph precise where code ran? | `observed_precision`; unconfirmed static edges in `a2_static_unconfirmed.jsonl` |
| Does the fix site exist in the graph and do the failing tests reach it? (H-A2) | `a2_gold.jsonl` status `in_graph_executed / in_graph_not_executed / absent_executed / absent_not_executed`, static reachability; task rates in `a2_summary.json` |
| Does symbol-only search lose to free text? (H-A4) | `a5/a5_summary.json` hit rates with Wilson intervals, `paired_mcnemar` |
| What produced these results? | `run_report.json` `manifest` (tasks.jsonl sha256, file counts and sizes, code commit, dirty flag); per task `a2_runs/<id>/env_freeze.txt`, `deps.json` |

## VERIFY-ON-DATA checklist (assumptions made without the archive)

1. **Node-id prefix rules** (`common.path_to_module_candidates`, `ModuleIndex`): `docs_src/` and
   `src/` leading segments are dropped. The rule is learned per file from the graph (first candidate
   that is a prefix of a node id) and written to `a1_gold_symbols.jsonl` as `prefix_rule`; check the
   histogram of rules and the `none` cases.
2. **Module of a node** (`a1_census.module_of_node`): nodes carry no file attribute, so the module is
   found by stripping trailing segments whose node text starts with `def`/`class`/`@`. Check on a
   few fastapi and rich nodes; nodes with empty `text` break the heuristic.
3. **Graph file pairs**: commit-named `graphs/<repo>_<commit>.json` are preferred; task-named twins
   may be empty (entrant note: "half the graph/embedding files are empty"). `find_graph_path`
   picks the first non-empty candidate; `a1_summary.tasks_graph_missing` must be 0.
4. **Edge types**: only `calls` expected; `a1_summary.edge_types_union` reports what is there. The
   harness filters `edge_type` with `==`, so `"CALLS"` returns nothing (code read from swegemma 0.2.7,
   `graph_utils.get_neighbor`); confirm on data with one call.
5. **Snapshot layout**: `.tgz` may or may not have a single top-level directory; `Snapshot` handles
   both. Check `a2_runs/*/status.json` for `editable_ok` per repo; fastapi (pdm-backend), rich
   (poetry-core), requests (setuptools, `src/` layout) build backends must be installable.
6. **Test dependencies** (`a2/run_task.py::install_deps`): best-effort list per repo. Read
   `deps.json` and `pytest_pre.log` for import errors; coverage per repo is a reported number, not
   assumed.
7. **Fail-to-pass detection** uses the plugin's outcomes (pre failed/error, post passed). Tasks
   with `FAIL_TO_PASS` in `tasks.jsonl` can be cross-checked against it.
8. **Test node ids** from pytest nodeids (`a2/analyze.test_node_id`): `tests/test_x.py::C::t[p]` ->
   `tests.test_x.C.t`. New tests from `test_patch` are never base-graph nodes; reachability falls back
   to the test class / module node and records `f2p_source_levels`.
9. **Embeddings**: one array per node id in the `.npz`; keys that are not node ids and nodes without
   vectors are counted in `a3_vectors_stats.csv`. Dimension assumed 256 (read from data).
10. **EmbeddingGemma prompts / hub ids** in `a3_encoder.py` (`DOC_PROMPTS`, `--models`): check the
    model cards; v2's exact hub id is unverified.
11. **Identifier extraction** (`common.extract_identifiers`): deterministic regexes and a stop list;
    `a1_identifiers.jsonl` lists every token and its resolution tier so false anchors can be audited.
12. **Containment pseudo-edges** are defined as `target.startswith(source + ".")`; the share is
    reported per graph and they are dropped from recall/precision.

## Known limitations of the sandbox this was written in

No competition data (Kaggle blocked, rules forbid mirrors), no Docker daemon, no GPU, no
huggingface.co. Hence: A3 is untested against a real encoder (only its vector-statistics path ran);
the Docker backend of A2 is untested (the venv backend is tested end to end); everything else ran on
the synthetic fixture (`pytest tests/`, 7 tests). The fixture includes an async def whose name
collides with a sync method, a module made only of async defs, a nested function, a generator, a
spurious static edge, a missed sync edge, `docs_src/` and `src/` files, a new method added by the
gold patch, and two tasks sharing one commit across shards.

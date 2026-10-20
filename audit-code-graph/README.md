# Execution-Grounded Audit of a Pre-Built Code Graph and Node Embeddings

**for Repository-Level Agentic Coding**: Saurav Bellam, The University of Texas at Austin

This repository holds the paper, the scripts that produced its numbers, and the result files those scripts wrote. This page tells you where every part of the paper comes from and how to check it. All paths below are relative to the `planA_results/` folder.

**In one paragraph.** The competition archive ships 127 commit-level code graphs and 256-d node vectors for 129 tasks from fastapi, rich, requests and httpx. We checked them against the source code and against the real call graph recorded by running each task's tests under a tracer. Every one of the 45,492 `async def` definitions is missing from the graph. The `calls` relation holds 37.9% of the executed non-test call edges, and only 21.4% of its edges whose endpoints ran were actually observed. The harness's bare-name resolver returns a different node with the same name for 41% of functions that are present. The vectors are a text embedding from an encoder we could not identify. Plain BM25 over the source finds the gold file in the top five for 65.9% of tasks, against 33.3% for the harness's retrieval.

---

## 1. Read

| What | File |
|---|---|
| Paper (PDF) | [`paper_draft/paper.pdf`](planA_results/paper_draft/paper.pdf) |
| Paper (LaTeX source) | [`paper_draft/paper.tex`](planA_results/paper_draft/paper.tex) |
| Figures (vector PDF) | `paper_draft/figs/fig*.pdf` |
| Script that draws every figure | `paper_draft/make_figs.py` |
| Long-form analysis notes behind the paper | `paper_draft/ANALYSIS_DRAFT.md` |
| Run log and folder-by-folder notes | [`planA_results/README.md`](planA_results/README.md) |

To rebuild the PDF: `cd planA_results/paper_draft && pdflatex paper.tex` (run it twice for references). The figures are already in `figs/`.

To redraw the figures, run `python3 make_figs.py` from `paper_draft/`. You need matplotlib and numpy. First set `U` at the top of the script to the absolute path of `planA_results/`, because it is set to the path used when the draft was written.

## 2. Check the pipeline

```
cd planA_results/scripts
pip install pytest numpy scipy
python3 -m pytest tests/test_pipeline.py -q
```

This runs the whole pipeline on a small synthetic archive (`tests/make_fixture.py`), so you don't need the competition data. The fixture includes an async def, a nested function, `src/` and `docs_src/` files, a spurious edge, a missing edge, and a method added by a gold patch. Result: **7 passed** (Python 3.13).

## 3. Where each section's results live

| Paper section | Result files (under `vm-a2/results/` unless noted) | Produced by (`scripts/planA/`) |
|---|---|---|
| §3.1 Coverage (Fig. 1, Fig. 2) | `a1_v2/a1_summary.json`, `a1_graphs.csv`, `a1_tasks.csv`, `a1_gold_symbols.jsonl`, `a1_identifiers.jsonl` | `a1_census.py` |
| §3.1 Id reconciliation | `a7/a7_summary.json`, `a7_class_dropped.csv`, `a7_provided_nodes.csv`, `a7_defs_absent.csv` | `a7_id_audit.py`* |
| §3.2 Resolver (Fig. 9) | `a1_v2/a1_summary.json` (`misresolution` block) | `a1_census.py`; the resolver copy is in `common.py` |
| §3.3 Recall and precision (Table 1, Fig. 3) | `a2_v3/a2_summary.json`, `a2_tasks.csv`, `a2_run_status.csv`, `a2_dyn_edges.jsonl`, `a2_static_unconfirmed.jsonl` | `a2/run_task.py`, `a2/trace_plugin.py`, `a2/analyze.py` |
| §3.3 Usable / non-usable tasks | `a2_nonusable/a2_nonusable.csv` (+ `README.md`, `make_nonusable_table.py`) | `a2_nonusable/make_nonusable_table.py` |
| §3.3 Why edges are missed (Fig. 4) | `a8/a8_summary.json`, `a8_edges.csv`, `a8_sample.csv` (80 edges with caller snippets) | `a8_edge_audit.py`* |
| §3.3 Reachability (Fig. 8) | `a2_v3/a2_summary.json`, `a2_gold.jsonl`, `a2_v3/a2_tasks.csv`, `a6_v3/a6_summary.json` | `a2/analyze.py` |
| §3.4 Vector statistics, encoder search | `a3/`, `a3_gemma2/` (`a3_summary.json`, `a3_per_model.csv`, `a3_vectors_stats.csv`) | `a3_encoder.py` |
| §3.4 Text identity, duplicates | `a3b/a3b_summary.json`, `a3b_checks_text_identity.csv`, `a3b_checks_text_identity_nodes.csv.gz`, `a3b_checks_duplicates.csv`; write-up in `a3b_results.md` | `a3b_gnn.py`* (`checks` stage) |
| §3.4 GNN sweep (Fig. 6) | `vm-gpu/results/a3b_gpu/a3b_gnn.csv`, `a3b_summary.json`, `a3b_gpu_analysis.txt` | `a3b_gnn.py`* (`fit`, `linkpred` stages) |
| §3.4 What cosine tracks (Fig. 7) | `vm-main/results/a4/a4_summary.json`, `a4_relations.csv`, `a4_knn.csv` | `a4_probe.py` |
| §3.5 Retrieval (Table 2, Fig. 5) | `a5_v3/a5_summary.json`, `a5_tasks.csv` | `a5_retrieval.py` |
| §3.6 Corrected graph | `a6_v3/graphs_v2/*.json.gz` (one per commit), `a6_v3/task_profile.csv`, `a6_v3/a6_summary.json` | `a6_release.py` |
| Per-task tests and traces | `a2_runs/<task>/` (`status.json`, pytest logs, junit XML, `trace_pre.json`, `trace_post.json`, dependency logs) | `a2/run_task.py` |

\* These scripts produced the results here, but they are not in this copy of `scripts/`. Their checksums are in `vm-a2/results/scripts_sha256_v3.txt` and `vm-gpu/results/a3b_gpu/provenance/scripts_sha256_a3b.txt`.

Every pooled number in the paper has its numerator, denominator and interval in the summary file listed for its section. Task-level rates carry Wilson intervals, pooled ratios carry task-cluster bootstrap intervals, and paired comparisons use the exact McNemar test (`scripts/planA/stats.py`).

## 4. Reproduce from scratch

The competition archive (tasks, graphs, embeddings, snapshots) is not redistributed here; download it from the competition. Then, on a Linux machine with Python 3.13 and network access:

```
cd planA_results/scripts
pip install -r planA/requirements.txt
python3 planA/run_all.py --data <archive> --out <results> --stages a1,a4,a5 --a1-shards 8 --jobs 8
python3 planA/a2/run_task.py --data <archive> --out <results>/a2_runs --backend venv --jobs 16 --keep
python3 planA/a2/analyze.py --data <archive> --runs <results>/a2_runs --a1 <results>/a1 --out <results>/a2
python3 planA/a6_release.py --data <archive> --a1 <results>/a1 --a2-runs <results>/a2_runs --a2-analysis <results>/a2 --out <results>/a6
python3 planA/a3_encoder.py --data <archive> --out <results>/a3 --graphs 10
```

`scripts/ops/run_main.sh` is the exact launcher used for A1, A4, A5 and the A3 statistics, and `scripts/ops/vm_setup.sh` prepares a VM. The A3 encoder step needs a Hugging Face token with access to EmbeddingGemma. The exact command for every result folder is in [`planA_results/README.md`](planA_results/README.md) §3 and §6. Host specs and package versions are in `provenance/` and `vm-gpu/results/a3b_gpu/provenance/`.

Run setup: Debian 13 with Python 3.13.5 (the harness's version), numpy 2.2.4 and scipy 1.15.3. A2 used per-task virtual environments, not Docker, with 16 parallel jobs. The GNN sweep ran on a separate GPU VM.

## 5. History and caveats worth knowing

- **Only final outputs are included.** The first run (Oct 10) used a wrong id scheme for nested definitions. That was fixed and every affected step was rerun on Oct 11. This repository holds the latest folder of each step (`a1_v2`, `a2_v3`, `a5_v3`, `a6_v3`, `a7`, `a8`, `a3b`), and the paper uses only these. The step logs of earlier passes are kept for the record.
- **rich_4006** hung on four test cases that ignore pytest-timeout. It was recovered by deselecting them and recording them as failed (`vm-a2/results/a2_runs/rich_4006/status.json`, `hung_tests`). Three other rich tasks were killed on the first attempt and recovered with a 60 s per-test timeout.
- **38 of 129 tasks are not usable** for the execution comparison. That is 20 clean runs with no expected fail-to-pass test, 3 with none confirmed, 9 collection errors, 2 import errors, and 4 requests tasks with no collectable tests. The collection and import errors reflect our dependency setup, not the tasks. Usable: 91 (fastapi 45, rich 39, requests 7).
- **No encoder was identified**, so free-text embedding retrieval (A5 mode c) was not run.
- The 80-edge hand-review sample in `a8/a8_sample.csv` has not been labelled yet.

## 6. Large files and what is not here

GitHub rejects files over 100 MB, so some outputs are gzipped. `gunzip -k` restores each one byte for byte. No script in this repository reads these files directly, so unzip them before use.

| File | Stored as | Uncompressed |
|---|---|---|
| Released typed graphs, 127 files | `vm-a2/results/a6_v3/graphs_v2/*.json.gz` | 662 MB |
| Per-definition census rows | `vm-a2/results/a1_v2/a1_async_audit.csv.gz` | 158 MB |
| Text-identity check, per node | `vm-a2/results/a3b/a3b_checks_text_identity_nodes.csv.gz` | 1.1 GB |

Not uploaded:
- The superseded first-pass outputs: `vm-main/results/a1`, `a5`; `vm-a2/results/a2`, `a2_v2`, `a2_before_rerun`, `a2_runs_killed*`, `a5_v2`, `a6`, `a6_v2`.
- The A1 shard files.
- The GNN sweep's fitted embedding arrays (`vm-gpu/results/a3b_gpu/embeddings/`, 2.2 GB). Its results are summarized in `a3b_gnn.csv`.

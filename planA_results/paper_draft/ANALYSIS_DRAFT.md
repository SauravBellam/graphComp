# Plan A results: analysis draft

Source: `planA_results/` (run of 2026-10-10, follow-up passes of 2026-10-11). All numbers below are
read from the summary/CSV files named in brackets; "latest" folders are `a1_v2`, `a2_v3`, `a5_v3`,
`a6_v3`, `a7`, `a8`, `a3`, `a3_gemma2`, `a3b`, `vm-gpu/a3b_gpu`, and `vm-main/a4`.

## 0. What was audited

The competition archive gives every task a pre-built **code graph** (one JSON per base commit, 127
graphs for 129 tasks: fastapi 67, rich 48, requests 13, httpx 1) and a **256-d embedding per node**.
The `swegemma` 0.2.7 harness exposes them through three tools: name resolution
(`resolve_node_name`, 4 tiers: exact → suffix → case-insensitive → substring), neighbour lookup
(`get_neighbor`, edge type filtered by `==`), and cosine search (`search_similar_code`).
Plan A asked, with executable evidence: (A1) is every definition in the source a node? (A2) does the
graph's `calls` relation match the *real* call graph recorded by running the tests? (A3/A3b) what
produced the vectors? (A4) what does cosine similarity track? (A5) how often does each retrieval
mode surface the fix location? (A6) can a corrected graph be built? (A7/A8) reconcile ids and audit
every edge.

Archive facts (`run_report.json`): 127 graphs (422 MB), 127 embedding files (468 MB), 130 snapshots
(21.5 GB), 124 wheels; `tasks.jsonl` sha256 `e4b3fd60…`. Both VMs: Debian 13, Python 3.13.5 (same
as harness), numpy 2.2.4, scipy 1.15.3. A2 used per-task venvs (no Docker), 16 parallel jobs.

## 1. A1 — static census of the provided graphs

**Graph shape** (`a1_graphs.csv`, 127 graphs): 367,985 nodes and 452,588 edges in total; the only
edge type is lowercase `calls`. Per repo (means per graph):

| repo | graphs | nodes | edges | isolated nodes | test-file nodes | edges from tests | containment pseudo-edges in `calls` | self-loops |
|---|---|---|---|---|---|---|---|---|
| fastapi | 67 | 3,968 | 2,458 | **68.7 %** | 68.4 % | 31.4 % | 8.5 % | 35 |
| rich | 47 | 1,970 | 5,625 | 3.5 % | 41.9 % | 26.6 % | 15.3 % | 5 |
| requests | 12 | 735 | 1,825 | 7.9 % | 60.5 % | 46.3 % | 20.7 % | 0 |
| httpx | 1 | 740 | 1,618 | 2.8 % | 19.5 % | 11.6 % | 29.8 % | 0 |

186,994 of 367,985 nodes (50.8 %) have no edge at all; in fastapi roughly 2,730 of 3,970 nodes per
graph are isolated. No dangling edges. 8.5–29.8 % of `calls` edges are containment pseudo-edges
(`target` starts with `source.`), not calls.

**Definitions present** (`a1_summary.json`, `omission_direct_*`, 127 graphs, all `.py` files parsed
with `ast`):

| kind | all files | source files only | test files only |
|---|---|---|---|
| sync function | 287,841 / 289,266 = 99.51 % | 90,792 / 90,792 = **100 %** | 197,049 / 198,474 = 99.28 % |
| class | 62,552 / 63,831 = 98.0 % | 47,258 / 48,348 = 97.75 % | 15,294 / 15,483 = 98.78 % |
| nested def | 14,483 / 18,178 = 79.7 % | 2,999 / 3,824 = 78.4 % | 11,484 / 14,354 = 80.0 % |
| **async function** | **0 / 45,492 = 0.0 %** (Wilson hi 0.01 %) | **0 / 28,113 = 0 %** | 0 / 17,379 = 0 % |

Async omission is total in every repo that has async code: fastapi 0/45,399, httpx 0/93, and 10,995
async defs sit in modules that have **no node at all** (`omission_inferred_module_files`). The
non-async gaps are explained by id reconciliation (A7): the 1,425 "absent" sync functions and 1,279
"absent" classes are present under a *class-dropped* id (e.g. a method stored as
`module.method` instead of `module.Class.method`; requests 1,425 functions = 19.75 % of its sync
functions; fastapi 1,279 classes). Nested defs are stored flattened (`matched_nested_flat`:
fastapi 8,033, rich 5,985, requests 277). requests graphs also contain 268 provided nodes that match
no definition (`unmatched`).

**What the name resolver does with a missing or ambiguous name** (`misresolution`, bare-name
queries as the harness would issue them):

- Sync function present, bare name: 118,638 / 287,841 = **41.2 %** resolve to a *different* node
  with the same short name (suffix tier); 73,723 of those wrong nodes are test functions. fastapi
  46.6 %, httpx 62.0 %, rich 28.5 %, requests 17.6 %.
- Class present, bare name: 29,967 / 62,552 = 47.9 % wrong node (fastapi 57.3 %).
- Async function (absent), bare name: 36,924 / 56,487 = **65.4 % wrong node** (28,961 suffix, 6,413
  substring, 1,550 case-insensitive); 26,851 of the wrong nodes are test nodes; 34.6 % no match.
  With the fully qualified name: 97.2 % no match, 2.9 % wrong node (httpx 82 %).
- Qualified names fix presence cases (5.25 % wrong for sync functions, 0.5 % in rich; 18.7 % in
  fastapi because of class-dropped ids).

**Per task** (`a1_tasks.csv`, 129 tasks): 711 gold-patch symbols, 470 (66.1 %, Wilson 62.6–69.5)
are nodes; 184 are new symbols added by the patch (cannot be in the base graph); 81 gold symbols are
async, **1 present** (1.2 %). 22 tasks (17.1 %) have an async gold function (fastapi 21/67 = 31.3 %,
httpx 1/1). In 21 tasks (16.3 %) a harness-style query for the gold function returns a wrong node
(fastapi 19/67 = 28.4 %). 11 tasks have no gold node in the graph at all; 7 tasks are module-level
only; 38 are multi-file. Issue text: median 418 characters, median 3 identifiers; of 1,014
extracted identifiers 578 resolve to nothing, 219 suffix, 195 substring, 17 ci, 5 exact; 121 resolve
to test nodes and 89 to a gold node. Anchors hit a gold node in 30 tasks. Graph distance *d*
(issue-mentioned node → gold node) is defined for 47 tasks (36.4 %): d=0 for 19, d=1 for 5, d=2 for 8,
d≥3 for 15; undefined for 82. Lexical visibility *v* (BM25 rank of the gold file over the real
snapshot source) ≤5 for 72 tasks; over graph node text ≤5 for 65 tasks.

## 2. A2 — execution oracle (real call graph)

**Runs** (`a2_run_status.csv`, `coverage`): 129 tasks; all 129 test and gold patches applied
exactly. Run classes: clean 114, collection_error 9, no_tests 4 (all requests), import_error 2.
First attempt: 4 rich tasks (3480, 3772, 4006, 4070) ran 10–17 min, up to ~36 GB RSS, SIGKILL;
rerun with a 60 s per-test timeout recovered three; rich_4006 hung on 4 escape-character test cases
in `tests/test_cells.py` that ignore pytest-timeout and was recovered on the 5th attempt with
per-test checkpoints and hung-test deselection. **Usable** tasks (both runs clean and ≥1 expected
fail-to-pass test confirmed): **91 / 129 = 70.5 %** (fastapi 45, rich 39, requests 7, httpx 0).
Non-usable 38 = 20 clean runs with no expected fail-to-pass test + 3 clean runs where the expected
tests were not confirmed (fastapi_14851, rich_3006, rich_4079) + 9 collection errors + 2 import
errors + 4 requests tasks without collectable tests. Fail-to-pass: 488 expected tests, 189 confirmed,
174 already passed pre-fix, 118 uncollected pre-fix, 7 failed post-fix; under the diff-aware
"truly added" rule 379 expected, 187 confirmed; strict tasks (every expected test confirmed) 55 under
either rule; 92 tasks pass all expected tests post-fix (harness rule).

**Recall of the real call graph** (`a2_summary.json`, usable tasks, pre-fix traces; task-cluster
bootstrap 95 % CI). "Representable" = dynamic edges whose both endpoints are nodes; "agent-facing" =
also counts edges to absent async/nested/sync definitions.

| scope | representable recall | agent-facing recall | miss causes |
|---|---|---|---|
| non-test, direct calls | 5,430 / 14,331 = **37.9 %** [34.4, 42.5] | 5,430 / 15,398 = 35.3 % | between existing nodes 89.3 %, absent nested 7.0 %, absent async 3.6 %, absent sync 0.07 % |
| non-test, any (incl. via library frames) | 36.8 % | 33.9 % | 88.2 / 6.8 / 4.9 / 0.1 % |
| gold-incident, direct | 603 / 1,043 = **57.8 %** [40.7, 70.0] | 50.4 % | between nodes 74.1 %, absent sync 13.3 %, absent async 6.9 %, absent nested 5.7 % |
| all (incl. test-patch defs) | 38.0 % | 32.9 % | |

Per-task mean representable recall is 53.5 % [49.4, 58.2] (pooled ratios are dominated by rich).
Per repo (non-test direct): **fastapi 75.5 %** [74.2, 77.1] (2,308/3,056; miss causes: absent async
28.4 %, absent nested 12.4 %), **rich 26.2 %** [25.6, 26.9] (2,695/10,288; 93.6 % between existing
nodes), **requests 43.3 %** (427/987; 96.7 % between nodes). Gold-incident: fastapi 71.3 %, rich
56.4 %, requests 42.6 %. The strict 55-task subset gives 35.9 % / 60.7 %; all 125 analysed tasks
39.7 % / 56.1 %; adding post-fix edges for the 11 tasks without a pre trace (97 tasks) 38.2 %.

**Observed precision** (static `calls` edges whose both endpoints executed and that were actually
seen): 8,013 / 37,433 = **21.4 %** [20.0, 22.8] (fastapi 16.5 %, rich 23.1 %, requests 34.3 %).
Node coverage of the traces: 17,154 / 102,985 = 16.7 % of non-test nodes executed by the target
tests (requests 39.4 %, rich 20.2 %, fastapi 13.0 %).

**Gold function status** (usable tasks, 245 gold symbols): in graph & executed 114, in graph & not
executed 56, absent & executed 4, absent & not executed 8, new symbol executed post-fix 59, new not
executed 18. Task level: gold in graph and executed by a fail-to-pass test 72 / 91 = **79.1 %**
[69.7, 86.2]; gold absent but executed 3 / 91 (3.3 %). From a test *node* in the graph, the gold
function is statically reachable in only 5 tasks (sources defined in 10: new tests are never base
nodes); from the executed *entry* functions (defined in 86 / 91), the gold is reachable in **69 / 72
= 95.8 %** (fastapi 32/35, rich 30/30, requests 7/7), is itself an entry in 32 / 72 (44.4 %), and
never reachable only undirected (0/72). Directed hop histogram: 0: 32, 1: 9, 2: 13, 3: 9, 4: 2,
5: 2, 6: 1, 7: 1, unreachable 3.

## 3. A8 — what the static edges are

(`a8_summary.json`, non-test edges, distinct (graph, src, dst)): 35,017 unconfirmed, 12,231
missed, 7,640 captured. Classification by how the callee appears in the caller's source:

| population | direct call | instantiation | self/cls call | obj.attr call | referenced (not called) | implicit dunder | string/comment only | no reference | explicit call-site share |
|---|---|---|---|---|---|---|---|---|---|
| captured (7,640) | 55.3 % | 19.0 % | – | 14.3 % | 0.3 % | 0.5 % | 4.5 % | 1.9 % | **88.6 %** |
| missed (12,231) | 10.0 % | 0.5 % | 15.9 % | 23.4 % | 28.3 % | 17.3 % | 1.3 % | 3.1 % | 50.0 % |
| unconfirmed (35,017) | 2.7 % | 7.6 % | 0 | 0.7 % | 1.8 % | 2.2 % | 26.7 % | **56.1 %** | 11.0 % |

So **82.8 % of unconfirmed static edges (no reference 56.1 % + text-only 26.7 %) are spurious**: the
caller's code never names the callee; requests 83 % + 13 %, fastapi 56 % + 32 %, rich 54 % + 22 %.
The generator never emits `self.method()` edges (0 unconfirmed, 15.9 % of misses), rarely
`obj.method()` (0.7 % vs 23.4 % of misses) and never implicit dunders (`__init__`, `__enter__`,
`__rich_console__`: 17.3 % of misses). Callee-name-shared share: captured 15.2 %, missed 39.7 %,
unconfirmed 21.1 %, consistent with a name-matching generator. 831 of 18,232 missed edges (all
scopes) go via library frames; 17,401 are direct.

## 4. A3 / A3b — the embeddings

**Vector statistics** (10 graphs, `a3_vectors_stats.csv`): 256-d, one vector per node, no extras.
Not unit-norm: mean norm 4.47 (fastapi, sd 1.37), 4.61 (rich), 5.30 (requests), 5.43 (httpx).
Strongly anisotropic: mean cosine to the mean vector 0.87 (fastapi) to 0.95 (rich); PC1 explains
44 % (fastapi), 50 % (requests), 68 % (rich), 75 % (httpx) of variance; mean pairwise cosine
0.75–0.91, 5th percentile 0.33–0.75.

**Encoder identification** (`a3_summary.json`, 10 graphs × 200 nodes, variants text/name/id/
name+text × prompts none/document/search_result/code_retrieval_query): best mean cosine to the
provided vector is **0.073** (EmbeddingGemma-300m, name, document prompt, httpx), 0.015
(EmbeddingGemma-2) and 0.036 (all-MiniLM-L6-v2); share ≥0.95 is 0 everywhere; pairwise Spearman
≤0.12. Identification threshold 0.99: **no encoder identified**; A5 mode (c) therefore not run.
EmbeddingGemma-2 needed pillow + torchvision and 2.5 h on CPU.

**Text identity across commits** (`a3b`, 3,358 same-repo graph pairs: fastapi 2,211, rich 1,081,
requests 66): nodes with byte-identical text have cosine ≥0.99 in 99.55 % of cases (min over pairs
98.4 %), 5th-percentile cosine 0.9996 (min 0.9916), exactly-1.0 share 90.0 % (the pre-registered
`share_eq1 < 0.99` rule fired on float32 noise and was replaced). Same text, unchanged neighbours:
0.9998; same text, changed neighbours: 0.971 (min 0.870), Spearman vs neighbour-Jaccard 0.098.
Neighbour-mean prediction 0.993 vs global-mean baseline 0.920 (homophily). Exact duplicate vectors:
16.7 % of nodes per graph on average (fastapi 31.1 %, httpx 3.2 %, rich 0.8 %, requests 0.1 %).
Isolated vs connected nodes separable by a centroid score, AUROC 0.975.

**GNN sweep** (`a3b_gpu`, L4 GPU, 127 graphs, 889 rows; 2-layer mean-aggregation GNN, 300 epochs,
80/20 node split, transductive). *fit* (predict provided vector; gain = share of gap above the
mean-vector baseline closed):

| repo | baseline | structural | text | both | both − text (mean; graphs where both > text) |
|---|---|---|---|---|---|
| rich (47) | 0.948 | 0.824 | **0.878** | 0.835 | −0.0022 (1/47) |
| fastapi (67) | 0.881 | 0.141 | **0.671** | 0.362 | −0.0369 (0/67) |
| requests (12) | 0.928 | 0.685 | **0.744** | 0.697 | −0.0034 (2/12) |
| httpx (1) | 0.935 | 0.793 | **0.924** | 0.827 | −0.0063 (0/1) |

Both > text in 3 / 127 graphs by at most +0.0013. Train-minus-held-out gap largest in structural
(fastapi +0.098) — memorisation by the per-node embedding. *linkpred* (embeddings trained only on
edges, Procrustes-aligned, held-out k-NN@10 overlap): random control 0.003–0.017, structural
0.005–0.016, **text 0.199 (fastapi) – 0.435 (httpx)**, both lower than text everywhere (fastapi
0.083). Text-arm held-out cosine for isolated vs connected nodes: rich 0.896 / 0.997, fastapi
0.945 / 0.996, requests 0.796 / 0.997. Verdict: the vectors are a text embedding of an unidentified
encoder; graph structure adds nothing; the small context effect is attributed to a wider text window.

## 5. A4 — what cosine tracks

(`a4_summary.json`, 127 graphs; AUROC of related pairs vs random pairs, mean cosine of random pairs
0.830): call-adjacent pairs 0.898 (mean cos 0.988), containment 0.903 (0.989), gold co-change 0.806
(51 graphs, 0.955), same module 0.682 (0.890), name overlap 0.675 (0.916). kNN composition
(`a4_knn.csv`, 500 nodes per graph, top-10 neighbours): **test-file nodes 61–77 % in fastapi**
(mean 67 %), 45 % rich, 60 % requests, 20 % httpx; same module 30–49 %; name overlap 14–26 %
(fastapi); call neighbours **≤3 %** everywhere; top-1 cosine 0.9987–0.9989.

## 6. A5 — retrieval of the fix location from the issue text

(`a5_summary.json`, `overall_all_tasks`, n = 129; a miss where no gold node exists; Wilson CIs)

| mode | gold node Hit@10 | gold file Hit@5 | MRR (node) | empty result |
|---|---|---|---|---|
| a0 resolved anchors only | 30 (23.3 %) | 40 (31.0 %) | 0.170 | 54 (41.9 %) |
| a harness (anchors + cosine) | 31 (24.0 %) | 43 (33.3 %) | 0.174 | 54 |
| d harness + 1-hop calls | 31 (24.0 %) | 45 (34.9 %) | 0.175 | 54 |
| b BM25 over graph node text | 38 (29.5 %) | 63 (48.8 %) | 0.170 | 0 |
| e BM25 over snapshot source | **53 (41.1 %)** | **85 (65.9 %)** | **0.232** | 0 |

Paired exact McNemar: harness vs BM25-graph file Hit@5 p = 0.0017 (a-only 9, b-only 29); harness vs
BM25-snapshot node Hit@10 p = 5.9e−5 (4 vs 26), file Hit@5 p < 1e−5 (5 vs 47); BM25-graph vs
BM25-snapshot node p = 2.8e−4, file p = 1e−5; cosine on top of anchors adds 1 node and 3 files
(p = 1.0 / 0.25); 1-hop adds 2 files (p = 0.5). Restricted to the 118 tasks with a gold node:
harness 26.3 % / 34.7 %, BM25-snapshot 44.9 % / 68.6 %. Per repo (all tasks), node Hit@10
harness → snapshot BM25: fastapi 14/67 → 21/67, rich 10/48 → 22/48, requests 6/13 → 9/13; file
Hit@5: fastapi 17 → 39, rich 16 → 33, requests 9 → 12.

## 7. A6 — corrected graph (v2) and task profile

(`a6_summary.json`, 127 graphs built from source with typed edges) Per graph means:

| repo | v2 nodes | modules | classes | async fn | fn | contains | imports | inherits | calls_static | calls_dynamic | tests | nodes not in provided graph |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastapi | 6,146 | 1,163 | 827 | 1,008 | 3,143 | 4,989 | 2,605 | 408 | 2,214 | 422 | 567 | 2,199 |
| rich | 2,208 | 196 | 243 | 1 | 1,728 | 1,999 | 1,055 | 76 | 4,759 | 979 | 1,683 | 237 |
| requests | 890 | 36 | 85 | 0 | 651 | 742 | 240 | 37 | 1,447 | 801 | 5,075 | 194 |
| httpx | 885 | 45 | 78 | 100 | 662 | 840 | 118 | 33 | 1,136 | 494 | 339 | 145 |

Task profile cells (d × v, 129 tasks): d0|v_high 15, d0|v_low 4, d1|v_high 4, d1|v_low 1,
d2+|v_high 16, d2+|v_low 7, d_undef|v_high 37, d_undef|v_low 44, d_undef|v_undef 1.

## 8. Main findings (ranked)

1. **Async omission is total**: 0 of 45,492 async defs (0 of 28,113 in source files) are nodes;
   81 of 711 gold symbols are async, 1 present; 31 % of fastapi tasks fix an async function.
2. **The provided `calls` relation is both incomplete and mostly wrong where it can be checked**:
   37.9 % of real non-test call edges are present (fastapi 75.5 %, rich 26.2 %, requests 43.3 %);
   21.4 % observed precision; 83 % of unconfirmed edges have no reference to the callee in the
   caller's source. Missed edges are dominated by `self.`/`obj.` method calls, callbacks and dunders.
3. **The resolver misleads on bare names**: 41 % of present sync functions and 65 % of absent async
   functions resolve to a different, same-named node, usually a test; 16 % of tasks' gold function
   resolves to a wrong node.
4. **The embeddings are a text embedding from an unidentified encoder** (best public match cosine
   0.07); identical text → identical vector (99.6 % ≥ 0.99); graph structure adds nothing (GNN
   both−text ≤ 0 in 124/127 graphs). Cosine neighbourhoods are 45–67 % test nodes and ≤3 % call
   neighbours; call-adjacent pairs are nonetheless separable from random (AUROC 0.90) via homophily.
5. **Plain BM25 over the real source beats the harness retrieval** on every metric (node Hit@10 41 %
   vs 24 %, file Hit@5 66 % vs 33 %, p < 1e−4); the harness returns nothing for 42 % of tasks.
6. **Where the gold function is a node, the real execution path does reach it** (95.8 % from entry
   functions, 44 % is the entry itself), so a corrected graph with dynamic edges (A6 v2) is a sound
   basis for Plan B.

## 9. Limitations recorded in the run

Per-task venvs instead of Docker; subprocesses spawned by tests not traced; via-library edges
excluded from `calls_dynamic`; 38 tasks non-usable for A2 (listed); rich_4006 needed hung-test
deselection; httpx has one task/graph; graphs within a repo are near-identical commits so per-graph
intervals are not independent; the A3b `share_eq1` rule misfired on float noise; A5 mode (c) not run;
encoder's input window not observable; A8 hand-review sample (80 edges) not yet labelled.

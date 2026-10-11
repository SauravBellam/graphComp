# Is the Map Right?

## An execution-grounded audit of the code graph and embeddings handed to local SWE agents

*Gemma 4 Developer Agent Competition, Paper Track. Draft of Oct 11, 2026.*

---

## Abstract

Coding agents are increasingly given a pre-built map of the repository: a call graph to traverse and an embedding index to search. Papers on such tools report downstream accuracy, but none checks the map itself against what the code does. We audit the graph and embeddings distributed to every entrant of the Gemma 4 Developer Agent Competition, 127 graphs covering 129 tasks from fastapi, rich, requests and httpx, using each task's own tests as the oracle. We parse every source file, run every task's target tests before and after the reference fix under a call tracer, and read the caller's source for every disputed edge. Four findings stand out. First, the graph contains no async functions: 0 of 45,492 async definitions are nodes, which removes 81 of 711 gold-patch symbols and touches 31% of fastapi tasks. Second, the harness's name resolver turns these omissions into wrong answers: a bare-name query for an absent async function returns a different node 65% of the time, usually a test. Third, the `calls` relation captures 37% of observed calls between existing nodes, and only 21% of static edges between executed functions were ever exercised; reading the source shows most unexercised edges are containment and inheritance links stored under the `calls` label, while most misses are unresolved method dispatch. Fourth, the embeddings are not reproducible from any EmbeddingGemma checkpoint, and the shipped symbol-to-symbol search finds the gold file half as often as BM25 over the source. The graph does connect failing tests to the code that must change: from where a failing test enters the library, the gold function is reachable in 69 of 72 tasks. We release the audit toolkit, a corrected typed graph per commit, and per-task annotations.

---

## 1. Introduction

The competition asks whether a 4-bit, 31-billion-parameter model on consumer hardware can fix real Python bugs. Alongside the tasks, the organizers ship two structural aids: a repository graph exposed through `get_code_neighbors` and `get_code_subgraph`, and per-node embeddings behind `search_similar_code`. Every entrant's agent can call them, and the hidden test set uses the same generator, so properties of these artefacts apply to every submission.

Recent work says structure helps agents when it is delivered well. In-band graph neighbours added 11.8 points of file accuracy over a strong baseline [6]; structural comments added 2.2 points of function-level accuracy and halved run-to-run variance [5]; language-server-resolved relations were worth 7 points for a small Gemma model [7]. Every one of these papers builds its own graph and reports what the agent achieves with it. None measures whether the graph is correct, and agents adopt corrupted tool returns more than a third of the time [17], so an error in the map may cost more than its absence.

We therefore ask a prior question: **is the map right?** We treat each task's test suite as ground truth for the code paths that matter, and measure, against it, what the provided graph contains, what its edges mean, what an agent's queries return, and what the embedding index can retrieve.

Our contributions:

1. A census of every definition in every snapshot against the graph, showing the systematic omission of async functions and how the harness resolver converts omissions into wrong answers (Section 5.1, 5.2).
2. A dynamic call-graph oracle over 129 tasks, before and after the reference fix, giving recall, observed precision and reachability of the gold functions (Section 5.3, 5.5).
3. A source-level classification of every disputed edge, which explains what the `calls` relation actually encodes (Section 5.4).
4. An encoder identification attempt and a retrieval audit of the shipped search tool (Section 5.6).
5. A released toolkit, typed graph and per-task profile (Section 6).

## 2. Related work

**Static call graphs for Python.** PyCG reports about 99% precision and 70% recall on its benchmarks by resolving assignments between identifiers [1]. Later work raises recall with LLM filtering [2], and comparisons of Code2flow, Pyan and WALA found none sound and many spurious edges [3]. These studies audit *tools*. We audit an *artefact handed to agents*, whose generator is undocumented.

**Execution as the oracle.** Measuring static recall against calls recorded during test execution is standard: Sui et al. report a median recall of 0.88 over 31 Java programs [4], and Helm et al. extend the baseline with fuzzing [18]. We follow this practice, using each task's own fail-to-pass tests.

**Structure for coding agents.** Graph tools offered separately tend to go unused, while the same structure placed in-band helps [5, 6, 8]. Dynamic traces lifted SWE-bench Verified resolution from 65.4% to 74.2%, but only when summarized [9]. Typed, resolved relations matter more than syntactic edges [7, 10]. Localization difficulty grows with the hop distance from issue-mentioned code to the fix [11]. None of these works measures the error rate of the graph it gives the agent.

**Embeddings and retrieval for agents.** How code is represented to a retriever changes what it finds by up to 40% Hit@5 [12], and compact free-text semantic search lets small localizers beat grep [13]. EmbeddingGemma 2 documents a native 256-dimensional truncation [14], the dimension of the competition's vectors. No study has measured what a symbol-to-symbol search loses compared with free text over the same repository.

## 3. The artefacts under audit

Each task carries a base commit, an issue text, a reference `patch` and a `test_patch`. The graph for a base commit is a NetworkX node-link file whose nodes carry an id (`module.Class.method`), a name and the full source text. Across the 127 graphs there is one edge type, `calls`, with 452,588 edges. Each node has one 256-dimensional float32 vector. The harness resolves a symbol query in four tiers (exact, dotted suffix, case-insensitive, substring) and breaks ties by fewest dots, then shortest id. `search_similar_code` resolves its query to a node this way and ranks other nodes by cosine; it cannot take free text. The harness runs at a 32,768-token context.

## 4. Method

All steps are CPU-only and run on two 8- and 16-core VMs under Python 3.13.5, the harness's interpreter version.

**Static census (A1).** For every snapshot we parse every `.py` file with `ast`, enumerate sync functions, async functions, classes and nested definitions, map each to the graph's id scheme, and record presence. For each definition we issue the queries an agent would type, the bare name and `Class.method`, through a line-for-line copy of the harness resolver and classify the outcome as *self*, *no match* or *wrong node*. Gold symbols are the innermost definitions enclosing each changed line of the reference patch.

**Id reconciliation (A7).** The graph's id scheme is undocumented. We classify every provided node against the parsed source and every parsed definition against the graph, which revealed that nested definitions appear under a flattened id: enclosing functions and the `<locals>` marker are dropped, classes are kept. All other steps use this scheme.

**Dynamic oracle (A2).** For each task we extract the snapshot, create a fresh virtual environment, install the package and its test requirements (best effort, network allowed), apply `test_patch`, and run the target test files under a `sys.setprofile` and `threading.setprofile` tracer, then apply the reference patch and run again. The tracer records every repository function executed per test and every caller-to-callee pair, directly or through library frames, counting coroutine and generator frames once. A task is *usable* when both runs are clean and at least one expected test fails before the fix and passes after; 91 of 129 tasks qualify (fastapi 45, rich 39, requests 7). Edges are attributed from the pre-fix trace, since the graph describes the base commit. We report **representable recall** (observed calls between two existing nodes that the graph holds), **agent-facing recall** (also counting calls to omitted definitions), and **observed precision** (static edges whose two endpoints both executed that were seen as calls). Precision outside executed code is unknown. Intervals are 95% task-cluster bootstrap for pooled ratios and Wilson for task-level rates.

**Reachability.** The failing tests are usually new in `test_patch`, so they are not nodes of the base graph. We seed the search instead from where each failing test enters the library: executed non-test nodes none of whose caller chains reaches another non-test node before test code, walking through frames that are not nodes. We then run a directed breadth-first search on `calls` to the gold functions those tests execute.

**Edge semantics (A8).** For every static edge never observed at runtime, every observed call the graph misses, and every captured call, we read the caller's own source and record how the callee appears: a bare call, `self.` call, call on another object, instantiation, reference without a call, mention only in a string or comment, implicit dunder dispatch, or no mention at all.

**Embeddings (A3 to A5).** We re-embed 200 nodes from each of 10 graphs with EmbeddingGemma 300m, EmbeddingGemma 2 and a MiniLM control, over node text, name, id and name plus text, with and without the documented prompts, truncated to 256 dimensions, and call an encoder identified at mean cosine 0.99 or more. We measure whether cosine separates call-adjacent, same-module and gold co-change pairs from random ones. Finally we compare retrieval modes per task: resolved issue identifiers only, the harness mode (identifiers plus cosine), harness plus one-hop neighbours, BM25 over node text, and BM25 over the real snapshot source.

## 5. Results

### 5.1 The graph has no async functions

| Definition kind (all files, 127 graphs) | Total | Nodes | Present |
| --- | --- | --- | --- |
| Sync function | 289,266 | 287,841 | 99.5% |
| Class | 63,831 | 62,552 | 98.0% |
| Nested definition | 18,178 | 14,483 | 79.7% |
| Async function | 45,492 | 0 | 0% |

Restricted to non-test source files, sync functions are 100% present and async functions 0 of 28,113. The missing classes and sync functions are id-scheme effects that A7 accounts for in full: fastapi lists 1,683 inner pydantic `Config` classes without their model class, and requests lists a subset of test methods without their test class. Every missing nested definition in fastapi is itself async, as is every one in rich except a single class defined inside a test function.

The omission falls on the code that needs fixing. Of 711 gold-patch symbols, 81 are async and none is a node; one appears present only because the nested async `FastAPI.setup.<locals>.openapi` flattens to the id of a different, sync method, `FastAPI.openapi`. 22 tasks have an async gold symbol, 21 of them in fastapi, which is 31% of fastapi tasks. In 3 usable tasks a gold function that the failing test executes has no node at all. fastapi is an async web framework; for its request handling path the graph is silent.

### 5.2 The resolver turns omissions into wrong answers

An agent that asks for an absent function does not get "not found". For a bare-name query on an absent async function, the harness resolver returns a different node in 65.4% of 56,487 cases, a test node in 26,851 of them; it returns nothing in the remaining 34.6%. The problem is not limited to absent code: for sync functions that *are* nodes, the bare name resolves to a different, same-named node 41.2% of the time (46.6% in fastapi, 62.0% in httpx), and 47.9% for classes, because the tie-break prefers short ids and fastapi's tutorial and test modules reuse names. For 21 of 129 tasks, the most specific query for some gold symbol lands on the wrong node.

### 5.3 The `calls` edges against execution

| Usable tasks (91) | Value | 95% CI |
| --- | --- | --- |
| Representable recall | 37.1% (8,032 of 21,653) | 34.8 to 40.2 |
| Agent-facing recall | 31.8% | 29.4 to 34.8 |
| Recall on gold-incident calls | 56.3% | 39.3 to 68.4 |
| Observed precision | 21.4% (8,013 of 37,433) | 20.0 to 22.8 |

Of the observed calls the graph misses, 79% run between two existing nodes, 11% touch an omitted sync definition, 6% a nested function and 4% an absent async function. Recall varies by repository: 65.6% in fastapi, 30.5% in rich, 34.7% in requests; precision is lowest in fastapi (16.5%) and highest in requests (34.3%). The tests executed 17% of non-test nodes, so all figures describe the code paths the tasks exercise.

These numbers contradict our hypothesis that the graph would be precise but incomplete, in the pattern known for conservative static analysers [1]. The graph is incomplete *and* most of its edges between executed functions were never exercised.

### 5.4 What the `calls` relation encodes

Reading the caller's source explains both numbers.

| How the callee appears in the caller (non-test edges) | Captured | Unexercised static | Missed |
| --- | --- | --- | --- |
| Explicit call site of any kind | 88.6% | 11.0% | 50.0% |
| Bare-name call `f()` | 55.3% | 2.7% | 10.0% |
| Instantiation `C()` | 19.0% | 7.6% | 0.5% |
| Call on another object `obj.f()` | 14.3% | 0.7% | 23.4% |
| `self.f()` | 0.0% | 0.0% | 15.9% |
| Reference without a call | 0.3% | 1.8% | 28.3% |
| Implicit dunder dispatch | 0.5% | 2.2% | 17.3% |
| Name only in a string or comment | 4.5% | 26.7% | 1.3% |
| No mention at all | 1.9% | 56.1% | 3.1% |

Captured edges are what a name-matching walker finds: bare calls and instantiations. Missed calls are method dispatch the walker does not resolve: `self.` calls, calls on objects, callbacks and dunders. The 35,017 unexercised non-test edges mostly have no textual basis in the caller. Classifying all of them structurally, 22.3% point from a method to its own class, 13.8% from a class to a subclass and 13.3% from a class to a base class; among missed calls, 45.5% run between two methods of the same class. We infer that the generator stores containment and inheritance, in both directions, under the label `calls`. That reading also explains a feature noticed early: 8 to 21% (30% in the single httpx graph) of `calls` edges per graph point from a symbol to its own member. An agent following "callers" and "callees" is in part walking a class hierarchy.

### 5.5 The graph connects failing tests to the fix

Despite the mislabelled and missing edges, the graph does link the code a failing test touches to the code that must change. An entry set exists for 86 of 91 usable tasks, and 72 tasks have a gold function that the failing tests execute and that is a node. In 69 of these 72 (95.8%, CI 88.5 to 98.6), every such gold function is reachable from the entry set along directed `calls` edges.

| Minimum hops, entry to gold | 0 | 1 | 2 | 3 | 4 to 7 | Unreachable |
| --- | --- | --- | --- | --- | --- | --- |
| Tasks | 32 | 9 | 13 | 9 | 6 | 3 |

We had hypothesized that gold functions would often be reachable at runtime but not in the static graph. On this data the hypothesis is rejected. The obstacles to an agent are omission (async code is absent, so it cannot be reached), mislabelling (half the frontier of a node is class structure), and distance on a hub-heavy graph, not missing connectivity.

### 5.6 Embeddings and retrieval

**The encoder is not public EmbeddingGemma.** No candidate approaches identification: best mean cosine to the provided vectors is 0.073 for EmbeddingGemma 300m (name, document prompt), 0.015 for EmbeddingGemma 2, and 0.036 for the MiniLM control, with pairwise rank correlation at most 0.17 for either EmbeddingGemma (0.21 for the control). The provided vectors are not unit-normalised, and are strongly anisotropic: mean cosine to the mean vector is 0.87 to 0.95, and top-1 neighbour cosine averages 0.999.

**What similarity tracks.** Cosine separates call-adjacent pairs from random pairs (mean AUROC 0.90 over 127 graphs) and containment pairs (0.90), gold co-change pairs less (0.81), same-module and name-overlap pairs weakly (0.68). But a node's ten nearest neighbours are 45% to 67% test nodes and only 2% to 6% call neighbours, so the tool mostly returns tests.

**Retrieval.** Over all 129 tasks:

| Mode | Gold node in top 10 | Gold file in top 5 | Returns nothing |
| --- | --- | --- | --- |
| Resolved issue identifiers | 30 | 40 | 54 |
| Harness: identifiers + cosine | 31 | 43 | 54 |
| Harness + 1-hop neighbours | 31 | 45 | 54 |
| BM25 over node text | 38 | 63 | 0 |
| BM25 over source files | 53 | 85 | 0 |

The cosine step changes three tasks, one-hop expansion two more. Free-text BM25 over the source finds the gold file in 47 tasks where the harness mode does not, and the reverse holds for 5 (exact McNemar p < 0.0001). Two tasks in five give the harness mode nothing to start from, because no identifier in the issue resolves to a node.

## 6. Released resources

- **Audit toolkit**: the census, id reconciliation, tracer and runner, analysis, edge-semantics classifier, encoder and retrieval audits, with a synthetic-fixture test suite (15 tests).
- **Typed graph v2**, one per base commit, in the harness's node-link format: all modules, classes, functions *including async*, and nested definitions under the provided scheme; typed edges `contains`, `imports`, `inherits`, `calls_static` (the provided edges, containment removed) and `calls_dynamic` with test provenance; `tests` edges from each test to the library code it executes.
- **Per-task profile**: gold coverage, async exposure, resolver hazards, entry distance, reachability, recall and precision.

The v2 graph keeps the provided ids so it can replace the shipped file in the harness without changes. Ninety-eight nested definitions per fastapi graph share a flattened id with a top-level one; the graph records them on that node rather than overwriting it.

## 7. Implications and limitations

**For agent builders.** Treat `get_code_neighbors` output as a mix of call and class-structure edges. Never trust a resolved name without checking the returned id, and fall back to text search for async code. A Skill script inside the sandbox can serve the typed graph legally.

**For benchmark designers.** Publish the graph's semantics. Separate containment, inheritance and calls into typed edges. Return "not found" instead of a substring match, and include async definitions.

**Limitations.** Precision is a lower bound measured only inside executed code, and our runs used per-task environments with network access rather than the official cached dependencies: 9 collection errors, 2 import errors and 4 requests tasks without collectable tests are our setup, not the tasks. Recall covers the paths the tests exercise. The generator's rules are inferred from its outputs, and the 80-edge hand-review sample is not yet labelled. The hidden test set uses the same generator, but we have not measured it. Results come from four repositories, two of which dominate the task count.

## 8. Reproducibility

Every number in this paper comes from files in the project repository, `planA_results/` on branch `governDynamics/hopeful-sagan-rzmbod`: census in `vm-a2/results/a1_v2/`, id reconciliation in `a7/`, dynamic oracle in `a2_v3/` (per-task runs in `a2_runs/`), edge semantics in `a8/`, embeddings in `a3/` and `vm-main/results/a4/`, retrieval in `a5_v3/`, release artefacts in `a6_v3/`. `scripts/ops/rerun_fixes.sh` and `scripts/ops/run_gaps.sh` regenerate them from the competition archive; script checksums are in `scripts_sha256_v3.txt`.

---

## References

1. Salis, V. et al. 2021. *PyCG: Practical Call Graph Generation in Python.* ICSE. https://consensus.app/papers/details/b2f145e1fe035745a8b387e3f4b74740/
2. Xiang, Y. et al. 2026. *InferCG: Enhancing Python Call Graph Generation via Static Analysis and LLMs.* TOSEM. https://consensus.app/papers/details/f9176ef478f6583eaef6c696919949a8/
3. Kummita, S. et al. 2021. *Qualitative and Quantitative Analysis of Callgraph Algorithms for Python.* https://consensus.app/papers/details/2565e0e7f78a5d05b74121697239f917/
4. Sui, L. et al. 2020. *On the Recall of Static Call Graph Construction in Practice.* ICSE. https://consensus.app/papers/details/93f20309e7855a5898b1f0f293a01442/

5. Lin, et al. 2026. *How Much Static Structure Do Code Agents Need? (CodeAnchor).* ISSTA. https://www.alphaxiv.org/abs/2606.26979
6. Hu, et al. 2026. *LARGER: Lexically Anchored Repository Graph Exploration and Retrieval.* https://www.alphaxiv.org/abs/2605.16352
7. Wei, et al. 2026. *SemNav: Semantic Navigation for Repository-Level Issue Localization.* https://www.alphaxiv.org/abs/2609.31176
8. Paipuru. 2026. *The Navigation Paradox in Large-Context Agentic Coding (CodeCompass).* https://www.alphaxiv.org/abs/2602.20048
9. Liu, et al. 2026. *DAIRA: Dynamic Analysis-enhanced Issue Resolution Agent.* https://www.alphaxiv.org/abs/2603.22048
10. Wang, et al. 2026. *GREPO: A Benchmark for GNNs on Repository-Level Bug Localization.* https://www.alphaxiv.org/abs/2602.13921
11. Liu, et al. 2025. *GraphLocator: Graph-guided Causal Reasoning for Issue Localization.* https://www.alphaxiv.org/abs/2512.22469
12. Caumartin, et al. 2026. *Retrieval-Oriented Code Representations in Agentic Bug Localization.* https://www.alphaxiv.org/abs/2607.11046
13. Lasbordes, et al. 2026. *Small Agents with Semantic Search: Efficient Multilingual Code Localization.* https://www.alphaxiv.org/abs/2610.05099
14. Google DeepMind. 2026. *EmbeddingGemma 2 model card.* https://huggingface.co/google/embeddinggemma-2
15. Gemma Team. 2026. *Gemma 4 Technical Report.* https://www.alphaxiv.org/abs/2607.02770
16. Li, et al. 2026. *RepoMirage: Probing Repository Context Reasoning in Code Agents with Perturbations.* https://www.alphaxiv.org/abs/2605.26177
17. Yang, et al. 2026. *Agents' Overreliance on Unreliable Tools.* https://www.alphaxiv.org/abs/2609.05587
18. Helm, D. et al. 2024. *Total Recall? How Good Are Static Call Graphs Really?* ISSTA. https://consensus.app/papers/details/c62e543287615a7abeb7430f76391389/

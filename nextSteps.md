# After Plan A: next steps and a graph-navigation metric for checkpoint-time evaluation

Written Oct 11, 2026 from the Plan A outputs in `planA_results/` and the Oct 7 research plans; corrected the same day against the paper draft. The work order for everything after shipping Plan A is `HANDOFF_planBC.md`.
Status language follows `compPhilosophy.md`: a number below is a Plan A measurement; anything
about agents is a hypothesis until a Plan B run exists.

## 1. What Plan A established (numbers from `planA_results/`)

| Question | Result | Source |
| --- | --- | --- |
| Are source definitions nodes of the provided graph? | sync functions 100%, classes 97.8%, **async functions 0 of 28,113**, nested 0 of 4,675 (source files, all 127 graphs) | `a1_summary.json` `omission_direct_source_files` |
| Does this touch the fix sites? | 81 of 711 gold symbols are async, 0 present; 22 of 129 tasks (21 of 67 fastapi) have an async gold function | `a1_summary.json` `gold_async`, `task_rates` |
| What does the harness resolver return for a short name? | present function queried by bare name: wrong node 41% (suffix tier, same name, 62% of those a test node); absent async def by bare name: wrong node 65%; 26 of 129 tasks have a gold function whose bare-name query lands on a wrong node | `a1_summary.json` `misresolution` |
| How much of the real call graph is in the provided `calls` edges? | recall 38% (between existing nodes), 31% agent-facing; 71 to 81% of misses are between nodes that both exist, so this is not only the async hole | `a2_summary.json` `usable_tasks` |
| Are the provided edges right where code ran? | observed precision 21% (lower bound; tests exercise narrow paths) | `a2_summary.json` `observed_precision` |
| Can the failing test reach the fix site through the provided graph? | Oct 10 analysis (test nodes as seeds): 5.6%. **Superseded by the paper's entry-set analysis: 69 of 72 tasks reachable along directed `calls`, gold is an entry point in 32.** See `HANDOFF_planBC.md` section 0 | `a2_v3` (paper), `a2_summary.json` (old) |
| Which encoder made the 256-d vectors? | none of EmbeddingGemma-300m, EmbeddingGemma-2, MiniLM (best mean cosine 0.07); vectors are not unit-norm (mean norm 4.5), PC1 holds 44 to 75% of variance, mean pairwise cosine 0.80 to 0.91 | `a3_summary.json` (both) |
| What does cosine track? | call adjacency AUROC 0.90, containment 0.90, gold co-change 0.81, same module 0.68, name overlap 0.68 | `a4_summary.json` |
| What does `search_similar_code` retrieve for the issue text? | harness mode: node Hit@10 27%, file Hit@5 34%, empty for 41% of tasks (no resolvable anchor); BM25 over node text: 33% / 52%, file-level gain p = 0.0008 (McNemar); adding 1-hop neighbours changes nothing | `a5_summary.json` |
| Plan B inputs | typed graph v2 for 127 commits; `task_profile.csv` with d x v cells: d undefined 82, d0 19, d1 5, d2+ 23; v high 72 | `a6_summary.json`, `task_profile.csv` |

Two things in this table were not in the Oct 7 plan and change what comes next:

1. The embeddings look structural, not textual. Cosine separates call-adjacent pairs (0.90) far better than name-overlap pairs (0.68), and no text encoder comes close. If the vectors are a function of the graph (node2vec-style or a GNN over the provided edges), then `search_similar_code` is a second view of `get_code_neighbors`, which is exactly what A5 saw (1-hop adds nothing). This is cheaply testable and should be settled before the write-up says "unknown encoder".
2. (Corrected Oct 11.) The paper's entry-set analysis shows the provided graph does route from where failing tests enter the library to the fix in 69 of 72 tasks. The obstacles are absent async code, class-structure edges labelled `calls`, and hub distance, not missing connectivity. B6 (v2-graph tools) stays in Plan B but is not promoted; see `HANDOFF_planBC.md`.

## 2. Next steps, in order

### 2.1 Ship Plan A (this week)

- Write the 3,000-word Plan A Writeup and submit it. Tie-break favours the earlier entry and every number is already on disk. Quote the `nontest` scope for recall and the `source_files` scope for omission (per `scripts/REVIEW.md`).
- Split out two short standalone Writeups as the plan intended: the async omission note (A1) and the retrieval note (A5: symbol-only search returns nothing for 41% of tasks, BM25 over the same node text beats it at file level).
- Release the v2 graphs and `task_profile.csv` with a data card (Best New Resource track). State the `calls_dynamic` provenance choice (direct edges only) and the A2 coverage (90 of 129 usable, per repo).

### 2.2 Close three Plan A loose ends (CPU, 1 to 2 days, parallel with writing)

- A3b, structural-encoder test. Fit node2vec (and a one-layer GNN) on each provided graph and compare to the vectors: cosine, Procrustes alignment, and the isolated-node check (a fastapi graph has about 2,900 isolated nodes; a structural encoder gives them degenerate or random vectors, a text encoder does not). Also regress cosine on graph distance. If it fits, A5's story becomes "the semantic tool is a graph tool" and H-A3 flips from "unknown" to "identified as structural".
- Confirm `edge_type` case handling with one real `get_code_neighbors(node, "CALLS")` call once the harness runs (B0). The code reading says it returns nothing; the Writeup should quote a run.
- A2 coverage note: 39 tasks are not usable (9 collection errors, 4 no-tests, 2 import errors, 1 killed, plus 23 clean runs where no expected fail-to-pass test was confirmed). List them in the Writeup appendix so the 90-task denominators are auditable.

### 2.3 B0 is the gate for everything else (GPU)

Nothing in Plans B or C, and nothing in the metric below, can be validated until `swegemma eval` runs on the 129 tasks with the per-turn tool log. Start B0 the day the Plan A draft is done. The budget check in the plan still holds: if one 129-task pass takes more than a GPU-day, drop to 2 seeds before dropping any arm.

### 2.4 Then Plan B as designed

Run B2 (baseline) and B3 (no graph tools) paired first, then B4 (push), B6 (v2-graph tools), B5. The full work order, with the strata the paper added (async gold, gold-is-entry-point), is `HANDOFF_planBC.md`.

## 3. The proposal: a graph-navigation metric that runs between training checkpoints

### 3.1 Verdict

Yes, with one reframing. "Improves graph usage" must not mean "calls the graph tools more". Plan A shows the provided graph is low-recall (31 to 38%), low-precision (21%) and misresolves 41% of bare-name queries, and the literature shows RL against graph walks cut echo but also cut distinct-evidence recall (GraphEcho [22]). A metric that rewards graph calls will be gamed in one epoch. The metric has to reward graph calls that land on the task's real structure and are acted on, and penalise calls that do not, and it has to be scored against the audited oracle (A2 traces, v2 graph), not against the provided graph.

Reframed, the metric fits three places at once: it is Plan B's process-metric block made precise, it is the dev-set signal Plan C needs for C2a/C4/C5 (nothing in the plan said how to pick a checkpoint or a prompt variant without a full resolution run), and it is a resource (metric plus stratified dev slice plus scorer) for the Best New Resource track. It touches Tasks & Benchmarks, Graph Reasoning and Tuning.

### 3.2 Definition (working name: graph-navigation scorecard; keep it a vector, not one number)

Inputs per trajectory: the harness trace (`traces/trace_<id>.json`, ATIF SessionTrace: ordered tool calls with arguments and returned JSON, file reads, edits, submit). Oracle per task, all from Plan A: gold files and functions (`a1_gold_symbols.jsonl`), the relevant region R = gold functions plus functions executed by fail-to-pass tests (`a2_runs/<task>/trace_pre.json`, `trace_post.json`), the v2 graph with `in_provided_graph` flags, hub in-degrees, and the d x v cell (`task_profile.csv`).

| Component | Definition | Why it is in | Oracle |
| --- | --- | --- | --- |
| Resolution precision | share of graph calls whose argument resolves (replaying the harness's 4-tier resolver offline) to the node the agent named, not a same-name collision | A1: 41% of bare-name queries land on a wrong node; a deployable fix exists (qualified names) | provided graph + resolver copy in `common.py` |
| Structural yield | share of graph calls whose returned nodes intersect R | separates useful calls from decorative ones | A2 traces, gold symbols |
| Link-following rate | share of graph calls followed within 2 steps by a `read_file` on a returned node's file | CodeAnchor's measure [5]; a call not acted on did nothing | trace order |
| Necessity match | agreement between "made at least one graph call" and the task's need label (d of at least 1 or v low = needed; d0 and v high = not needed); later replaced by the counterfactual label from B2 vs B3 | the adoption paradox [20]: calls should rise where structure is needed, not everywhere | `task_profile.csv` |
| Waste rate | share of graph calls returning a node already seen, or a hub (in-degree above the per-repo 95th percentile) whose neighbours are mostly test callers | echo walks and hub capture [22], Plan A hub table | v2 graph degrees |
| Hallucination channel | out-of-list tool names, malformed arguments, unresolvable node arguments, with vLLM parser errors logged separately from model non-calls [36] | Plan C1, Flat Score [11] | harness log |
| Localization outcome | steps to first gold-file read; gold-file coverage; second-file discovery on the 38 multi-file tasks | Plan B's primary metrics; the thing the components should predict | gold files |

Report each as a task-level rate with a Wilson interval, per d x v cell and per repo. Do not average them into one score for the paper. For checkpoint selection pick one primary: structural yield restricted to tasks whose need label is "needed", with the hallucination channel as a veto.

### 3.3 Running it between checkpoints

- Dev slice: 24 tasks from `task_profile.csv`, stratified over the d x v cells and the three repos, A2-usable only, held out from any training trajectories. Hold a second disjoint 24 for the final check.
- Budget: `max_tool_calls` 15, `max_time_minutes` 10, 1 seed, temperature 0.2. The metric needs the trace, not the resolution, so Phase 2 cost is irrelevant and the run is about 1 GPU-hour instead of a GPU-day.
- Scorer: a CPU script that takes a results directory and the Plan A oracle and writes the scorecard CSV. It can be written and fixture-tested now, before B0, against the documented trace format, and run on the first B2 outputs the hour they exist.
- Use: after every C2a prompt variant, every C4 probe threshold, every C5 LoRA epoch, run the slice and keep the candidate only if primary rises without the hallucination channel rising.

### 3.4 Validation, which is the actual experiment

The metric is worth nothing until it predicts resolution. The validation is a babystep chain:

1. Compute the scorecard on B2 and B3 (already planned runs, no new GPU). Report the correlation of each component with gold-file coverage and with resolution, pooled and per cell, across 129 tasks x 3 seeds. Hypothesis: structural yield and link-following correlate with localization; raw graph-call count does not.
2. Compute it on B6 (v2 graph). Hypothesis: resolution precision and structural yield rise on the same prompts because the map improved, which shows the metric separates map quality from model behaviour.
3. Compute it on B4/B5 (push). Hypothesis: graph calls fall, link-following of the brief rises, outcome does not fall; a metric that only counted calls would call this a regression.
4. Only then use it for selection in Plan C. The resolution check on the held-out 24 (and the full 129 at the end) is the SWE-bench-style test the user asked for: the swegemma harness is a SWE-bench-shaped evaluation (same two-container, pytest-pass criterion), and a public-SWE-bench run would require building graphs and embeddings for new repos (the v2 builder can, the embeddings cannot without the encoder), so keep that as a stretch and say so.

### 3.5 About "existing SOTA approaches"

Inside the fixed harness, the comparable approaches are agent configurations, not external agents: the sample submission (root coder plus read-only analyzer sub-agent), tools-off, grep-first, push brief, qualified-name prompting, v2-graph tools. Scoring these with the scorecard is Plan B. Running OpenHands or mini-SWE-agent with Gemma-4-31B would be a separate harness with no graph tools, so the metric would not apply; do not spend the time. Runs on the smaller Gemma 4 variants in the local CLI are cheap and show whether the components move with model size (Quality criterion), if GPU time allows.

### 3.6 Risks

- Small numbers: 129 tasks, resolution 10 to 35%, so component-to-resolution correlations will have wide intervals. Localization outcomes are the primary targets; resolution is reported with intervals.
- Goodhart: once used for selection the metric is a target. The held-out slice and the final full run are the guard; the GraphEcho result [22] is the cautionary example to cite.
- The need label from d x v is defined for only 47 tasks (d) and should be replaced by the counterfactual label from B2 vs B3 as soon as those exist.
- Serving-stack confounds [36]: parser-level failures must be kept out of the hallucination channel or the metric measures vLLM, not Gemma.

## 4. Calendar check

Today is Oct 11; the deadline is Nov 12. Plan A write-up by about Oct 16 (ahead of the Oct 20 plan), A3b and the scorer in the same week on CPU, B0 started by Oct 15, B2/B3 by Oct 25, scorecard validation on them by Oct 28, B6 and push arms by Nov 4, Plan C selection runs Nov 4 to 10. First cut if time runs short stays C5, then B5.

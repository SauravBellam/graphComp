# graphComp handoff 2: everything after "Ship Plan A"

Written Oct 11, 2026 for the implementing model. Supersedes section 7 of `HANDOFF.md` from step 5
onward and section 2 of `nextSteps.md`. Plan A is done and its paper ("Is the Map Right?", draft of
Oct 11) is being submitted. This document is the work order for what follows: the Plan A loose ends,
B0, Plan B, the graph-navigation scorecard, and Plan C. The research design is fixed by the Oct 7
plans (reproduced in `HANDOFF.md`); this document only orders, scopes and updates it with what Plan A
and the paper revision learned. Read `compPhilosophy.md` first; its rules apply to every run below.

## 0. Facts that changed since the Oct 7 plan

The paper's final analysis (results folders `a1_v2`, `a7`, `a2_v3`, `a8`, `a5_v3`, `a6_v3`) lives in
the paper repository `https://github.com/SauravBellam/is-the-map-right` under `planA_results/`,
with `papers/verify_paper1_numbers.py` checking 66 claims against the files and
`papers/for_judges.pdf` mapping every section to its files. **This repository's `planA_results/` is
the Oct 10 snapshot (`a1`, `a2`, `a5`, `a6` with the wrong nested-id scheme) and must not be used
for Plan B inputs.** The revision changed four things that the Oct 7 plan and `nextSteps.md` still
state the old way. Use the paper's numbers.

| Item | Oct 7 plan / early run | Paper (use this) |
| --- | --- | --- |
| Usable A2 tasks | 90 | 91 (fastapi 45, rich 39, requests 7) |
| Reachability from failing tests to gold | 5.6% statically reachable (test nodes as seeds) | 69 of 72 (95.8%) reachable from the *entry set* (where the failing tests enter the library) along directed `calls`; gold is itself an entry point in 32 tasks and within 3 hops in 63 |
| Nested definitions | 0 present | 79.7% present under a flattened id (`<locals>` dropped); id reconciliation step A8 |
| What `calls` means | call edges, high precision expected | a mix of calls, containment and inheritance: 22.3% of unexercised edges are method-to-own-class, 13.8% class-to-subclass, 13.3% class-to-base; 56.1% of unexercised edges have no textual basis in the caller |

Consequences for the plans:

- The provided graph *can* route from a failing test's entry point to the fix. So the "better map"
  arm (B6) is no longer justified by routing failure. It is justified by the three obstacles the
  paper names: absent async code (31% of fastapi tasks), mislabeled class-structure edges, and
  hub distance. B6 stays in Plan B but after B2/B3/B4, not before.
- H-B2 (pull benefit only at d of at least 2 with low v) now has a sharper mechanism to test:
  on tasks with an async gold function, graph tools cannot help at all; on tasks where the gold
  function is an entry point (32 of 72), the graph is one hop from a test the agent can run. Add
  both as strata (see B1).
- Every Plan B and C step that reads `task_profile.csv` or the v2 graphs must take them from the
  paper repository (`vm-a2/results/a6_v3/task_profile.csv`, `a6_v3/graphs_v2/*.json.gz`, gzipped,
  662 MB uncompressed). The Oct 10 copy here (`vm-a2/results/a6/`) has the pre-reconciliation
  nested counts and the old reachability columns.

## 1. Plan A loose ends (CPU; do these while B0 is being set up)

### 1.1 Two repositories: keep them straight

The paper, its LaTeX, figures, `figure_data.json`, the final result folders, the revised scripts
(`a7_id_audit.py`, `a8_edge_audit.py`, `ops/rerun_fixes.sh`, `ops/run_gaps.sh`), the 15-test
fixture suite and `verify_paper1_numbers.py` (66 PASS, 0 FAIL, 9 INFO) are in
`SauravBellam/is-the-map-right`. This repository (`governDynamics`) holds the philosophy, the
plans, the harness reference and the Oct 10 run. Do two things: add a line at the top of
`planA_results/README.md` here pointing to the paper repository and marking this folder as the
superseded first run (done Oct 11); and start `planB_results/` and `planC_results/` in whichever
repository the Plan B paper will live in, with the same layout and a verifier script like
`verify_paper1_numbers.py` from day one. Everything in sections 2 to 5 that names a Plan A file
means the paper repository's copy.

### 1.2 Structural-encoder test (A3b, half a day)

A4 shows cosine separates call-adjacent and containment pairs at AUROC 0.90 but name-overlap pairs
at 0.68, no text encoder comes within cosine 0.1, and the top-1 neighbour cosine averages 0.999.
Hypothesis to test: the vectors are a function of the graph, not of the text.

1. Same text, different commit. The 127 graphs share thousands of unchanged functions across
   commits of the same repo. For node ids with byte-identical `text` in two graphs, compute the
   cosine between their two vectors. A text encoder gives 1.0; a structural or per-graph encoder
   gives less, and how much less tracks how much the surrounding graph changed. Report the
   distribution per repo. This single test is decisive and costs minutes.
2. Duplicate vectors. Count exact-duplicate vectors per graph and what the duplicated nodes have
   in common (identical text, identical neighbour sets, isolated).
3. Isolated nodes (fastapi has about 2,900 per graph): is their vector distribution
   distinguishable from connected nodes? A structural encoder gives them a degenerate cluster.
4. Fit node2vec on the provided `calls` graph (dimension 256) and compare after orthogonal
   Procrustes; report cosine and rank correlation like A3 did.

Output: `a3b_summary.json`, one paragraph for the paper's section 5.6 or a standalone note. If (1)
gives cosines well below 1 for identical text, the paper's "not reproducible from EmbeddingGemma"
becomes "not a text embedding", which is the stronger and more useful claim.

### 1.3 Two checks that need the running harness (fold into B0)

- `get_code_neighbors(node, edge_type="CALLS")`: the harness README's examples use upper case, the
  data holds lowercase `calls`, and the code reading in `scripts/REVIEW.md` says the filter is a
  plain `==`. Make one call each way and record the result. If uppercase returns nothing, that is an
  agent-facing hazard for the paper's "for agent builders" paragraph and for every Plan B prompt.
- `get_code_subgraph` edge `type` field: README shows `"CALLS"`; confirm what it returns.

### 1.4 Non-usable task accounting

129 minus 91 is 38 tasks. The paper and the judges' guide name 9 collection errors, 2 import errors
and 4 requests tasks without collectable tests (15). The other 23 are clean runs where no expected
fail-to-pass test was confirmed (rich_4006 is now clean with 4 hung tests deselected and recorded
as failed). List all 38 with their run class and reason in `a2_run_status.csv` terms in the paper
repository's `planA_results/README.md`, so the 91 denominator is auditable by a judge.

## 2. B0: harness bring-up and logging (GPU; the gate for everything below)

Goal: `swegemma eval` runs the 129 tasks with `gemma-4-31b-it-qat-w4a16-ct` on hardware that fits
the weights (not a T4), with per-turn logs that Plan B, the scorecard and Plan C can read.

1. Serve the model with vLLM as the harness does (`tool_call_parser=gemma4`,
   `reasoning_parser=gemma4`, `max_model_len=32768`, `enable_lora=True`). Record GPU type, VRAM,
   vLLM version and the exact launch line in `planB_results/provenance/`.
2. Run the sample submission on 5 tasks (one per d x v cell where possible) with the sample budgets.
   Confirm: traces land in `results/<run>/traces/trace_<id>.json` (ATIF SessionTrace), each tool
   call carries name, arguments and the returned JSON, and the `logs/<id>.log` transcript shows
   parser errors distinguishable from model non-calls (the serving-stack confound, ref [36] of the
   lit review). If parser errors are not separable from the trace, add a vLLM request log and join
   on timestamps; do not proceed without this.
3. Run the gold patches through Phase 2 (`--skip-agent-patch` for the baseline-fails check, then
   the reference patch) on all 129 tasks. Every task must resolve with the gold patch. Tasks that do
   not are excluded from every Plan B denominator and listed.
4. Time one full 129-task pass of B2's configuration at 1 seed. The plan assumes about one
   GPU-day. If it is slower, cut seeds from 3 to 2 before cutting any arm.
5. Make the two `edge_type` calls from section 1.3 and record them.

Deliverable: `planB_results/b0/` with the timing, the gold-patch resolution table, the five sample
traces, and a `B0.md` stating what was verified. No Plan B arm starts before B0.md exists.

## 3. Plan B: pull, push, or ignore (GPU)

Design as in the Oct 7 plan (B1 to B6); changes marked.

### B1 Task profile (CPU, from `a6_v3/task_profile.csv`)

Columns Plan B stratifies on: `cell` (d x v), `multi_file`, repo, `a2_usable`. **Add** two
strata from the paper's findings: `gold_has_async` (22 tasks; graph tools cannot reach the gold
function) and `gold_is_entry_point` (32 of 72; gold is one hop from a test the agent can run). Add
the B0 gold-patch-resolves flag. Publish the resulting 129-row table as `planB_results/b1/profile.csv`.

### B2 Baseline

Single agent, all 9 tools, the sample prompt minus the sub-agent, temperature 0.2, fixed budgets
(propose `max_tool_calls` 50, `max_time_minutes` 30, 3 seeds; adjust from B0 timing). This is the
reference arm for everything; its traces feed the scorecard (section 4) and Plan C's C1.

### B3 Remove graph tools

B2 without `get_code_neighbors`, `search_similar_code`, `get_code_subgraph`; same seeds, paired by
task. If adoption in B2 is near zero, B3 measures the cost of the tools' presence (prompt tokens and
context), which is itself a finding; say so.

### B4 Push (research harness)

B2 plus a structural brief prepended to the problem statement: 1-hop callers and callees of the
resolved issue identifiers from the **v2** graph (not the provided one: the v2 graph includes
async definitions and separates `contains`, `inherits` and `calls_static`), test callers filtered,
grouped by file, capped at about 400 tokens. Record the brief per task so the scorecard can score
"link-following of the brief".

### B5 Push, harness-legal

The same brief produced by a Skill script inside the sandbox (`skills/<name>/SKILL.md` plus a `.py`
script, the only legal way to ship custom structure; see `HARNESS_README.md` section 2.2 and 2.4).
Confirm first that a skill script can read the graph file in the sandbox and that its output
reaches the model; if not, B5 degrades to "the brief is precomputed and shipped as text per repo",
which is still legal because the hidden set uses the same generator but is weaker. State which.

### B6 Better map

B2 with the three graph tools backed by the v2 graph. Requires a harness patch that loads
`graphs_v2/<key>.json` in place of the provided graph and maps the `edge_type` filter to the v2
edge types. Not deployable in a submission; it is a research arm.

### Metrics and analysis (unchanged from the plan, plus the scorecard)

Primary: steps to first gold-file read, gold-file read coverage, second-file discovery on the 38
multi-file tasks. Secondary: resolution (paired McNemar), pass^3 consistency. Process: the
scorecard of section 4. Analysis: mixed-effects logistic regression with task random effects,
condition x d x v, reported per repo as well as pooled. Report every rate with its interval and the
denominator.

Order of runs: B0, B2, B3 (paired, same week), then B4, then B6, then B5. First cut if time runs
short: B5, then B6.

## 4. The graph-navigation scorecard (CPU scorer; Plan B's process block, Plan C's dev signal)

Purpose: a per-trajectory measure of whether graph tool use was *correct*, scored against the Plan
A oracle, cheap enough to run on a small slice between training checkpoints. It is not a single
number and it must never reward graph calls as such: the provided graph has 31 to 37% recall, 21%
observed precision and misresolves 41% of bare-name queries, so "more graph calls" is not a goal.

### 4.1 Inputs

- Trajectory: `traces/trace_<id>.json` from the harness (ordered tool calls with arguments and
  returned JSON, file reads, edits, submit) plus the transcript log for parser errors.
- Oracle per task, all from the paper repository: gold files and functions
  (`a1_v2/a1_gold_symbols.jsonl`), the relevant region R = gold functions plus functions executed
  by fail-to-pass tests (`a2_runs/<task>/trace_pre.json` and `trace_post.json`), the v2 graph
  (`a6_v3/graphs_v2/*.json.gz`) with `in_provided_graph` and degrees, the entry set and entry
  distance (`a2_v3/a2_gold.jsonl`, `entry_nodes` in `analyze.py`), the B1 profile.

### 4.2 Components (each a task-level rate with a Wilson interval; aggregate per cell and per repo)

| Component | Definition | Oracle |
| --- | --- | --- |
| Resolution precision | share of graph calls whose argument, replayed through the harness's 4-tier resolver (`common.resolve_node_name`), returns the node the agent named rather than a same-name collision; report the collision-to-test-node share separately | provided graph |
| Structural yield | share of graph calls whose returned nodes intersect R | A2 traces, gold symbols |
| Link-following rate | share of graph calls followed within 2 tool calls by a `read_file` on a returned node's file (for B4/B5: reads on files named in the brief) | trace order |
| Necessity match | agreement between "made at least one graph call" and the need label; label v1 = d of at least 1 or v low; label v2 (replace as soon as B2/B3 exist) = tools-arm found the gold file and no-tools arm did not, per task and seed | B1 profile, then B2 vs B3 |
| Waste rate | share of graph calls returning a node already seen in the trajectory, or a hub (in-degree above the per-repo 95th percentile) whose neighbours are mostly test nodes | v2 graph degrees |
| Hallucination channel | out-of-list tool names, malformed arguments, unresolvable node arguments; vLLM parser failures logged separately and excluded | harness log |
| Async hazard hits | graph calls whose argument names a gold or R function that has no node (async or nested), and what the resolver returned instead | A1 audit |
| Localization outcome | steps to first gold-file read, gold-file coverage, second-file discovery | gold files |

Primary for checkpoint selection: structural yield on tasks whose need label is "needed", with the
hallucination channel as a veto (a candidate that raises yield and raises the channel is rejected).

### 4.3 Implementation

`scripts/planB/scorecard.py --results <swegemma results dir> --oracle <planA_results> --profile
<b1 profile> --out <dir>`: writes `scorecard_tasks.csv` (one row per task x seed x component),
`scorecard_summary.json` (rates with intervals per cell and repo) and `scorecard_calls.jsonl` (one
row per graph call with its classification, so every number can be traced to a call). Build it
against the trace format documented in `HARNESS_README.md` section 9.2 and a synthetic trace
fixture (add to `scripts/tests/`), before B0 finishes. Run it on the five B0 sample traces the hour
they exist and fix whatever the real format breaks.

### 4.4 Dev slice for checkpoint-time runs

24 tasks from the B1 profile: stratified over d x v cells, the three main repos, `gold_has_async`
and `gold_is_entry_point`; A2-usable and gold-patch-resolving only; disjoint from any trajectory
used for training (C5) and from a second held-out 24 used only for the final check. Budget per
run: `max_tool_calls` 15, `max_time_minutes` 10, 1 seed, temperature 0.2. Only the trace is needed,
so a slice run costs on the order of 1 GPU-hour. Write the two slices to `planB_results/b1/
dev_slice.txt` and `holdout_slice.txt` before any Plan C run.

### 4.5 Validation chain (this is the experiment; the metric is worthless until it passes)

1. Score B2 and B3 (no new GPU time). Report each component's correlation with gold-file coverage
   and with resolution, pooled and per cell. Hypothesis: structural yield and link-following
   correlate with localization; raw graph-call count does not.
2. Score B6. Hypothesis: resolution precision and structural yield rise on unchanged prompts because
   the map improved; the metric separates map quality from model behaviour.
3. Score B4/B5. Hypothesis: graph calls fall, link-following of the brief rises, outcome does not
   fall. A call-count metric would call this a regression.
4. Only after 1 to 3 is the scorecard used for selection in Plan C. The resolution check on the
   held-out slice and the final 129-task run is the SWE-bench-style test; the swegemma harness is a
   SWE-bench-shaped evaluation (two containers, pytest-pass criterion).

Out of scope, and say so in the write-up: running external agents (OpenHands, mini-SWE-agent) with
the model, since they have no graph tools; running public SWE-bench, since the v2 builder can make
graphs for new repos but the embeddings cannot be reproduced (A3). Runs on smaller Gemma 4
variants in the local CLI are a cheap Quality-criterion extension if GPU time remains.

## 5. Plan C: teaching the agent when to read the map (GPU)

Design as in the Oct 7 plan (C1 to C5), with the scorecard as the selection signal.

- C1 Channel census (CPU, from B2/B3 logs): valid-call rate, out-of-list names, unresolvable graph
  arguments, parser failures kept separate, error-budget curve over the tool-call budget. This is
  the scorecard's hallucination channel plus the budget curve; ship it as the short tool-channel
  note the plan scheduled for Oct 22.
- C2a Static repair (deployable): allowed-name list and an end-of-prompt structure checklist in the
  system prompt. **Add** the Plan A fix: instruct qualified names (`module.Class.method`) for every
  graph call, since bare names misresolve 41% of the time and qualified names 5%. Select the
  variant on the dev slice by the scorecard primary.
- C2b Dynamic repair (research harness): re-inject the tool inventory after each failed call.
- C3 Necessity probe: first-turn hidden states of the w4a16 model, L2 logistic probe, repo-held-out
  folds, AUROC with interval; label v2 from section 4.2 once B2/B3 exist.
- C4 Probe-and-prefill: if AUROC is at least 0.75, prefill one sentence naming the graph tool when
  the probe fires; adoption and payoff by cell; scorecard on the dev slice.
- C5 Adoption LoRA (stretch): rank 16, loss only on segments where a graph call preceded gold-file
  discovery, data from B2/B4 runs plus converted public trajectories; check tool-call validity
  first (ref [15]); evaluate each epoch on the dev slice with the scorecard, final check on the
  held-out slice and the full 129. **Check before committing**: whether PEFT training runs on the
  QAT compressed-tensors checkpoint; fallback is training on BF16 weights and serving on w4a16,
  which itself needs a check.

## 6. Compute, environment, unknowns

- GPU: the 4-bit 31B weights are about 19 GB; a T4 (16 GB) cannot serve them. Use Kaggle's 4x L4
  or a rented GPU with at least 24 GB and a generation the w4a16 kernels support (unverified; B0
  step 1 settles it).
- CPU: A3b and the scorecard run on the existing vm-main spec (8 vCPU, 31 GB). The archive (about
  22 GB) must be on the same machine as any Plan B harness run.
- Unknowns, check before relying on them: `edge_type` case handling (1.3); skill scripts' access to
  the graph file in the sandbox (B5); PEFT on the QAT checkpoint (C5); per-run GPU time (B0);
  whether parser errors are separable from model non-calls in the trace (B0 step 2).

## 7. Calendar (deadline Nov 12, 11:59 PM UTC)

| Date | Milestone |
| --- | --- |
| Oct 13 | Plan A paper submitted; 1.1 pushed; A3b started; scorecard scorer written against the documented format |
| Oct 15 | B0 complete (`B0.md`); async-omission and retrieval notes submitted as standalone Writeups |
| Oct 22 | B2 and B3 done (3 seeds or 2 per B0 timing); scorecard validation step 1; C1 note submitted |
| Oct 29 | B4, B6 done; scorecard steps 2 and 3; C2a variants selected on the dev slice |
| Nov 4 | Plan B Writeup submitted; B5 if time; C3 probe done |
| Nov 10 | Plan C Writeup submitted (C4, C5 if it ran) |
| Nov 12 | Choose the two Writeups to be judged (expected: Plan B for Overall, Plan A for Resource) |

First cut if time runs short: C5, then B5, then B6. Every completed step is a reportable result; a
Writeup for a plan ships as soon as its completed steps stand alone.

## 8. Repository conventions for the next runs

- Results go under `planB_results/` and `planC_results/` with the same layout as `planA_results/`:
  raw outputs untouched, `scripts/` with the exact code, `provenance/` with host specs, package
  versions, launch lines and sha256 of the code as run, and a README that maps every reported
  number to a file and field.
- Every pooled number carries numerator, denominator and an interval from `scripts/planA/stats.py`.
- Keep `nextSteps.md` as the short summary; this file is the work order. Update section 0 of this
  file whenever a number the plans depend on changes.

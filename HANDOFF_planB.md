# Plan B handoff: pull, push, or ignore, and the scorecard that predicts it

Written Oct 11, 2026 for the implementing model. This is the complete work order for Plan B. It
replaces section 3 and section 4 of `HANDOFF_planBC.md` and assumes the compute below; it does not
change the research design fixed in the Oct 7 plan (reproduced in `HANDOFF.md`). Read
`compPhilosophy.md` first: one component per experiment, ground truth before claims, every
number with numerator, denominator and interval.

**The goal of Plan B, stated once.** Map when the provided graph pays off for a 4-bit Gemma agent
(pull vs push vs ignore, by structural distance and lexical visibility), and in doing so find the
scorecard version that best predicts that payoff from a trajectory alone. The scorecard is not a
side product: the Plan B Writeup's resource is the scorecard plus its dev slice, and its empirical
result is the payoff surface the scorecard was validated against. Every GPU-hour below serves one
of those two.

## 0. Compute, and what it forces

Two single L4 GPUs (24 GB each): one GCP `g2-standard-8` (8 vCPU, 32 GB RAM, persistent disk,
no session limit) and one Kaggle L4 notebook (session-limited, no persistent state between
sessions). A quota increase is pending; nothing below depends on it.

What a single L4 forces:

- The competition model (`gemma-4-31b-it-qat-w4a16-ct`, weights 16 to 19 GB) leaves about 5 GB
  for KV cache. Whether vLLM serves it at the full 32,768-token context on one L4 is **unknown and
  is the first thing B0 measures**. If it must drop to a shorter context, record the number, run
  every arm at that context, and state in the Writeup that the harness ran at 4x L4 and we ran at
  one. Resolution rates then do not compare across papers; paired comparisons within Plan B do.
- No BF16 control (62 GB). Dropped.
- Sandbox concurrency 2 on the GCP box (each container 2 vCPU / 4 GiB; vLLM needs the rest).
- The two machines get different jobs. **GCP runs every full 129-task arm**, sharded with
  `--shard-index k --num-shards M` and resumable, since `task_results.jsonl` is append-only.
  **Kaggle runs the 24-task dev-slice jobs**, which finish inside one session. Results from both are
  copied to one results tree and merged by `instance_id`.

GPU-day accounting (one 129-task arm at one seed is assumed to take about one GPU-day; B0 replaces
this guess with a measurement):

| Job | Where | GPU-days |
| --- | --- | --- |
| B0 bring-up, gold-patch verification, timing | GCP | 1 |
| B2 baseline, 3 seeds | GCP | 3 |
| B3 no graph tools, 3 seeds | GCP | 3 |
| B4 push brief, 2 seeds | GCP | 2 |
| B6 v2-graph tools, 2 seeds | GCP | 2 |
| B5 harness-legal push, 1 seed | GCP | 1 |
| Forced-call arm (anti-Goodhart), dev slice | Kaggle | 0.2 |
| Qualified-names prompt, dev slice | Kaggle | 0.2 |
| Thinking budget 0 vs 4096, dev slice | Kaggle | 0.4 |
| Serialization flat / grouped / tree, dev slice | Kaggle | 0.6 |
| Context 16k vs 32k (or whatever B0 allows), dev slice | Kaggle | 0.4 |
| Model-size sweep 12B / 9B / E4B, B2 and B3 on dev slice | Kaggle | 1.2 |
| Final benchmark check: B2 and B3 on all 129 tasks for 12B / 9B / E4B (BF16; smaller models run faster) | Kaggle + GCP | 3 |
| Held-out slice resolution checks (2 uses) | Kaggle | 0.4 |
| **Total** | | **about 18.5** |

Two GPUs over the 24 days from Oct 13 to Nov 6 give at most 48 GPU-days; assume half is lost to
setup, failures and Kaggle session limits. The plan fits with margin. If B0 measures more than
1.5 GPU-days per arm, drop B2/B3 to 2 seeds first, then drop B5, then the dev-slice size sweep
(the final benchmark check in section 4.4 test 6 is not cut: it decides the paper).

## 1. B0: bring-up (GCP first, then the same on Kaggle)

Deliverable: `planB_results/b0/B0.md` plus the files it cites. No arm starts before it exists.

1. Serve the model with vLLM exactly as the harness does (`tool_call_parser=gemma4`,
   `reasoning_parser=gemma4`, `enable_auto_tool_choice`, `enable_lora=True`, `max_model_len=32768`,
   `gpu_memory_utilization=0.80`). Record whether it starts; if not, lower `max_model_len` in
   steps of 4,096 until it does and record the largest that works. Record GPU, driver, vLLM
   version, launch line, in `planB_results/provenance/`.
2. Run the sample submission on 5 tasks chosen one per d x v cell. Confirm the trace
   (`traces/trace_<id>.json`) carries, per tool call: tool name, arguments, the full returned
   JSON string (not only a parsed summary), turn index, and token counts; confirm compaction
   events are visible; confirm the transcript log (`logs/<id>.log`) distinguishes vLLM parser
   failures from model turns with no tool call. **If raw tool returns or compaction points are
   not in the trace, patch the harness to log them before anything else runs**; the scorecard's
   waste, serialization and context components need both.
3. Verify the 129 gold patches resolve in Phase 2 (`--skip-agent-patch` to confirm baseline
   failure, then the reference patch). Any task that does not resolve is excluded from every
   denominator and listed in `B0.md`.
4. Time one full B2 pass at one seed (this is also B2 seed 1; do not throw it away).
5. Make the two `edge_type` calls (`"CALLS"` vs `"calls"`) and one `get_code_subgraph` call,
   record the raw returns.
6. On Kaggle: repeat steps 1 and 2 only, and measure how many dev-slice tasks fit in one session.

## 2. B1: task profile and slices (CPU, from the paper repository)

Inputs: `is-the-map-right/planA_results/vm-a2/results/a6_v3/task_profile.csv` and
`a2_v3/a2_gold.jsonl`. Do not use this repository's Oct 10 copy.

Build `planB_results/b1/profile.csv`, one row per task, with: repo; `cell` (d x v);
`multi_file`; `a2_usable`; `gold_has_async` (22 tasks); `gold_is_entry_point` (32 of 72);
`entry_hops` (0 to 7 or none); `n_anchors` (0 for 54 tasks); B0 gold-patch-resolves flag.

Slices, written to `planB_results/b1/` and never changed afterwards:

- `dev_slice.txt`: 24 tasks, stratified over cell, repo (fastapi, rich, requests), `gold_has_async`
  and `gold_is_entry_point`; A2-usable and gold-resolving only. This is where scorecard versions
  are compared and where every Kaggle ablation runs.
- `holdout_slice.txt`: a disjoint 24 with the same stratification. Touched twice in Plan B
  (section 5, step 6) and otherwise never.
- The remaining 81 tasks plus both slices form the full-129 arms; the slices are not excluded
  from them, they are only the subsets the selection rule may look at.

## 3. The arms (GCP, full 129 tasks)

Common settings for every arm: temperature 0.2, `thinking_budget` 4096 with `include_thoughts`,
`max_output_tokens` 16384, `max_tool_calls` 50, `max_time_minutes` 30, `max_turns` 100, seeds
fixed as 1, 2, 3 and identical across paired arms. The system prompt is the sample submission's
root prompt with the sub-agent removed, saved verbatim under `planB_results/prompts/`. One run
directory per arm and seed: `planB_results/runs/<arm>_s<seed>/` with the harness's own layout.

| Arm | What changes from B2 | Seeds | Purpose |
| --- | --- | --- | --- |
| **B2** baseline | nothing | 3 | reference; pass^3; scorecard variance |
| **B3** no graph tools | the three graph tools removed from `tools:` | 3 | pull benefit; counterfactual need label |
| **B4** push | structural brief (section 3.1) prepended to the problem statement; graph tools kept | 2 | push vs pull |
| **B6** better map | harness patched to load `graphs_v2/<key>.json` in place of the provided graph; `edge_type` mapped to v2 types | 2 | map quality vs model behaviour |
| **B5** push, harness-legal | the B4 brief produced by a Skill script inside the sandbox | 1 | deployability of B4 |

### 3.1 The structural brief (B4, B5)

From the v2 graph: resolve the issue's identifiers with the harness resolver copy; for each
resolved non-test node list 1-hop callers and callees (edge types `calls_static`, `calls_dynamic`,
`inherits`), drop test nodes, group by file, show `module.Class.method` ids and one line of each
definition's signature, cap at 400 tokens. Save each task's brief to
`planB_results/briefs/<task>.md` before the run so the scorecard can score reads against it.
Tasks with no resolved identifier (54) get an empty brief; they are the natural control.

### 3.2 B5 feasibility check (do before B5 is scheduled)

Confirm in the sandbox that a Skill `.py` script can read the task's graph file and that its
output reaches the model turn. If the graph is not readable from inside the sandbox, B5 becomes
"precomputed brief shipped as text", which is still legal; record which variant ran.

## 4. The scorecard: candidates, not one definition

The scorer is `scripts/planB/scorecard.py`. It reads a results directory, the Plan A oracle and the
B1 profile, and writes `scorecard_calls.jsonl` (one row per graph call with every classification),
`scorecard_tasks.csv` (task x seed x component) and `scorecard_summary.json` (rates with Wilson
intervals per cell and repo). It is built against the documented trace format and a synthetic trace
fixture (`scripts/tests/`) before B0 finishes, and run on the five B0 traces the hour they exist.

### 4.1 Oracle per task (all from the paper repository)

Gold files and functions (`a1_v2/a1_gold_symbols.jsonl`); executed-by-fail-to-pass function sets
(`a2_runs/<task>/trace_pre.json`, `trace_post.json`); the v2 graph with `in_provided_graph`,
in-degrees and test flags; entry set and entry hops (`a2_v3`); the B1 profile; the B4 brief.

### 4.2 Components and their free parameters

Every component is a task-level rate. The parameters are what the candidate versions vary.

| Component | Definition | Parameters |
| --- | --- | --- |
| Resolution precision | graph calls whose argument, replayed through the resolver, returns the node the agent named (not a same-name collision) | none |
| Structural yield | graph calls whose returned nodes intersect R | R = gold only; gold + executed; gold + executed + 1 hop |
| Link-following | graph calls followed by a `read_file` on a returned node's file (B4/B5: reads on files in the brief) | window 1, 2 or 3 tool calls |
| Necessity match | agreement of "made a graph call" with the need label | label v1 = d of at least 1 or v low; v2 = B2 found gold file and B3 did not, per task |
| Waste | graph calls returning an already-seen node, or a hub whose neighbours are mostly tests | hub = in-degree above the 90th or 95th per-repo percentile; "mostly" = 50% or 75% |
| Hallucination channel | out-of-list tool names, malformed or unresolvable arguments; parser failures excluded | none |
| Async hazard | graph calls naming a gold or R function with no node, and what came back | none |
| Localization outcome | steps to first gold-file read; gold-file coverage; second-file discovery | none (these are targets, not features) |

### 4.3 Candidate versions (pre-registered here; add to this list, never silently replace)

| Version | Primary | Role |
| --- | --- | --- |
| v0 | graph-call count | the strawman every other version must beat |
| v1 | structural yield (R = gold + executed), window 2, label v1, hub 95% / 50% | the design from the Oct 11 proposal |
| v2 | v1 with label v2 | counterfactual need label |
| v3 | v1 with R = gold only | strict yield |
| v4 | v1 with yield conditioned on "needed" tasks only and hallucination veto | the selection-rule form |
| v5 | link-following alone | delivery-only |
| v6 | logistic combination of all components fitted on B2 seed 1 of the 81 non-slice tasks | the learned composite; reported but suspected to overfit |

Each version is a named function in `scorecard.py` and a line in `scorecard_versions.md` with its
date, parameters, and which test in 4.4 it was added to pass.

### 4.4 The selection criteria (what "performs best" means, fixed before any score is computed)

A version is scored on six tests in two stages. Tests 1 to 5 are the development stage: they
narrow the candidates using cached traces and the two 24-task slices. Test 6 is the decision
stage: the version named in the paper is the one that performs best **on the full 129-task SWE
benchmark in the harness, across all models run** (the 31B competition model and the 12B, 9B and
E4B variants). Nothing in tests 1 to 5 is reported as the final result; they decide only which
versions reach test 6.

1. **Within-arm prediction.** On B2 (all seeds, 129 tasks): AUROC of the primary for predicting
   gold-file coverage above the median, and for resolution, with task-cluster bootstrap intervals.
   Must beat v0.
2. **Arm ordering.** The version must order B3 < B2 and B2 <= B6 on the primary, and must not
   call B4 worse than B2 when B4's localization is at least B2's. Pass or fail.
3. **Seed stability.** Intraclass correlation of the primary across the three B2 seeds, per task.
   A version whose ICC is below the ICC of gold-file coverage itself is dropped.
4. **Anti-Goodhart.** The forced-call arm (prompt instructs a graph call every turn; dev slice,
   one seed) must score *worse* than B2 on the primary. v0 fails this by construction; any version
   that passes it has separated calls from useful calls.
5. **Selection accuracy (held-out slice).** Among the dev-slice ablation arms (section 6), each
   version picks a winner. The winners are run on the held-out slice for resolution and gold-file
   coverage. Versions whose pick is within the interval of the best pick advance to test 6; the
   rest are reported as eliminated.
6. **Full-benchmark decision, all models.** For every version that reached this stage, and for
   every model (31B w4a16 from B2/B3; 12B, 9B, E4B from their own full-129 B2 and B3 runs):
   (a) per-task AUROC of the primary for resolution and for gold-file coverage on all 129 tasks,
   with task-cluster bootstrap intervals; (b) the B3-vs-B2 ordering holds for that model;
   (c) the model ordering by scorecard primary agrees with the model ordering by resolution
   (Kendall tau over the four models). The version with the highest mean AUROC across models
   that also passes (b) and (c) for every model is the final scorecard. Ties go to the version
   with fewer components. This test is run once, after everything else, and its result is the
   paper's headline for the scorecard.

The winning version is frozen as `scorecard.py --version final` and is the one Plan C uses.
Report every version's results on all six tests in the Writeup, including the failures.

### 4.5 Scorecard ablations: which part of it is doing the work

Run these on the final version after test 6, on the same cached traces (CPU only, no new runs).
Report them as a table in the Writeup; a scorecard whose performance cannot be attributed to
named components is not a resource anyone can reuse.

1. **Leave-one-component-out.** Recompute tests 1, 2 and 6(a) with each component removed from the
   primary (for single-component primaries, this is the same as test 4.4 on the remaining
   components as a group). The drop in AUROC per component is its contribution.
2. **Single-component.** Each component alone as the primary, tests 1 and 6(a). Shows which
   components carry signal on their own and which only help in combination.
3. **Add-one-in from v0.** Starting from graph-call count, add components in the order of their
   single-component AUROC and record the curve. The knee of the curve is the minimal scorecard.
4. **Parameter sensitivity.** For each free parameter in 4.2 (R definition, link-following window,
   need label, hub percentile and test share), recompute test 6(a) over the parameter's range on
   the final version. A parameter the AUROC is flat over is reported as "does not matter" and set
   to the simplest value.
5. **Per-cell and per-model attribution.** Repeat 1 and 2 split by d x v cell and by model. The
   expected pattern, stated as a hypothesis: resolution precision and the async hazard matter most
   for the 31B model on fastapi; necessity match matters most on d of at least 2 with low v;
   waste matters most at the shortest context. Any other pattern is reported as found.
6. **Shapley attribution for the composite (v6 only, if it is the final version).** Exact Shapley
   values over the eight components on test 6(a), since eight components give 256 subsets, which
   is cheap on cached traces.
7. **Oracle swap.** Recompute yield and waste with the *provided* graph as the oracle instead of
   the Plan A oracle. If the scorecard still works, the audited map was not needed for the metric;
   if it fails, the Plan A oracle is what makes it work. Either answer is worth a paragraph.

## 5. Order of work and the validation chain

1. B0 on GCP (section 1). Write the scorer in parallel; run it on the five B0 traces.
2. B1 profile and slices.
3. B2 seeds 1 to 3, then B3 seeds 1 to 3, paired. Compute tests 1 and 3 for every version.
   Build label v2 from B2 vs B3; recompute v2 and v4.
4. Kaggle, as soon as B2 seed 1 exists: forced-call arm on the dev slice (test 4), then the
   qualified-names prompt (the Plan A fix for the 41% misresolution; this is also C2a's first
   variant).
5. B4 seeds 1 to 2, then B6 seeds 1 to 2. Test 2 for every version. B5 if B5 feasibility passed.
6. Kaggle ablations (section 6) on the dev slice. Every version picks a winner; run the winners on
   the held-out slice (test 5). Eliminate versions that fail.
6b. Full-129 B2 and B3 for 12B, 9B and E4B (Kaggle for the smaller ones, GCP when free). Test 6
   across all four models. Freeze the final version. Run the scorecard ablations (4.5).
7. Payoff surface: mixed-effects logistic regression with task random effects, condition x d x v,
   on steps-to-first-gold-read, gold-file coverage, second-file discovery, resolution; per repo and
   pooled. The final scorecard's components enter as covariates in a second model to show which
   process behaviours carry the condition effects.
8. Writeup.

## 6. Dev-slice ablations (Kaggle) and what each tells the scorecard

Each changes one thing. Each is one seed on the 24-task dev slice unless noted.

| Ablation | Change | Expected scorecard reading | If the reading is absent |
| --- | --- | --- | --- |
| Forced call | prompt demands a graph call every turn | primary falls vs B2; waste rises | the version is counting calls; drop it |
| Qualified names | prompt instructs `module.Class.method` arguments | resolution precision rises from about 59% toward 95%; yield rises | resolver replay is wrong; check `common.resolve_node_name` parity |
| Thinking 0 vs 4096 | `thinking_budget` | necessity match moves; resolution precision does not | the two components are confounded |
| Serialization flat / grouped-by-file / tree | harness patch on `get_code_neighbors` output format | link-following moves; yield constant by construction | link-following is not measuring delivery |
| Context 16k vs max | `max_model_len` | waste rises before localization falls | waste is not an early warning; redefine it |
| Size sweep 12B / 9B / E4B (BF16, B2 and B3) | model | necessity match and yield rise with size; waste falls | the scorecard does not track competence |

The size sweep is run first on the dev slice (to catch serving problems cheaply) and then on all
129 tasks for B2 and B3, because test 6 needs the full benchmark for every model.

Serialization and context need the same harness patch mechanism as B6; schedule them after B6's
patch exists. The size sweep is not optional: its full-129 runs are the input to test 6.

## 7. Reporting rules

- Every rate: numerator, denominator, Wilson interval; every paired comparison: exact McNemar;
  every pooled ratio: task-cluster bootstrap (`scripts/planA/stats.py`).
- Per repo as well as pooled; fastapi (67) and rich (48) dominate and must not be averaged away.
- Resolution is reported with intervals and is secondary; 129 tasks at 10 to 35% cannot carry a
  main claim.
- The single-L4 context limit from B0, if any, is stated in the abstract.
- Results layout mirrors `planA_results/`: raw harness outputs untouched, `scripts/` with the code
  as run, `provenance/` with hashes and launch lines, a verifier script that re-derives every
  Writeup number from the files, started on day one.

## 8. Risks

- The model does not serve at 32k on one L4. Then every arm runs at the largest context that
  works, and the Writeup says so; the paired design survives, the absolute rates do not transfer.
- Adoption in B2 is near zero. Then B3 measures the cost of the tools' presence, the forced-call
  arm becomes the only source of graph calls to score, and the scorecard's validation rests on
  B4/B6 and the ablations. Say so; it is a finding (the navigation paradox holds for Gemma).
- Kaggle sessions die mid-slice. Shard the slice into 6-task chunks and merge.
- Label v2 has few positives (tools helped and no-tools did not). Report its count; if under 10,
  v2 and v4 are reported but cannot win test 5.
- The learned composite (v6) wins test 1 and loses test 5. That is the expected outcome and the
  reason test 5 exists.

## 9. Calendar

| Date | Milestone |
| --- | --- |
| Oct 13 | B0 started on GCP; scorer written against fixture |
| Oct 15 | B0.md; B1 slices; B2 seed 1 running |
| Oct 20 | B2 and B3 complete; tests 1 and 3; label v2 |
| Oct 22 | Forced-call and qualified-names on Kaggle; test 4; tool-channel note submitted |
| Oct 27 | B4 and B6 complete; test 2 |
| Oct 31 | Dev-slice ablations done; held-out runs; test 5 |
| Nov 2 | Full-129 B2/B3 for 12B, 9B, E4B done; test 6 across all models; final version frozen; scorecard ablations (4.5) |
| Nov 4 | Plan B Writeup submitted |
| Nov 5 to 10 | Plan C with the frozen scorecard |

## 10. How the frozen scorecard feeds Plan C, with weights

Plan C needs three things from Plan B: a selection signal for prompt variants and checkpoints
(C2a, C4, C5), a per-segment reward for the adoption LoRA (C5), and a need label for the
necessity probe (C3). All three are derived from the final scorecard and its ablations, and all
three carry weights that are **fixed from Plan B's results and never retuned inside Plan C**.

### 10.1 Selection signal (C2a prompt variants, C4 probe thresholds, C5 epochs)

The signal is a weighted sum over components, computed on the 24-task dev slice:

    S = sum_i w_i * z_i   subject to the veto below

where `z_i` is component `i` standardized against the B2 distribution (mean 0, sd 1 across B2
tasks and seeds, so every component is on the same scale), and `w_i` comes from the ablations:

- `w_i` = the leave-one-out drop in test 6(a) AUROC for component `i` (section 4.5 item 1),
  normalized so the weights sum to 1. A component whose drop is within its bootstrap interval of
  zero gets `w_i = 0` and is removed from `S`.
- Sign: components where higher is better (resolution precision, yield, link-following,
  necessity match) enter positively; waste and the async hazard enter negatively.
- Veto: the hallucination channel is not weighted. A candidate whose channel rate exceeds B2's
  by more than its Wilson interval is rejected before `S` is compared.
- Localization outcomes are not in `S`. They are what `S` is validated against, and a selection
  signal that includes its own target is circular.

Pre-registered default, used only if the ablations are not finished when Plan C starts:
yield 0.35, resolution precision 0.20, necessity match 0.20, link-following 0.15, waste -0.10,
with the veto. The Writeup states which weights were actually used and where they came from.

### 10.2 Per-segment reward for the adoption LoRA (C5)

C5 computes loss only on high-value segments (the SWE-Prime recipe). A segment is one assistant
turn that issues a graph call plus the turn that consumes its result. Its value is

    r_seg = w_yield * [returned nodes intersect R]
          + w_follow * [next read_file lands on a returned node's file]
          + w_prec  * [argument resolved to the intended node]
          - w_waste * [returned node already seen or hub-of-tests]
          - w_async * [argument named a function with no node]

with the same `w_i` as 10.1, restricted to the components that have a per-call definition
(necessity match is per task and is not in `r_seg`). Segments with `r_seg` above the B2 median
are trained on; the rest contribute no loss. Tool-call validity is checked first, as the lit
review's SFT failure case requires, and the training trajectories exclude both slices.

### 10.3 Need label for the probe (C3, C4)

The probe's positive class is the counterfactual label v2 from Plan B (tools-arm found the gold
file, no-tools arm did not). If v2 has fewer than 10 positives, fall back to label v1 (d of at
least 1 or v low) and say so. The probe is not trained on scorecard values; the scorecard enters
C3 only through the label, and C4 only through the selection signal when choosing the prefill
threshold.

### 10.4 What Plan C may not do

Retune `w_i`, change a component's parameters, or swap the scorecard version. Any such change is a
new scorecard version, goes through tests 1 to 6 again, and is reported as Plan C's finding, not
as a selection choice. The held-out slice is not used in Plan C; Plan C's final check is the full
129 with the frozen scorecard and resolution side by side.

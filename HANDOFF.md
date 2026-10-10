# graphComp: handoff for the implementing model

Written Oct 10, 2026. Read this whole section first; the full research plan and literature review follow below it, unchanged from the shared doc.

## 1. The job

Write and run the experiment code for **Plan A** first (all CPU), then **Plan B** and **Plan C** (GPU). The research design, hypotheses, babystep order and metrics are fixed in the plan below. Your job is implementation and execution, not redesign. If you find the design is wrong or infeasible, say so and propose the smallest change; do not silently substitute a different experiment.

Read `compPhilosophy.md` before writing anything. Its rules apply to code too: every experiment adds exactly one component, ground truth before claims, careful language ("we hypothesize", "untested") until a result exists.

## 2. Competition facts that constrain the work

- Paper Track of the Gemma 4 Developer Agent Competition. Deadline **Nov 12, 2026, 11:59 PM UTC**. 3,000 words max per Writeup.
- **Unlimited Writeups may be submitted; the team selects 2 to be evaluated.** Ship each plan (and small standalone findings) as its own Writeup as soon as it stands alone. Ties go to the earlier entry.
- Fixed model: `gemma-4-31b-it-qat-w4a16-ct`, served by vLLM (`tool_call_parser=gemma4`, `reasoning_parser=gemma4`), `max_model_len=32768`, on 4x L4 in the official harness. Optional LoRA (max 8 adapters, rank <= 128, submission < 3 GiB).
- Full harness reference: `archive/HARNESS_README.md`. Rules summary: `rules.md` in git HEAD (`git show HEAD:rules.md`). Competitor scan: `git show HEAD:competitor_scan.md`. Prior lit review: `git show HEAD:lit_review.md`. These files were deleted from the working tree but are in commit 09dadc1.

## 3. Compute and environment (as told by the user)

- **CPU work (Plan A, B1):** Google Cloud VM(s). The user can spin up several machines, so CPU steps should support sharding across machines (`--shard i/N`).
- **GPU work (Plans B, C):** Kaggle notebooks (the competition's 4x L4) and rented GPUs. **Do not plan on a T4**: the 4-bit 31B weights are ~19 GB (Gemma 4 report, Table 3) and exceed a T4's 16 GB, and the w4a16 kernels may need a newer GPU generation (unverified).
- Data: `archive/` in this folder (~22 GB: snapshots 21 GB, graphs 403 MB, embeddings 447 MB, wheels 27 MB). Must be copied to the GCP VM (e.g. `gcloud storage cp` via a bucket, or download from Kaggle directly on the VM).
- The CPU steps are independent of the "3 experiments at a time" limit as understood so far (that limit is assumed to mean GPU runs; confirm with the user if it matters).

### Suggested CPU specs (estimates, not measured)

| Step | Machine | Disk | Notes |
| --- | --- | --- | --- |
| A1 + B1 census | 4 vCPU, 16 GB (e.g. e2-standard-4) | 60 GB | Measured: 6 tasks in 9 s on a laptop; all 129 should take minutes |
| A3 encoder ID, A4 probe | same box, CPU torch | +10 GB for models | EmbeddingGemma v1 (`google/embeddinggemma-300m`) is gated: needs a Hugging Face token with access accepted. v2 (`google/embeddinggemma-2`) is open |
| A2 dynamic tracing | 16 vCPU, 64 GB (e.g. e2-standard-16), Docker | 150 GB | Run ~7 containers in parallel at 2 vCPU / 4 GB each (the harness's limits). Embarrassingly parallel across tasks; shard across VMs |

## 4. Current status (Oct 10, updated later the same day)

- **No experiments have run on the real data yet.** The schedule in the plan is ~3 days behind.
- The earlier partial `planA/` draft, `compPhilosophy.md` and `archive/` were **not in this repository**
  (only this file was committed), and the cloud sandbox cannot reach Kaggle or huggingface.co. The
  archive lives on the user's machine at `/Users/sauravbellam/Desktop/Projects/graphComp/archive`.
- A **complete Plan A implementation now exists in `planA/`** (A1/B1, A2 tracer + sharded runner +
  analysis, A3, A4, A5, A6, and `run_all.py`, which runs every independent step in parallel). It was
  written against the harness source (swegemma 0.2.7, obtained from public entrant repos) and the
  HARNESS_README, and tested end to end on a synthetic archive (`pytest tests/`). Read
  `planA/README.md` first: it lists every `VERIFY-ON-DATA` assumption that must be checked on the
  real archive before any number is reported, and the exact commands to run.
- Facts settled from the harness source: graph JSON is networkx node-link (`edges` key; nodes
  `id`,`name`,`text`; edges `source`,`target`,`type`), embeddings `.npz` keyed by node id,
  `get_code_neighbors(edge_type=...)` compares with `==` (so `"CALLS"` matches nothing when data
  holds `calls`; unknown 8.2 is settled by code, confirm on data), `search_similar_code` resolves the
  query via the 4-tier `resolve_node_name` (exact, `.`/`;` suffix, case-insensitive, substring; ties
  to fewest dots then shortest id), Phase-2 pytest is `python3 -s -m pytest <test files> -p no:anyio
  -o timeout=0 -o python_classes="Test* *Test" -q` with a harness-written `pytest.ini`/`conftest.py`.
- Another entrant's notes (ns-0437/gemma4-developer-agent, CLAUDE.md point 10) independently claim:
  "only `calls` edges, no async functions, ~half the graph/embedding files are empty (one of each
  hard-link pair), embeddings have one dominant direction". Finding 5.1 is therefore corroborated but
  not yet ours to report until A1 runs on all 127 graphs.

## 5. Findings from the first test run (verify across all graphs before reporting)

These came from one or two fastapi graphs only. They are leads, not results.

1. **No `async def` nodes in the fastapi graph checked (fastapi_14962's base commit).** 0 nodes whose text starts with `async def`. In `fastapi/routing.py`, 8 of 46 defs are missing from the graph, and all 8 are `async def` (e.g. `fastapi.routing.serialize_response`, `run_endpoint_function`, `APIRouter._startup`). `serialize_response` is a gold-patch function for `fastapi_14962`. If this holds across graphs, the provided graph systematically omits async functions, which matters a lot for fastapi (an async framework). This is potentially the headline result of Plan A. Check all 127 graphs and all three repos.
2. **`docs_src` modules lose their prefix in node ids.** `docs_src/custom_response/tutorial010_py310.py::read_items` appears as `custom_response.tutorial010_py310.read_items`-style ids (top-level dir dropped). `common.path_to_module` does not yet handle this; gold matching must try dropping the leading `docs_src.` segment (and confirm the exact rule from the data).
3. **Nested functions (`f.<locals>.g`) are not graph nodes**, e.g. `fastapi.routing.get_request_handler.<locals>.app`. Count them separately as "nested, not representable" rather than as graph misses.
4. **Most fastapi nodes are tests and most are isolated.** In one fastapi graph: 4,106 nodes, 2,326 edges, 77.6% test nodes, 2,958 isolated nodes (2,591 of them under `tests`). Edge type is only `calls` everywhere checked.
5. **Identifier extraction finds little in many fastapi issues** (several issues are PR titles, e.g. "Add support for Server Sent Events"), so d is often undefined. Report the share of tasks with defined d; do not impute.
6. Snapshots: all 129 present (fastapi 67, rich 48, requests 13, httpx 1). requests uses a `src/` layout (`src/requests/utils.py` -> module `requests.utils`).

## 6. Design notes for A2 (dynamic call-graph oracle)

- Build a Docker image from `archive/docker/Dockerfile.sandbox` (python:3.13-slim + pytest + `imp.py`/`telnetlib.py` shims). Python 3.13 gives `code.co_qualname`, which maps frames to graph-style ids (`module.Class.method`).
- Per task: extract the snapshot, install the package, apply `test_patch`, run only the test files the `test_patch` touches. Run twice: **pre-fix** (base + test_patch; this is the oracle for the base-commit graph's precision/recall) and **post-fix** (+ gold patch; gives which tests flip fail->pass and whether they execute the gold functions).
- **Test dependencies are a known problem.** `archive/wheels/` has only ~40 packages; the official harness streams extra cached test site-packages that are not in this dataset. fastapi tests need more (and requests tests need `pytest-httpbin`). For the audit we are not bound by the offline rule: let the container use the network and install the repo's own test requirements at that commit (`requirements-tests.txt`, `requirements-dev.txt`, poetry dev groups), best effort, logging what failed. Report per-task test coverage honestly.
- Tracing: a small pytest plugin using `sys.setprofile` + `threading.setprofile` (fastapi's TestClient runs the app in another thread via anyio). Keep only frames whose file is under the workspace. Record two edge sets: direct (immediate parent frame is in the repo) and via-library (nearest repo ancestor). For coroutines and generators, record the edge only on first entry of a frame object, not on every resume. Record per-test executed-function sets and junit outcomes. Override repo pytest addopts (`-o addopts=`), disable cacheprovider, and use a subprocess timeout.
- Analysis normalization: drop static containment pseudo-edges (target starts with source + "."); collapse `Class.__init__` to `Class` on both sides when the class is a node. Report recall against dynamic edges, and "observed precision" only over static edges whose both endpoints executed (precision outside executed code is unknown; say so).
- Gold reachability: from fail-to-pass test nodes, static BFS on directed `calls` vs. dynamic execution. Test node id from a pytest nodeid: path -> module, `::` -> `.`, strip `[params]`.

## 7. Order of work

1. Fix and fully run A1/B1 on all 129 tasks; confirm finding 5.1 (async) across all graphs. This alone may be a standalone Writeup.
2. A3 (encoder ID) and A4 (embedding content probe) in parallel.
3. A2 on the GCP VM(s), sharded.
4. A5, A6, then the Plan A Writeup.
5. Plan B (B0 harness validation on a GPU capable of running the model), then Plan C.

Keep all results as CSV/JSON with the scripts that produced them, so each Writeup can cite exact numbers and the public notebook can reproduce them.

## 8. Things still unknown (check, don't assume)

- Which encoder produced the 256-d embeddings (A3).
- Whether `get_code_neighbors(edge_type=...)` is case-sensitive (README examples say `"CALLS"`, data says `calls`).
- Whether PEFT training works on the QAT compressed-tensors checkpoint (Plan C5).
- Real per-run GPU time for a 129-task harness pass (B0).

---

# graphComp Literature Review & Research Plans

Oct 7, 2026 · @Sunny

## Scope and conventions

This review builds on the Oct 6 `lit_review.md` (107 references), which survives only in the repo's git history. It does three new things: reads the newest published surveys of the field in full, re-reads the prior review's key abstract-only sources at full-text depth, and fills gaps in the four competition topics. It ends with research plans sized for three concurrent experiments.

- **Sources:** alphaXiv (primary; full text, newest work), Consensus (peer-reviewed lookup), Hugging Face (papers, datasets, models).
- **Evidence tags:** **\[full\]** = read in full text or alphaXiv's full report; **\[abs\]** = abstract only; **\[prior\]** = carried over from the Oct 6 review without re-reading.
- **"Ground truth"** means a result from a controlled comparison in a published source. A single paper's result is labeled as one paper's result, not as settled.
- **Numbering:** references restart at \[1\] here; the Oct 6 numbers are not reused.

## Part 1: What the latest reviews establish

The newest reviews agree on five things that matter for a 4-bit Gemma-4-31B agent with fixed graph tools. Interface design moves results about as much as model choice, most of all for weaker models. Graph structure offered as a separate tool tends to go unused, while the same structure placed inside what the agent already reads helps. Finding the right file is necessary but far from sufficient. Pass/fail scores hide process failures. And Gemma-4-31B has a measured, quantization-amplified tool-hallucination habit.

### 1.1 What was read

| Source | Date | What it is | Depth |
| --- | --- | --- | --- |
| Project `lit_review.md` (git HEAD 09dadc1) | Oct 6, 2026 | Our prior survey, 107 refs, no gap analysis | Read in full; used as baseline |
| Li et al., Advances and Frontiers of LLM-based Issue Resolution \[1\] | Jan 2026 | Newest dedicated survey of the field (175 papers) | \[full\] incl. appendix tables |
| Fan et al., Empirical Study of Harness Design \[2\] | Sep 2026 | Controlled ablation of planning, action space, context management; 176 settings | \[full\] |
| Hu et al., Beyond the Model: Harness Effects \[3\] | Sep 2026 | 10 open models x 2 harnesses x 3 benchmarks + 5-component ablation | \[full\] |
| Re-read at full depth from the prior review | 2025-26 | \[4\] \[5\] \[6\] \[7\] \[8\] \[9\] \[10\] \[11\] | \[full\] |

No newer dedicated survey exists on alphaXiv as of Oct 7, 2026. The two September harness studies \[2\]\[3\] serve as the most recent field-wide reviews for a fixed-harness setting like ours.

### 1.2 Ground truths (controlled results, with numbers)

**Harness and interface**

1. **Harness effects rival model effects, and shrink as the model gets stronger on issue repair.** On SWE-bench Verified the spread across harnesses at a fixed model is 19.4 points, close to the 22.8-point spread across models at a fixed harness. OpenHands' gain over mini-SWE-agent falls from +13.6 points (Claude 3.7 Sonnet) to +5.5 (Claude 4 Sonnet) \[3\]. \[full\]
2. **Structured tools help weak models and can hurt strong ones.** Structured file tools beat bash-only by +15.0 points on SWE-bench Verified for Nemotron-3 30B; the 550B model does better with bash-only (+3.6) at 53% lower cost \[2\]. In a separate 11,700-trajectory study, "atomic" tools were the only design that raised repeat-run consistency for all three actors, with the largest gain for the weakest (Qwen3-Coder-30B pass^9 0.020 to 0.094) \[4\]. The mechanism: fewer malformed commands and broken edits. \[full\]
3. **Planning is a scaffold for weak models and an efficiency aid for strong ones.** For Nemotron-3 30B, planning added +11.6 points; without it, 68.6% of runs ended without editing a file \[2\]. Lightweight scratchpad or hypothesis tools changed almost nothing: actors just restated their existing reasoning \[4\]. \[full\]
4. **Context management matters only when context binds.** The managed vs. unmanaged gap on SWE-bench Verified is 35.7 points at a 32k budget and 2.7 at 128k; unmanaged runs overflow on 78.7% of tasks at 32k \[2\]. A recall tool for elided observations gave no reliable benefit: agents rarely used it \[2\]. Context compression *hurt* repository generation by 3.9 to 4.9 points \[3\]. \[full\] Our harness runs at 32,768 tokens with compaction at 14,336, squarely in the regime where this matters.
5. **Sub-agents: task-specific ones help, generic ones hurt** (+5.9 vs. -0.7 points on ProgramBench for Qwen3.7-Max) \[3\]. \[full\]

**Code structure for agents**

6. **A graph offered as a separate tool tends to go unused.** In a pilot on 20 SWE-bench tasks, an agent given an optional call-graph tool mostly kept using grep \[5\]. Under matched conditions on SWE-bench Lite, LocAgent's graph tools trailed a grep-first Codex agent by 23.7 points Func@5 (59.5% vs. 83.2%) \[5\]. Graph-tool agents CoSIL and LocAgent also trailed the best non-graph agents on LocBench and MuLocBench \[6\]. \[full\]
7. **The same structure placed in-band helps.** LARGER injects confidence-filtered graph neighbors into the agent's own grep results: +11.8 points file Acc@5 on LocBench over the strongest baseline (GPT-5.2), with lower runtime and tokens; removing graph expansion is the largest ablation drop (-13.5%) \[6\]. CodeAnchor writes callers/callees as source comments: +2.2 points Func@5, -1.6 rounds, run-to-run variance roughly halved, about 10% more input tokens \[5\]. \[full\]
8. **More structure saturates, and direction matters with repo size.** Dense data-flow/config tags added no Func@5 over plain call/inheritance topology but cost +4.9 rounds; in large hub-heavy repos, "who calls me" links alone did best, because forward links from hub functions flood search results \[5\]. \[full\]
9. **Resolved relations plus short semantic summaries help small models specifically.** SemNav, using Gemma-4 E4B, raised File Hit@10 on SWE-bench Lite from 68.3% to 82.7%; removing language-server resolution cost 7.0 points, removing semantic cards 6.0 \[7\]. \[full\]
10. **Representation is a first-class choice.** Role-aware file summaries beat file paths by up to 40% Hit@5 at 10 to 21 times less text than raw source; 7B and 14B models could not use summaries reliably \[10\]. \[full\]
11. **Learned graph models beat LLM agents at function-level localization when trained across repos.** GNNs on GREPO (86 repos, 47,294 tasks) reached 31.5 Hit@5 on average, above LocAgent and Agentless, and improved steadily with more training repos \[9\]. \[full\]

**Localization vs. resolution**

12. **RL on bash alone matches graph-tool localizers, and the policy collapses to `rg` + `sed`** \[8\]. \[full\]
13. **Better localization buys only part of the resolution gap.** Qwen3-4B resolves 13.4% alone, 17.2% with CodeScout locations, 19.6% with oracle locations; Qwen3-Coder-30B goes 45.2%, 46.0%, 52.0% \[8\]. SemNav's oracle-file context still resolves only 39% (Gemma 4B repair) and 67% (Qwen3.5-35B repair) \[7\]. \[full\]

**Measurement and the target model**

14. **Pass/fail scores mask process damage.** On τ²-bench, 4-bit AWQ looked lossless on score, yet Gemma-4-31B's out-of-list tool calls rose from 19.5% to 38.3% of all calls (649 to 1,646 events) \[11\]. The 10-error budget absorbed it; tightening to 2 errors exposed a 16.7-point gap. Quantization amplified failures the model already had (0.97 correlation, 0.18% novel). Re-injecting the tool inventory after a failed call cut hallucinations from 1,646 to 82 and added +5.9 points \[11\]. \[full\]
15. **Public benchmarks are noisy and partly memorized.** Solution leakage and weak tests inflate SWE-bench results \[13\]; models localize Verified files from issue text alone far better than on fresher sets \[14\]. \[prior\]
16. **Training landscape.** Outcome-reward RL dominates; dense 7 to 32B models become competitive with domain rewards; SFT recipes center on data scaling, curricula and rejection sampling. For scale: SWE-Lego-Qwen3-8B reaches 44.4% on Verified with SFT alone \[1\]. \[full\]

### 1.3 Future work the authors name

| Source | Stated future work or open problem |
| --- | --- |
| Survey \[1\] | Efficiency-aware evaluation (cost alongside resolve rate); fine-grained process rewards; autonomous context management; intrinsic validation via regression tests and dependency analysis; decontamination; lighter sandboxes for RL |
| Harness design \[2\] | Planning and action-space ablations were run only at 128k; interactions with tight budgets are untested. Crossover points need more models, harnesses and languages. One run per task |
| Harness effects \[3\] | More components; more model families; benchmarks beyond ProgramBench |
| Tool architecture \[4\] | Graph and embedding capabilities were *deliberately excluded* to isolate architecture; capability and architecture remain entangled for semantic-retrieval tools |
| CodeAnchor \[5\] | End-to-end repair was not evaluated; other agents and languages; a benchmark sorted by structural depth (authors suspect deeper faults benefit more); tags as fast path with graph queries as fallback |
| LARGER \[6\] | Adaptive neighbor count and confidence threshold by repository size |
| SemNav \[7\] | Large gap to oracle context remains; repair, not localization, is the limiting stage |
| Flat Score \[11\] | Same-recipe requantization; constrained decoding of tool names; reruns at tightened budgets; other benchmarks |

## Part 2a: Tuning & Optimization

LoRA SFT on agent trajectories reliably helps, RL adds little on top for tool calling, and what you train on matters as much as how. The current limit for our setting: nobody has shown tuning that makes a small model *use structural tools it would otherwise ignore*.

**Ground truths**

- **LoRA SFT is the safe default for tool calling.** Across Qwen3 0.6B to 32B, LoRA SFT beat zero-shot in all 18 in-distribution comparisons and was best in 15; GRPO's transfer advantage was under 1 point; SFT then GRPO showed no stable benefit. At 4B, LoRA beat full fine-tuning on all three datasets (e.g. 88.0% vs. 80.4% on xLAM) \[17\]. \[full\]
- **Training corpus choice moves accuracy by up to 55 points**, comparable to the method choice \[17\]. \[full\]
- **At small data scale, quantity beats curation; curation pays off later.** For Qwen2.5-Coder-7B QLoRA on 500 to 2,000 trajectories, doubling data cut held-out loss 12.7%, about 18 times the effect of quality filtering at 500; a significant quality effect appeared only at 2,000 (3.6%, p=0.016). Error-retry rate alone captured most of the quality signal \[15\]. \[full\]
- **SFT can fail to teach the tool-call format itself.** All 16 fine-tuned models in \[15\] produced zero well-formed `<tool_call>` JSON because tool-call turns were 0 to 3% of training actions. \[full\]
- **Selective loss on good segments beats more data.** SWE-Prime trained on 10% of 32,161 successful trajectories, computing loss only on LLM-judged high-value segments, and beat SFT on all of them by up to 24.2% relative on Verified, with fewer turns and fewer repeated calls \[16\]. \[full\]
- **Small RL-trained localizers beat larger untrained models; the policy collapses to `rg` + `sed`** \[8\]. Action-level credit that rewards *discovering* gold locations lifted 4B/8B models above 8x larger baselines \[prior: A²Agent\].
- **Models often know when a tool is needed before they act on it.** Tool necessity is linearly decodable from pre-generation hidden states (AUROC 0.89 to 0.96, six models); a probe plus a one-sentence prefill cut tool calls 48% at a 1.7% accuracy cost, beating prompting and SFT \[18\]. \[full\]
- **Multi-LoRA by role is an established pattern** (reasoner/executor/summarizer LoRA groups) \[19\]. \[abs\] Our harness supports up to 8 adapters at rank 128 under 3 GiB.
- **Repository SKILL documents give gains smaller than run-to-run noise at single-repo scale** (+4.9 points GEPA, +0.1 SkillOpt on Kotlin repos) \[38\]. \[abs + conclusion\]

**Where the limit is**

- Tuning work targets resolution or localization accuracy. None targets the *adoption* of a provided structural tool by a small model, even though adoption is the bottleneck in \[20\] and \[5\].
- No PEFT study uses a QAT w4a16 base like ours; \[15\] used NF4 QLoRA on a 7B model, and \[11\] shows 4-bit damage hides in process metrics.
- With 129 training tasks, our own trajectory pool is far below the 500-trajectory floor where \[15\] saw even quantity effects, so any tuning plan needs external trajectories (Hugging Face holds several, e.g. \[45\]\[46\]) or synthetic tasks.

## Part 2b: Code Comprehension

Structure helps agents when it is delivered where they already look, and when it is semantically resolved rather than syntactic; graph *quality* is assumed, not measured. The current limit: no study audits the code graph handed to an agent against what the code actually does at runtime, and none asks what a provided embedding index can and cannot retrieve.

**Ground truths**

- **Delivery beats availability** (Part 1, items 6 to 8): in-band neighbors in grep output \[6\] and structural comments \[5\] help; separate graph tools are skipped. \[full\]
- **When the graph tool *is* used, it is near-perfect on hidden dependencies.** On 30 FastAPI tasks, a 1-hop IMPORTS/INHERITS/INSTANTIATES tool reached 99.4% architectural coverage on "hidden" tasks vs. 76.2% vanilla and 78.2% BM25 (Claude Sonnet 4.5). But 58% of trials made zero graph calls despite explicit instructions; adoption was 0% on mid-difficulty structural tasks; coverage without the tool (80.2%) equaled vanilla \[20\]. \[full\]
- **Resolved relations matter more than syntactic edges.** Removing language-server resolution from SemNav cost 7.0 points File Hit@10 with Gemma-4 E4B \[7\]. In GREPO's GNN ablation, removing temporal anchors cut Hit@5 from 31.5 to 19.1, while removing contain or inherit edges cost about 1 point and removing call edges changed Hit@5 by +0.5, within noise \[9\]. \[full\]
- **Static Python call graphs are high-precision, low-recall.** PyCG: about 99.2% precision, 69.9% recall on its benchmarks \[24\]; LLM-filtered candidate edges add 13.9% recall \[25\]; older tools contain many spurious edges and none is sound \[26\]. Agents tolerate missing edges: CodeAnchor's PyCG tags helped despite unsoundness, and under 3% of gold functions used dynamic dispatch \[5\]. \[abs / full\]
- **Dynamic traces help repair agents, but only when summarized.** DAIRA's `sys.settrace` tool raised SWE-bench Verified from 65.4% to 74.2% (DeepSeek V3.2); raw traces without semantic summaries gave 65.8%, no better than baseline \[35\]. \[full\]
- **Embeddings need structure and a good encoder, not a big one.** Frozen PLM features fed to a GNN beat both GNN-only and frozen-PLM baselines; the 0.5B encoder gave the best AUPRC, and GNN-only models always trailed \[23\]. Compact semantic search let sub-2B localizers beat grep-based ones and cut latency 44% \[21\]. \[full / abs\]
- **Representation of code for the model is a first-class choice**: role-aware summaries vs. paths vs. raw source (Part 1, item 10) \[10\]; SemNav's issue-conditioned cards cut working context 48.2% \[7\]. \[full\]
- **Agents over-trust tool returns.** Across 14 models, corrupted tool returns were adopted over a third of the time, up to 68% for web search, often overriding answers the model had right without tools \[37\]. \[abs\] A wrong graph edge may therefore cost more than a missing one.

**Where the limit is**

- Every graph-for-agents paper builds its own graph and reports downstream accuracy. None measures the precision and recall of the graph it gives the agent against dynamic execution, or links graph errors to agent errors.
- Embedding search in agent papers takes free-text queries \[21\]. Our harness's `search_similar_code` resolves the query to an existing node name first (`HARNESS_README.md`), so it is symbol-to-symbol similarity. No study measures what that design loses.
- The encoder behind the competition's 256-d vectors is unknown. EmbeddingGemma 2's model card lists native 256-d Matryoshka truncation \[33\]; v1 (embeddinggemma-300m, gated, card not readable here) is believed to as well but is unverified. Whether the vectors can be reproduced is directly testable.

## Part 2c: Tasks & Benchmarks

The field has moved from pass/fail toward process and structure-aware evaluation, and has shown that standard benchmarks barely demand multi-file reasoning. The current limit: difficulty-by-structure has been measured for *localizers* (frontier models), but nobody has measured how the *benefit of graph tools* changes with structural difficulty, or done so for a small local model.

**Ground truths**

- **Pass/fail benchmarks are noisy and partly memorized** (Part 1, item 15) \[13\]\[14\], and pass/fail masks process damage (item 14) \[11\]. \[prior / full\]
- **Most "repository" successes are local.** GPT-5 resolved 53.8% of SWE-bench Verified instances after inspecting one file and 88.0% within three. Semantics-preserving perturbations that force cross-file reasoning cut eight frontier models from 66.8% to 49.8% resolved, and to 25.3% when the structural step became the explicit task. Agents explored more (4.8 to 13.2 files) but failed to turn exploration into edits; a short structural hint lifted GPT-5 from 27.6% to 62.6% \[27\]. \[full\]
- **Localization difficulty grows with symptom-to-cause distance.** GraphLocator defines it as the max over gold functions of the min hop distance from issue-mentioned nodes on a typed repository graph, and shows recall falls with distance for every localizer tested (Claude-3.5) \[34\]. \[full\]
- **Category labels hide task demands.** A Spread-Novelty-Centrality profile computed from gold patches separates every pair of five benchmarks; resolved runs concentrate in low-SNC tasks across 14,922 trajectories \[28\]. \[partial\]
- **Hidden-dependency tasks need a structural tool; semantic ones do not.** BM25 reached 100% coverage on keyword-discoverable tasks; a graph tool reached 99.4% on hidden ones \[20\]. \[full\]
- **Process metrics now exist**: exploration quality (SWE-Explore, used in \[7\]), repeat-run consistency pass^k \[4\], link-following rate and run-to-run variance \[5\], echo-walk rate for graph agents \[22\], per-channel tool-error rates and error-budget curves \[11\]. \[full / prior\]
- **Local tool-use measurement is fragile.** Serving stacks (vLLM, Ollama, SGLang, llama.cpp) handle the same tool request differently; refused requests get logged as model non-calls; pooled vs. per-episode rates differed by up to 55 points \[36\]. \[full\] Our harness runs vLLM with the `gemma4` tool and reasoning parsers, so tool-call validity must be logged separately from model choice.

**Where the limit is**

- GraphLocator's distance analysis covers localizers with a frontier model and its own graph \[34\]. CodeAnchor explicitly calls for a benchmark sorted by structural depth to test whether structure helps more on deeper faults \[5\]. No one has crossed *structural distance* with *lexical visibility* and *tool availability* to find where graph tools start paying off.
- The competition's own data has no difficulty annotation. The hidden test set is built "under a similar pipeline", with graphs from the same generator, so a per-task structural profile defined on the provided graph could be computed for those tasks too (whether its effect transfers is untested).

## Part 2d: Graph Reasoning

LLMs can traverse small graphs placed in context, but they are sensitive to how the graph is written down, and as agents they explore graphs wastefully. The current limit: almost all of this work uses knowledge, citation or synthetic graphs; budgeted exploration of *code* graphs, with their hub functions and many redundant paths, is unstudied.

**Ground truths**

- **Serialization changes answers.** The right encoder adds 4.8 to 61.8% on graph tasks \[42\]. LLM graph reasoners are not invariant to node relabeling, edge order or syntax; fine-tuning reduces relabeling sensitivity but *increases* format sensitivity, and larger models are more robust \[29\]. For code, tree-formatted subgraphs beat edge lists, DOT and JSON \[12\]. \[prior / abs + conclusion\]
- **Gemma-4-31B is a capable in-context graph walker**: 82.3 F1 on GraphWalks under 128k tokens without thinking, vs. 32.8 for Gemma 3 27B \[31\]. \[full\] Whether this transfers to call-graph JSON returned by a tool is untested.
- **Graph agents mistake repeated paths for new evidence.** Redundant paths to one source raised the echo-walk rate in every frozen agent tested (Qwen3 1.7B to 8B, GPT-5-mini) by 5.8 to 38.8 points. Collapsing equivalent paths before exploration gave 0% echo and 83% accuracy; RL training cut echo but also cut distinct-source recall and transfer accuracy \[22\]. \[full\]
- **Explicit search state beats implicit state.** Replacing an agent's history with a persistent list of confirmed facts and open questions, plus a programmatic "exhaustion" stop, improved multi-hop QA by up to 11.4% and saved up to 39% of tokens (GPT-4o-mini) \[30\]. \[abs + conclusion\]
- **Learned graph models beat LLM traversal for localization when trained** \[9\]; graph structure *inside* the LLM works only with a large backbone (CGM: 7B variant 4.0% on Lite) \[41\], and graph tokens can become ignored attention sinks \[43\]. \[full / prior\]
- **Graph agents over-trust and under-adopt.** Separate graph tools go unused (Part 2b); tool returns are adopted even when wrong \[37\]. \[full / abs\]

**Where the limit is**

- Code call graphs are hub-heavy: in the competition's requests graph, `Session` has in-degree 77 and a test class 74 (checked locally). CodeAnchor found hub forward-links flood search \[5\]; GraphEcho shows redundant paths cause echo walks \[22\]. Nobody has measured echo walks or hub capture for an agent calling `get_code_neighbors` on a real repository.
- How tool output is serialized for a small model (flat neighbor list, as the harness returns, vs. tree vs. grouped by file) is unmeasured in any SWE harness.
- Surveys point to graph-native capability substrates and self-evolving graphs as next steps \[32\]; these are far from a five-week, babystep scope.

## What the released dataset actually contains

The provided graph is a single-edge-type, test-heavy call graph, and the semantic-search tool cannot take free text; both shape what any graph experiment can show. Checked on the local files on Oct 7, except where marked.

| Item | What we found | Why it matters |
| --- | --- | --- |
| Tasks | 129: fastapi 67, rich 48, requests 13, httpx 1; `hints_text` empty for all | Per-repo effects will dominate small samples |
| Problem statements | Median 418 characters; 38 under 200; 24 contain PR-template text (16 rich, 8 fastapi), e.g. `fastapi_15661` is a near-empty PR template | Weak issue text leaves lexical search little to anchor on |
| Gold patches | 91 of 129 touch one source file; median 12 lines changed (from `compPhilosophy.md`) | Most tasks are local, as in \[27\] |
| Graph files | 127 distinct graphs (one per base commit); nodes carry only `id`, `name`, full source `text` | No file, class or module node types |
| Edge types | Only `calls` | Imports, inheritance and containment, which \[5\]\[9\]\[20\] found useful, are absent as typed edges |
| Containment inside `calls` | 8 to 21% of `calls` edges go from a symbol to its own member (e.g. class to `__init__`) | "Calls" partly encodes containment |
| Test code | 26 to 46% of edges start in test modules; in one requests graph, 454 of 754 nodes are tests | Neighbor lists of library functions fill with test callers |
| Density | fastapi: \~3,300 to 4,300 nodes but only \~2,000 to 2,700 edges; rich: \~1,970 nodes, \~5,600 edges; requests: \~750 nodes, \~1,800 edges; 59 of 754 requests nodes isolated | Graph usefulness will differ sharply by repo |
| Hubs | requests: `Session` in-degree 77, `TestRequests` 74 | Hub capture and echo walks are likely \[5\]\[22\] |
| Embeddings | One float32 vector of 256 dims per node, keyed by node id | Encoder unknown; 256 matches EmbeddingGemma's truncation size \[33\] |
| `search_similar_code` | Resolves the query to an existing node name, then ranks by cosine (`HARNESS_README.md`) | Symbol-to-symbol only; issue text cannot be searched |
| `get_code_neighbors` filter | README examples use `"CALLS"`, `"DEFINED_IN"`, `"IMPORTS"`; data holds lowercase `calls` only | Unknown: a typed filter may silently return nothing |
| Harness limits | 32,768-token context, compaction at 14,336; `read_file` 150 lines; outputs cut at 5,000 chars; vLLM `gemma4` tool parser | The tight-context regime where \[2\] saw the largest management effects |
| Skills | Submissions may ship `skills/` with Python scripts run in the sandbox (`rules.md`) | A legal way to add a custom structural tool inside the official harness |

Open unknowns to settle first: which encoder made the vectors; whether `edge_type` filtering is case-sensitive; what fraction of gold-patch functions exist as graph nodes; how often issue-mentioned names resolve to nodes.

## Part 3: Gap map and chosen direction

**Direction: "the map we were handed."** We study how a 4-bit local agent gets, uses and misses the repository graph and embeddings the competition provides. Nine open gaps converge on this one object, every one is testable on files already on disk, and together they touch all four topics. It is measurement-first, so it scores on Novelty's "highlight important properties of existing methods" without inventing yet another navigator.

| # | Open gap (as of Oct 7, 2026) | Closest prior work and what it left open | Topics | Plan |
| --- | --- | --- | --- | --- |
| 1 | No audit of a graph *as given to agents* against runtime behavior | CodeAnchor bounds the unsoundness of its own tags \[5\]; DAIRA uses traces as a tool \[35\]; static-CG studies audit tools, not agent inputs \[24\]\[26\] | Code Comprehension, Tasks | A |
| 2 | What symbol-only "semantic" search loses vs. free-text search | Agent embedding-search papers all take free text \[10\]\[21\] | Code Comprehension | A |
| 3 | Unknown encoder behind the 256-d vectors | None | Code Comprehension (resource) | A |
| 4 | Graph-tool adoption by a small, 4-bit open model | Measured only for Claude Sonnet 4.5 \[20\] and Codex \[5\] | Graph Reasoning, Tuning | B |
| 5 | Tool *benefit* as a function of structural distance x lexical visibility | Distance vs. localizer accuracy, Claude-3.5, own graph \[34\]; hidden vs. semantic tasks, 30 hand-made tasks \[20\] | Tasks, Graph Reasoning | B |
| 6 | Push vs. pull of the *same* structure for a small model in one harness | Push works for frontier models \[5\]\[6\]; a grep-able test map works for 4-bit Qwen \[TDAD, 39\] but is not neighbor structure | Graph Reasoning, Code Comprehension | B |
| 7 | Gemma-4-31B's tool-hallucination channel inside an SWE harness, QAT w4a16 recipe | Measured on τ²-bench with AWQ INT4; recipe confound left open \[11\] | Tuning, Tasks | B (instrument), C |
| 8 | Interventions that raise adoption only where structure is needed | Hidden-state necessity probes for calculator and search tools \[18\]; none for code-graph tools | Tuning | C |
| 9 | Echo walks and hub capture when an agent explores a real call graph | Synthetic and scientific graphs \[22\]; hub effects on grep tags \[5\] | Graph Reasoning | B (metric) |

**Why not the alternatives.** A new GNN localizer needs far more than 129 tasks and GREPO already exists \[9\]. Graph-in-model (CGM) needs a large backbone and weeks \[41\]. The "AST graph, localize, compress" pitch is the most crowded one in the competitor scan, with unvalidated mock-harness claims. Leaderboard tuning without a question is ruled out by `compPhilosophy.md`.

**The main risk, stated up front.** The strongest competitor paper found its localizer put the gold file in the top 5 for 14 of 18 tasks that still failed (`competitor_scan.md`), and oracle locations close only part of the resolution gap \[8\]\[7\]. So Plan B measures localization, multi-file completeness and resolution separately. If the graph pays off only far from the issue text, or only for co-change sites, that is the finding.

**How the three plans relate.** Plan A is CPU-only and ships first; its audited, typed graph and per-task structural profile feed Plan B. Plan B's agent logs feed Plan C. Each is written to stand alone as a paper.

**Submission strategy.** Teams may submit any number of Writeups and select 2 for evaluation, and more findings are encouraged. So each plan ships as its own Writeup as soon as it stands alone, and smaller self-contained findings ship too (for example the encoder identification from A3, or the tool-channel census from C1). The two we select at the end will most likely be Plan B (Overall Best Paper) and Plan A (Best New Resource), but the choice waits for results. Early entry also helps the tie-break, which favors the paper entered first.

## Plan A: Is the map right? An execution-grounded audit of the graph and embeddings handed to SWE agents

**Pitch.** Before asking whether an agent uses the competition's graph, measure whether the graph matches what the code does at runtime, and what the embedding index can actually retrieve. CPU-only, all inputs on disk, first paper out. Topics: Code Comprehension (primary), Tasks & Benchmarks. Award fit: Best New Resource (audited, typed graph plus audit toolkit) or Overall.

### A.1 What has already been proven

**Static Python call graphs are precise but miss many real calls.** PyCG computes assignment relations between identifiers of functions, variables, classes and modules across files, then resolves each call to the functions it may invoke. On a micro-benchmark of small programs and a macro-benchmark of real packages it reached about 99.2% precision and 69.9% recall, at 0.38 s per 1,000 lines \[24\]. The idea: Python names bind at runtime, so a static tool either stays conservative (precise, low recall) or guesses (more recall, spurious edges). Later tools trade along that line: JARVIS adds flow-sensitive type inference (+84% precision and at least +20% recall over PyCG on its benchmarks) \[47\]; InferCG generates permissive name-and-arity candidates and lets an LLM filter them (+13.9% average recall over PyCG) \[25\]. An evaluation of Code2flow, Pyan and WALA found none sound and many spurious edges \[26\]. \[abs\]

**Execution is the standard oracle for call-graph truth.** For 31 Java programs, Sui et al. recorded calls while running built-in and synthesized tests and measured static recall against that oracle: median 0.884, with built-in tests exposing more behavior than generated ones \[49\]. Helm et al. extend this with fuzzing-built dynamic baselines \[48\]. Test suites are therefore the accepted instrument, and every competition task ships one. \[abs\]

**Agents tolerate missing edges.** CodeAnchor injected PyCG-plus-AST call/inheritance tags into source as comments. The authors measured unsoundness directly: under 3% of gold functions used dynamic dispatch, and their extractors captured 94.2% of statically visible imports and class hierarchies. Tags still raised Func@5 by 2.2 points and halved run-to-run variance; agents treated tags as probabilistic cues \[5\]. \[full\]

**Runtime structure helps repair agents, but only when compressed.** DAIRA gives the agent a tracer (`sys.settrace`, restricted to repository modules, recording only calls, returns and exceptions) that runs the agent's reproduction script, then has an LLM turn the trace into an execution tree and role summary. On SWE-bench Verified with DeepSeek V3.2 it lifted resolution from 65.4% to 74.2%; feeding raw traces instead gave 65.8%, no gain \[35\]. \[full\]

**Wrong structure is dangerous because agents trust tools.** Across 14 models with corrupted tool returns, agents adopted the corrupted content more than a third of the time for every tool and 68% for web search, often overriding answers they had right without the tool \[37\]. \[abs\]

**How code is represented to a retriever changes what it finds.** Same retrievers, different text: role-aware file summaries beat file paths by up to 40% Hit@5 at 10 to 21 times less text than raw source \[10\]. Compact free-text semantic search let sub-2B localizers beat grep agents while cutting latency 44% \[21\]. Small frozen encoders plus a graph model beat either alone \[23\]. At 256 dimensions, EmbeddingGemma 2 keeps 76.2 on MTEB-code vs. 78.7 at full 768 \[33\], so 256-d vectors are near full quality *when queried with text*. \[full / abs / model card\]

**Typed, resolved edges matter to agents.** IMPORTS/INHERITS/INSTANTIATES edges drove CodeCompass's hidden-dependency gains \[20\]; language-server-resolved relations were worth 7.0 points File Hit@10 for Gemma-4 E4B \[7\]. \[full\]

### A.2 The gap

Every graph-for-agents paper builds its own graph and reports only downstream accuracy. Nobody has audited a graph *as distributed to agents* (here, to every competitor) against the runtime behavior of the code it describes, or linked its omissions to the functions that need fixing. Nobody has measured what a symbol-to-symbol embedding tool can retrieve compared with free-text search over the same vectors. The encoder behind the vectors is undocumented.

### A.3 Our addition

The first execution-grounded audit of a benchmark-supplied code graph and embedding index, using each task's own tests as the oracle, released as a corrected, typed graph with per-task structural annotations. Hypotheses, each testable and each publishable whichever way it lands:

- **H-A1** The provided `calls` edges are high-precision and low-recall, and recall is lowest in fastapi, whose decorators and dependency injection bind calls at runtime.
- **H-A2** A meaningful share of gold-patch functions are unreachable from the failing tests in the static graph but reachable in the dynamic one.
- **H-A3** The vectors are reproducible with a public EmbeddingGemma checkpoint at 256-d.
- **H-A4** Symbol-only search (as the harness implements it) retrieves gold functions far less often than free-text search over the same vectors.

### A.4 Babystep chain (each step adds one component)

| Step | Adds | Measures | Compute | Est. time |
| --- | --- | --- | --- | --- |
| A1 | Static census | Share of gold-patch functions present as nodes; containment inside `calls`; test-node share; degree and hubs per repo; how often issue identifiers resolve to nodes under the harness's 4-tier name matching | CPU | 1 day |
| A2 | Dynamic oracle | Run each task's target tests (with `test_patch`, before and after the gold patch) under `sys.setprofile` filtered to the repo package; map frames to node ids; precision and recall of `calls` edges, overall and incident to gold functions; static vs. dynamic reachability from tests to gold functions; report executed-code coverage, since precision is only a lower bound outside it | CPU + Docker (4 GB, 2 vCPU per container, like the harness) | 3 to 4 days |
| A3 | Encoder provenance | Embed node `text` with EmbeddingGemma v1 and v2, 256-d truncation, with and without the documented prefixes; cosine to provided vectors (identified if mean at or above 0.99) | CPU or small GPU | 0.5 day |
| A4 | Content probe | Does vector similarity track call adjacency, same file, name overlap, or co-change among a task's gold functions? | CPU | 0.5 day |
| A5 | Retrieval-mode audit | Gold-function Hit@10 and file Hit@5 for: (a) harness mode, issue identifiers resolved to nodes then cosine; (b) BM25 over node text; (c) free-text issue embedding, if A3 succeeds; (d) mode (a) plus 1-hop neighbors | CPU | 1 day |
| A6 | Release | Typed graph v2 in the harness's JSON format: contains, imports and inherits from AST; dynamic call edges with test provenance; test nodes marked; test-to-code links. Per-task profile (coverage, distance, visibility) for Plan B | CPU | 2 days |

A1, A3 and A4 can start tomorrow in parallel. A2 is the long pole and the core result.

### A.5 Paper shape (under 3,000 words)

Title candidate: *Is the Map Right? Auditing the Code Graph and Embeddings Given to Local SWE Agents.* Sections: the provided artifacts; execution-grounded audit method; graph precision and recall by repo; gold-function reachability; what the vectors are and what they retrieve; the released typed graph; implications for agents (one paragraph linking to Plan B).

### A.6 Risks and what we still learn

- Some snapshots may not run their tests cleanly offline. The harness reports verified fail-to-pass behavior for all tasks, so failures here are our setup; report coverage per repo.
- Tests exercise narrow paths, so recall is "recall within executed code". That is still the slice the fix must touch.
- If A3 fails, report the nearest candidates and continue A4 and A5 with BM25 standing in for free text.

## Plan B: Pull, push, or ignore. When does the provided graph pay off for a 4-bit Gemma agent?

**Pitch.** Map the "payoff surface" of repository structure for one small local model in the official harness: how often it calls the graph tools, and how much they help as a function of how far the fix sits from the issue text (structural distance) and how visible it is to lexical search. Then test whether pushing the same structure into context beats leaving it as a tool. Topics: Graph Reasoning (primary), Tasks & Benchmarks, Code Comprehension. Award fit: Overall Best Paper.

### B.1 What has already been proven

**The navigation paradox: a graph tool is nearly perfect when used and usually not used.** Paipuru parsed the FastAPI RealWorld app (about 3,500 lines, 40 files) with Python's `ast` into 71 file nodes and 255 IMPORTS, INHERITS and INSTANTIATES edges, served through an MCP tool returning a file's 1-hop structural neighborhood. Thirty edit tasks were split into G1 (required files findable by keywords), G2 (reachable by 2 to 4 import hops from a named file) and G3 (no lexical overlap; reachable only through structure). Claude Code with Sonnet 4.5 ran three arms over 258 trials: vanilla, BM25 rankings prepended, and the graph tool with an instruction to call it first. On G3 the graph arm reached 99.4% architectural coverage vs. 76.2% vanilla and 78.2% BM25. Yet 58% of graph-arm trials made zero graph calls; adoption was 22% on G1, 0% on G2 and 100% on G3 only after moving a checklist to the end of the prompt. With the tool used, coverage was 99.5%; without, 80.2%, the same as vanilla \[20\]. The idea: agents skip a tool whenever cheap search looks sufficient, a rational but often wrong judgment. \[full\]

**Grep-first agents beat graph-tool agents unless structure arrives in-band.** Under matched conditions on 274 SWE-bench Lite tasks, LocAgent's graph tools reached 59.5% Func@5 vs. 83.2% for grep-first Codex; a pilot with an optional call-graph tool saw it rarely used \[5\]. Writing the same call/inheritance facts as source comments raised Func@5 by 2.2 points, raised the share of navigation steps that followed a structural link from 0.15 to 0.18 up to 0.21 to 0.24, and halved run-to-run variance. About 27% of tag-following hops landed on a gold entity. Denser data-flow tags saturated, and large hub-heavy repos did best with "who calls me" links only \[5\]. LARGER appends confidence-filtered graph neighbors to each grep result: LocBench file Acc@5 87.0% vs. 75.2% for the best baseline (GPT-5.2), with lower runtime and tokens than Codex; removing graph expansion was the largest ablation drop, -13.5% \[6\]. \[full\]

**Difficulty grows with symptom-to-cause distance.** GraphLocator extracts keywords from the issue, maps them to graph nodes by exact name, and defines distance as the maximum over gold functions of the shortest hop distance to any mentioned node; recall falls with distance for every localizer tested, all with Claude-3.5 on GraphLocator's own typed graph \[34\]. RepoMirage shows why this matters: 53.8% of GPT-5's Verified wins needed one file, and forcing cross-file reasoning cut eight frontier models from 66.8% to 49.8% \[27\]. \[full\]

**Structure pushed as a grep-able file works at our scale.** TDAD built an AST code-test graph and exported a plain-text test map, delivered through a 20-line SKILL file. With Qwen3-Coder-30B at 4-bit and a 32K context on 100 Verified tasks, test-level regressions fell from 6.08% to 1.82% with resolution 31% to 29%; verbose TDD instructions alone *raised* regressions 42%. As a skill in OpenCode with Qwen3.5-35B-A3B on 25 tasks, resolution rose from 24% to 32% \[39\]. \[full\]

**Weak models gain most from structure, and tight context matters.** Structured file tools added +15.0 points for Nemotron-3 30B but hurt the 550B model; at 32k, unmanaged runs overflowed on 78.7% of tasks \[2\]. Atomic tools gave the largest consistency gain to the weakest actor \[4\]. Harness gains shrink as models strengthen on issue repair \[3\]. \[full\]

**Measure process, not only outcome.** Pass/fail can hide doubled tool hallucination for Gemma-4-31B \[11\]; serving stacks can turn refused requests into fake non-calls \[36\]; redundant graph paths cause echo walks \[22\]. \[full\]

### B.2 The gap

Every result above uses a frontier model, a hand-built graph, or a localization-only metric. None measures, for one small 4-bit model in one harness: how often *provided* graph tools are called; whether benefit depends jointly on structural distance and lexical visibility; whether pushing the same neighbors into context beats pulling them; and whether any of it moves resolution rather than file coverage.

### B.3 Our addition: the payoff surface

For each task, Plan A gives distance *d* (GraphLocator's definition, on the provided graph, with deterministic identifier extraction instead of an LLM) and visibility *v* (best BM25 rank of a gold file using the issue text). We estimate adoption(d, v), the pull benefit (tools vs. no graph tools), and the push benefit (same structure in context), on localization, multi-file completeness and resolution.

- **H-B1** Adoption is low and does not rise where structure is needed (the paradox \[20\] holds for Gemma).
- **H-B2** Pull benefit is near zero when d = 0 or v is high, and positive only for d of at least 2 with low v.
- **H-B3** Push beats pull in the low-v cell; push costs a little in high-v cells at 32k context.
- **H-B4** The graph helps multi-file tasks find their *second* file (co-change sites) more than it helps find the first.

### B.4 Babystep chain

| Step | Adds | Measures | Compute |
| --- | --- | --- | --- |
| B0 | Local harness and logging | `swegemma eval` on the 129 tasks; gold patches must resolve; per-turn log of tool name, arguments, validity, parser errors kept separate from model non-calls \[36\] | 1 GPU, setup |
| B1 | Task profile | d, v, multi-file flag per task (from A1/A6) | CPU |
| B2 | Baseline | Single agent, all 9 tools, sample prompt minus sub-agent, temperature 0.2, fixed budgets; 3 seeds | GPU |
| B3 | Remove graph tools | Same as B2 without the 3 graph tools; paired by task and seed | GPU |
| B4 | Push | B2 plus a short structural brief (1-hop callers and callees of resolved issue symbols, test callers filtered, grouped by file) prepended to the problem statement | GPU |
| B5 | Push, harness-legal | The same brief produced by a Skill script inside the sandbox, so it is deployable in a real submission | GPU |
| B6 | Better map | B2 with tools backed by Plan A's audited, typed graph | GPU |

Primary metrics: steps to first gold-file read, gold-file read coverage, second-file discovery on multi-file tasks. Secondary: resolution (McNemar, paired), pass^3 consistency \[4\]. Process: graph-call rate, out-of-list tool names, unresolvable node arguments, echo-walk rate on `get_code_neighbors` \[22\]. Analysis: mixed-effects logistic regression with task random effects, condition x d x v.

### B.5 Paper shape and risks

Title candidate: *Pull, Push, or Ignore: When Repository Graphs Pay Off for a 4-bit Local Coding Agent.* Risks: with 129 tasks and resolution likely 10 to 35%, resolution differences need large effects, so the continuous localization metrics are primary and resolution is reported with intervals. Per-repo imbalance (67 fastapi, 1 httpx) means results are reported per repo as well as pooled. If adoption is near zero, B3 becomes a test of the tools' mere presence (prompt and context cost), which is itself a finding.

## Plan C: Teaching a 4-bit agent when to read the map

**Pitch.** If Plan B shows that the graph helps only where the agent rarely reaches for it, the next question is how to make adoption track need. Test a ladder of interventions, cheapest first: prompt repair, a hidden-state probe that prefills a nudge, and an adoption LoRA. Along the way, measure Gemma-4-31B's tool-hallucination channel inside an SWE harness for the first time. Topics: Tuning & Optimization (primary), Graph Reasoning. Award fit: Overall, or folded into Plan B's paper.

### C.1 What has already been proven

**Quantization amplifies a failure Gemma-4-31B already has, and pass/fail hides it.** Jang et al. ran τ²-bench (114 tasks x 4 trials = 456 episodes per arm) at BF16, FP8 and INT4 across Gemma-4 and Qwen-3.6 checkpoints, greedy decoding, with a 10-error budget per episode. No final-score change survived correction. Underneath, Gemma-4-31B's out-of-list tool calls in the telecom domain rose from 19.5% to 38.3% of its calls (649 to 1,646 events), budget-exhausting episodes from 5 to 46, and all 22 hallucinated names at INT4 were real tools on the *user's* side of the interface. The set of hallucinated names barely changed (0.97 correlation; 0.18% novel); quantization multiplied existing near-miss decisions, which the authors explain with a logit-margin account. Re-injecting the tool inventory after each failed call cut the channel from 1,646 events to 82 and added +5.9 points, and only in the damaged cell. Their INT4 for Gemma was an AWQ-style asymmetric group-32 recipe; a same-recipe or QAT comparison is listed as missing \[11\]. \[full\]

**Models often know when a tool is needed but do not act on it.** When2Tool built 18 environments with easy, boundary and hard tiers of tool necessity. A logistic probe on the last input token's hidden states (all layers concatenated) predicted necessity with AUROC 0.89 to 0.96 across Qwen3 1.7B to 32B and Llama 8B and 70B, even where verbal reasoning failed. Prefilling one steering sentence ("I need to use a tool for this question" or the opposite) when the probe fires cut tool calls 48% at a 1.7% accuracy cost, with under 1 ms overhead; prompting cut calls indiscriminately and Reason-then-Act collapsed Llama's tool calling \[18\]. \[full\]

**Prompt placement and inventories change adoption.** Moving a tool-use checklist to the end of the prompt raised hidden-dependency adoption from 85.7% to 100% for Sonnet 4.5 \[20\]. An explicit allowed-name list recovers most tool-call fidelity for small local models; constrained decoding removes parse failures but can cause endless loops \[36\]. \[full\]

**LoRA SFT is the dependable tuning tool, with known traps.** LoRA SFT beat RL in-distribution across 0.6B to 32B and beat full fine-tuning at 4B \[17\]. When tool calls were 0 to 3% of training actions, 16 of 16 LoRA models failed to emit valid tool-call JSON \[15\]. Computing loss only on high-value segments of 10% of trajectories beat training on all of them \[16\]. Rewarding *discovery* of gold locations, not just final answers, lifted small localizers \[40\]. RL against redundant graph walks reduced repetition but also reduced distinct-evidence recall and transfer accuracy \[22\]. \[full / prior\]

### C.2 The gap

No study tries to make a small agent call a *code-structure* tool where structure is needed and skip it where it is not. Tool-necessity probes exist for calculators and search \[18\]; tool-hallucination measurements for Gemma-4 exist for a customer-service benchmark under AWQ \[11\]. Neither has been done for a QAT w4a16 Gemma in an SWE harness with graph tools.

### C.3 Our addition

- **H-C1** Gemma-4-31B w4a16 shows a measurable out-of-list tool channel in the swegemma harness (e.g. calling shell-like names), and the harness's nudges and budgets absorb it the way τ²'s error budget did.
- **H-C2** A static allowed-name list plus an end-of-prompt structure checklist (both legal in a real submission) cuts that channel and raises graph adoption, but not selectively.
- **H-C3** Whether a task needs structure (Plan B's low-v, d of at least 2 cell, or a counterfactual label: tools found the gold file and no-tools did not) is linearly decodable from Gemma's first-turn hidden states.
- **H-C4** Probe-and-prefill raises adoption selectively, improving the payoff where structure is needed without the context cost elsewhere.

### C.4 Babystep chain

| Step | Adds | Measures | Compute |
| --- | --- | --- | --- |
| C1 | Channel census | From B2 and B3 logs: valid-call rate, out-of-list names, graph-tool arguments that fail to resolve, serving-level failures kept separate; error-budget style curve over the tool-call budget \[11\] | CPU, no new runs |
| C2a | Static repair | Allowed-name list and end-of-prompt checklist in the system prompt (harness-legal) | GPU |
| C2b | Dynamic repair | Re-inject the tool inventory after each failed call (research harness) \[11\] | GPU |
| C3 | Necessity probe | First-turn hidden states of the w4a16 model; L2 logistic probe; repo-held-out cross-validation; AUROC with interval | 1 GPU, hours |
| C4 | Probe-and-prefill | If AUROC is at least 0.75, prefill one sentence naming the graph tool when the probe fires; adoption and payoff by cell | GPU |
| C5 (stretch) | Adoption LoRA | Rank-16 LoRA, loss only on segments where a graph call preceded gold-file discovery \[16\]; data from Plan B runs plus converted public trajectories \[45\]\[46\]; check tool-call validity first \[15\] | GPU, multi-day |
| C1b (optional) | Precision control | Repeat C1 on BF16 Gemma-4-31B if 64 GB of VRAM is available, answering the open recipe question in \[11\] for QAT | Large GPU |

### C.5 Risks

- With 129 tasks, the probe has few positives; use repo-held-out folds and report the interval, and treat a null as informative.
- Probe-and-prefill and dynamic repair are research-harness interventions, not deployable in the declarative competition format; C2a and Plan B's skill route are the deployable ones. Say so in the paper.
- Unknown: whether PEFT training runs directly on the QAT compressed-tensors checkpoint. Check before committing C5; the fallback is training on BF16 weights and serving on w4a16, which itself needs a check.

## Running three experiments at a time

The three plans fit the five weeks left with never more than three runs in flight: Plan A's write-up ships Oct 20, a short tool-channel note Oct 22, Plan B Nov 4 and Plan C Nov 10, leaving two days to choose the two to be judged.

| Dates (2026) | Slot 1 | Slot 2 | Slot 3 |
| --- | --- | --- | --- |
| Oct 8-9 | A1 | A3 | B1 |
| Oct 10-12 | A2 (to Oct 14) | A4 | B0 |
| Oct 13-17 | A2, A5 | B2 | B3 |
| Oct 18-20 | A6 | C1 | (writing A) |
| Oct 21-25 | B4 | B6 | C2 |
| Oct 27-Nov 3 | B5 | C3, C4 | C5 (stretch) |

Write-ups: Plan A Oct 20; tool-channel note Oct 22; Plan B Nov 4; Plan C Nov 10; deadline Nov 12 (select 2 to judge). The schedule is now ~3 days behind (no runs had started by Oct 10).

Week 1 is all CPU (A1 to A4, B1) while the GPU harness is validated (B0), so results start arriving before any GPU run. The paired GPU runs B2 and B3 share a week so their seeds match.

- **Assumption to check in B0:** one 129-task run of one configuration and seed takes about one GPU-day. If slower, drop from 3 seeds to 2 before dropping any step.
- **First cut if time runs short:** C5 (adoption LoRA), then B5. Every completed step is already a reportable result.
- **Each write-up stands alone:** Plan A needs nothing from B or C; Plan B uses A's task profile; Plan C uses B's logs.

## References

Tags: \[full\] read in full text or alphaXiv full report; \[abs\] abstract only; \[prior\] carried from the Oct 6 review; \[card\] model or dataset card. Links are alphaXiv, Consensus or Hugging Face as returned by the tools.

1. Li et al. 2026. [Advances and Frontiers of LLM-based Issue Resolution in Software Engineering: A Comprehensive Survey](https://www.alphaxiv.org/abs/2601.11655) \[full\]
2. Fan et al. 2026. [An Empirical Study of Harness Design for Coding Agents](https://www.alphaxiv.org/abs/2609.20804) \[full\]
3. Hu et al. 2026. [Beyond the Model: Demystifying Harness Effects in Software Engineering Agents](https://www.alphaxiv.org/abs/2609.32459) \[full\]
4. Xu et al. 2026. [The Devil Is in the Interface: How Tool Architecture Shapes Coding Agent Behavior (COLM 2026)](https://www.alphaxiv.org/abs/2608.11386) \[full\]
5. Lin et al. 2026. [How Much Static Structure Do Code Agents Need? (CodeAnchor, ISSTA 2026)](https://www.alphaxiv.org/abs/2606.26979) \[full\]
6. Hu et al. 2026. [LARGER: Lexically Anchored Repository Graph Exploration and Retrieval](https://www.alphaxiv.org/abs/2605.16352) \[full\]
7. Wei et al. 2026. [SemNav: Semantic Navigation for Repository-Level Issue Localization](https://www.alphaxiv.org/abs/2609.31176) \[full\]
8. Sutawika et al. 2026. [CodeScout: An Effective Recipe for RL of Code Search Agents](https://www.alphaxiv.org/abs/2603.17829) \[full\]
9. Wang et al. 2026. [GREPO: A Benchmark for GNNs on Repository-Level Bug Localization](https://www.alphaxiv.org/abs/2602.13921) \[full\]
10. Caumartin et al. 2026. [Retrieval-Oriented Code Representations in Agentic Bug Localization](https://www.alphaxiv.org/abs/2607.11046) \[full\]
11. Jang et al. 2026. [Flat Score, Amplified Failures: How the Error Budget Masks Damage in Quantized LLM Agents](https://www.alphaxiv.org/abs/2607.27275) \[full\]
12. Chen et al. 2025. [LocAgent: Graph-Guided LLM Agents for Code Localization (ACL 2025)](https://www.alphaxiv.org/abs/2503.09089) \[prior\]
13. Aleithan et al. 2024. [SWE-Bench+: Enhanced Coding Benchmark for LLMs](https://consensus.app/papers/details/a2fde21b76f5544c9edca661a2b6e868/?utm_source=claude_desktop) \[prior\]
14. Prathifkumar et al. 2025. [Does SWE-Bench-Verified Test Agent Ability or Model Memory?](https://consensus.app/papers/details/b4db7508813d5b5bb87f40a948b9ddbf/?utm_source=claude_desktop) \[prior\]
15. Han 2026. [A Systematic Evaluation of Trajectory Data Curation for LoRA Fine-Tuning of Code Agents](https://www.alphaxiv.org/abs/2607.17205) \[full\]
16. Zheng et al. 2026. [SWE-Prime: Fewer Trajectories, Better Performance](https://www.alphaxiv.org/abs/2608.27449) \[full\]
17. Laskar et al. 2026. [SFT or RL for Tool-Calling Agents? A Controlled Study Across Data, Method, and Scale](https://www.alphaxiv.org/abs/2609.17848) \[full\]
18. Sun et al. 2026. [LLM Agents Already Know When to Call Tools, Even Without Reasoning (When2Tool)](https://www.alphaxiv.org/abs/2605.09252) \[full\]
19. Han et al. 2025. [MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles](https://consensus.app/papers/details/002636daddcc5dc3a7790b281da631fe/?utm_source=claude_desktop) \[abs\]
20. Paipuru 2026. [The Navigation Paradox in Large-Context Agentic Coding (CodeCompass)](https://www.alphaxiv.org/abs/2602.20048) \[full\]
21. Lasbordes et al. 2026. [Small Agents with Semantic Search: Efficient Multilingual Code Localization](https://www.alphaxiv.org/abs/2610.05099) \[abs\]
22. Wang et al. 2026. [GraphEcho: Structural Redundancy and Evidence Provenance in LLM Graph Agents](https://www.alphaxiv.org/abs/2609.17695) \[full\]
23. El Idrissi et al. 2026. [PLMGH: What Matters in PLM-GNN Hybrids for Code Classification and Vulnerability Detection](https://www.alphaxiv.org/abs/2604.25599) \[full\]
24. Salis et al. 2021. [PyCG: Practical Call Graph Generation in Python (ICSE)](https://consensus.app/papers/details/b2f145e1fe035745a8b387e3f4b74740/?utm_source=claude_desktop) \[abs\]
25. Xiang et al. 2026. [InferCG: Enhancing Python Call Graph Generation via Static Analysis and LLMs (TOSEM)](https://consensus.app/papers/details/f9176ef478f6583eaef6c696919949a8/?utm_source=claude_desktop) \[abs\]
26. Kummita et al. 2021. [Qualitative and Quantitative Analysis of Callgraph Algorithms for Python](https://consensus.app/papers/details/2565e0e7f78a5d05b74121697239f917/?utm_source=claude_desktop) \[abs\]
27. Li et al. 2026. [RepoMirage: Probing Repository Context Reasoning in Code Agents with Perturbations](https://www.alphaxiv.org/abs/2605.26177) \[full\]
28. Shayanfar et al. 2026. [What Does an Agentic Software Engineering Benchmark Measure? (SNC profile)](https://www.alphaxiv.org/abs/2609.01271) \[abs + method\]
29. TUM, Cambridge and Technion authors (corr. F. Frasca) 2025. [Lost in Serialization: Invariance and Generalization of LLM Graph Reasoners](https://www.alphaxiv.org/abs/2511.10234) \[abs + conclusion\]
30. 2026. [The Context Gathering Decision Process: A POMDP Framework for Agentic Search](https://www.alphaxiv.org/abs/2605.07042) \[abs + conclusion\]
31. Gemma Team 2026. [Gemma 4 Technical Report](https://www.alphaxiv.org/abs/2607.02770) \[full, relevant sections\]
32. Feng et al. 2026. [Graph Engineering in the Era of LLM Agents (survey)](https://www.alphaxiv.org/abs/2608.21156) \[open-challenges section\]
33. Google DeepMind 2026. [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2) \[card\]
34. Liu et al. 2025. [GraphLocator: Graph-guided Causal Reasoning for Issue Localization](https://www.alphaxiv.org/abs/2512.22469) \[full\]
35. Liu et al. 2026. [DAIRA: Dynamic Analysis-enhanced Issue Resolution Agent](https://www.alphaxiv.org/abs/2603.22048) \[full\]
36. Tang and Zheng 2026. [Measuring the Serving Stack Instead of the Model: Hidden Confounds in Local Tool-Use Evaluation](https://www.alphaxiv.org/abs/2609.26693) \[full\]
37. Yang et al. 2026. [Agents' Overreliance on Unreliable Tools](https://www.alphaxiv.org/abs/2609.05587) \[abs\]
38. 2026. [Skill Issue: Lessons from Optimizing Repository SKILLs for Coding Agents (JetBrains Research)](https://www.alphaxiv.org/abs/2609.12742) \[abs + conclusion\]
39. Alonso et al. 2026. [TDAD: Test-Driven Agentic Development via Graph-Based Impact Analysis](https://www.alphaxiv.org/abs/2603.17973) \[full\]
40. Kim et al. 2026. [A²Agent: Action-Aware RL for Code Localization Agents](https://www.alphaxiv.org/abs/2608.29831) \[prior\]
41. Tao et al. 2025. [Code Graph Model (CGM)](https://www.alphaxiv.org/abs/2505.16901) \[prior\]
42. Fatemi et al. 2023. [Talk like a Graph: Encoding Graphs for Large Language Models](https://www.alphaxiv.org/abs/2310.04560) \[prior\]
43. 2026. [When Graph Tokens Sink](https://www.alphaxiv.org/abs/2606.03712) \[prior, abs\]
44. 2026. [SWE-Explore: Benchmarking How Coding Agents Explore Repositories](https://www.alphaxiv.org/abs/2606.07297) \[abs\]
45. Nebius. [SWE-agent-trajectories dataset](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) \[card\]
46. Kwai-Klear. [SWE-smith mini\_swe\_agent\_plus trajectories, 66k](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k) \[card\]
47. Huang et al. 2023. [Scalable and Precise Application-Centered Call Graph Construction for Python (JARVIS)](https://consensus.app/papers/details/0b07fcf181c755a98c72831f75ae1a22/?utm_source=claude_desktop) \[abs\]
48. Helm et al. 2024. [Total Recall? How Good Are Static Call Graphs Really? (ISSTA)](https://consensus.app/papers/details/c62e543287615a7abeb7430f76391389/?utm_source=claude_desktop) \[abs\]
49. Sui et al. 2020. [On the Recall of Static Call Graph Construction in Practice (ICSE)](https://consensus.app/papers/details/93f20309e7855a5898b1f0f293a01442/?utm_source=claude_desktop) \[abs\]

**Project files used:** `compPhilosophy.md`, `archive/HARNESS_README.md`, `archive/tasks.jsonl`, `archive/graphs/`, `archive/embeddings/`, `archive/sample_submission/`, and from git HEAD 09dadc1: `lit_review.md`, `competitor_scan.md`, `rules.md`, and the three competition PDFs.

*Consensus search notice (reproduced as required by the tool):* Upgrade to Consensus Pro to return 20 results per search instead of 10, and include more data like study design and key takeaways for every result.: https://consensus.app/pricing/?utm\_source=claude\_desktop

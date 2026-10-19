# Plan B results (living document)

Last updated: Oct 14, 2026, after FC_s1 (notebook 03). Update this file whenever a run lands: add it to section 2, revise
section 3 only when the evidence changes, and move a run from section 5 to section 2.

Plan B asks whether the 31B agent pulls the provided code graph through its three graph tools, whether pushing the same
structure into its context does better, and whether either moves resolution (pull / push / ignore). Raw outputs are in
`planB_results/` (index: `planB_results/README.md`); rules in `compPhilosophy.md`; status in `HANDOFF.md`.

Conventions: resolution is counted on the environment's own `sub_valid` tasks; intervals are Wilson 95% unless marked;
paired arm comparisons use exact McNemar on the same tasks, same build, same seed. **Noise bar:** two runs of the same
B2 condition disagree on about 3 of 23 dev tasks (section 2.3), so a dev-slice arm needs about 6 tasks moving one way
before it counts.

## 1. Headline so far

1. **The 31B agent ignores the graph tools, and forcing their use does not help.** Unprompted, 10.9% of tasks make any
   graph call (0.36% of tool calls). Ordering a graph call "every turn" raises that to every task but only about 7% of
   turns, adds wrong-argument calls, and resolves 8/23 vs 10/23 (within noise).
2. **The context limit is the largest single loss.** With 16,384 output tokens reserved, the agent has 16,384 tokens of
   input; 16 valid tasks of 103 died there in B2_s1 and **none resolved**. Compaction does fire (every 15 calls, at
   about 14k tokens), but tasks that cross 16,384 before the next check die.
3. **Multi-file tasks are nearly unsolved** (3/30 vs 35/73 single-file).
4. **The scorecard test against rewarding graph calls for their own sake (Plan B 4.4 test 4) passed only for v7 (TQS).**
   v0-v5 rank FC above B2; they do not advance as selection signals.

## 2. Runs

### 2.1 Validity and setup (B0, B1, notebook 01)
Sources: `planB_results/b0/`, `planB_results/b1/`, `planB_results/kaggle/nb01_B2_31B_all129_oldbuild/B2_s1_evaluation.md`.
- Kaggle GPU image with the test dependencies installed (inline-snapshot 0.36.0, executing 2.3.0, dirty-equals 0.11):
  **sub_valid 103/129** (fastapi 60/67, rich 43/48, requests 0/13, httpx 0/1). The test-dependency fix took fastapi from
  29 to 60 valid tasks.
- requests and httpx tests import the installed copy, not `/workspace` (B0 import check), so they are not scorable.
- Official slices (fixed Oct 12, seed 20261012): dev 24, holdout 24. Dev task `fastapi_14356` is invalid on Kaggle (gold
  patch fails), so dev denominators are 23.
- Validity is environment-specific (rich_4070 and rich_3782 are valid on the GPU image but not on CPU).

### 2.2 B2_s1: baseline, 31B, all 129 tasks, old build (notebook 01)
Source: `planB_results/kaggle/nb01_B2_31B_all129_oldbuild/`.
- Resolved **38/103 = 36.9% [28.2, 46.5]**; dev 11/23, holdout 10/24, rest 17/56.
- By task: single-file 35/73 vs multi-file 3/30; no async in gold 35/85 vs async 3/18; gold not entry point 29/69 vs
  entry point 9/34.
- End reasons (valid tasks): none 33/60 resolved, tool budget 5/26, **context exceeded 0/16**, time limit 0/1.
- Graph adoption 14/129 = 10.9% [6.6, 17.4]; 15 graph calls in 4,207 tool calls; 11/15 resolved to the intended node.
  Adopters resolve 3/9 vs non-adopters 35/94 (self-selected, uninformative).
- Gold-file read 119/129; valid tasks that never read the gold file resolve 0/9.
- Scorecard 0.3.2: no version predicts resolution (AUROC 0.48-0.59) and none is length-confounded.

### 2.3 B2-WH2_s1: build check, 31B, dev, new build (notebook 14)
Source: `planB_results/kaggle/nb14_B2_31B_dev_newbuild/`.
- The competition wheelhouse was rebuilt on Oct 13 (same versions, new files). Validity re-check 24/24 agree.
- Resolved 10/23 vs 11/23 for B2_s1 on the same tasks; 20/23 agree (B2_s1 only: fastapi_14266, fastapi_14430; WH2 only:
  rich_3052; McNemar p = 1.00). **No build effect; this pair defines the noise bar.**
- Adoption 3/24; context exceeded 2/24; gold-file read 22/24. Every new-build arm pairs with this run.

### 2.4 FC_s1: forced graph call every turn, 31B, dev, new build (notebook 03)
Source: `planB_results/kaggle/nb03_FC_31B_dev_newbuild/` (prompt paragraph in `provenance/system_prompt.md`).

| Measure (dev, 24 tasks) | B2-WH2 | FC |
| --- | --- | --- |
| Resolved (23 valid) | 10 | 8 |
| Tasks with a graph call | 3 | 24 |
| Graph calls / model calls | 3 / 818 tool calls | 57 / 820 model calls (about 7% of turns) |
| Graph calls with unresolvable arguments | 1 / 3 | 19 / 57 |
| Gold file read; median step of first read | 22; 2 | 20; 3.5 |
| Context exceeded | 2 | 6 |
| TQS (v7) median | 0.497 | 0.461 |

- Pairs: B2-WH2 only fastapi_12942, rich_3052, rich_3521; FC only fastapi_14430; McNemar p = 0.63, under the noise bar.
- FC solved the 7 tasks every 31B run solves plus fastapi_14430; it lost the 3 marginal ones.
- **Compliance was low.** The model met "call a graph tool first on every turn" on about 7% of turns (median 2 graph
  calls per task).
- **The extra context-exceeded endings are not caused by large graph outputs** (0.5-7k characters in rich_3521). In the two
  early deaths (rich_3521 after 8 calls, fastapi_12942 after 12) the prompt reached 12-13k tokens and one long thinking
  output pushed it past 16,384 before the first compaction check. B2 came within 700 tokens of the same limit on rich_3521
  (15,682) and survived because compaction fired in time. Inference: FC's extra turns push more tasks over a limit that
  already exists.
- Anti-Goodhart test (pre-registered, Plan B 4.4 test 4): v0-v5 score FC higher than B2 (fail); v7 scores it lower, mean
  difference -0.042 [-0.081, -0.009] (bootstrap 95%), 17 of 24 tasks lower (sign test p = 0.064); largest drops in the
  TQS components scp (-0.13) and ped (-0.10).

### 2.5 E4B screen: five arms, dev, one L4 on the GCP VM
Source: `planB_results/vm/e4b_dev_screen_B2-FC-QN-T0-O8K/summary_dev.md`. Pairs only with VM runs.

| Arm | Resolved (23) | Tasks with a graph call | Graph calls / tool calls | Graph calls resolving correctly | Unresolvable graph arguments | Context exceeded |
| --- | --- | --- | --- | --- | --- | --- |
| B2 | 3 | 11 | 22 / 302 | 36% | 9 | 5 |
| FC | 2 | 24 | 76 / 380 | 38% | 39 | 1 |
| QN (qualified names) | 2 | 13 | 42 / 318 | 52% | 16 | 2 |
| T0 (thinking 0) | 1 | 14 | 34 / 338 | 47% | 15 | 1 |
| O8K (8k output reserve) | 2 | 13 | 25 / 369 | 48% | 11 | 0 |

- E4B is at the floor (3/23, all tasks 31B also solves), so resolution cannot rank arms.
- E4B calls graph tools more readily than 31B (46% of tasks vs 12%) but resolves fewer of its calls to the right node.
- QN raises the correct-resolution rate (36% to 52%), as Plan A predicted (bare names misresolve 41%, qualified 5%).
- FC raises the number of graph calls, not their accuracy, and quadruples wrong-argument calls.
- O8K removes context-exceeded endings (5 to 0) on E4B.
- E4B quits early (median 11 tool calls vs about 30 for 31B; gold file read 15/24 vs 22/24).

### 2.6 12B: does not serve (notebook 11)
Source: `planB_results/kaggle/nb11_B2_12B_dev_newbuild_FAILED-serving-tp4/`. vLLM 0.19.1 has no native `gemma4_unified`; the
Transformers fallback crashed at tp 4 (`ValueError: Expected hidden_size to be 3840, but found: 256`). No task ran.
**12B is dropped** (Oct 14).

### 2.7 LoRA training data from B2_s1 (Plan C input)
Source: `planB_results/lora_data/rft_31b_B2s1/manifest.json`. 17 resolved trajectories outside both slices (fastapi 8, rich 9),
540 assistant turns; 234 usable before the first compaction (195 train / 39 val, split by task). One graph call in the
whole set: this data cannot teach graph adoption.

## 3. What the evidence says about the Plan B hypotheses

| Hypothesis | Status | Evidence |
| --- | --- | --- |
| H-B1: adoption is low and does not rise where structure is needed | **Supported (31B)** | 14/129 tasks; 8 of 9 valid adopters are in d_undef / d2+ cells, but adoption is too rare to test "tracks need" |
| H-B2: pull benefit only for d >= 2 with low visibility | Untestable with pull as is | Too few graph calls for B3 to differ from B2; FC shows that forcing calls does not create a benefit |
| H-B3: push beats pull in the low-visibility cell | **Not yet run** | Notebook 16 (B4) |
| H-B4: the graph helps find the second file | Not yet testable | 3/30 multi-file tasks resolved; needs B4 / B6 on multi-file tasks |

Cross-cutting conclusions:
- Prompt pressure changes whether the model touches the graph at all, not how often or how well it uses it. Any real test of the
  graph's content has to put the content in front of the model (B4/B5) or train adoption (C5).
- The 16,384-token input limit costs more resolution than anything the graph arms have moved. It caps every arm and adds
  noise to every paired comparison.
- Graph-call accuracy is a measured weak point on both models (31B: 61-73%; E4B: 36%); qualified names help.

## 4. How this compares with the literature

- **Structure works when it is delivered, not when the agent is told to fetch it.** LARGER injects graph neighbours into grep
  results and loses most when graph expansion is removed; CodeAnchor puts call-graph links into source comments; neither
  adds a tool. CodeScout's RL policy collapses to `rg` and `sed` and still matches graph-tool agents (all three as summarised
  in `lit_review.md`, section B1/B4). FC took the opposite route and the model mostly ignored it.
- **Rules in the prompt are followed partly; rules the harness enforces are followed.** In a study of agents querying a failing
  source, the agents judged its results useless 97-100% of the time and still did not stop: "a stopping rule or call cost in the
  prompt is followed at most partly. Stopping follows the evidence only when the harness enforces an integration step"
  ([Judged Useless, Queried Anyway, p. 1](https://www.alphaxiv.org/pdf/2610.06191?page=1)). FC's 7% per-turn compliance fits
  this; B4 is the harness-level version.
- **Unneeded or unreliable tools have costs even when rarely called.** Making an unnecessary tool available cut the answer rate
  from 98.2% to 63.5%, "even when the tool is rarely called"
  ([When Tools Get in the Way, p. 1](https://www.alphaxiv.org/pdf/2609.14157?page=1)). Agents adopt corrupted tool returns at
  high rates and call tools more when the returns are bad
  ([Agents' Overreliance on Unreliable Tools, p. 1](https://www.alphaxiv.org/pdf/2609.05587?page=1)). With 41% bare-name
  misresolution (Plan A), forced graph calls are often noise.
- **Where graph tools do help, the use was learned or built into the pipeline.** LocAgent's gains come from SFT on graph-tool
  trajectories; ARISE credits its data-flow graph, not the tool schema (`lit_review.md`, B1). This supports C5 (adoption LoRA)
  over prompt pressure.
- Understanding, not editing, is where agents spend effort: in CABRA, the number of reading and analysis tool calls predicts
  SWE-bench Verified accuracy better than lines edited ([Code Understanding is a Bottleneck, p. 1](https://www.alphaxiv.org/pdf/2610.10610?page=1)).
  Consistent with our gold-file-read result (0/9 resolved without reading it).

## 5. Runs prepared, not yet run (Kaggle, new build)

Notebooks in `planB_results/kaggle_upload/notebooks/`, generated by `scripts/planB/notebook_builder/make_nb15_17.py` from
notebook 14 (same runner; only CONFIG, the task list and the B4 brief option differ). Inputs as notebook 14. Run 15 first or
in parallel; 16 and 17 pair with it task by task.

| # | Arm | Tasks | Est. time | Question | Prediction stated before the run |
| --- | --- | --- | --- | --- | --- |
| 15 | B2-SUB: B2 on the union of 16's and 17's tasks | 43 (14 dev) | ~2.5 h | paired baseline; second B2 sample on 14 dev tasks | dev tasks: about 3 flips vs notebook 14 |
| 16 | B4 push: structural brief prepended to the issue | 28 | ~1.5 h | does delivered structure help where pulled structure was ignored? | earlier first gold-file read; resolution within noise unless 6+ tasks move |
| 17 | O8K: max_output_tokens 8192 | 21 context-heavy | ~1.5 h | does the context limit, not the task, cause those failures? | context-exceeded endings fall to near 0; some of those tasks resolve |

Why these task sets:
- **B4 cannot be tested on dev.** The pre-registered brief needs code elements named in the issue; most dev issues are short PR
  titles ("fix fonts"), so only 9/24 dev tasks get a brief, and B2 already resolves all 8 valid ones. Over all tasks, 51/129 get
  a brief; on valid non-holdout tasks 28 do, 23 of those briefs name a gold file, and 11 of those 23 were not resolved by B2_s1.
  Brief rules and per-task diagnostics: `scripts/planB/build_b4_briefs.py`, `planB_results/kaggle_upload/b4_briefs/`.
- **O8K on dev can move at most about 2 tasks**, so it runs on every valid non-holdout task that hit the context limit in any
  31B run (B2_s1, B2-WH2, FC).
- No task from the holdout slice is used (the notebooks assert this).

Still open after these: B3 (expected equal to B2, since graph calls are rare), seeds 2-3 for the noise bar, QN on 31B (low
leverage until adoption rises), a context-hygiene prompt arm (line ranges instead of whole files, tail long outputs), B5 (the
B4 brief produced inside the sandbox, so it is legal in a submission) if B4 shows a benefit.

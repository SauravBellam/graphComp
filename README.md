# graphComp: post-training Gemma 4 31B into a SWE agent

This repository is our entry for the Kaggle **Gemma 4 Developer Agent** competition (prediction track plus paper track). The
competition asks for an agent that fixes real bugs in Python repositories (fastapi, rich, requests, httpx), running offline.
The submitted model must be `gemma-4-31b-it-qat-w4a16-ct`, optionally with a LoRA adapter. The agent runs inside the
competition's `swegemma` harness, whose tools are `run_command`, `read_file`, `edit_file`, `write_file`, `get_status`,
`submit_patch`, plus three code-graph tools.

The work went through three phases:

| Phase | Dates (2026) | Question | Where |
| --- | --- | --- | --- |
| Plan A | Oct 10-11 | Is the competition's pre-built code graph and its node embeddings accurate? | [`governDynamics`](https://github.com/SauravBellam/governDynamics) repo, `planA_results/` |
| Plan B | Oct 12-14 | Does the 31B agent use the graph tools, and does pushing structure into its context help? | notebooks 01-17, [`results/plan_b_results.md`](results/plan_b_results.md) |
| **Post-training (SFT)** | Oct 14-19 | Can supervised fine-tuning on verified agent trajectories make the 31B a better agent? | notebooks 19-21, `scripts/planB/{lora,external,notebook_builder}`, this README |

Most of this README covers the third phase. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) lists every experiment and its
result. [`docs/NOTEBOOKS.md`](docs/NOTEBOOKS.md) indexes all Kaggle notebooks.

---

## 1. Where things stand (Oct 19, 2026)

| Item | Status |
| --- | --- |
| Best measured agent | **Base 31B: 14/27 on our holdout** (notebook 19; Wilson 95% [34.0%, 69.3%]). On all 129 tasks the base agent resolved 38/103 valid tasks (36.9%, notebook 01). |
| Training data | **36,442 verified trajectories over 13,928 tasks** that fit the 32k context, converted into our harness's tool format from public Qwen3-Coder-480B trajectories. Plus **82 verified 31B trajectories on 46 competition tasks**. |
| 31B LoRA | Training notebook 21 runs on Kaggle in resumable 11.75 h sessions. **Sessions 1 and 2 trained 0 steps** (out of GPU memory, explained in §4.5). Session 3 started Oct 19 with the memory fixes. No LoRA has a measured resolution yet. |
| Submission | Every session writes a submittable `submission.zip`: the LoRA if it trained and vLLM applies it, the base model otherwise. The final-submission cap is 8 min per task. |

The honest summary: the data pipeline is complete and verified, but the trainer is not yet proven at 31B scale on 4×L4.
No fine-tuned model has a measured resolution yet.

---

## 2. Background from Plan B (why we moved to post-training)

All numbers are from `results/` (Kaggle, 4×L4, 31B w4a16, the competition's vLLM build).

- **Base agent:** 38/103 valid tasks resolved (36.9% [28.2, 46.5]). Single-file tasks 35/73, multi-file tasks 3/30.
- **The model ignores the graph tools.** 10.9% of tasks make any graph call (0.36% of tool calls). Forcing a graph call
  every turn gave 8/23 vs 10/23 on dev, within noise.
- **The context limit was the largest single loss.** 16 valid tasks hit 16,384 input tokens and none resolved. Dropping
  the output reserve to 8192 removed those endings on E4B.
- **Noise bar:** two runs of the same condition disagree on about 3 of 23 dev tasks. A dev comparison needs about 6
  tasks moving one way before it counts.
- **12B could not be served** by the competition's vLLM at tp 4. The rules also allow only the 31B as the submitted
  model, so every adapter we train is a 31B adapter.

Prompt pressure changed whether the model touched a tool, not how well it worked. The literature points the same way:
tool use that helps is learned (e.g. LocAgent's SFT on graph-tool trajectories), not prompted. That moved us to
fine-tuning.

---

## 3. Post-training, first attempt: self-generated trajectories (notebook 19, Oct 15)

**Design** (`notebooks/19_README.md` has the full rationale and citations):

1. The 31B generates **teacher trajectories** under the exact deployed agent: 53 non-slice tasks × seeds 11-13, 8 min and
   30 calls each.
2. `build_rft_data.py` keeps resolved trajectories only: at most 3 per task, none from dev or holdout tasks.
3. `train_lora.py` v2 trains a **31B QLoRA** (4-bit nf4, rank 32 / alpha 64, lr 1e-4, assistant-only loss, step-level
   error masking). It uses the deployed system prompt, so train matches deploy. k_proj is excluded on the k_eq_v
   full-attention layers, which vLLM would otherwise serve wrongly.
4. A 27-task **holdout** covering all 9 task categories compares BASE31 and LORA31. The winner is submitted under a
   budget-safe deployment config.

**Outcome:**

- **Trajectories:** 280 attempts gave 45 verified trajectories (42 with accurate tool use, 44 used for training);
  `results/trajectories/graphcomp_traj_v1/`.
- **31B LoRA failed to train.** Accelerate moved the `logits_to_keep` tensor to GPU 0; fixed in commit `427477c`.
- **12B student failed to load.** Kaggle's copy of `gemma-4-12b-it-qat-q4_0-unquantized` v2 has a broken safetensors
  file.
- **BASE31 scored 14/27** on the holdout and became the submission. Its 5-min / 30-call budget guarantees the 12 h limit
  even if every task hits its cap.

**Lessons:**

- Self-distillation from the 31B is weak by itself; SWE-Gym saw resolve rates fall when training on a model's own
  trajectories.
- About 45 trajectories is far too little data.

**Notebook 20** (45 h iterative RFT campaign, `make_nb20.py`) was built but never run; the outside-data route below
replaced it. Its fixed 31B LoRA training code is what notebook 21 builds on.

---

## 4. Post-training, main attempt: outside verified trajectories → 31B LoRA (notebook 21)

### 4.1 Rules the user set for the data

- **Teacher must be stronger than Gemma 4 31B.** In practice that means **Qwen3-Coder-480B-A35B only**.
  - Excluded: SWE-smith (teacher models borderline or weaker) and Kwai (unknown teacher). The converter
    still supports both.
- **Verified only:** every trajectory passed the task's tests. No heuristic labels.
- **Deduplication:** exact and near-duplicate trajectories are removed. Different solution routes for the same problem
  are kept.
- **Training tasks:** dev and holdout tasks may be trained on. The final model is the student; there is no BASE31
  comparison.
- **Competition data last:** competition trajectories are trained at the end of every session.

### 4.2 Converting trajectories into our harness by replay (`scripts/planB/external/convert_trajs.py`, exp `ca1b26ab`)

**Sources:**

- `togethercomputer/CoderForge-Preview` (filtered_reward1): 155,144 trajectories over SWE-rebench, SWE-smith and R2E-Gym
  tasks.
- `nebius/SWE-rebench-openhands-trajectories` (resolved == 1): 32,161 trajectories.

These are OpenHands-style traces. We could not just rename their tools, because the observations would no longer be
what our tools return. Each trajectory is therefore **replayed on a real checkout** of its task:

- **Editor calls** become our `read_file` (150-line / 10k-char chunks), `edit_file` (via the harness's own
  `apply_replacement`) and `write_file`.
- **Directory views** become `run_command find`; bash keeps the source's output.
- **Every file view is compared line by line** with the source trace. A stale or mismatched view rejects the trajectory.
- **OpenHands soft timeouts are merged:** a wait becomes the full output, and a Ctrl-C becomes `TimeoutExceeded`.
- **Literature filters (SWE-Lego 2601.01426, SWE-HERO 2604.01496):**
  - git hacking (commits after the base);
  - a final patch that modifies tests;
  - 3 or more editor errors;
  - a null source change.

**Result** (run c9b10d84):

- **CoderForge:** 124,174 / 155,144 kept, over 32,208 tasks.
- **nebius:** 25,326 / 32,161 kept, over 3,511 tasks. 97.6% of nebius final diffs equal the source `model_patch` exactly,
  which checks the replay.
- **Main rejects:**
  - ≥3 editor errors: 9,966;
  - modifies tests: 8,614;
  - stale file view: 8,581;
  - interactive input: 2,762;
  - no finish: 2,716;
  - git hacking: 1,984.

Every row carries provenance labels: source dataset, trajectory id, model, scaffold, and the filters it passed.

### 4.3 The SFT mix (`scripts/planB/lora/build_sft_mix.py`, exp `5303fc9a`)

- **Duplicates:**
  - Exact duplicates are the same task and the same tool-call sequence.
  - Near-duplicates are the same task with route similarity ≥ 0.9 (difflib over (tool, target) sequences); the
    trajectory with fewer calls is kept.
- **Dropped:** rows with `pip install` / downloads, since deployment is offline.
- **Result:** 149,517 rows read; 12,822 install rows and 705 near-duplicates dropped. **135,990 kept over 32,466
  tasks.**
- **Fit:** **36,442 fit the 32k-token window, over 13,928 tasks** (CoderForge 33,562, nebius 2,867, own 13).
- **Kaggle datasets (private):**
  - `sauravbellam/graphcomp-sft-mix`: the fit set.
  - `sauravbellam/graphcomp-external-trajs-all`: every converted row.
  - `sauravbellam/graphcomp-comp-trajs`: 82 verified 31B trajectories on 46 competition tasks.

### 4.4 Training design (notebook 21, `scripts/planB/notebook_builder/make_nb21.py`)

**Loss** (literature-checked; links in the notebook's second cell):

- Loss on assistant turns only, with the whole conversation as context.
- No loss on turns whose tool call errored (step-level error masking).
- **Turns with no reasoning text are context only.** The thinking-on Gemma 4 chat template cannot target them. Our GPU
  functional test found this bug; about 10 such turns per converted trajectory.

**Resumable sessions:** one session per weekly Kaggle quota (4×L4 hours count double, so 11.75 h ≈ 23.5 quota-hours).

- A session runs 11 h 45 min and stops training ~0.85 h before the end, so it can save, check the adapter and write the
  submission.
- `NB21_STATE.json` records which trajectories and tasks were trained, competition visit counts, measured tokens/s,
  out-of-memory windows and the session log.
- Each session attaches the previous one's output (`kernel_sources`) and continues from its adapter (`--init-adapter`).
- The trainer checkpoints atomically every N minutes (`--save-every-min`), so a crash keeps progress.

**Data order inside a session:**

- **New problems first** (tasks never trained), **shorter trajectories first**, mixed round-robin over length quartiles.
- **~8% replay** of trajectories from earlier sessions, spread through the slice. Repeated identical texts make models
  fragile (2609.33559); replay prevents forgetting (SWE-Lego's cumulative curriculum).
- **Competition trajectories form the tail and are trained last** (`--tail-n`). The trainer jumps to the tail before the
  deadline, so every session ends on competition data.
  - Each session takes the least-visited half of the competition trajectories, with a cap of 3 visits per trajectory.

**Deployment config:** 8 min per task (user's choice, Oct 19), 30 tool calls, 80 turns.

- If all ~120 hidden tasks hit the 8-min cap, the run takes ~17.2 h, over the 12 h limit. The notebook prints this risk
  instead of asserting.
- It relies on most tasks finishing early: the 31B averaged ~6.3 min at concurrency 4.

**Adapter check:** the LoRA is served exactly as the scorer serves it, and it must change a greedy answer, which catches
no-op adapters. If there is no adapter, the base model is submitted.

### 4.5 What went wrong at 31B scale and the fixes

Memory per GPU is the binding constraint. The 31B in 4-bit, split over 4 L4s (24 GB each), leaves little room for long
trajectories. The last GPU also holds the 262,144-token vocabulary output layer.

| Session | What happened | Root cause | Fix (experiment, VM-tested) |
| --- | --- | --- | --- |
| s1 (Oct 18) | 32k, 24k and 18k windows all ran out of memory at the first backward pass; 0 steps; ~28 min lost per retry while the 31B reloaded | `--max-target 8000` means 8000 × 262k-vocab logits, plus stored float32 cross-entropy chunks (~17 GB) on GPU 3 | Checkpointed cross-entropy chunks; target tokens 2048/1536/1024 per window; in-process OOM recovery with no model reload (exp `b70d7942`, run 4e2bd751, 18/18 checks) |
| s2 (Oct 19) | First real OOM at a 25.9k-token sample, then every sample down to 4.9k OOMed; 0 steps | GPU 3 held lm_head + logits + 15 layers; after the first OOM the memory was not returned (leak), so recovery cascaded | Custom device map (last GPU: embeddings/lm_head + 6 layers, the rest spread evenly); leak check after an OOM → save + exit 75 → the notebook continues at the next window from the saved adapter; windows 24k/20k/16k; speeds from 0-step sessions ignored (exp `b83c0e28`, run 4fd70af7, 18/18 checks) |
| s3 (Oct 19) | running | first session with all the above | check: steps > 0, tokens/s, which window trained, OOM events |

**How the trainer is tested** (`scripts/planB/lora/test_train_resume.py`):

- The model is a 4-layer random-weight copy of the 31B's own config (same tokenizer, chat template and k_eq_v
  full-attention layer), trained on 200 real converted trajectories on the VM's L4.
- **Case A:** deadline, tail jump and checkpointing.
- **Case B:** resume from A's adapter.
- **Case C:** a simulated OOM, which must recover in-process.
- **Case D:** peak memory of the old vs checkpointed loss. At 8000 targets, 20.2 → 17.2 GB; at 2048 targets, 6.1 GB.
- The tiny model cannot show whether real 31B activations at 24k fit on GPU 3; session 3 is that test.

### 4.6 Side experiments that fed the data

- **External task import** (exp `394a0d99` and validity v2-v4): 160 same-repo SWE tasks from SWE-bench, SWE-rebench and
  SWE-bench-extra were converted to the competition layout, with snapshots, code graphs (98-100% node recall vs the
  competition's graphs) and embeddings.
  - Validity rose from 8/160 → 30 → 38/160 as missing test dependencies went onto a sandbox path (trustme,
    pytest-asyncio, pytest-trio).
  - The rest cannot be fixed on the competition image (Python 3.12 vs old requests, fastapi tasks without tests).
- **requests/httpx re-grader** (exp `da9699f7`, `b2017c91`): these tasks were unscorable because the tests imported the
  installed library. A `src/` path shim plus pytest-httpbin made **10/14 valid** (from 0).
- **Synthetic procedural bugs** (exp `eb125260`): SWE-smith-style mutations at competition snapshots, validated
  gold/empty. A smoke run on httpx gave 12/13 valid. The full run stopped on VM disk space and is paused.
- **Forced graph-call rate search on E4B** (exp `c2358a8e`) and **tool tracing** (exp `d4e6af3b`): cancelled or not run
  once post-training took priority. The 31B tool-trace baseline is in `results/trajectories/graphcomp_traj_v1/tool_trace/`.

### 4.7 Not done (on purpose or blocked)

- **Teacher runs on the imported external tasks that no converted trajectory covers** (planned teacher: GLM-5.3, an open-weight
  model). Paused: no inference credits.
- **No proprietary-model-generated training data:** provider terms restrict training other models on their outputs. The user
  chose the open-weight route.
- **No resolution measurement of any LoRA yet:**
  - The VM's PyPI vLLM cannot serve Gemma 4 LoRAs, so adapters can only be evaluated on Kaggle.
  - Notebook 21 does not run a holdout comparison, by the user's decision.

---

## 5. Repository map

```
README.md                    this file
docs/EXPERIMENTS.md          every experiment: question, runs, result, branch
docs/NOTEBOOKS.md            every Kaggle notebook: what it ran, status
notebooks/                   all Kaggle notebooks (01-21, 21s smoke), 19_README.md
notebooks/kaggle_kernels/    kernel-metadata.json for notebook 21 sessions 1-3 (inputs, machine, chaining)
results/                     Plan B summaries, notebook 19 summary, trajectory-set manifests, B0/B1 setup records
scripts/planB/
  run_arm.py                 runs one agent arm in the harness (gold / empty validity, budgets, remote teachers)
  external/                  convert_trajs.py (replay converter), import_external.py, code_graph.py, embed_graph.py
  lora/                      train_lora.py, build_rft_data.py, build_sft_mix.py, test_train_resume.py
  notebook_builder/          make_nb19.py, make_nb20.py, make_nb21.py (notebooks are generated; edit these, not the .ipynb)
  vm/                        dev_runs.py, fc_search.py, merge_lora.py (single-L4 VM jobs)
vm_job.sh                    the experiment entry point on the VM (one JOB block per experiment branch)
rules.md, lit_review.md, competitor_scan.md   competition rules digest, literature review, public-notebook scan
```

**Branches.** Each `orx/<slug>` branch is one experiment, with the exact code its runs used (see
`docs/EXPERIMENTS.md`). `main` is this write-up, built on the latest experiment branch
(`orx/31b-memory-lighter-gpu-3-device-map-oom-leak-det`).

## 6. Reproducing

- **Notebooks are generated.** Edit the builder and run
  `python scripts/planB/notebook_builder/make_nb21.py` (add `--smoke` for 21s). Then copy the notebook into a kernel
  folder with the session's `kernel-metadata.json` and run `kaggle kernels push -p <folder>`.
- **Session chaining.** For the next session, copy the folder, bump the `-sN` id, and set `kernel_sources` to the
  previous session.
- **Data.** Rebuild with `convert_trajs.py` (needs the source datasets from Hugging Face) and then `build_sft_mix.py`.
  The Kaggle datasets above are the built outputs.
- **Trainer test.** `python scripts/planB/lora/test_train_resume.py --data <trajs.jsonl> --work <dir>` on any CUDA GPU.

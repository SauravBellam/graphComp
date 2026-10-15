"""Build notebook 19: teacher trajectories -> 31B LoRA + E4B student LoRA -> holdout comparison -> budget-safe submission.

Reuses notebook 15's setup cells (inputs, offline install, test deps, provenance, run_arm.py) and TPU T2's task list; embeds
build_rft_data.py and train_lora.py. Re-run after editing any embedded script:
    python3 scripts/planB/notebook_builder/make_nb19.py
"""
import json, re
from pathlib import Path
R = Path("/Users/sauravbellam/Desktop/Projects/graphComp")
U = R / "planB_results/kaggle_upload"
nb = json.load(open(U / "notebooks/15_B2_subset_newbuild_s1.ipynb"))
t2 = json.load(open(U / "TPU_runs/T2_tpu_rft_datagen.ipynb"))
line = [l for l in "".join(t2["cells"][3]["source"]).split("\n") if l.startswith("REST =")][0]
REST = json.loads(line[len("REST = "):].split("#")[0])
SAMPLE_PROMPT = (R / "archive/sample_submission/prompts/system.md").read_text()
C = nb["cells"]
src = lambda i: "".join(C[i]["source"])
def sub1(s, old, new):
    assert s.count(old) == 1, old[:80]
    return s.replace(old, new)
def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}
def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

# ------------------------------------------------------------------ the deployed agent's prompt addendum
ADDENDUM = """

## Budget: always finish inside the time limit
The task prompt states your time allowance. When it runs out the session stops and the edits in /workspace are graded as
they are; an unedited tree always scores zero.
- `get_status` is free (it does not use a tool call): call it every 4-5 tool calls and read `time_seconds_remaining`.
- Use at most the first 40% of the time to locate the code. By then make your best edit at the most likely source location,
  even if you are unsure; you can correct it later.
- With less than 30% of the time left: no more exploration. Run at most one targeted test, fix only what it shows, then call
  `submit_patch`.
- With less than 60 seconds left: call `submit_patch` immediately.
- Keep outputs short (`| head -40`, `sed -n 'START,ENDp' FILE`): long outputs fill the context window and end the session.

## Graph tools
`get_code_neighbors` and `get_code_subgraph` resolve fully qualified names only (for example
`fastapi.routing.APIRoute.get_route_handler`); bare names often resolve to the wrong symbol. Use them only with a qualified name
you have already seen in the code.
"""

HEADER = """# 19 — teacher trajectories → 31B LoRA + two students (E4B, 12B) → holdout comparison → budget-safe submission

Offline (Internet off), GPU L4 × 4. Rationale and every source: `19_README.md` next to this notebook.

**What it does, in order (each stage is skipped if an attached earlier version of this notebook already finished it):**
1. **Teacher trajectories.** The competition model `gemma-4-31b-it-qat-w4a16-ct` runs the *deployed agent* (sample prompt +
   the budget section, nine tools, `max_output_tokens` 8192, thinking 4096) on the valid, non-slice tasks, seeds 11, 12, 13.
   With notebook 01's seed-1 run, `build_rft_data.py` keeps resolved trajectories (at most 3 per task) → `graphcomp_arm/rft/`.
2. **LoRAs on that data** (`train_lora.py`, QLoRA; choices and sources below):
   - **31B** on `gemma-4-31b-it-qat-q4_0-unquantized`, the 4-bit model split over all 4 GPUs (time budget `TRAIN31_MAX_H`);
     no k_proj LoRA on the full-attention layers (attention_k_eq_v).
   - **Two students at the same time:** E4B on GPUs 0-1 and 12B on GPUs 2-3, each data-parallel over its 2 GPUs.
3. **Holdout comparison**, all arms with the same deployed agent: `BASE31` (no adapter, control) and `LORA31` on one 31B server;
   then `STUDENT_E4B` and `STUDENT_12B` **at the same time**, each on its own GPU and vLLM port (every agent's tool calls stay
   sequential). The holdout covers every kind of task: the official 24 (fastapi + rich, single- and multi-file) plus one task of
   each category the official slice lacks (moved out of the training pool), plus one requests and one httpx task that run as a
   budget check only (their tests import the installed library in this sandbox, so they cannot be scored here).
   12B failed to serve at tp 4 in notebook 11; here it is tried at tp 1, and a failure only drops that arm.
4. **Submission** = the arm with the most scored holdout tasks resolved (ties → BASE31, LORA31, E4B, 12B), excluding any arm
   whose adapter vLLM did not apply. `/kaggle/working/submission/` + `submission.zip`, with the deployment budget.

**Deployment budget (12-hour limit).** Hidden tasks run one after another; a submission sets per-task caps only. Cap =
`(720 − 15 serve − 20 margin) / 120 tasks − 0.5 setup ≈ 5 min`, 30 tool calls: 120 tasks all hitting the cap still finish in
about 11.4 h. The prompt makes the agent edit early and watch `get_status` (free) for `time_seconds_remaining`; on a timeout
the harness grades the working tree as it is.

**Time:** setup 10 min · teacher ≤ 4.6 h · 31B training ≤ 2.6 h · students ≤ 0.8 h (parallel) · holdout ≈ 0.9 h (31B arms)
+ 0.9 h (students, parallel) · about 10.5 h. A stage that would not fit stops cleanly; attach this output to the next version.

**Inputs:** competition data; `metric/gemma-4-developer-agent-wheelhouse`; models `google/gemma-4`: `gemma-4-31b-it-qat-w4a16-ct`,
`gemma-4-e4b-it-qat-w4a16-ct`, `gemma-4-12b-it-qat-w4a16-ct`, and (Transformers) `gemma-4-31b-it-qat-q4_0-unquantized`,
`gemma-4-e4b-it-qat-q4_0-unquantized`, `gemma-4-12b-it-qat-q4_0-unquantized`; **notebook 01's output**; datasets
`graphcomp-planb` and `graphcomp-train-wheels`; for a resumed run, this notebook's previous version output.
"""

TRAINING_CHOICES = """## Training choices (literature check, Oct 15)
- **Assistant-only loss with the whole conversation as context; split by task.** The tool-calling SFT baseline of SFT or RL for
  Tool-Calling Agents? (arXiv 2609.17848) puts the loss on assistant turns only, with LoRA rank 32, alpha 64, lr 1e-4, 2 epochs,
  all linear layers (read from the alphaXiv summary). Turns are packed into windows so one forward pass trains several turns.
- **Step-level error masking** (SWE-Lego, arXiv 2601.01426, alphaXiv summary): turns whose tool call failed keep their place in the context but get
  no loss; test and reproduction failures are not masked (here: run_command results are never masked).
- **Same prompt and serialization in training and deployment** (2609.17848 used the same system prompts and serialization
  in training and evaluation): `--system-prompt` and `--deploy-budget`.
- **Quantity over filtering at small scale** (Systematic Evaluation of Trajectory Data Curation for LoRA Fine-Tuning of Code
  Agents, arXiv 2607.17205, alphaXiv summary: going from 500 to 1,000 trajectories lowered held-out loss about 18x more than
  quality filtering did; resolved-only gating was not better than random at 500): up to 3 resolved trajectories per task instead of 2, every seed kept.
- **Vanilla LoRA, tune the learning rate** (Learning Rate Matters: Vanilla LoRA May Suffice, arXiv 2602.04998, alphaXiv summary:
  LoRA variants reach vanilla LoRA's peak once each one's learning rate is tuned): plain LoRA, lr 1e-4, cosine with warmup.
- **Teacher stronger than student** (SWE-Gym, arXiv 2412.21139, p. 5: fine-tuning on the policy's own trajectories
  lowered the resolve rate from 15.3% to 8.7%); the 31B LoRA is
  close to self-distillation, which is why BASE31 is in the comparison and wins ties.
"""

# ------------------------------------------------------------------ config cell (nb15 cell 2)
c = src(2)
c = sub1(c, 'VARIANT_TAG = "B2-SUB"           # names the run directory: runs/<VARIANT_TAG>_s<SEED>',
            'VARIANT_TAG = "NB19"             # names the run directory: runs/NB19 (parts end in _s<seed>)')
c = sub1(c, "SEED = 1                      # Plan B seeds 1, 2, 3 (identical across paired arms)",
            "SEEDS = [11, 12, 13]          # teacher replicates; with notebook 01's seed 1 that is 4 attempts per task\n"
            "SEED = SEEDS[0]\n"
            "RECHECK_VALIDITY = False      # notebook 14: 24/24 agree across builds; the rebuilt wheel only split the pytest timeout\n"
            "SESSION_BUDGET_H = 11.3       # Kaggle stops GPU sessions at 12 h; output is saved only if the notebook ends\n"
            "DATAGEN_END_H = 4.6           # no new teacher chunk after this\n"
            "TRAIN31_MAX_H, TRAIN_STUDENT_MAX_H, EVAL_ARM_H = 2.6, 0.8, 0.9\n"
            "# ---- the deployed agent (also used for the teacher runs and the holdout arms)\n"
            "N_TASKS_ASSUMED, SERVE_START_MIN, MARGIN_MIN, SETUP_MIN = 120, 15, 20, 0.5\n"
            "DEPLOY_TIME_MIN = int(((720 - SERVE_START_MIN - MARGIN_MIN) / N_TASKS_ASSUMED - SETUP_MIN) * 2) / 2   # 5.0\n"
            "DEPLOY_BUDGET = dict(max_tool_calls=30, max_time_minutes=DEPLOY_TIME_MIN, max_turns=80, timeout_seconds=240)\n"
            "assert N_TASKS_ASSUMED * (DEPLOY_BUDGET['max_time_minutes'] + SETUP_MIN) + SERVE_START_MIN + MARGIN_MIN <= 720\n"
            "CONC_SLOWDOWN = 1.6           # estimate: a task at concurrency 4 runs about this much slower than alone\n"
            "# ---- students: trained side by side (2 GPUs each), then served side by side (1 GPU and port each)\n"
            "STUDENTS = {'STUDENT_E4B': dict(key='e4b', slug='gemma-4-e4b-it-qat-w4a16-ct', train_gpus=[0, 1], master_port=29510, eval_gpu=0, port=8001,\n"
            "                            max_target=2500, max_len=18432),\n"
            "            # 12B in 4 bits (~9 GB with embeddings and towers) + activations of a full window would reach ~20 GB of 22.5 on one L4\n"
            "            'STUDENT_12B': dict(key='12b', slug='gemma-4-12b-it-qat-w4a16-ct', train_gpus=[2, 3], master_port=29520, eval_gpu=1, port=8002,\n"
            "                            max_target=1500, max_len=14336)}\n"
            "")
c = re.sub(r"^TASK_LIST = \[.*\]$", lambda m: f"TASK_LIST = {REST!r}   # 56: sub_valid in nb01, in neither official slice", c, count=1, flags=re.M)
c = sub1(c, "MAX_OUTPUT_TOKENS = 16384", "MAX_OUTPUT_TOKENS = 8192 ")
c = re.sub(r"^PROMPT_ADDENDUM = ''$", lambda m: "PROMPT_ADDENDUM = " + repr(ADDENDUM), c, count=1, flags=re.M)
c = sub1(c, "BUDGET = dict(max_tool_calls=50, max_time_minutes=30, max_turns=100, timeout_seconds=300)",
            "BUDGET = dict(max_tool_calls=30, max_time_minutes=8, max_turns=80, timeout_seconds=240)   # in-notebook runs (concurrency 4)")
c = sub1(c, 'RUN_NAME = f"{VARIANT_TAG}_s{SEED}"', 'RUN_NAME = VARIANT_TAG')
c = c.replace("SESSION_BUDGET_H = 11.0       # stop starting chunks when the next one might not finish before this (Kaggle limit is 12 h)\n", "")
setup = [code(c)] + [C[i] for i in range(3, 16)]   # helpers .. oracle pack / scorecard import
assert "PROMPT_ADDENDUM = " + repr(ADDENDUM) in c and "SESSION_BUDGET_H = 11.3" in c and c.count("SESSION_BUDGET_H =") == 1

# resume: copy an attached earlier version of THIS notebook's output (everything: parts, data, adapters, holdout arms)
RESUME = """# STEP 0 - resume: copy an attached earlier version of this notebook's output
_prior = [dp for dp, dns, fns in walk(INPUT, maxdepth=8) if dp.name == "graphcomp_arm" and (dp / "NB19_STATE.json").exists()]
for _p in _prior:
    shutil.copytree(_p, OUT, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.zip"))
    plog(f"resume: copied earlier output {_p}")
STATE_FILE = OUT / "NB19_STATE.json"
STATE = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
def save_state(**kw):
    STATE.update(kw)
    STATE_FILE.write_text(json.dumps(STATE, indent=1))
save_state(sessions=STATE.get("sessions", 0) + 1)
def left_h():
    return SESSION_BUDGET_H - (time.time() - _T0) / 3600
plog(f"state: {STATE}")
"""

SUBMISSION = C[16]   # write_submission + SUB (uses BUDGET, MAX_OUTPUT_TOKENS, PROMPT_ADDENDUM)
VALIDITY = code(sub1(src(17), "    if STACK_CHANGED:   #", "    if STACK_CHANGED and not RECHECK_VALIDITY:\n"
                "        plog(f'validity re-check skipped (RECHECK_VALIDITY=False) on build {STACK_ID}; using notebook 01 validity')\n"
                "    elif STACK_CHANGED:   #"))
ORDER_CELL = C[18]

TODO = r'''# STEP 2 - holdout set (official slice + one task per missing category) and what is left of the teacher runs
INFRA_RE = re.compile(r"APIConnectionError|ServiceUnavailable|Connection refused|ConnectError|InternalServerError|"
                      r"BadGatewayError|EngineDeadError|Server disconnected|RemoteProtocolError", re.I)
if set(ORDER) & (set(DEV) | set(HOLDOUT)):
    raise SystemExit(f"STOP: TASK_LIST touches an official slice: {sorted(set(ORDER) & (set(DEV) | set(HOLDOUT)))}")
PROFILE = {r["instance_id"]: r for r in csv.DictReader(open(B1_PROFILE))}
VALID = {i for i, r in PROFILE.items() if r["sub_valid"] == "True"}
# "at least one of each example": the official holdout (14 fastapi + 10 rich, all valid) covers 6 of the 9 task categories
# (cell = graph distance x visibility). One valid task of each missing category moves from the training pool into the holdout
# (never trained on: removed from the teacher runs here and from the training data in step 5). requests and httpx are never
# scorable in this sandbox (their tests import the installed copy, B1): one task of each runs as a budget / behaviour check only.
_cells_in = {PROFILE[t]["cell"] for t in OFFICIAL_HOLDOUT}
HOLDOUT_EXTRA = sorted({PROFILE[t]["cell"]: t for t in sorted(TASK_LIST, reverse=True) if PROFILE[t]["cell"] not in _cells_in}.values())
COVERAGE_ONLY = [min(t for t in ALL_IDS if t.startswith(r_ + "_")) for r_ in SHADOWED_REPOS]
HO = [t for t in OFFICIAL_HOLDOUT if t in VALID] + HOLDOUT_EXTRA      # scored
HO_RUN = HO + COVERAGE_ONLY                                            # run
ORDER = [t for t in ORDER if t not in HOLDOUT_EXTRA]
plog(f"holdout: {len(HO)} scored ({len(HO) - len(HOLDOUT_EXTRA)} official + {HOLDOUT_EXTRA} for categories "
     f"{sorted({PROFILE[t]['cell'] for t in HOLDOUT_EXTRA})}) + coverage-only {COVERAGE_ONLY}; teacher pool now {len(ORDER)} tasks")

def part_records(seed):
    recs = {}
    for pdir in sorted(PARTS.glob(f"*_s{seed}")):
        for r in read_jsonl(pdir / "task_results.jsonl"):
            iid = r.get("instance_id")
            infra = bool(INFRA_RE.search(json.dumps({k: v for k, v in r.items() if isinstance(v, str)})))
            prev = recs.get(iid)
            if prev is None or (prev[2] and not infra):
                recs[iid] = (pdir, r, infra)
    return recs

def remaining(seed):
    done = {i for i, (_, _, infra) in part_records(seed).items() if not (infra and RETRY_INFRA_ERRORS)}
    return [t for t in ORDER if t not in done]

TODO = {s: remaining(s) for s in SEEDS}
plog("teacher runs: " + ", ".join(f"s{s} {len(ORDER) - len(v)}/{len(ORDER)} done" for s, v in TODO.items()))
'''

VLLM = r'''# STEP 3 - vLLM helpers: one server per port; a server can be pinned to some GPUs (two students side by side)
import traceback, threading
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
SERVERS = {}          # port -> VllmServer
def kill_gpu_procs():
    subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; "
                   "pkill -9 -f 'VLLM::|vllm.entrypoints|EngineCore'", shell=True)
    time.sleep(10)
def stop_servers():
    for port, srv in list(SERVERS.items()):
        try:
            srv.stop()
        except Exception as e:
            plog(f"vLLM :{port} stop failed: {e}")
        SERVERS.pop(port, None)
    kill_gpu_procs()
def start_server(model_dir, tp, adapter_sub=None, port=8000, gpus=None):
    """Serve as the scoring harness does (gemma4 parsers, thinking on); adapters discovered from a submission folder.
    gpus: CUDA device ids for this server only (the variable is set while vLLM's process is spawned)."""
    adapters = discover_adapters(str(adapter_sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS) if adapter_sub else None
    cfg = VllmConfig(model=str(model_dir), port=port, tool_call_parser="gemma4", reasoning_parser="gemma4",
                     default_chat_template_kwargs={"enable_thinking": True}, max_model_len=SERVE["max_model_len"],
                     dtype="bfloat16", gpu_memory_utilization=SERVE["util"], enable_auto_tool_choice=True,
                     enable_lora=bool(adapters), max_loras=8, max_lora_rank=128, tensor_parallel_size=tp,
                     startup_timeout=SERVER_STARTUP_TIMEOUT)
    srv = VllmServer(cfg, adapter_manifest=adapters)
    plog(f"starting vLLM :{port} on GPUs {gpus or 'all'}: {shlex.join(srv.build_cmd())[:400]}")
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    t0 = time.time()
    try:
        srv.start()
    finally:
        if gpus is not None:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old
    SERVERS[port] = srv
    plog(f"vLLM :{port} up after {time.time() - t0:.0f} s")
    return srv
def server_alive(port=8000):
    import urllib.request
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=20).read()
        return True
    except Exception:
        return False
def agent_cmd(sub, ids, results, model_dir, alias, concurrency=CONCURRENCY, timeout_s=None, port=8000):
    return (f"cd {GC} && " + (f"timeout -k 60 {int(timeout_s)} " if timeout_s else "") + f"{sys.executable} run_arm.py "
            f"--mode agent --tasks {TASKS_JSONL} --snapshots {SNAPSHOTS} "
            f"--submission {sub} --results {results} --sandbox subprocess --task-ids {' '.join(ids)} "
            f"--max-tool-calls {BUDGET['max_tool_calls']} --max-time-minutes {BUDGET['max_time_minutes']} "
            f"--max-turns {BUDGET['max_turns']} --timeout-seconds {BUDGET['timeout_seconds']} --concurrency {concurrency} "
            f"--compaction-interval {COMPACTION_INTERVAL} --model-path {model_dir} --model-alias {alias} --port {port}")
'''

DATAGEN = """# STEP 4 - teacher runs: deployed agent on 31B, seed by seed, chunks of CHUNK tasks (stops at DATAGEN_END_H)
SESSION_TAG = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
chunk_secs, stop_reason = [], "all teacher runs done"
if any(TODO.values()):
    try:
        start_server(MODEL_DIR, TP)
        for seed in SEEDS:
            sub = write_submission(OUT / f"submissions/teacher_s{seed}", TOOLS, seed)
            todo = list(TODO[seed])
            while todo:
                est_h = (1.25 * max(chunk_secs[-3:]) if chunk_secs else 0.5 * 3600) / 3600
                if (time.time() - _T0) / 3600 + est_h > DATAGEN_END_H:
                    stop_reason = f"DATAGEN_END_H ({DATAGEN_END_H} h) reached; the rest resumes next session"
                    break
                if not server_alive():
                    plog("!!! vLLM not answering; restarting")
                    stop_servers(); start_server(MODEL_DIR, TP)
                chunk, todo = todo[:CHUNK], todo[CHUNK:]
                part = PARTS / f"p{len(list(PARTS.iterdir())):03d}_{SESSION_TAG}_{RUN_NAME}_s{seed}"
                t0 = time.time()
                with Progress(part / "task_results.jsonl", len(chunk), f"teacher s{seed}", every=120):
                    sh(agent_cmd(sub, chunk, part, MODEL_DIR, MODEL_SLUG), log=part.parent / f"{part.name}.log")
                chunk_secs.append(time.time() - t0)
                recs = part_records(seed)
                plog(f"    s{seed}: {len(recs)}/{len(ORDER)} tasks, resolved {sum(bool(r.get('resolved')) for _, r, _ in recs.values())}; "
                     f"chunk {chunk_secs[-1] / 60:.1f} min; session {(time.time() - _T0) / 3600:.2f} h")
            if stop_reason != "all teacher runs done":
                break
    except Exception as e:
        stop_reason = f"error: {type(e).__name__}: {e}"
        traceback.print_exc()
TODO = {s: remaining(s) for s in SEEDS}
save_state(teacher_complete=not any(TODO.values()))
plog(f"teacher runs: {stop_reason}; left: {{s: len(v) for s, v in TODO.items()}} -> " + str({s: len(v) for s, v in TODO.items()}))
"""

MERGE_BUILD = """# STEP 5 - merge per seed, then build the training data (notebook 01's seed 1 + every teacher seed)
MERGED_DIRS = []
for seed in SEEDS:
    recs = part_records(seed)
    if not recs:
        continue
    M = RUN_ROOT / f"merged_s{seed}"
    shutil.rmtree(M, ignore_errors=True)
    for d in ("traces", "logs", "patches", "test_outputs"):
        (M / d).mkdir(parents=True)
    with open(M / "task_results.jsonl", "w") as f:
        for iid in ORDER:
            if iid not in recs:
                continue
            pdir, r, infra = recs[iid]
            f.write(json.dumps({**r, "_part": pdir.name, "_infra_error": infra}) + "\\n")
            for d in ("traces", "logs", "patches", "test_outputs"):
                for p in ([q for q in (pdir / d).iterdir() if re.search(rf"(?<![0-9A-Za-z]){re.escape(iid)}(?![0-9])", q.name)] if (pdir / d).exists() else []):
                    (shutil.copytree if p.is_dir() else shutil.copy)(p, M / d / p.name)
    rr = [r for r in read_jsonl(M / "task_results.jsonl") if r["instance_id"] in VALID]
    k_ = sum(bool(r.get("resolved")) for r in rr)
    lo, hi = sc.wilson(k_, len(rr))
    plog(f"teacher s{seed}: resolved {k_}/{len(rr)} valid = {k_ / max(len(rr), 1):.1%} [{lo:.1%}, {hi:.1%}]")
    MERGED_DIRS.append(M)
B2S1 = next((dp for dp, dns, fns in walk(INPUT, maxdepth=9, skip=("snapshots", "wheels", "embeddings", "graphs", "__pycache__", ".git",
             "traces", "logs", "test_outputs", "patches", "parts")) if dp.name == "merged_B2_s1" and "task_results.jsonl" in fns), None)
plog(f"notebook 01 seed-1 run: {B2S1 or 'NOT FOUND (attach notebook 01 output)'}")
(GC / "official_slices").mkdir(exist_ok=True)   # build_rft_data.py reads ../official_slices next to lora/
(GC / "official_slices/dev_slice.txt").write_text("\\n".join(OFFICIAL_DEV) + "\\n")
(GC / "official_slices/holdout_slice.txt").write_text("\\n".join(OFFICIAL_HOLDOUT) + "\\n")
RFT = OUT / "rft"
DEPLOY_PROMPT = GC / "deploy_system.md"
DEPLOY_PROMPT.write_text((Path(SAMPLE_SUB) / "prompts/system.md").read_text() + PROMPT_ADDENDUM)
runs = ([B2S1] if B2S1 else []) + MERGED_DIRS
if runs:
    sh(f"cd {GC} && {sys.executable} lora/build_rft_data.py --runs {' '.join(map(str, runs))} --profile {B1_PROFILE} "
       f"--out {RFT} --max-per-task 3", log=OUT / "provenance/build_rft_data.log")
TRAIN_DATA = RFT / "train.jsonl"
if TRAIN_DATA.exists():   # the categories moved into the holdout (step 2) must not be trained on (notebook 01 ran them at seed 1)
    _kept = [l for l in TRAIN_DATA.read_text().splitlines() if l.strip() and json.loads(l)["instance_id"] not in set(HOLDOUT_EXTRA)]
    TRAIN_DATA.write_text("\\n".join(_kept) + "\\n")
n_traj = len(read_jsonl(TRAIN_DATA))
plog(f"training data: {n_traj} trajectories -> {TRAIN_DATA}")
"""

TRAIN = r'''# STEP 6 - train the LoRAs: 31B split over all 4 GPUs, then the two students side by side (E4B on GPUs 0-1, 12B on 2-3)
stop_servers()
_models = [dp for dp, dns, fns in walk(INPUT, maxdepth=9) if "config.json" in fns]
def _find(*keys):
    # "-assistant" checkpoints are speculative-decoding drafters, never the model itself
    return next((dp for dp in _models if all(k in str(dp).lower() for k in keys) and "assistant" not in str(dp).lower()), None)
BASE31_TRAIN = _find("31b", "unquantized")
for _s in STUDENTS.values():
    _s["train_dir"], _s["serve_dir"] = _find(_s["key"], "unquantized"), _find(_s["slug"])
plog(f"models: 31B train {BASE31_TRAIN} | " + " | ".join(f"{k}: train {v['train_dir']}, serve {v['serve_dir']}" for k, v in STUDENTS.items()))
# training libraries: the attached graphcomp-train-wheels (peft 0.21.2, accelerate 1.15.0) into a private folder, so the
# serving / sandbox environment is unchanged; without it the image's own peft / accelerate are used
TRAIN_LIB = Path("/tmp/trainlib")
_wh = sorted(glob.glob(str(INPUT / "**/peft-*.whl"), recursive=True)) + sorted(glob.glob(str(INPUT / "**/accelerate-*.whl"), recursive=True))
if _wh and not TRAIN_LIB.exists():
    sh(f"{sys.executable} -m pip install --no-deps --no-index --target {TRAIN_LIB} {' '.join(_wh)}", log=OUT / "provenance/pip_train.log")
TRAIN_ENV = {"PYTHONPATH": str(TRAIN_LIB)} if TRAIN_LIB.exists() else {}
plog(f"training libraries: {'graphcomp-train-wheels ' + str([Path(w).name for w in _wh]) if TRAIN_ENV else 'image (peft 0.19.1, accelerate 1.13.0)'}")
DEPLOY_ARGS = (f"--system-prompt {DEPLOY_PROMPT} --deploy-budget " + shlex.quote(json.dumps({
    "time_minutes": DEPLOY_BUDGET["max_time_minutes"], "tool_calls": DEPLOY_BUDGET["max_tool_calls"],
    "turns": DEPLOY_BUDGET["max_turns"], "timeout_seconds": DEPLOY_BUDGET["timeout_seconds"]})))
RESERVE_H = 2 * EVAL_ARM_H + 0.4          # the 31B holdout arms, then both students at once, and the zip
def _budget(max_h, name):
    b = min(max_h, left_h() - RESERVE_H)
    if not TRAIN_DATA.exists() or n_traj == 0:
        plog(f"{name}: no training data"); return None
    if b < 0.25:
        plog(f"{name}: skipped, {left_h():.1f} h left (reserve {RESERVE_H:.1f} h): attach this output and run again"); return None
    return b
def _report(name, out_dir, rc, t0):
    ok = (out_dir / "adapter" / "adapter_config.json").exists()
    plog(f"{name}: exit {rc} after {(time.time() - t0) / 3600:.2f} h; adapter {'saved' if ok else 'MISSING'}")
    for l_ in read_jsonl(out_dir / "train_log.jsonl")[-2:]:
        print(l_)
    return ok
# --- 31B
OUT31 = OUT / "lora31"
ADAPTER31 = OUT31 / "adapter"
OK31 = (ADAPTER31 / "adapter_config.json").exists()
if not OK31 and BASE31_TRAIN is not None and (b := _budget(TRAIN31_MAX_H, "lora31")):
    shutil.rmtree(OUT31, ignore_errors=True); t0 = time.time()
    rc = sh(f"cd {GC}/lora && timeout -k 60 {int((b + 0.4) * 3600)} {sys.executable} train_lora.py --data {TRAIN_DATA} "
            f"--base {BASE31_TRAIN} --out {OUT31} --device-map auto --max-target 8000 --no-eval-start "
            f"--served-model google/{MODEL_SLUG} {DEPLOY_ARGS} --time-budget-h {b:.2f}", log=OUT / "provenance/train_lora31.log", env=TRAIN_ENV)
    kill_gpu_procs()
    OK31 = _report("lora31", OUT31, rc, t0)
# --- students in parallel: each its own GPUs, its own torchrun rendezvous port, its own log
procs = {}
for name, s in STUDENTS.items():
    s["out"] = OUT / name.lower()
    s["ok"] = (s["out"] / "adapter" / "adapter_config.json").exists()
    if s["ok"] or s["train_dir"] is None:
        plog(f"{name}: {'adapter already in this output; reused' if s['ok'] else 'training checkpoint not attached; skipped'}")
        continue
    b = _budget(TRAIN_STUDENT_MAX_H, name)
    if not b:
        continue
    shutil.rmtree(s["out"], ignore_errors=True)
    cmd = (f"cd {GC}/lora && CUDA_VISIBLE_DEVICES={','.join(map(str, s['train_gpus']))} timeout -k 60 {int((b + 0.3) * 3600)} "
           f"{sys.executable} -m torch.distributed.run --nproc_per_node {len(s['train_gpus'])} --master_port {s['master_port']} "
           f"train_lora.py --data {TRAIN_DATA} --base {s['train_dir']} --out {s['out']} --max-target {s['max_target']} --max-len {s['max_len']} "
           f"--served-model google/{s['slug']} {DEPLOY_ARGS} --time-budget-h {b:.2f}")
    log_ = open(OUT / f"provenance/train_{name}.log", "w")
    procs[name] = (subprocess.Popen(cmd, shell=True, stdout=log_, stderr=subprocess.STDOUT, env={**os.environ, **TRAIN_ENV}), time.time())
    plog(f"{name}: training started on GPUs {s['train_gpus']}: {cmd[:300]}")
while any(p.poll() is None for p, _ in procs.values()):
    time.sleep(120)
    plog("    students: " + " | ".join(f"{n}: " + (json.dumps(read_jsonl(STUDENTS[n]['out'] / 'train_log.jsonl')[-1:])[:160] if p.poll() is None else f"exit {p.returncode}")
                                     for n, (p, _) in procs.items()))
kill_gpu_procs()
for name, (p, t0) in procs.items():
    STUDENTS[name]["ok"] = _report(name, STUDENTS[name]["out"], p.returncode, t0)
    if not STUDENTS[name]["ok"]:
        print(open(OUT / f"provenance/train_{name}.log").read()[-3000:])
save_state(lora31=OK31, **{n.lower(): s["ok"] for n, s in STUDENTS.items()})
'''

EVAL = r'''# STEP 7 - holdout comparison: BASE31 + LORA31 on one 31B server, then both students at once (one GPU and port each)
def make_sub(dest, model_slug, adapter_dir=None, budget=None, seed=1):
    """The deployed agent; budget=None -> the in-notebook BUDGET (concurrency 4), else the deployment budget, no seed."""
    d = write_submission(dest, TOOLS, seed)
    y = (d / "agent.yaml").read_text().replace(f"model: {MODEL_SLUG}\n", f"model: {model_slug}\n")
    if adapter_dir is not None:
        shutil.copytree(adapter_dir, d / "adapters" / "agent_lora", dirs_exist_ok=True)
        y = y.replace(f"model: {model_slug}\n", f"model: {model_slug}\nadapter: agent_lora\n")
    (d / "agent.yaml").write_text(y)
    if budget is not None:
        (d / "configs/sampling.yaml").write_text(re.sub(r"seed: .*\n", "", (d / "configs/sampling.yaml").read_text()))
        (d / "eval_config.yaml").write_text("evaluation:\n" + "".join(f"  {k}: {v}\n" for k, v in budget.items()))
    return d
ARMS = {"BASE31": dict(model=MODEL_DIR, slug=MODEL_SLUG, adapter=None, tp=TP, port=8000, gpus=None),
        "LORA31": dict(model=MODEL_DIR, slug=MODEL_SLUG, adapter=ADAPTER31 if OK31 else None, tp=TP, port=8000, gpus=None)}
for name, s in STUDENTS.items():
    ARMS[name] = dict(model=s["serve_dir"], slug=s["slug"], adapter=(s["out"] / "adapter") if s["ok"] else None, tp=1,
                      port=s["port"], gpus=[s["eval_gpu"]])
RESULTS = dict(STATE.get("holdout", {}))
# the two students run their holdout at the same time, each on its own GPU with the same concurrency as the 31B arms (4): the
# machine (g2-standard-24, 24 vCPUs) carries 8 sandboxes, and each model sees exactly the load the 31B arms saw
STUDENT_CONCURRENCY = CONCURRENCY
def adapter_changes_output(port, name="agent_lora"):
    """One greedy request to the base model and to the adapter: identical text and logprobs -> the adapter is a no-op."""
    import urllib.request
    def post(path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{port}" + path, data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=300).read())
    ids = [m["id"] for m in post("/v1/models")["data"]]
    base = next(i for i in ids if i != name)
    outs = []
    for mdl in (base, name):
        r_ = post("/v1/chat/completions", {"model": mdl, "temperature": 0, "max_tokens": 48, "logprobs": True, "top_logprobs": 1,
             "chat_template_kwargs": {"enable_thinking": False},
             "messages": [{"role": "user", "content": "Fix this bug and reply with the function only: def add(a, b): return a - b"}]})
        ch = r_["choices"][0]
        outs.append((ch["message"].get("content"), [round(t["logprob"], 4) for t in (ch.get("logprobs") or {}).get("content") or []]))
    plog(f"adapter probe :{port}: base {outs[0][0]!r:.80} | adapter {outs[1][0]!r:.80} | identical: {outs[0] == outs[1]}")
    return outs[0] != outs[1]
def arm_done(arm):
    return {r["instance_id"] for r in read_jsonl(RUN_ROOT / f"holdout_{arm}/task_results.jsonl")} >= set(HO_RUN)
def score_arm(arm, server_log=None):
    A = ARMS[arm]
    rr = {r["instance_id"]: r for r in read_jsonl(RUN_ROOT / f"holdout_{arm}/task_results.jsonl")}
    if server_log and Path(server_log).exists():
        shutil.copy(server_log, RUN_ROOT / f"vllm_{arm}.log")
    lp = RUN_ROOT / f"vllm_{arm}.log"
    ignored = A["adapter"] is not None and lp.exists() and "will be ignored" in lp.read_text(errors="replace")
    k_ = sum(bool(rr[t].get("resolved")) for t in HO if t in rr)
    k_off = sum(bool(rr[t].get("resolved")) for t in HO if t in rr and t in OFFICIAL_HOLDOUT)
    RESULTS[arm] = {"resolved": k_, "finished": sum(t in rr for t in HO), "n": len(HO), "adapter_ignored": ignored,
                    "official": [k_off, len([t for t in HO if t in OFFICIAL_HOLDOUT])],
                    "per_task": {t: bool(rr[t].get("resolved")) for t in HO if t in rr},
                    "coverage": {t: {k: rr[t].get(k) for k in ("duration_seconds", "tool_calls", "agent_patch_size", "error")}
                                 for t in COVERAGE_ONLY if t in rr}}
    lo, hi = sc.wilson(k_, len(HO))
    plog(f"HOLDOUT {arm}: {k_}/{len(HO)} = {k_ / max(len(HO), 1):.1%} [{lo:.1%}, {hi:.1%}] (official 24: {k_off}); "
         f"{RESULTS[arm]['finished']} finished; coverage {RESULTS[arm]['coverage']}{' ADAPTER IGNORED BY vLLM' if ignored else ''}")
    save_state(holdout=RESULTS)
def run_arm_holdout(arm):
    A = ARMS[arm]
    res_dir = RUN_ROOT / f"holdout_{arm}"
    todo = [t for t in HO_RUN if t not in {r["instance_id"] for r in read_jsonl(res_dir / "task_results.jsonl")}]
    if todo and A["adapter"] is not None and not adapter_changes_output(A["port"]):
        RESULTS[arm] = {"resolved": 0, "finished": 0, "n": len(HO), "adapter_ignored": True, "per_task": {}, "official": [0, 0], "coverage": {}}
        plog(f"!!! {arm}: the adapter does not change the output (vLLM did not apply it); arm skipped")
        save_state(holdout=RESULTS); return
    if todo:
        sub = make_sub(OUT / f"submissions/eval_{arm}", A["slug"], A["adapter"])
        with Progress(res_dir / "task_results.jsonl", len(todo), f"holdout {arm}", every=120):
            sh(agent_cmd(sub, todo, res_dir, A["model"], A["slug"], timeout_s=EVAL_ARM_H * 3600, port=A["port"],
                         concurrency=STUDENT_CONCURRENCY if arm in STUDENTS else CONCURRENCY),
               log=RUN_ROOT / f"holdout_{arm}.log")
    srv = SERVERS.get(A["port"])
    score_arm(arm, srv.log_path if srv is not None else None)
try:
    arms31 = ["BASE31"] + (["LORA31"] if OK31 else [])
    if not all(arm_done(a) for a in arms31):
        if left_h() > EVAL_ARM_H + 0.3:
            start_server(MODEL_DIR, TP, make_sub(OUT / "submissions/_serve31", MODEL_SLUG, ADAPTER31) if OK31 else None)
            for a in arms31:
                run_arm_holdout(a)
        else:
            plog(f"31B holdout arms skipped: {left_h():.1f} h left; attach this output and run again")
    else:
        for a in arms31:
            score_arm(a)
    stop_servers()
    # students: both servers up (each on its own GPU and port), then both arms at the same time
    st = [n for n, s in STUDENTS.items() if s["ok"] and s["serve_dir"] is not None]
    if any(not arm_done(n) for n in st) and left_h() > EVAL_ARM_H + 0.3:
        up = []
        for n in st:
            if arm_done(n):
                continue
            A = ARMS[n]
            try:
                start_server(A["model"], 1, make_sub(OUT / f"submissions/_serve_{n}", A["slug"], A["adapter"]), port=A["port"], gpus=A["gpus"])
                up.append(n)
            except Exception as e:     # 12B failed to serve at tp 4 in notebook 11 (vLLM Transformers fallback); recorded, not fatal
                plog(f"!!! {n}: vLLM did not start ({type(e).__name__}: {str(e)[:300]}); arm skipped")
                RESULTS[n] = {"resolved": 0, "finished": 0, "n": len(HO), "adapter_ignored": False, "serve_failed": True,
                              "per_task": {}, "official": [0, 0], "coverage": {}}
                save_state(holdout=RESULTS)
        threads = [threading.Thread(target=run_arm_holdout, args=(n,)) for n in up]
        for t_ in threads:
            t_.start()
        for t_ in threads:
            t_.join()
    else:
        for n in st:
            if arm_done(n):
                score_arm(n)
except Exception as e:
    traceback.print_exc()
    plog(f"!!! holdout stage failed: {type(e).__name__}: {e}")
finally:
    stop_servers()
'''

PICK = r'''# STEP 8 - choose and write the submission (deployment budget), and the summary
PRIORITY = ["BASE31", "LORA31"] + list(STUDENTS)        # tie-break: no adapter without evidence, then the bigger model
cands = {a: r for a, r in RESULTS.items() if r["finished"] == r["n"] and not r["adapter_ignored"] and not r.get("serve_failed")
         and a in ARMS and (a == "BASE31" or ARMS[a]["adapter"] is not None)}
best = max(cands, key=lambda a: (cands[a]["resolved"], -PRIORITY.index(a))) if cands else "BASE31"
expected = ["BASE31"] + (["LORA31"] if OK31 else []) + [n for n, s in STUDENTS.items() if s["ok"]]
COMPLETE = all(a in RESULTS and (a in cands or RESULTS[a].get("serve_failed") or RESULTS[a]["adapter_ignored"]) for a in expected)
final = {}
for a in PRIORITY:
    A = ARMS[a]
    if a == "BASE31" or A["adapter"] is not None:
        final[a] = make_sub(OUT / f"submissions/final_{a}", A["slug"], A["adapter"], budget=DEPLOY_BUDGET)
FINAL = Path("/kaggle/working/submission")
shutil.rmtree(FINAL, ignore_errors=True)
shutil.copytree(final[best], FINAL)
_zip = Path(shutil.make_archive("/kaggle/working/submission", "zip", FINAL))   # agent.yaml at the zip root
_size = sum(f.stat().st_size for f in FINAL.rglob("*") if f.is_file())
assert _size < 3 * 2**30, f"submission is {_size / 2**30:.2f} GiB (limit 3 GiB)"
worst_h = (N_TASKS_ASSUMED * (DEPLOY_BUDGET["max_time_minutes"] + SETUP_MIN) + SERVE_START_MIN) / 60
lines = [f"# Notebook 19 summary ({now()})", "",
         f"- Teacher runs complete: {STATE.get('teacher_complete')}; training trajectories: {n_traj}",
         f"- LoRAs trained: 31B {OK31}, " + ", ".join(f"{n} {s['ok']}" for n, s in STUDENTS.items()),
         f"- Holdout: {len(HO)} scored tasks (official 24 + {HOLDOUT_EXTRA}); coverage-only {COVERAGE_ONLY} (not scorable here)",
         "- Results (Wilson 95%):"]
for a, r in RESULTS.items():
    lo, hi = sc.wilson(r["resolved"], r["n"])
    lines.append(f"  - {a}: {r['resolved']}/{r['n']} [{lo:.1%}, {hi:.1%}] (official 24: {r['official'][0]}); finished {r['finished']}; "
                 f"adapter ignored {r['adapter_ignored']}{'; SERVE FAILED' if r.get('serve_failed') else ''}; coverage {r.get('coverage')}")
if "BASE31" in RESULTS:
    for a in RESULTS:
        if a != "BASE31":
            b, x = RESULTS["BASE31"]["per_task"], RESULTS[a]["per_task"]
            only_a = sorted(t for t in x if x[t] and not b.get(t)); only_b = sorted(t for t in b if b[t] and not x.get(t))
            lines.append(f"  - {a} vs BASE31 on the same tasks: {a} only {len(only_a)} {only_a}; BASE31 only {len(only_b)} {only_b}")
lines += ["", f"**SUBMISSION: {best}**" + ("" if COMPLETE else " (comparison INCOMPLETE: attach this output and run again)"),
          f"- deployment budget {DEPLOY_BUDGET}; worst case if all {N_TASKS_ASSUMED} tasks hit the cap: {worst_h:.1f} h of 12",
          f"- {FINAL} ({_size / 1e6:.1f} MB) and {_zip}",
          "- Noise bar: two runs of one condition disagree on about 3 of 23 tasks; a smaller gap is not evidence."]
(OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
save_state(submission=best, complete=COMPLETE)
print("\n".join(lines))
print((FINAL / "agent.yaml").read_text()); print((FINAL / "eval_config.yaml").read_text())
'''

ZIP = """# STEP 9 - zip the run folder (Output tab)
stop_servers()
zp = Path("/kaggle/working/graphcomp_nb19.zip")
if zp.exists():
    zp.unlink()
shutil.make_archive(str(zp.with_suffix("")), "zip", OUT)
plog(f"zip: {zp} {zp.stat().st_size / 1e6:.1f} MB")
"""

scripts = []
for rel in ("lora/build_rft_data.py", "lora/train_lora.py"):
    scripts.append(code(f"%%writefile /kaggle/working/gc/{rel}\n" + (R / "scripts/planB" / rel).read_text()))
STARTED = """# first cell: proves the kernel is running, and prints "CELL END" after every cell from here on (also after %%writefile cells)
import datetime as _dt, sys as _sys
print(f"NOTEBOOK STARTED {_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC | Python {_sys.version.split()[0]}", flush=True)
_cell_n = [0]
def _cell_end_marker(result=None):
    _cell_n[0] += 1
    failed = bool(result is not None and (result.error_in_exec or result.error_before_exec))
    print(f"CELL END {_cell_n[0]}{' (FAILED)' if failed else ''} | {_dt.datetime.now(_dt.timezone.utc):%H:%M:%S} UTC", flush=True)
_ip = get_ipython()
for _f in [f for f in _ip.events.callbacks["post_run_cell"] if getattr(f, "__name__", "") == "_cell_end_marker"]:
    _ip.events.unregister("post_run_cell", _f)
_ip.events.register("post_run_cell", _cell_end_marker)
"""
cells = ([md(HEADER), md(TRAINING_CHOICES), code(STARTED)] + setup[:3] + [code(RESUME)] + setup[3:] + [SUBMISSION, VALIDITY, ORDER_CELL]
         + [code("# STEP 1 - script folders\n(GC / 'lora').mkdir(exist_ok=True)\n")] + scripts
         + [code(TODO), code(VLLM), code(DATAGEN), code(MERGE_BUILD), code(TRAIN), code(EVAL), code(PICK), code(ZIP)])
for c_ in cells:
    if c_["cell_type"] == "code":
        c_["outputs"], c_["execution_count"] = [], None
nb["cells"] = cells
out = U / "notebooks/19_teacher_lora31_student_holdout_submission.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print(out, len(cells), "cells", out.stat().st_size // 1000, "KB")

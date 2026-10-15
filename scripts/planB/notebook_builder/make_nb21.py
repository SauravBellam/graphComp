"""Build notebook 21: 31B LoRA (the student) on the SFT mix (build_sft_mix.py) -> adapter check -> submission.

Derived from the built notebook 19 (run make_nb19.py first), with notebook 20's script versions (its 31B training fixes): same setup, resume, validity, holdout set, vLLM helpers, deployed
agent, holdout comparison, submission choice and zip. Removed: teacher generation (step 4), merging teacher runs (step 5) and the
students. Step 5 now reads the attached SFT mix; step 6 trains only the 31B, with the training window stepped down on OOM.
    python3 scripts/planB/notebook_builder/make_nb21.py
"""
import json, sys
SMOKE = "--smoke" in sys.argv     # 21s: shortest end-to-end check: train the 31B LoRA on ONE trajectory, write the submission
from pathlib import Path
R = Path("/Users/sauravbellam/Desktop/Projects/graphComp")
U = R / "planB_results/kaggle_upload"
HERE = Path(__file__).resolve().parent
nb = json.load(open(U / "notebooks/19_teacher_lora31_student_holdout_submission.ipynb"))
C = nb["cells"]
src = lambda i: "".join(C[i]["source"])
first = lambda i: src(i).split("\n", 1)[0]


def sub1(s, old, new):
    assert s.count(old) == 1, old[:80]
    return s.replace(old, new)


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


assert first(26).startswith("# STEP 4 - teacher runs") and first(27).startswith("# STEP 5 - merge") and first(28).startswith("# STEP 6")

HEADER = """# 21 — 31B LoRA (the student) on the SFT mix → adapter check → submission

Offline (Internet off), GPU L4 × 4. Built by `scripts/planB/notebook_builder/make_nb21.py` from notebook 19 (setup, resume,
deployed agent, submission writer) with notebook 20's scripts (its 31B training fixes).

**Resumable, one 11 h 30 min session at a time** (attach the previous version's output to continue; each session ends with a
submittable `submission.zip`):
1. **This session's slice** of the attached `graphcomp-sft-mix` (`train_fit.jsonl`: test-verified trajectories ≤ 32k tokens,
   duplicates removed): trajectories not trained in an earlier session, **new problems first** (the shortest trajectory of
   each never-trained problem), **shortest first**, sized to the tokens the measured training speed fits before the deadline
   (× 1.5; the trainer stops on time anyway). **Competition trajectories go last, every session** (our own on the 129 tasks +
   external ones on the four competition repos): the model sees them last.
2. **31B LoRA** (`train_lora.py`, QLoRA, 4-bit model split over the 4 GPUs): **continues the previous session's adapter**;
   1 pass in that order (shuffled only within blocks of 32 windows); LR decays to 30% of the peak, not 0, so the competition
   tail still moves the model; stops at a step boundary before the training deadline (session start + 11.5 h − 0.85 h), and
   checkpoints the adapter every 30 min (atomically), so even a killed run leaves an adapter. The training window starts at
   32,768 tokens and steps down on out-of-memory; a resume that fails to load falls back to a fresh LoRA once.
3. **Adapter check**: vLLM serves the adapter as the scorer does; a no-op adapter is reported (the output stays submittable).
4. **Submission** = the LoRA with the deployment budget (≈ 5 min / 30 tool calls per task: 120 tasks fit in 12 h), plus the
   state for the next session (trained trajectory ids, problems, measured tokens/s).

**Inputs:** as notebook 19 (competition data, wheelhouse, `gemma-4-31b-it-qat-w4a16-ct` + `gemma-4-31b-it-qat-q4_0-unquantized`,
notebook 01's output, `graphcomp-planb`, `graphcomp-train-wheels`) **plus `graphcomp-sft-mix`**; for a resumed run, this
notebook's previous version output.
"""

DATA_NOTES = """
## Data choices (Oct 15)
- **Verified only, teachers stronger than the student**: every external trajectory passed its task's tests and comes from
  Qwen3-Coder-480B (the user's rule: only models stronger than Gemma 4 31B).
- **Filters from the same teacher's distillation papers** (applied in `convert_trajs.py`): no git history beyond the base
  commit (SWE-Lego arXiv 2601.01426, SWE-HERO arXiv 2604.01496), no test-file edits, < 3 editor errors, a real source change.
- **Train = deploy**: trajectories longer than the served 32k context could not happen at deployment, so they are not used;
  installs (`pip install`, downloads) are dropped because deployment is offline.
- **Diversity first**: one trajectory per task before any task gets a second (different routes for one task are kept,
  duplicates are removed upstream).
"""

# ------------------------------------------------------------------ config (cell 3)
c = src(3).replace("NB19", "NB21")
c = sub1(c, "SEEDS = [11, 12, 13]          # teacher replicates; with notebook 01's seed 1 that is 4 attempts per task\nSEED = SEEDS[0]",
         "SEEDS = []                    # no teacher runs in this notebook\nSEED = 1")
c = sub1(c, "DATAGEN_END_H = 4.6           # no new teacher chunk after this\n", "")
# every session lasts at most SESSION_BUDGET_H (user: 11 h 30 min) and always ends with a submittable output
c = sub1(c, "SESSION_BUDGET_H = 11.3", "SESSION_BUDGET_H = 1.6 " if SMOKE else "SESSION_BUDGET_H = 11.5")
c = sub1(c, "TRAIN31_MAX_H, TRAIN_STUDENT_MAX_H, EVAL_ARM_H = 2.6, 0.8, 0.9",
         f"TRAIN31_MAX_H, TRAIN_STUDENT_MAX_H, EVAL_ARM_H = {0.6 if SMOKE else 10.0}, 0, {0.1 if SMOKE else 0.5}   # EVAL_ARM_H = the adapter check (31B load + 1 request)\n"
         "TRAIN_MAX_LENS = [32768, 24576, 18432]   # training window (tokens); the next one only after an out-of-memory failure\n"
         f"MAX_TRAIN_TRAJ = {1 if SMOKE else 100000}         # cap per session (the token budget below is the real limit)\n"
         f"POST_TRAIN_H = {0.3 if SMOKE else 0.85}             # after training: adapter check (31B load + 1 request), submission, zip, margin\n"
         "LOAD_H = 0.6                  # 31B load inside the trainer before the first step (notebook 20: ~35 min)\n"
         "TOK_PER_S_GUESS = 150         # training tokens / s until a session has measured it (train_meta.json)\n"
         "DATA_OVERSHOOT = 1.5          # select this many times the tokens expected to fit (the trainer stops at the deadline)\n"
         f"SMOKE = {SMOKE}                  # True: the single shortest trajectory, 1 epoch, 1 window per step, no adapter check\n"
         "VAL_FRAC = 0.02               # of tasks, for the validation loss (2% of thousands is enough and keeps eval short)")
i0 = c.index("STUDENTS = {'STUDENT_12B'")
i1 = c.index("EVAL_STUDENTS = False")
c = c[:i0] + "STUDENTS = {}                 # no students in this notebook\n" + c[i1:]
assert "NB21" in c and "STUDENTS = {}" in c and "TRAIN_MAX_LENS" in c

RESUME = src(6).replace("NB19", "NB21")

DATA = r'''# STEP 5 - training data for THIS session: untrained trajectories, own first, then new problems shortest-first, ...
import random, hashlib
MIX = next((Path(dp) for dp, dns, fns in walk(INPUT, maxdepth=7) if "train_fit.jsonl" in fns and "manifest.json" in fns), None)
plog(f"SFT mix: {MIX or 'NOT FOUND: attach the graphcomp-sft-mix dataset'}")
if MIX:
    plog("mix manifest: " + json.dumps({k: v for k, v in json.loads((MIX / 'manifest.json').read_text()).items() if k != 'inputs'})[:1500])
DEPLOY_PROMPT = GC / "deploy_system.md"
DEPLOY_PROMPT.write_text((Path(SAMPLE_SUB) / "prompts/system.md").read_text() + PROMPT_ADDENDUM)
task_of = lambda r: r["instance_id"].split("@")[0]
def traj_key(r):   # stable across sessions (train_lora.py records the same key in trained_ids.json)
    if not r.get("source_traj_id"):
        calls = [json.dumps(tc["function"], sort_keys=True) for m in r["messages"] for tc in (m.get("tool_calls") or [])]
        r["source_traj_id"] = "own:" + task_of(r) + ":" + hashlib.sha1("\\n".join(calls).encode()).hexdigest()[:12]
    return r["source_traj_id"]
TRAINED = set(STATE.get("trained_ids", []))
TRAINED_TASKS = set(STATE.get("trained_tasks", []))
_rows = [json.loads(l) for l in open(MIX / "train_fit.jsonl")] if MIX else []
_rows = [r for r in _rows if r.get("est_tokens", 0) <= max(TRAIN_MAX_LENS)]
for r in _rows:
    traj_key(r)                         # every row gets its stable key (also the competition tail)
# COMPETITION trajectories (our own on the 129 tasks + external ones on the four competition repos) are trained LAST in every
# session (user, Oct 15: "save our competition dataset for the end, we want it to see that last"), every session again
COMP_REPOS = {"fastapi/fastapi", "tiangolo/fastapi", "textualize/rich", "psf/requests", "encode/httpx",
              "fastapi", "rich", "requests", "httpx"}
is_comp = lambda r: r.get("mix_source") == "rft" or str(r.get("repo", "")).lower() in COMP_REPOS
_comp = sorted([r for r in _rows if is_comp(r)], key=lambda r: r.get("est_tokens", 0))
_rows = [r for r in _rows if not is_comp(r) and traj_key(r) not in TRAINED]
# the rest: NEW problems (never trained) first, the shortest trajectory of each, shortest first; then everything else shortest
_rows.sort(key=lambda r: r.get("est_tokens", 0))
_first, _seen = [], set(TRAINED_TASKS)
for r in _rows:
    if task_of(r) not in _seen:
        _first.append(r); _seen.add(task_of(r))
_firstset = {id(r) for r in _first}
_queue = _first + [r for r in _rows if id(r) not in _firstset]
# size this session's slice: tokens the trainer can process before the deadline, times DATA_OVERSHOOT
TRAIN_DEADLINE = _T0 + (SESSION_BUDGET_H - POST_TRAIN_H) * 3600
_tps = STATE.get("tok_per_s") or TOK_PER_S_GUESS
_train_s = max(0.0, TRAIN_DEADLINE - time.time() - LOAD_H * 3600)
_comp_tok = sum(r.get("est_tokens", 0) for r in _comp)
_budget_tok = max(0.0, _tps * _train_s * DATA_OVERSHOOT - _comp_tok)
_pick, _tok = [], 0
for r in _queue:
    if len(_pick) >= MAX_TRAIN_TRAJ or (_pick and _tok + r.get("est_tokens", 0) > _budget_tok):
        break
    _pick.append(r); _tok += r.get("est_tokens", 0)
TAIL_N = len(_comp)
_pick = _pick + _comp                 # the trainer always trains these, at the end (--tail-n)
if SMOKE:
    _pick, TAIL_N = (_comp or _queue)[:1], 0
TRAIN_DATA = OUT / "sft" / f"train_session{STATE.get('sessions', 1)}.jsonl"
TRAIN_DATA.parent.mkdir(parents=True, exist_ok=True)
with open(TRAIN_DATA, "w") as f:
    for r in _pick:
        f.write(json.dumps(r) + "\n")
n_traj = len(_pick)
_new = len({task_of(r) for r in _pick} - TRAINED_TASKS)
plog(f"session {STATE.get('sessions')}: {len(TRAINED)} trajectories / {len(TRAINED_TASKS)} problems trained before; "
     f"this slice {n_traj} trajectories (~{(_tok + _comp_tok) / 1e6:.1f}M tokens; {_new} new problems; the last {TAIL_N} = "
     f"competition trajectories, trained last) for "
     f"{_train_s / 3600:.1f} h at {_tps:.0f} tok/s x {DATA_OVERSHOOT}; {len(_queue)} untrained left -> {TRAIN_DATA}")
del _rows, _queue, _first
'''

# ------------------------------------------------------------------ step 6: only the 31B, training window stepped down on OOM
T = src(28)
T = sub1(T, "# STEP 6 - train the LoRAs: 31B split over all 4 GPUs, then the two students side by side (E4B on GPUs 0-1, 12B on 2-3)",
         "# STEP 6 - train the 31B LoRA (4-bit model split over all 4 GPUs); training window stepped down on out-of-memory")
old31 = T[T.index("if not OK31 and BASE31_TRAIN is not None"):T.index("# --- students in parallel")]
T = T.replace(old31, '''# resume: the previous session's adapter (copied in by the resume step) is the starting point; it stays the submission
# if this session trains nothing. The trainer stops at TRAIN_DEADLINE at a step boundary and always saves (also every 30
# min, atomically), so a session always leaves an adapter.
PREV = OUT / "lora31_prev"
if ADAPTER31.exists() and STATE.get("adapter_session") != STATE.get("sessions"):
    shutil.rmtree(PREV, ignore_errors=True); shutil.move(str(OUT31), str(PREV))
INIT = PREV / "adapter" if (PREV / "adapter" / "adapter_config.json").exists() else None
OK31, L = False, None
for L in ([18432] if SMOKE else TRAIN_MAX_LENS):
    if BASE31_TRAIN is None or not n_traj:
        break
    left_s = TRAIN_DEADLINE - time.time()
    if left_s < (LOAD_H + 0.25) * 3600:
        plog(f"lora31: {left_s / 3600:.2f} h to the training deadline: not enough for the load + training"); break
    for init in ([INIT, None] if INIT else [None]):   # a resume that fails to load falls back to a fresh LoRA once
        shutil.rmtree(OUT31, ignore_errors=True); t0 = time.time()
        _log = OUT / f"provenance/train_lora31_s{STATE.get('sessions')}_len{L}{'_resume' if init else ''}.log"
        b = max(0.25, (TRAIN_DEADLINE - time.time()) / 3600 - LOAD_H)
        rc = sh(f"cd {GC}/lora && timeout -k 60 {int(TRAIN_DEADLINE - time.time() + 900)} {sys.executable} train_lora.py "
                f"--data {TRAIN_DATA} --base {BASE31_TRAIN} --out {OUT31} --device-map auto --max-len {L} --max-target 8000 "
                f"--val-frac {VAL_FRAC} --no-eval-start --served-model google/{MODEL_SLUG} {DEPLOY_ARGS} --time-budget-h {b:.2f} "
                f"--epochs 1 --keep-order --stop-at {TRAIN_DEADLINE:.0f} --save-every-min 30 --seed {STATE.get('sessions', 1)}"
                + (f" --tail-n {TAIL_N} --lr-floor 0.3" if TAIL_N else "")
                + (f" --init-adapter {init}" if init else "") + (" --grad-accum 1" if SMOKE else ""), log=_log, env=TRAIN_ENV)
        kill_gpu_procs()
        OK31 = _report(f"lora31 (window {L}{', resumed' if init else ''})", OUT31, rc, t0)
        _tail = _log.read_text(errors="replace")[-30000:] if _log.exists() else ""
        if OK31 or not init or re.search(r"out of memory|OutOfMemoryError", _tail, re.I):
            break
        plog("!!! lora31: resuming from the previous adapter failed; training a fresh LoRA instead (see " + str(_log) + ")")
    if OK31 or not re.search(r"out of memory|OutOfMemoryError", _tail, re.I):
        break
    plog(f"lora31: out of memory with a {L}-token window; trying the next one")
if OK31:
    _ids = json.loads((ADAPTER31 / "trained_ids.json").read_text()) if (ADAPTER31 / "trained_ids.json").exists() else []
    _meta = json.loads((OUT31 / "train_meta.json").read_text()) if (OUT31 / "train_meta.json").exists() else {}
    _byid = {}
    for l_ in open(TRAIN_DATA):
        r_ = json.loads(l_); _byid[r_["source_traj_id"]] = task_of(r_)
    save_state(trained_ids=sorted(TRAINED | set(_ids)), trained_tasks=sorted(TRAINED_TASKS | {_byid[i] for i in _ids if i in _byid}),
               tok_per_s=_meta.get("tok_per_s") or STATE.get("tok_per_s"), lora31_window=L, adapter_session=STATE.get("sessions"),
               adapter_applies=None, session_log=STATE.get("session_log", []) + [{"session": STATE.get("sessions"), "trained": len(_ids),
               "stopped": _meta.get("stopped"), "tok_per_s": _meta.get("tok_per_s"), "steps": _meta.get("steps"), "resumed": bool(INIT)}])
    plog(f"lora31: this session trained {len(_ids)} trajectories at {_meta.get('tok_per_s')} tok/s (stopped: {_meta.get('stopped')}); "
         f"total {len(STATE['trained_ids'])} trajectories over {len(STATE['trained_tasks'])} problems")
elif INIT:                          # nothing new this session: the previous adapter remains the submission
    shutil.rmtree(OUT31, ignore_errors=True); shutil.copytree(PREV, OUT31); OK31 = True
    plog("lora31: no new training this session; the previous session's adapter is kept as the submission")
''')
assert "TRAIN_MAX_LENS" in T

# ------------------------------------------------------------------ steps 7-8: adapter check, then the submission (no BASE31, no holdout)
_ev = src(29)
DEFS = "\n".join(_ev[_ev.index(a):_ev.index(b)] for a, b in (("def make_sub", "ARMS = {"), ("def adapter_changes_output", "def arm_done")))
CHECK = r'''# STEP 7 - adapter check: serve the LoRA as the scorer does; it must change a greedy answer (a no-op adapter is not submitted)
''' + DEFS + r'''
if not OK31:
    raise SystemExit("STOP: no trained adapter (see provenance/train_lora31_len*.log); attach this output and run again")
if SMOKE:
    save_state(adapter_applies="not checked (smoke)")
elif STATE.get("adapter_applies") is None:
    _chk = make_sub(OUT / "submissions/check_LORA31", MODEL_SLUG, ADAPTER31)
    stop_servers(); start_server(MODEL_DIR, TP, _chk)
    try:
        save_state(adapter_applies=bool(adapter_changes_output(8000)))
    finally:
        stop_servers()
if STATE["adapter_applies"] is False:
    plog("!!! WARNING: vLLM serves the adapter as a no-op (identical greedy output to the base model); the submission is still "
         "written (it behaves like the base model), check the adapter before relying on it")
'''
FINAL = r'''# STEP 8 - write the submission (the LoRA, deployment budget) and the summary
FINAL = Path("/kaggle/working/submission")
shutil.rmtree(FINAL, ignore_errors=True)
shutil.copytree(make_sub(OUT / "submissions/final_LORA31", MODEL_SLUG, ADAPTER31, budget=DEPLOY_BUDGET), FINAL)
_zip = Path(shutil.make_archive("/kaggle/working/submission", "zip", FINAL))   # agent.yaml at the zip root
_size = sum(f.stat().st_size for f in FINAL.rglob("*") if f.is_file())
assert _size < 3 * 2**30, f"submission is {_size / 2**30:.2f} GiB (limit 3 GiB)"
worst_h = (N_TASKS_ASSUMED * (DEPLOY_BUDGET["max_time_minutes"] + SETUP_MIN) + SERVE_START_MIN) / 60
lines = [f"# Notebook 21 summary ({now()})", "",
         f"- Training trajectories: {n_traj} (SFT mix {MIX}); 31B training window {STATE.get('lora31_window')}",
         f"- Train log (last): {json.dumps(read_jsonl(OUT31 / 'train_log.jsonl')[-1:])[:400]}",
         f"- Adapter changes the served output: {STATE.get('adapter_applies')}",
         f"- Sessions: {STATE.get('session_log')}; trained so far {len(STATE.get('trained_ids', []))} trajectories over "
         f"{len(STATE.get('trained_tasks', []))} problems; attach this output to the next version to continue",
         "", "**SUBMISSION: LORA31** (no base-model arm, no holdout comparison: every task was training data)",
         f"- deployment budget {DEPLOY_BUDGET}; worst case if all {N_TASKS_ASSUMED} tasks hit the cap: {worst_h:.1f} h of 12",
         f"- {FINAL} ({_size / 1e6:.1f} MB) and {_zip}"]
(OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
save_state(submission="LORA31")
print("\n".join(lines))
print((FINAL / "agent.yaml").read_text()); print((FINAL / "eval_config.yaml").read_text())
'''

# scripts: this branch's copies, which carry notebook 20's 31B fixes (train_lora.py: logits_to_keep as a list so accelerate does
# not move it to GPU 0 (the nb19 31B crash), checkpoint integrity preflight, compaction-aware prompts; run_arm.py: src/ shim)
SCRIPTS = HERE.parent
assert first(13).startswith("%%writefile /kaggle/working/gc/run_arm.py") and first(22).endswith("lora/build_rft_data.py") \
    and first(23).endswith("lora/train_lora.py")
run_arm = code("%%writefile /kaggle/working/gc/run_arm.py\n" + (SCRIPTS / "run_arm.py").read_text())
lora = [code(f"%%writefile /kaggle/working/gc/{rel}\n" + (SCRIPTS / rel).read_text())
        for rel in ("lora/build_rft_data.py", "lora/train_lora.py", "tool_trace.py")]
assert first(19).startswith("# STEP 1/7 - validity") and first(20).startswith("# STEP 2/7 - task order")
cells = ([md(HEADER), md(src(1) + DATA_NOTES), C[2], code(c)] + C[4:6] + [code(RESUME)] + C[7:13] + [run_arm]
         + (C[14:19] + [C[21]] if SMOKE else C[14:22]) + lora
         + [C[25], code(DATA), code(T), code(CHECK), code(FINAL), C[31]])
for cell in cells:
    if cell["cell_type"] == "code":
        cell["source"] = "".join(cell["source"]).replace("NB19_STATE", "NB21_STATE").splitlines(keepends=True)
        cell["outputs"], cell["execution_count"] = [], None
nb["cells"] = cells
if SMOKE:
    nb["cells"][0] = md("# 21s — SMOKE TEST of notebook 21: one trajectory → 31B LoRA → submission (≈ 1–1.5 GPU-hours)\n\n"
                        "Question: does notebook 21's path train the 31B LoRA and write a submittable `submission.zip`? "
                        "Trains on the single shortest trajectory of the attached `graphcomp-sft-mix-smoke` (18k window, 1 "
                        "epoch, 1 window per step), then writes `/kaggle/working/submission/` + `submission.zip` with the "
                        "adapter. Skipped to save time: the validity pass and the vLLM adapter check. Most of the time is the "
                        "31B load (~35 min). Inputs: as notebook 21 (notebook 01's output not needed).\n\n"
                        + "".join(nb["cells"][0]["source"]))
out = U / ("notebooks/21s_SMOKE_sft_mix_lora31.ipynb" if SMOKE else "notebooks/21_sft_mix_lora31_submission.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out} ({len(cells)} cells)")

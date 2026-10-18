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
   (× 1.5; the trainer stops on time anyway), **mixed over length quartiles** rather than strictly shortest-first (SWE-Gym's
   easy-task bias); from session 2, **~8% replay** of earlier sessions' trajectories spread through the slice (SWE-Lego's
   cumulative curriculum; the replay result of arXiv 2609.33559). **Competition trajectories go last, every session** (our own
   verified 31B runs + external ones on the four competition repos), **rotated**: the least-visited half each session, at most
   3 visits each, so every session ends on competition data while no text is drilled (arXiv 2609.33559).
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
c = sub1(c, "SESSION_BUDGET_H = 11.3", "SESSION_BUDGET_H = 1.6 " if SMOKE else "SESSION_BUDGET_H = 11.75")   # user, Oct 18: 11 h 45 min
c = sub1(c, "TRAIN31_MAX_H, TRAIN_STUDENT_MAX_H, EVAL_ARM_H = 2.6, 0.8, 0.9",
         f"TRAIN31_MAX_H, TRAIN_STUDENT_MAX_H, EVAL_ARM_H = {0.6 if SMOKE else 10.0}, 0, {0.1 if SMOKE else 0.5}   # EVAL_ARM_H = the adapter check (31B load + 1 request)\n"
         "TRAIN_MAX_LENS = [32768, 24576, 18432]   # training window (tokens); the next one only after an out-of-memory failure\n"
         f"MAX_TRAIN_TRAJ = {1 if SMOKE else 100000}         # cap per session (the token budget below is the real limit)\n"
         f"POST_TRAIN_H = {0.3 if SMOKE else 0.85}             # after training: adapter check (31B load + 1 request), submission, zip, margin\n"
         "LOAD_H = 0.6                  # 31B load inside the trainer before the first step (notebook 20: ~35 min)\n"
         "TOK_PER_S_GUESS = 150         # training tokens / s until a session has measured it (train_meta.json)\n"
         "DATA_OVERSHOOT = 1.5          # select this many times the tokens expected to fit (the trainer stops at the deadline)\n"
         "FIRST_OVERSHOOT = 4.0         # the same before any session has measured the speed (an unused slice costs only rendering)\n"
         "REPLAY_FRAC = 0.08            # tokens of earlier sessions' outside trajectories per token of new ones (session 2+)\n"
         "COMP_MAX_VISITS = 3           # a competition trajectory is trained in at most this many sessions (rotation below)\n"
         "COMP_PER_SESSION = 0.5        # share of the competition trajectories in each session's tail (least-visited first)\n"
         "LENGTH_MIX = True             # new data round-robin over length quartiles, not strictly shortest-first\n"
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
def _scan(path):
    """Light records (key, task, repo, est_tokens, source, byte offset): the 3.7 GB file is never held in memory."""
    out_ = []
    with open(path, "rb") as fh:
        while True:
            off = fh.tell(); line = fh.readline()
            if not line:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("est_tokens", 0) > max(TRAIN_MAX_LENS):
                continue
            out_.append({"source_traj_id": traj_key(r), "instance_id": r["instance_id"], "repo": r.get("repo", ""),
                         "est_tokens": r.get("est_tokens", 0), "mix_source": r.get("mix_source", "rft"),
                         "_file": str(path), "_off": off})
    return out_
_rows = _scan(MIX / "train_fit.jsonl") if MIX else []
# our verified 31B trajectories on the competition tasks (graphcomp-comp-trajs: notebooks 01 + 19, dev/holdout included)
COMP_FILE = next((Path(dp) / "competition_trajs.jsonl" for dp, dns, fns in walk(INPUT, maxdepth=7) if "competition_trajs.jsonl" in fns), None)
plog(f"competition trajectories: {COMP_FILE or 'NOT FOUND: attach graphcomp-comp-trajs (only the mix copies are used)'}")
_known = {r["source_traj_id"] for r in _rows}
for r in (_scan(COMP_FILE) if COMP_FILE else []):
    if r["source_traj_id"] not in _known:          # same actions = the same trajectory (the mix holds 17 of them)
        r["mix_source"] = "rft"; _rows.append(r); _known.add(r["source_traj_id"])
def _load(r):
    with open(r["_file"], "rb") as fh:
        fh.seek(r["_off"]); full = json.loads(fh.readline())
    full["source_traj_id"] = r["source_traj_id"]; full["mix_source"] = r["mix_source"]
    return full
# COMPETITION trajectories (our own on the 129 tasks + external ones on the four competition repos) are trained LAST in every
# session (user, Oct 15: "save our competition dataset for the end, we want it to see that last"), every session again
COMP_REPOS = {"fastapi/fastapi", "tiangolo/fastapi", "textualize/rich", "psf/requests", "encode/httpx",
              "fastapi", "rich", "requests", "httpx"}
is_comp = lambda r: r.get("mix_source") == "rft" or str(r.get("repo", "")).lower() in COMP_REPOS
_comp_all = sorted([r for r in _rows if is_comp(r)], key=lambda r: r.get("est_tokens", 0))
# repeat cap (literature check, Oct 18: re-showing the same texts many times leaves a model fragile, a fresh solution per visit
# does not, arXiv 2609.33559): a competition trajectory is trained in at most COMP_MAX_VISITS sessions. Sessions before this
# rule (no comp_visits in the state) count as one visit for every competition trajectory they trained.
COMP_VISITS = dict(STATE.get("comp_visits") or {k: 1 for k in (r["source_traj_id"] for r in _comp_all) if k in TRAINED})
# rotation: each session's tail is the least-visited COMP_PER_SESSION of them (every session still ENDS on competition data, as
# the user asked, while each text repeats about half as often); only if all are at the cap does the tail go over it (logged)
_n_tail = max(1, round(len(_comp_all) * COMP_PER_SESSION)) if _comp_all else 0
_under = [r for r in _comp_all if COMP_VISITS.get(r["source_traj_id"], 0) < COMP_MAX_VISITS]
_comp = sorted(_under or _comp_all, key=lambda r: (COMP_VISITS.get(r["source_traj_id"], 0), r.get("est_tokens", 0)))[:_n_tail]
_comp.sort(key=lambda r: r.get("est_tokens", 0))
COMP_OVER_CAP = not _under and bool(_comp_all)
_replay_pool = [r for r in _rows if not is_comp(r) and traj_key(r) in TRAINED]
_rows = [r for r in _rows if not is_comp(r) and traj_key(r) not in TRAINED]
def _mix_lengths(rs):
    """Round-robin over length quartiles (short, medium, long, longest, short, ...) instead of strictly shortest-first:
    SWE-Gym (arXiv 2412.21139) warns that successful-trajectory data is biased toward easy tasks; shortest-first would
    sharpen that inside one session."""
    if not LENGTH_MIX or len(rs) < 8:
        return sorted(rs, key=lambda r: r.get("est_tokens", 0))
    rs = sorted(rs, key=lambda r: r.get("est_tokens", 0))
    q = [rs[i * len(rs) // 4:(i + 1) * len(rs) // 4] for i in range(4)]
    out_ = []
    for i in range(max(map(len, q))):
        out_ += [b[i] for b in q if i < len(b)]
    return out_
# the rest: NEW problems (never trained) first, one trajectory each (the shortest of each problem), mixed over lengths; then
# second trajectories etc., mixed the same way
_rows.sort(key=lambda r: r.get("est_tokens", 0))
_first, _seen = [], set(TRAINED_TASKS)
for r in _rows:
    if task_of(r) not in _seen:
        _first.append(r); _seen.add(task_of(r))
_firstset = {id(r) for r in _first}
_queue = _mix_lengths(_first) + _mix_lengths([r for r in _rows if id(r) not in _firstset])
# size this session's slice: tokens the trainer can process before the deadline, times DATA_OVERSHOOT
TRAIN_DEADLINE = _T0 + (SESSION_BUDGET_H - POST_TRAIN_H) * 3600
_tps = STATE.get("tok_per_s") or TOK_PER_S_GUESS
_over = DATA_OVERSHOOT if STATE.get("tok_per_s") else FIRST_OVERSHOOT
_train_s = max(0.0, TRAIN_DEADLINE - time.time() - LOAD_H * 3600)
_comp_tok = sum(r.get("est_tokens", 0) for r in _comp)
_budget_tok = max(0.0, _tps * _train_s * _over - _comp_tok)
# replay (SWE-Lego's cumulative curriculum, arXiv 2601.01426; 2609.33559's replay): REPLAY_FRAC of the new-data tokens are
# trajectories trained in earlier sessions, spread through the slice (not at its end)
_replay_tok = _budget_tok * REPLAY_FRAC / (1 + REPLAY_FRAC) if _replay_pool else 0.0
_new_budget = _budget_tok - _replay_tok
_pick, _tok = [], 0
for r in _queue:
    if len(_pick) >= MAX_TRAIN_TRAJ:
        break
    if _pick and _tok + r.get("est_tokens", 0) > _new_budget:
        continue                      # too long for what is left: a shorter one may still fit
    _pick.append(r); _tok += r.get("est_tokens", 0)
_replay, _rtok = [], 0
for r in random.Random(STATE.get("sessions", 1)).sample(_replay_pool, len(_replay_pool)):
    if _rtok + r.get("est_tokens", 0) <= _replay_tok:
        _replay.append(r); _rtok += r.get("est_tokens", 0)
if _replay:                           # spread evenly through the new data
    step_ = max(1, len(_pick) // (len(_replay) + 1))
    for k_, r in enumerate(_replay):
        _pick.insert(min(len(_pick), (k_ + 1) * step_ + k_), r)
_tok += _rtok
TAIL_N = len(_comp)
COMP_KEYS = [r["source_traj_id"] for r in _comp]
_pick = _pick + _comp                 # the trainer always trains these, at the end (--tail-n)
if SMOKE:
    _pick, TAIL_N = (_comp or _queue)[:1], 0
TRAIN_DATA = OUT / "sft" / f"train_session{STATE.get('sessions', 1)}.jsonl"
TRAIN_DATA.parent.mkdir(parents=True, exist_ok=True)
with open(TRAIN_DATA, "w") as f:
    for r in _pick:
        f.write(json.dumps(_load(r)) + "\n")
n_traj = len(_pick)
_new = len({task_of(r) for r in _pick} - TRAINED_TASKS)
plog(f"session {STATE.get('sessions')}: {len(TRAINED)} trajectories / {len(TRAINED_TASKS)} problems trained before; "
     f"this slice {n_traj} trajectories (~{(_tok + _comp_tok) / 1e6:.1f}M tokens; {_new} new problems; the last {TAIL_N} = "
     f"competition trajectories, trained last; {len(_replay)} replayed from earlier sessions; "
     f"tail = least-visited {len(_comp)} of {len(_comp_all)} competition trajectories"
     f"{' (ALL at the ' + str(COMP_MAX_VISITS) + '-visit cap: repeating anyway so the session ends on competition data)' if COMP_OVER_CAP else ''}) for "
     f"{_train_s / 3600:.1f} h at {_tps:.0f} tok/s x {_over}; {len(_queue)} untrained left -> {TRAIN_DATA}")
del _rows, _queue, _first, _replay_pool
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
# windows that ran out of memory in an earlier session are skipped, and a session never starts above the window that last
# trained (session 1, Oct 18: 32k ran out of memory on GPU 3, which also holds the 262k-vocabulary output layer)
_oom = set(STATE.get("oom_windows", []))
_cap = STATE.get("lora31_window") or max(TRAIN_MAX_LENS)
_lens = [l for l in TRAIN_MAX_LENS if l <= _cap and l not in _oom] or [min(TRAIN_MAX_LENS)]
plog(f"lora31: training windows to try {_lens} (out of memory before: {sorted(_oom) or 'none'})")
for L in ([18432] if SMOKE else _lens):
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
    save_state(oom_windows=sorted(set(STATE.get("oom_windows", [])) | {L}))
    plog(f"lora31: out of memory with a {L}-token window; trying the next one")
if OK31:
    _ids = json.loads((ADAPTER31 / "trained_ids.json").read_text()) if (ADAPTER31 / "trained_ids.json").exists() else []
    _meta = json.loads((OUT31 / "train_meta.json").read_text()) if (OUT31 / "train_meta.json").exists() else {}
    _byid = {}
    for l_ in open(TRAIN_DATA):
        r_ = json.loads(l_); _byid[r_["source_traj_id"]] = task_of(r_)
    _cv = dict(COMP_VISITS)
    for k_ in set(COMP_KEYS) & set(_ids):
        _cv[k_] = _cv.get(k_, 0) + 1
    save_state(comp_visits=_cv, trained_ids=sorted(TRAINED | set(_ids)), trained_tasks=sorted(TRAINED_TASKS | {_byid[i] for i in _ids if i in _byid}),
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
if not OK31:   # still submittable: the base model, flagged (attach this output and run again to train)
    plog("!!! NO ADAPTER this session (see provenance/train_lora31_*.log): the submission is the BASE 31B, no LoRA")
    save_state(adapter_applies="no adapter")
elif SMOKE:
    save_state(adapter_applies="not checked (smoke)")
elif left_h() < 0.6:                 # the submission must still be written inside the session
    plog(f"adapter check skipped: {left_h():.2f} h left"); save_state(adapter_applies="not checked (time)")
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
shutil.copytree(make_sub(OUT / "submissions/final_LORA31", MODEL_SLUG, ADAPTER31 if OK31 else None, budget=DEPLOY_BUDGET), FINAL)
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
         "", ("**SUBMISSION: LORA31**" if OK31 else "**SUBMISSION: BASE 31B (no adapter was trained; see the logs)**")
         + " (no holdout comparison: every task is training data)",
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
ZIP = code("".join(C[31]["source"]).replace("graphcomp_nb19.zip", "graphcomp_nb21.zip"))
cells = ([md(HEADER), md(src(1) + DATA_NOTES), C[2], code(c)] + C[4:6] + [code(RESUME)] + C[7:13] + [run_arm]
         + C[14:19] + [C[21]] + lora
         + [C[25], code(DATA), code(T), code(CHECK), code(FINAL), ZIP])
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

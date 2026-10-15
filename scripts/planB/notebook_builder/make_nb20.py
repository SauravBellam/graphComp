"""Build notebook 20: a ~45 GPU-hour iterative rejection-sampling campaign over several resumed Kaggle sessions.

Reuses notebook 19's tested cells (setup, offline install, test deps, provenance, scorecard, submission writer, validity,
task order, holdout definition, vLLM helpers, make_sub / adapter probe) from the notebook that actually ran, and embeds
this branch's run_arm.py (src/ + SWE_EXTRA_PYTHONPATH shim), build_rft_data.py (accurate-tool selection, time order),
train_lora.py and tool_trace.py. Re-run after editing any embedded script:
    python3 scripts/planB/notebook_builder/make_nb20.py
"""
import json, re
from pathlib import Path

R = Path("/Users/sauravbellam/Desktop/Projects/graphComp")          # data, notebooks (not in git)
U = R / "planB_results/kaggle_upload"
HERE = Path(__file__).resolve().parents[1]                           # scripts/planB on this branch
nb = json.load(open(U / "notebooks/19_teacher_lora31_student_holdout_submission.ipynb"))
C = nb["cells"]
src = lambda i: "".join(C[i]["source"])
def first_line(i):
    return src(i).splitlines()[0]
def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}
def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
def sub1(s, old, new):
    assert s.count(old) == 1, old[:80]
    return s.replace(old, new)
# the nb19 cells this builder relies on, checked by their first line so a changed nb19 fails loudly here
EXPECT = {2: "# first cell", 3: "# STEP 0/7 - setup: config", 4: "# STEP 0/7 - setup: helper", 5: "# STEP 0/7 - setup: locate",
          6: "# STEP 0 - resume", 13: "%%writefile /kaggle/working/gc/run_arm.py", 18: "# STEP 0/7 - setup: write the arm",
          19: "# STEP 1/7 - validity", 20: "# STEP 2/7 - task order", 24: "# STEP 2 - holdout set", 25: "# STEP 3 - vLLM helpers",
          29: "# STEP 7 - holdout comparison"}
for i, p in EXPECT.items():
    assert first_line(i).startswith(p), (i, first_line(i))

NEW_TASKS = ["httpx_3672", "requests_6644", "requests_7205", "requests_7309", "requests_7315", "requests_7328", "requests_7427",
             "requests_7433", "requests_7502", "requests_7505"]   # valid on the VM with the src/ shim + pytest-httpbin (Oct 15)

HEADER = """# 20 — iterative rejection-sampling campaign (~45 GPU-hours, several sessions) → best LoRA → budget-safe submission

**What it does, every session (resumable; attach the previous version's output):**
0. **Train first** (first session; notebook 19's `LORA31` never trained): round 1 = a 31B LoRA on the verified, accurate
   trajectories already collected (notebook 01 B2 s1 + notebook 19's teacher passes), scored on the holdout before sampling.
1. **Sample** with the current best adapter (base model if no round beat `BASE31`).
   Tasks come in *passes*: never-solved tasks first (fewest attempts first), then solved ones from lowest solve rate and
   slowest solve upward ("hard problems that took a while"). Every pass attempts every task once, then a new pass starts.
2. **Per-task time cap** is recomputed before every chunk from the sampling time left and the tasks left in the pass, so
   every task gets at least one attempt (bounds `CAP_MIN_RANGE`); tool calls stay at 30.
3. **Every attempt is saved** (`runs/NB20/parts/*`, indexed in `attempts.jsonl`), passed or not.
4. **Round** (when training is still active): training data = verified trajectories with *accurate tool use* (every source
   edit lands on the gold files/functions and a reproduction or test runs after the last edit; `tool_trace.py`), fastest
   first, at most `MAX_PER_TASK_TRAIN` per task, never dev / holdout; **retrain from the base 31B** (training windows
   reshuffled every epoch, with a different shuffle seed each round); score the new adapter on
   the same 27-task holdout as notebook 19 (same 8-min / 30-call budget, seed 1). A round that does not beat the best stops
   training (after at least `MIN_NEW_ROUNDS` new rounds); later sessions keep sampling with the best adapter.
5. **Submission** = the best round's adapter (or none) with the deployment budget.

**New tasks:** the 10 requests/httpx tasks made valid on the VM (src/ shim in `run_arm.py`; `pytest-httpbin` from the
`graphcomp-testdeps-httpbin` dataset, unpacked onto the sandbox `PYTHONPATH`, never installed). Their validity is
re-checked on this image (gold passes, empty fails) before they join the pool.

**Inputs:** competition data, `metric/gemma-4-developer-agent-wheelhouse`, models `gemma-4-31b-it-qat-w4a16-ct` (serving) and
`gemma-4-31b-it-qat-q4_0-unquantized` (training), **notebook 01's output**, **notebook 19's output**, `graphcomp-train-wheels`,
`graphcomp-testdeps-httpbin`, and from session 2 on **this notebook's previous version output**. GPU L4 × 4.

**Literature:** iterative RFT helps for a couple of rounds and then plateaus, and naive on-policy self-training can hurt
(SWE-Gym, arXiv 2412.21139); single-attempt online updates destabilize, so rounds retrain from the base on verified data
(ASCENT 2610.05303; ReSAIL 2609.39306 on collapse across cycles); process quality of successful coding trajectories
matters for SFT (TQS 2610.13152); function-level selection separates success from failure (TRACEPROBE 2607.06184).

**Training data and repeats:** several *different* verified trajectories per task are kept (up to `MAX_PER_TASK_TRAIN`), but
exact route copies are dropped (`--dedupe-routes 1.0`, fastest kept) and epochs stay at ≤2: drilling the same solution
repeatedly makes models fragile while a new solution per visit does not (arXiv 2609.33559); at this data scale keep all
verified data rather than a tiny filtered subset (2607.17205); loss on assistant turns with the full context (SWE-Prime
2608.27449); LoRA rather than full fine-tuning (2609.17848). **Compaction:** `run_arm.py` now logs every ADK compaction
summary (`compactions.jsonl`) and `train_lora.py` rebuilds the post-compaction prompts from it; notebook 19's runs predate
this, so their turns after the first compaction (445 of them) are still left out.
"""

CONFIG = r'''# STEP 0 - NB20 campaign config
CAMPAIGN_H = 45.0              # total GPU hours over all sessions of this notebook
ROUND_SAMPLE_H = 6.0           # sampling hours in a session that also trains and scores a round (then ~2.6 h train + ~0.9 h holdout)
MIN_NEW_ROUNDS, MAX_ROUNDS = 1, 4    # rounds after notebook 19's LORA31; training stops when a round does not beat the best
CAP_MIN_RANGE = (4.0, 12.0)    # per-task agent time cap bounds (minutes), recomputed before every chunk
SETUP_MIN_PER_TASK = 0.7       # sandbox setup + verification per task, outside the agent's time cap
MAX_PER_TASK_TRAIN = 3         # verified, accurate trajectories per task in a round's training data (fastest first)
SEED_BASE = 2000               # sampling seeds: SEED_BASE + attempt number (every attempt a different seed)
HOLDOUT_BUDGET = dict(max_tool_calls=30, max_time_minutes=8, max_turns=80, timeout_seconds=240)   # = notebook 19's holdout
NEW_TASKS = ''' + repr(NEW_TASKS) + r'''
RETRY_NOTES = True             # failed-before tasks get a description of earlier failed attempts (not the verbatim calls)
RETRY_NOTE_SHARE = 2           # notes on every 2nd chunk only, so their effect is measured (attempts.jsonl "noted")
BUDGET = dict(BUDGET, max_tool_calls=30)
plog(f"NB20 campaign: {CAMPAIGN_H} h total, rounds {MIN_NEW_ROUNDS}..{MAX_ROUNDS}, cap {CAP_MIN_RANGE} min, new tasks {len(NEW_TASKS)}")
'''

RESUME = r'''# STEP 0 - resume: this notebook's earlier output (NB20_STATE.json), else start from notebook 19's output
_prior = [dp for dp, dns, fns in walk(INPUT, maxdepth=8) if dp.name == "graphcomp_arm" and (dp / "NB20_STATE.json").exists()]
for _p in _prior:
    shutil.copytree(_p, OUT, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.zip"))
    plog(f"resume: copied earlier NB20 output {_p}")
NB19_OUT = next((dp for dp, dns, fns in walk(INPUT, maxdepth=8) if dp.name == "graphcomp_arm" and (dp / "NB19_STATE.json").exists()), None)
NB19_STATE = json.loads((NB19_OUT / "NB19_STATE.json").read_text()) if NB19_OUT else {}
assert NB19_OUT is not None, "attach notebook 19's output (holdout results, LORA31 adapter, teacher trajectories)"
STATE_FILE = OUT / "NB20_STATE.json"
STATE = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
def save_state(**kw):
    STATE.update(kw)
    STATE_FILE.write_text(json.dumps(STATE, indent=1))
save_state(sessions=STATE.get("sessions", 0) + 1)
_CAMPAIGN_BEFORE = STATE.get("campaign_h_done", 0.0)
def left_h():
    return SESSION_BUDGET_H - (time.time() - _T0) / 3600
def campaign_left_h():
    return CAMPAIGN_H - _CAMPAIGN_BEFORE - (time.time() - _T0) / 3600
def mark_campaign():
    save_state(campaign_h_done=round(_CAMPAIGN_BEFORE + (time.time() - _T0) / 3600, 3))
plog(f"state: session {STATE['sessions']}, campaign used {_CAMPAIGN_BEFORE:.1f} of {CAMPAIGN_H} h; notebook 19 output {NB19_OUT}")
'''

TESTDEPS = r'''# STEP 1 - pytest-httpbin + deps for the requests tasks: unpacked onto the sandbox PYTHONPATH, never installed
_w = sorted(glob.glob(str(INPUT / "**/pytest_httpbin-*.whl"), recursive=True))
TESTDEPS = Path("/kaggle/working/testdeps")      # not under /tmp: the sandbox rewrites /tmp paths
if _w:
    if not (TESTDEPS / "pytest_httpbin").exists():
        TESTDEPS.mkdir(parents=True, exist_ok=True)
        for whl in sorted(glob.glob(str(Path(_w[0]).parent / "*.whl"))):
            zipfile.ZipFile(whl).extractall(TESTDEPS)
    os.environ["SWE_EXTRA_PYTHONPATH"] = str(TESTDEPS)
    plog(f"test deps: {len(glob.glob(str(Path(_w[0]).parent / '*.whl')))} wheels unpacked into {TESTDEPS}; SWE_EXTRA_PYTHONPATH set")
else:
    plog("!!! graphcomp-testdeps-httpbin not attached: the new requests tasks will fail their gold check and stay out of the pool")
'''

def holdout_part():
    s = src(24)
    return s[:s.index("def part_records")]

POOL = holdout_part() + r'''
# ---- new tasks: re-check validity on this image (gold passes, empty fails); cached in STATE
NEW_VALID = STATE.get("new_valid")
if NEW_VALID is None:
    _vd = OUT / "validity_new"
    _rc = lambda mode, ids, d: (f"cd {GC} && {sys.executable} run_arm.py --mode {mode} --tasks {TASKS_JSONL} --snapshots {SNAPSHOTS} "
                                f"--submission {SUB} --results {d} --sandbox subprocess --task-ids {' '.join(ids)} "
                                f"--timeout-seconds 300 --concurrency {CONCURRENCY}")
    sh(_rc("gold", NEW_TASKS, _vd / "gold"), log=_vd.parent / "provenance/validity_new_gold.log")
    _gold = [r["instance_id"] for r in read_jsonl(_vd / "gold/task_results.jsonl") if r.get("resolved")]
    if _gold:
        sh(_rc("empty", _gold, _vd / "empty"), log=_vd.parent / "provenance/validity_new_empty.log")
    _emp = {r["instance_id"] for r in read_jsonl(_vd / "empty/task_results.jsonl") if r.get("resolved")}
    NEW_VALID = [t for t in _gold if t not in _emp]
    save_state(new_valid=NEW_VALID)
plog(f"new tasks valid on this image: {len(NEW_VALID)}/{len(NEW_TASKS)} {NEW_VALID}")
POOL = list(ORDER) + [t for t in NEW_VALID if t not in ORDER and t not in set(HO) | set(OFFICIAL_DEV)]
# training-data profile: notebook 01's validity plus the re-checked new tasks; never-train list: the extra holdout tasks
with open(B1_PROFILE) as f_, open(OUT / "profile_nb20.csv", "w", newline="") as g_:
    _r = csv.DictReader(f_); _w_ = csv.DictWriter(g_, fieldnames=_r.fieldnames); _w_.writeheader()
    for row in _r:
        if row["instance_id"] in NEW_VALID:
            row["sub_valid"], row["sub_reason"] = "True", "valid with src/ shim + pytest-httpbin (NB20 check)"
        _w_.writerow(row)
(GC / "never_train.txt").write_text("\n".join(HOLDOUT_EXTRA) + "\n")
# ---- external tasks (scripts/planB/external/import_external.py output, attached as a dataset): same layout as the
# competition data (tasks.jsonl, snapshots/, graphs/, embeddings/); wheels come from the competition data; validity is
# re-checked on this image like the new tasks
SOURCE = {t: (TASKS_JSONL, SNAPSHOTS) for t in ALL_IDS}
TASK_TEXT = {t: tasks_by_id[t]["problem_statement"] for t in ALL_IDS}
EXT_ROWS = []
for _m in sorted(glob.glob(str(INPUT / "**/graphcomp_external.json"), recursive=True)):
    _src = Path(_m).parent
    _meta = json.loads(Path(_m).read_text())
    _dst = Path("/kaggle/working/extdata") / _src.name
    if not _dst.exists():
        _dst.mkdir(parents=True)
        for _d in ("snapshots", "graphs", "embeddings"):
            os.symlink(_src / _d, _dst / _d)
        os.symlink(DATA_ROOT / "wheels", _dst / "wheels")
        shutil.copy(_src / "tasks.jsonl", _dst / "tasks.jsonl")
    _rows = {json.loads(l)["instance_id"]: json.loads(l) for l in open(_dst / "tasks.jsonl") if l.strip()}
    assert not set(_rows) & set(ALL_IDS), "external ids collide with competition ids"
    _key = f"ext_valid::{_src.name}"
    if _key not in STATE:
        _vd = OUT / f"validity_ext/{_src.name}"
        _ids = [t for t in _meta.get("valid", list(_rows))]
        _rc = lambda mode, ids, d: (f"cd {GC} && {sys.executable} run_arm.py --mode {mode} --tasks {_dst / 'tasks.jsonl'} "
                                    f"--snapshots {_dst / 'snapshots'} --submission {SUB} --results {d} --sandbox subprocess "
                                    f"--task-ids {' '.join(ids)} --timeout-seconds 300 --concurrency {CONCURRENCY}")
        sh(_rc("gold", _ids, _vd / "gold"), log=OUT / f"provenance/validity_ext_{_src.name}.log")
        _g = [r["instance_id"] for r in read_jsonl(_vd / "gold/task_results.jsonl") if r.get("resolved")]
        if _g:
            sh(_rc("empty", _g, _vd / "empty"), log=OUT / f"provenance/validity_ext_{_src.name}.log")
        _e = {r["instance_id"] for r in read_jsonl(_vd / "empty/task_results.jsonl") if r.get("resolved")}
        save_state(**{_key: [t for t in _g if t not in _e]})
    for t in STATE[_key]:
        SOURCE[t] = (_dst / "tasks.jsonl", _dst / "snapshots")
        TASK_TEXT[t] = _rows[t]["problem_statement"]
        EXT_ROWS.append(_rows[t])
        if t not in POOL:
            POOL.append(t)
    plog(f"external source {_src.name}: {len(STATE[_key])}/{len(_rows)} valid on this image -> pool {len(POOL)}")
with open(GC / "all_tasks.jsonl", "w") as f_:          # gold patches for the accurate-tool check: competition + external
    for t in list(tasks_by_id.values()) + EXT_ROWS:
        f_.write(json.dumps(t) + "\n")
if EXT_ROWS:
    with open(OUT / "profile_nb20.csv", "a", newline="") as g_:
        _w2 = csv.DictWriter(g_, fieldnames=list(csv.DictReader(open(B1_PROFILE)).fieldnames))
        for r_ in EXT_ROWS:
            _w2.writerow({"instance_id": r_["instance_id"], "repo": r_["instance_id"].split("_")[0], "sub_valid": "True",
                          "sub_reason": "external task, valid on this image (NB20 check)"})
plog(f"pool: {len(POOL)} tasks ({len(ORDER)} from notebook 19 + {len(POOL) - len(ORDER)} new / external)")
'''

SAMPLER_HEAD = "# STEP 4 - sampling: passes over the pool with the current best adapter; every attempt saved and indexed\n"
SAMPLER = r'''import statistics, hashlib
from collections import defaultdict, Counter
ATTEMPTS = OUT / "attempts.jsonl"
B2S1 = next((dp for dp, dns, fns in walk(INPUT, maxdepth=9, skip=("snapshots", "wheels", "embeddings", "graphs", "__pycache__", ".git",
             "traces", "logs", "test_outputs", "patches", "parts")) if dp.name == "merged_B2_s1" and "task_results.jsonl" in fns), None)
NB19_PARTS = sorted((NB19_OUT / "runs/NB19/parts").glob("*/")) if (NB19_OUT / "runs/NB19/parts").exists() else []
def history():
    """Per task: [(resolved, seconds)] over notebook 01 seed 1, notebook 19's teacher runs and every NB20 attempt."""
    h = defaultdict(list)
    for d in ([B2S1] if B2S1 else []) + NB19_PARTS:
        for r in read_jsonl(d / "task_results.jsonl"):
            h[r["instance_id"]].append((bool(r.get("resolved")), float(r.get("duration_seconds") or 0)))
    for r in read_jsonl(ATTEMPTS):
        h[r["instance_id"]].append((bool(r["resolved"]), float(r["seconds"] or 0)))
    return h
def pass_order(pool):
    h = history()
    unsolved = sorted([t for t in pool if not any(ok for ok, _ in h[t])], key=lambda t: (len(h[t]), t))
    solved = [t for t in pool if any(ok for ok, _ in h[t])]
    rate = lambda t: sum(ok for ok, _ in h[t]) / len(h[t])
    slow = lambda t: statistics.median([s for ok, s in h[t] if ok])
    return unsolved + sorted(solved, key=lambda t: (rate(t), -slow(t), t))
sys.path.insert(0, str(GC))
import tool_trace
def failed_traces():
    """Per task: trace files of earlier attempts that ran normally and did not resolve the task."""
    f = defaultdict(list)
    for d in ([B2S1] if B2S1 else []) + NB19_PARTS + sorted(q for q in PARTS.glob("p*") if q.is_dir()):
        for r in read_jsonl(d / "task_results.jsonl"):
            if not r.get("resolved") and not INFRA_RE.search(json.dumps({k: v for k, v in r.items() if isinstance(v, str)})):
                f[r["instance_id"]].append((d / "traces" / f"trace_{r['instance_id']}.json", r))
    return f
def failure_note(iid, fails):
    """Describe what earlier failed attempts did, in words; never their verbatim calls or patches (verbatim failed calls
    raise the chance of repeating them, a runtime description removes most of that: arXiv 2608.23651; conditioning a
    retry on the failed attempt helps: FC-SWE, arXiv 2610.07898)."""
    edited, ends, untested = Counter(), Counter(), 0
    for tp, r in fails:
        rows = tool_trace.trace_run(json.load(open(tp)), {"files": [], "funcs": set()})[0] if tp.exists() else []
        files = {x["target"].split()[0] for x in rows if x["purpose"] == "edit-src"}
        edited.update(files)
        err = str(r.get("error") or "").lower()
        ends["ran out of time" if "timeout" in err or "timed out" in err else "used all tool calls" if "tool" in err and "budget" in err
             else "submitted a patch that failed the tests" if r.get("agent_patch_size") else "made no change"] += 1
        last = max((x["i"] for x in rows if x["purpose"] == "edit-src"), default=None)
        untested += last is not None and not any(x["purpose"] in ("reproduce", "test") and x["i"] > last for x in rows)
    lines = [f"## Notes from earlier attempts at this issue", f"{len(fails)} earlier attempt(s) did not pass the hidden tests."]
    if edited:
        lines.append("They edited: " + ", ".join(f"{p} ({n})" for p, n in edited.most_common(4)) + ".")
    lines.append("They ended: " + ", ".join(f"{n} {k}" for k, n in ends.most_common()) + ".")
    if untested:
        lines.append(f"{untested} of them did not run any reproduction or test after their last edit.")
    lines.append("The cause may be somewhere else, or need a different kind of fix. Form a different hypothesis before "
                 "editing the same places again, and check your fix by running code before you submit.")
    return "\n".join(lines)
def cap_minutes(n_left, sample_h_left):
    raw = sample_h_left * 60 * CONCURRENCY / max(1, n_left) - SETUP_MIN_PER_TASK
    return round(min(max(raw, CAP_MIN_RANGE[0]), CAP_MIN_RANGE[1]) * 2) / 2
# ---- rounds: 0 = base, 1 = notebook 19's LORA31 (copied here so later sessions keep it)
if "rounds" not in STATE:
    h19 = NB19_STATE.get("holdout", {})
    rounds = [{"round": 0, "adapter": None, "resolved": h19.get("BASE31", {}).get("resolved"), "per_task": h19.get("BASE31", {}).get("per_task", {})}]
    if (NB19_OUT / "lora31/adapter/adapter_config.json").exists() and "LORA31" in h19:
        shutil.copytree(NB19_OUT / "lora31/adapter", OUT / "rounds/r1/adapter", dirs_exist_ok=True)
        rounds.append({"round": 1, "adapter": str(OUT / "rounds/r1/adapter"), "resolved": h19["LORA31"]["resolved"],
                       "per_task": h19["LORA31"].get("per_task", {})})
    best = max(rounds, key=lambda r: (r["resolved"] if r["resolved"] is not None else -1, -r["round"]))
    save_state(rounds=rounds, best_round=best["round"], training_stopped=False, pass_todo=[], attempt_seq=0, passes=0)
NEED_H = TRAIN31_MAX_H + EVAL_ARM_H + 0.4      # one round: train (incl. the ~35 min 31B load) + holdout + margin
def refresh_rounds():
    global ROUND_BY_N, BEST, TRAINING, SAMPLE_ROUND
    ROUND_BY_N = {r["round"]: r for r in STATE["rounds"]}
    BEST = ROUND_BY_N[STATE["best_round"]]
    TRAINING = (not STATE["training_stopped"]) and sum("n_train" in r for r in STATE["rounds"]) < MAX_ROUNDS
    SAMPLE_ROUND = len(STATE["rounds"])            # the round these attempts feed (data for the next adapter)
    plog("rounds so far: " + ", ".join(f"r{r['round']} {r['resolved']}/{len(HO)}" for r in STATE["rounds"]) + f" | best r{BEST['round']}")
refresh_rounds()
'''

SAMPLER_LOOP = r'''# STEP 5 - sampling loop
_el = (time.time() - _T0) / 3600
SAMPLE_END_H = (min(_el + ROUND_SAMPLE_H, SESSION_BUDGET_H - NEED_H) if TRAINING
                else SESSION_BUDGET_H - 0.6)   # session hours at which sampling stops (a round still fits after it)
chunk_secs, stop_reason = [], "sampling window over"
try:
    sub = None
    while True:
        el = (time.time() - _T0) / 3600
        window = min(SAMPLE_END_H - el, campaign_left_h() - (TRAIN31_MAX_H + EVAL_ARM_H if TRAINING else 0.6))
        est = 1.25 * max(chunk_secs[-3:]) / 3600 if chunk_secs else 0.5
        if window < est:
            stop_reason = f"window {window:.2f} h < next chunk {est:.2f} h"; break
        if not STATE["pass_todo"]:
            save_state(pass_todo=pass_order(POOL), passes=STATE["passes"] + 1)
            plog(f"pass {STATE['passes']}: {len(STATE['pass_todo'])} tasks, never-solved first: {STATE['pass_todo'][:6]} ...")
        cap = cap_minutes(len(STATE["pass_todo"]), window)
        BUDGET["max_time_minutes"] = cap
        _s0 = SOURCE[STATE["pass_todo"][0]]                    # one task source per run_arm call
        chunk = [t for t in STATE["pass_todo"] if SOURCE[t] == _s0][:CHUNK]
        seq = STATE["attempt_seq"]
        sub = make_sub(OUT / "submissions/sample", MODEL_SLUG, Path(BEST["adapter"]) if BEST["adapter"] else None, seed=SEED_BASE + seq)
        if not SERVERS or not server_alive():
            stop_servers(); start_server(MODEL_DIR, TP, sub if BEST["adapter"] else None)
        part = PARTS / f"p{len([q for q in PARTS.glob('p*') if q.is_dir()]):04d}_r{SAMPLE_ROUND}_seed{SEED_BASE + seq}"
        plog(f"chunk: {chunk} | cap {cap} min, 30 calls | adapter r{BEST['round']} | window {window:.2f} h | campaign left {campaign_left_h():.1f} h")
        noted, _ft = set(), failed_traces()
        prior = {t for t in chunk if _ft.get(t)}
        cmd = agent_cmd(sub, chunk, part, MODEL_DIR, MODEL_SLUG).replace(f"--tasks {TASKS_JSONL} --snapshots {SNAPSHOTS}",
                                                                         f"--tasks {_s0[0]} --snapshots {_s0[1]}")
        if RETRY_NOTES and seq % RETRY_NOTE_SHARE == 0:
            briefs = {t: {"brief": failure_note(t, _ft[t]), "problem_sha256": hashlib.sha256(TASK_TEXT[t].encode()).hexdigest()}
                      for t in chunk if _ft.get(t)}
            if briefs:
                bp = OUT / f"briefs/{part.name}.json"; bp.parent.mkdir(exist_ok=True); bp.write_text(json.dumps(briefs, indent=1))
                cmd += f" --briefs {bp}"
                noted = set(briefs)
        t0 = time.time()
        with Progress(part / "task_results.jsonl", len(chunk), f"sample r{SAMPLE_ROUND}", every=120):
            sh(cmd, log=part.parent / f"{part.name}.log")
        chunk_secs.append(time.time() - t0)
        done, retry = [], []
        with open(ATTEMPTS, "a") as f:
            for r in read_jsonl(part / "task_results.jsonl"):
                infra = bool(INFRA_RE.search(json.dumps({k: v for k, v in r.items() if isinstance(v, str)})))
                (retry if infra else done).append(r["instance_id"])
                f.write(json.dumps({"instance_id": r["instance_id"], "resolved": bool(r.get("resolved")), "seconds": r.get("duration_seconds"),
                                    "tool_calls": r.get("tool_calls"), "cap_min": cap, "seed": SEED_BASE + seq, "part": part.name,
                                    "adapter_round": BEST["round"], "sample_round": SAMPLE_ROUND, "infra_error": infra,
                                    "noted": r["instance_id"] in noted, "prior_failure": r["instance_id"] in prior, "time": now()}) + "\n")
        missing = [t for t in chunk if t not in done and t not in retry]
        save_state(pass_todo=retry + missing + [t for t in STATE["pass_todo"] if t not in chunk], attempt_seq=seq + 1)
        mark_campaign()
        k = sum(r["resolved"] for r in read_jsonl(ATTEMPTS)[-len(chunk):])
        plog(f"    chunk {chunk_secs[-1] / 60:.1f} min: {k}/{len(chunk)} resolved; infra retry {retry}; pass left {len(STATE['pass_todo'])}")
except Exception as e:
    stop_reason = f"error: {type(e).__name__}: {e}"
    traceback.print_exc()
stop_servers()
mark_campaign()
_att = read_jsonl(ATTEMPTS)
plog(f"sampling stopped ({stop_reason}); attempts so far {len(_att)}, resolved {sum(a['resolved'] for a in _att)}, "
     f"tasks ever solved {len({a['instance_id'] for a in _att if a['resolved']})}/{len(POOL)}")
'''

def defs_from_29():
    s = src(29)
    out = []
    for name, nxt in (("def make_sub", "ARMS = {"), ("def adapter_changes_output", "def arm_done")):
        i, j = s.index(name), s.index(nxt)
        out.append(s[i:j])
    return "\n".join(out)

ROUND_DEFS = r'''# STEP 4b - rounds: build data -> train a LoRA from the base 31B -> score on the holdout -> keep or stop
_models = [dp for dp, dns, fns in walk(INPUT, maxdepth=9) if "config.json" in fns]
BASE31_TRAIN = next((dp for dp in _models if "31b" in str(dp).lower() and "unquantized" in str(dp).lower() and "assistant" not in str(dp).lower()), None)
TRAIN_LIB = Path("/tmp/trainlib")
_wh = sorted(glob.glob(str(INPUT / "**/peft-*.whl"), recursive=True)) + sorted(glob.glob(str(INPUT / "**/accelerate-*.whl"), recursive=True))
if _wh and not TRAIN_LIB.exists():
    sh(f"{sys.executable} -m pip install --no-deps --no-index --target {TRAIN_LIB} {' '.join(_wh)}", log=OUT / "provenance/pip_train.log")
TRAIN_ENV = {"PYTHONPATH": str(TRAIN_LIB)} if TRAIN_LIB.exists() else {}
(GC / "official_slices").mkdir(exist_ok=True)   # build_rft_data.py reads ../official_slices next to lora/
(GC / "official_slices/dev_slice.txt").write_text("\n".join(OFFICIAL_DEV) + "\n")
(GC / "official_slices/holdout_slice.txt").write_text("\n".join(OFFICIAL_HOLDOUT) + "\n")
DEPLOY_PROMPT = GC / "deploy_system.md"
DEPLOY_PROMPT.write_text((Path(SAMPLE_SUB) / "prompts/system.md").read_text() + PROMPT_ADDENDUM)
DEPLOY_ARGS = (f"--system-prompt {DEPLOY_PROMPT} --deploy-budget " + shlex.quote(json.dumps({
    "time_minutes": DEPLOY_BUDGET["max_time_minutes"], "tool_calls": DEPLOY_BUDGET["max_tool_calls"],
    "turns": DEPLOY_BUDGET["max_turns"], "timeout_seconds": DEPLOY_BUDGET["timeout_seconds"]})))
def eval_holdout(name, adapter_dir):
    """Score an adapter on notebook 19's holdout with notebook 19's holdout budget and seed (comparable with BASE31 / LORA31)."""
    global BUDGET
    res_dir = RUN_ROOT / f"holdout_{name}"
    todo = [t for t in HO_RUN if t not in {r["instance_id"] for r in read_jsonl(res_dir / "task_results.jsonl")}]
    saved, BUDGET = BUDGET, dict(HOLDOUT_BUDGET)
    try:
        if todo:
            sub_ = make_sub(OUT / f"submissions/eval_{name}", MODEL_SLUG, adapter_dir, seed=1)
            stop_servers(); start_server(MODEL_DIR, TP, sub_)
            if not adapter_changes_output(8000):
                plog(f"!!! {name}: the adapter does not change the output (vLLM did not apply it)"); return None
            with Progress(res_dir / "task_results.jsonl", len(todo), f"holdout {name}", every=120):
                sh(agent_cmd(sub_, todo, res_dir, MODEL_DIR, MODEL_SLUG, timeout_s=EVAL_ARM_H * 3600), log=RUN_ROOT / f"holdout_{name}.log")
    finally:
        BUDGET = saved
        stop_servers()
    rr = {r["instance_id"]: r for r in read_jsonl(res_dir / "task_results.jsonl")}
    if not all(t in rr for t in HO):
        plog(f"!!! {name}: holdout incomplete ({sum(t in rr for t in HO)}/{len(HO)})"); return None
    per = {t: bool(rr[t].get("resolved")) for t in HO}
    k = sum(per.values()); lo, hi = sc.wilson(k, len(HO))
    plog(f"HOLDOUT {name}: {k}/{len(HO)} [{lo:.1%}, {hi:.1%}]")
    return {"resolved": k, "per_task": per}
def run_round():
  n = SAMPLE_ROUND
  if not TRAINING:
    plog("training rounds stopped earlier (or MAX_ROUNDS reached): this session only samples")
  elif left_h() < NEED_H or campaign_left_h() < NEED_H:
    plog(f"round skipped: {left_h():.1f} h left in the session, {campaign_left_h():.1f} h in the campaign (need {NEED_H:.1f})")
  elif BASE31_TRAIN is None:
    plog("!!! 31B unquantized training checkpoint not attached: no round")
  else:
    RD = OUT / f"rounds/r{n}"
    runs = ([B2S1] if B2S1 else []) + NB19_PARTS + sorted(p for p in PARTS.glob("p*") if (p / "task_results.jsonl").exists())
    _runs_file = RD / "runs.txt"; RD.mkdir(parents=True, exist_ok=True); _runs_file.write_text("\n".join(map(str, runs)) + "\n")
    sh(f"cd {GC} && {sys.executable} lora/build_rft_data.py --runs {' '.join(map(str, runs))} --profile {OUT / 'profile_nb20.csv'} "
       f"--out {RD / 'data'} --max-per-task {MAX_PER_TASK_TRAIN} --accurate-tools {GC / 'all_tasks.jsonl'} --order time "
       f"--exclude {GC / 'never_train.txt'} --dedupe-routes 1.0", log=RD / "build_rft_data.log")
    TRAIN_DATA = RD / "data/train.jsonl"
    n_traj = len(read_jsonl(TRAIN_DATA))
    plog(f"round {n}: {n_traj} training trajectories from {len(runs)} run folders")
    if n_traj:
        t0 = time.time()
        b = min(TRAIN31_MAX_H, left_h() - EVAL_ARM_H - 0.4)
        stop_servers()
        rc = sh(f"cd {GC}/lora && timeout -k 60 {int((b + 0.4) * 3600)} {sys.executable} train_lora.py --data {TRAIN_DATA} "
                f"--base {BASE31_TRAIN} --out {RD / 'lora'} --device-map auto --max-target 8000 --no-eval-start --seed {n} "
                f"--served-model google/{MODEL_SLUG} {DEPLOY_ARGS} --time-budget-h {b:.2f}", log=RD / "train_lora.log", env=TRAIN_ENV)
        kill_gpu_procs()
        ok = (RD / "lora/adapter/adapter_config.json").exists()
        plog(f"round {n}: training exit {rc} after {(time.time() - t0) / 3600:.2f} h; adapter {'saved' if ok else 'MISSING'}")
        res = eval_holdout(f"R{n}", RD / "lora/adapter") if ok else None
        if res is not None:
            rounds = STATE["rounds"] + [{"round": n, "adapter": str(RD / "lora/adapter"), "n_train": n_traj, **res}]
            better = res["resolved"] > (BEST["resolved"] or -1)
            new_rounds = sum("n_train" in r for r in rounds)
            stop = (not better) and new_rounds >= MIN_NEW_ROUNDS
            save_state(rounds=rounds, best_round=n if better else BEST["round"], training_stopped=stop)
            plog(f"round {n}: {res['resolved']} vs best r{BEST['round']} {BEST['resolved']} -> "
                 f"{'NEW BEST' if better else 'not better'}{'; training stops, sampling continues' if stop else ''}")
  mark_campaign()
  refresh_rounds()


# Train first: with no trained adapter yet (notebook 19's LORA31 never trained), round 1 trains on the verified trajectories
# already collected (notebook 01 B2 s1 + notebook 19 teacher passes) before any sampling, so this session samples with it.
if TRAINING and not any("n_train" in r for r in STATE["rounds"]) and not ROUND_BY_N[STATE["best_round"]]["adapter"]:
    plog("train first: no trained adapter yet -> round 1 on the trajectories collected so far")
    run_round()
'''

ROUND = r'''# STEP 6 - round on everything sampled so far (retrained from the base 31B)
if any(f"_r{SAMPLE_ROUND}_seed" in p.name for p in PARTS.glob("p*") if p.is_dir()):
    run_round()
else:
    plog(f"no new attempts for round {SAMPLE_ROUND}: no round this session")
'''

FINAL = r'''# STEP 7 - submission from the best round (deployment budget) and the summary
ROUND_BY_N = {r["round"]: r for r in STATE["rounds"]}
BEST = ROUND_BY_N[STATE["best_round"]]
FINAL = Path("/kaggle/working/submission")
shutil.rmtree(FINAL, ignore_errors=True)
shutil.copytree(make_sub(OUT / "submissions/final", MODEL_SLUG, Path(BEST["adapter"]) if BEST["adapter"] else None, budget=DEPLOY_BUDGET), FINAL)
_zip = Path(shutil.make_archive("/kaggle/working/submission", "zip", FINAL))
_att = read_jsonl(ATTEMPTS)
_by = defaultdict(list)
for a in _att:
    _by[a["instance_id"]].append(a)
lines = [f"# Notebook 20 summary ({now()})", "",
         f"- Sessions {STATE['sessions']}; campaign GPU hours used {STATE.get('campaign_h_done', 0):.1f} of {CAMPAIGN_H}; passes {STATE['passes']}",
         f"- NB20 attempts {len(_att)} (all saved in runs/NB20/parts); resolved {sum(a['resolved'] for a in _att)}; "
         f"tasks ever solved {sum(any(a['resolved'] for a in v) for v in _by.values())}/{len(POOL)}",
         "- Rounds (holdout, same budget and seed as notebook 19):"]
_rt = [a for a in _att if a.get("prior_failure")]
for _lab, _g in (("with retry note", [a for a in _rt if a["noted"]]), ("without (same kind of retry)", [a for a in _rt if not a["noted"]])):
    lines.insert(-1, f"- Retries of failed-before tasks {_lab}: {sum(a['resolved'] for a in _g)}/{len(_g)} resolved")
for r in STATE["rounds"]:
    lo, hi = sc.wilson(r["resolved"] or 0, len(HO))
    lines.append(f"  - r{r['round']}: {r['resolved']}/{len(HO)} [{lo:.1%}, {hi:.1%}]" + (f"; trained on {r['n_train']}" if "n_train" in r else "")
                 + (" (base, no adapter)" if r["adapter"] is None else ""))
lines += ["", f"**SUBMISSION: round {BEST['round']}**" + (" (base model, no adapter)" if BEST["adapter"] is None else ""),
          f"- training stopped: {STATE['training_stopped']}; deployment budget {DEPLOY_BUDGET}",
          "- Noise bar: two runs of one condition disagree on about 3 of 23 tasks; a smaller gap is not evidence."]
(OUT / "SUMMARY_NB20.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
'''

ZIP = r"""# STEP 8 - zip the run folder (Output tab): every attempt, every round's data and adapter, the state for the next session
stop_servers()
mark_campaign()
zp = Path("/kaggle/working/graphcomp_nb20.zip")
if zp.exists():
    zp.unlink()
shutil.make_archive(str(zp.with_suffix("")), "zip", OUT)
plog(f"zip: {zp} {zp.stat().st_size / 1e6:.1f} MB; campaign {STATE.get('campaign_h_done')} h of {CAMPAIGN_H}; attach this output to the next version")
"""

# ---- assemble
cfg = sub1(src(3), 'VARIANT_TAG = "NB19"', 'VARIANT_TAG = "NB20"')
run_arm = "%%writefile /kaggle/working/gc/run_arm.py\n" + (HERE / "run_arm.py").read_text()
scripts = [code("# STEP 1 - script folders\n(GC / 'lora').mkdir(exist_ok=True)\n")]
for rel, dest in (("lora/build_rft_data.py", "lora/build_rft_data.py"), ("lora/train_lora.py", "lora/train_lora.py"),
                  ("tool_trace.py", "tool_trace.py")):
    scripts.append(code(f"%%writefile /kaggle/working/gc/{dest}\n" + (HERE / rel).read_text()))
cells = ([md(HEADER), C[2], code(cfg), code(CONFIG), C[4], C[5], code(RESUME)] + C[7:13] + [code(run_arm)] + C[14:21]
         + scripts + [code(TESTDEPS), code(POOL), C[25], code(SAMPLER_HEAD + defs_from_29() + "\n" + SAMPLER), code(ROUND_DEFS), code(SAMPLER_LOOP), code(ROUND), code(FINAL), code(ZIP)])
for c_ in cells:
    if c_["cell_type"] == "code":
        c_["outputs"], c_["execution_count"] = [], None
nb["cells"] = cells
out = U / "notebooks/20_iterative_rft_campaign.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print(out, len(cells), "cells", out.stat().st_size // 1000, "KB")

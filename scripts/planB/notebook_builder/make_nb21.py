"""Build notebook 21: 31B LoRA on the SFT mix (build_sft_mix.py) -> BASE31 vs LORA31 holdout -> budget-safe submission.

Derived from the built notebook 19 (run make_nb19.py first), with notebook 20's script versions (its 31B training fixes): same setup, resume, validity, holdout set, vLLM helpers, deployed
agent, holdout comparison, submission choice and zip. Removed: teacher generation (step 4), merging teacher runs (step 5) and the
students. Step 5 now reads the attached SFT mix; step 6 trains only the 31B, with the training window stepped down on OOM.
    python3 scripts/planB/notebook_builder/make_nb21.py
"""
import json
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

HEADER = """# 21 — 31B LoRA on the SFT mix → holdout comparison (BASE31 vs LORA31) → budget-safe submission

Offline (Internet off), GPU L4 × 4. Built by `scripts/planB/notebook_builder/make_nb21.py` from notebook 19.

**What it does, in order (each stage is skipped if an attached earlier version of this notebook already finished it):**
1. **Training data** = the attached `graphcomp-sft-mix` dataset (`build_sft_mix.py`): our verified 31B RFT trajectories plus
   public test-verified Qwen3-Coder-480B trajectories (CoderForge-Preview, nebius SWE-rebench-OpenHands) replayed into this
   harness's tools (`convert_trajs.py`), exact and near-duplicate trajectories removed. Dev and holdout tasks may be trained on
   (`TRAIN_ON_HOLDOUT`, user, Oct 15); holdout tasks that were trained on are listed in the summary, because on those the
   BASE31 vs LORA31 comparison favours the LoRA. From `train_fit.jsonl` (whole trajectory ≤ 32k tokens, the served context) the notebook takes our own trajectories
   first, then one trajectory per new task, then second ones, up to `MAX_TRAIN_TRAJ` (more than one session can train).
2. **31B LoRA** (`train_lora.py`, QLoRA, 4-bit model split over the 4 GPUs, time budget). The training window starts at 32,768
   tokens and steps down (24,576, then 18,432) only if a run fails out of memory.
3. **Holdout comparison**, same deployed agent: `BASE31` (no adapter) and `LORA31`, notebook 19's holdout (official 24 + one task
   per missing category; one requests and one httpx task as a budget check only).
4. **Submission** = the arm with more scored holdout tasks resolved (ties → BASE31), deployment budget (≈ 5 min / 30 tool calls
   per task: 120 tasks fit in 12 h).

**Time:** setup ≈ 0.5 h · training ≤ `TRAIN31_MAX_H` (whatever is left after the holdout + 0.4 h reserve) · holdout
≈ 0.9 h (both 31B arms, one server). A stage that would not fit stops cleanly; attach this output to the next version to resume.

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
c = sub1(c, "TRAIN31_MAX_H, TRAIN_STUDENT_MAX_H, EVAL_ARM_H = 2.6, 0.8, 0.9",
         "TRAIN31_MAX_H, TRAIN_STUDENT_MAX_H, EVAL_ARM_H = 9.0, 0, 0.9   # training gets what the holdout arms leave\n"
         "TRAIN_MAX_LENS = [32768, 24576, 18432]   # training window (tokens); the next one only after an out-of-memory failure\n"
         "MAX_TRAIN_TRAJ = 4000         # more than one session trains; picked own-data first, then one per task, then seconds\n"
         "VAL_FRAC = 0.02               # of tasks, for the validation loss (2% of thousands is enough and keeps eval short)\n"
         "TRAIN_ON_HOLDOUT = True       # user, Oct 15: holdout tasks may be trained on; the summary lists them (comparison bias)")
i0 = c.index("STUDENTS = {'STUDENT_12B'")
i1 = c.index("EVAL_STUDENTS = False")
c = c[:i0] + "STUDENTS = {}                 # no students in this notebook\n" + c[i1:]
assert "NB21" in c and "STUDENTS = {}" in c and "TRAIN_MAX_LENS" in c

RESUME = src(6).replace("NB19", "NB21")

DATA = r'''# STEP 5 - training data: the attached SFT mix, minus the holdout; own trajectories first, then one per task, ...
import random
MIX = next((Path(dp) for dp, dns, fns in walk(INPUT, maxdepth=7) if "train_fit.jsonl" in fns and "manifest.json" in fns), None)
plog(f"SFT mix: {MIX or 'NOT FOUND: attach the graphcomp-sft-mix dataset'}")
if MIX:
    plog("mix manifest: " + json.dumps({k: v for k, v in json.loads((MIX / 'manifest.json').read_text()).items() if k != 'inputs'})[:1500])
DEPLOY_PROMPT = GC / "deploy_system.md"
DEPLOY_PROMPT.write_text((Path(SAMPLE_SUB) / "prompts/system.md").read_text() + PROMPT_ADDENDUM)
_never = set() if TRAIN_ON_HOLDOUT else set(HO_RUN) | set(OFFICIAL_HOLDOUT)   # dev-slice tasks are always allowed (user, Oct 15)
_rows = [json.loads(l) for l in open(MIX / "train_fit.jsonl")] if MIX else []
_rows = [r for r in _rows if r["instance_id"].split("@")[0] not in _never and r.get("est_tokens", 0) <= max(TRAIN_MAX_LENS)]
random.Random(0).shuffle(_rows)
_own = [r for r in _rows if r.get("mix_source") == "rft"]
_by_task = {}
for r in _rows:
    if r.get("mix_source") != "rft":
        _by_task.setdefault(r["instance_id"].split("@")[0], []).append(r)
_pick, _round = list(_own), 0
while len(_pick) < MAX_TRAIN_TRAJ and any(len(v) > _round for v in _by_task.values()):
    _pick += [v[_round] for v in _by_task.values() if len(v) > _round][: MAX_TRAIN_TRAJ - len(_pick)]
    _round += 1
TRAIN_DATA = OUT / "sft" / "train.jsonl"
TRAIN_DATA.parent.mkdir(parents=True, exist_ok=True)
with open(TRAIN_DATA, "w") as f:
    for r in _pick:
        f.write(json.dumps(r) + "\n")
n_traj = len(_pick)
assert not ({r["instance_id"].split("@")[0] for r in _pick} & _never), "holdout task in the training data"
TRAINED_HOLDOUT = sorted({r["instance_id"].split("@")[0] for r in _pick} & set(HO_RUN))
if TRAINED_HOLDOUT:
    plog(f"NOTE: {len(TRAINED_HOLDOUT)} holdout tasks are in the training data (LORA31 has seen them): {TRAINED_HOLDOUT}")
plog(f"training data: {n_traj} trajectories ({len(_own)} own, {n_traj - len(_own)} external over "
     f"{len({r['instance_id'].split('@')[0] for r in _pick})} tasks, {_round} round(s)) of {len(_rows)} eligible -> {TRAIN_DATA}")
del _rows, _by_task
'''

# ------------------------------------------------------------------ step 6: only the 31B, training window stepped down on OOM
T = src(28)
T = sub1(T, "# STEP 6 - train the LoRAs: 31B split over all 4 GPUs, then the two students side by side (E4B on GPUs 0-1, 12B on 2-3)",
         "# STEP 6 - train the 31B LoRA (4-bit model split over all 4 GPUs); training window stepped down on out-of-memory")
old31 = T[T.index("if not OK31 and BASE31_TRAIN is not None"):T.index("# --- students in parallel")]
T = T.replace(old31, '''for L in TRAIN_MAX_LENS:
    if OK31 or BASE31_TRAIN is None:
        break
    b = _budget(TRAIN31_MAX_H, f"lora31 (window {L})")
    if not b:
        break
    shutil.rmtree(OUT31, ignore_errors=True); t0 = time.time()
    _log = OUT / f"provenance/train_lora31_len{L}.log"
    rc = sh(f"cd {GC}/lora && timeout -k 60 {int((b + 0.4) * 3600)} {sys.executable} train_lora.py --data {TRAIN_DATA} "
            f"--base {BASE31_TRAIN} --out {OUT31} --device-map auto --max-len {L} --max-target 8000 --val-frac {VAL_FRAC} "
            f"--no-eval-start --served-model google/{MODEL_SLUG} {DEPLOY_ARGS} --time-budget-h {b:.2f}", log=_log, env=TRAIN_ENV)
    kill_gpu_procs()
    OK31 = _report(f"lora31 (window {L})", OUT31, rc, t0)
    _tail = _log.read_text(errors="replace")[-30000:] if _log.exists() else ""
    if OK31 or not re.search(r"out of memory|OutOfMemoryError", _tail, re.I):
        break
    plog(f"lora31: out of memory with a {L}-token window; trying the next one")
save_state(lora31_window=L if OK31 else None)
''')
assert "TRAIN_MAX_LENS" in T

PICK = src(30).replace("Notebook 19", "Notebook 21")
PICK = sub1(PICK, "f\"- Teacher runs complete: {STATE.get('teacher_complete')}; training trajectories: {n_traj}\",",
            "f\"- Training trajectories: {n_traj} (SFT mix {MIX}); 31B training window {STATE.get('lora31_window')}\",\n"
            "         f\"- Holdout tasks trained on (comparison favours LORA31 there): {TRAINED_HOLDOUT or 'none'}\",")

# scripts: this branch's copies, which carry notebook 20's 31B fixes (train_lora.py: logits_to_keep as a list so accelerate does
# not move it to GPU 0 (the nb19 31B crash), checkpoint integrity preflight, compaction-aware prompts; run_arm.py: src/ shim)
SCRIPTS = HERE.parent
assert first(13).startswith("%%writefile /kaggle/working/gc/run_arm.py") and first(22).endswith("lora/build_rft_data.py") \
    and first(23).endswith("lora/train_lora.py")
run_arm = code("%%writefile /kaggle/working/gc/run_arm.py\n" + (SCRIPTS / "run_arm.py").read_text())
lora = [code(f"%%writefile /kaggle/working/gc/{rel}\n" + (SCRIPTS / rel).read_text())
        for rel in ("lora/build_rft_data.py", "lora/train_lora.py", "tool_trace.py")]
cells = ([md(HEADER), md(src(1) + DATA_NOTES), C[2], code(c)] + C[4:6] + [code(RESUME)] + C[7:13] + [run_arm] + C[14:22] + lora
         + C[24:26] + [code(DATA), code(T), C[29], code(PICK), C[31]])
for cell in cells:
    if cell["cell_type"] == "code":
        cell["source"] = "".join(cell["source"]).replace("NB19_STATE", "NB21_STATE").splitlines(keepends=True)
        cell["outputs"], cell["execution_count"] = [], None
nb["cells"] = cells
out = U / "notebooks/21_sft_mix_lora31_holdout_submission.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out} ({len(cells)} cells)")

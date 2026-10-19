import base64
import json
import os
from pathlib import Path

V = json.loads(os.environ.get("NB_VARIANT", "{}"))
V = {"file": "graphcomp_arm.ipynb", "tag": "B2", "arm": "B2", "seed": 1, "scope": "all", "validity": True,
     "addendum": "", "max_out": 16384, "think": 4096, "model": "31b", "title": "", "job": "", **V}

UP = Path("/mnt/user-data/uploads/graphComp")
OUTD = Path("/mnt/user-data/outputs/graphComp")
SA = ["".join(c["source"]) for c in json.load(open(UP / "graphcomp_b0_sessionA.ipynb"))["cells"]]
B1 = ["".join(c["source"]) for c in json.load(open(OUTD / "graphcomp_b1_gold129.ipynb"))["cells"]]
helpers, install = SA[2], SA[5]
assert "def sh(" in helpers and "WHEELHOUSE_DIR" in install
b1cfg = next(s for s in B1 if "progress logging (every cell" in s)
LOG = b1cfg[b1cfg.index("# ============================== progress logging"):]
LOG = LOG.replace('_cell.update(label="STEP 0/5 - setup: config and progress logging", t=_T0)\nplog(">>> START  STEP 0/5 - setup: config and progress logging (this cell)")',
                  '_cell.update(label="STEP 0/7 - setup: config and progress logging", t=_T0)\nplog(">>> START  STEP 0/7 - setup: config and progress logging (this cell)")')
assert "STEP 0/7" in LOG and "class Progress" in LOG
PB = Path(__file__).with_name("progress_block.py").read_text()
i0 = LOG.index("class Progress"); i1 = LOG.index("_cell.update(label=\"STEP 0/7")
LOG = LOG[:i0] + PB + LOG[i1:]

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md("""
# graphComp Plan B — validity check + agent arm on Kaggle, one notebook (B2; B3 with `ARM = "B3"`), resumable

One GPU session does both jobs, back to back, in the environment the agent runs in:

1. **Validity (first session only, about 40 min, CPU work).** Which tasks can be resolved at all in this sandbox:
   one gold-patch pass over the 115 fastapi/rich tasks, then an empty-patch pass over the tasks the gold patch resolves.
   requests and httpx (14 tasks) are not run: their tests import the notebook's own installed copy, not the task's code
   (B1 import check, Oct 12, identical for every task of each repo), so no patch can change their result. Writes
   `validity/` (validity per task, profile with `sub_valid`, dev and holdout slices, the full package list).
2. **The agent arm** over all 129 tasks, as many as fit in the session; re-run with the previous version's output
   attached until done. Validity is then read from the attached output, not re-run, and the package list must match it
   exactly (the CPU B1 run showed a different image changes test results: `rich_4070`).

**Test dependencies (Oct 12 decision).** In subprocess mode the harness does not install test dependencies
(`swegemma/harness/container_setup.py` line 529), and the Kaggle image lacks `inline_snapshot` and `dirty_equals`, so
34 fastapi tasks could not even collect their tests. This notebook installs both into the notebook's Python from the
competition's own `wheels/` folder (offline) before anything runs, and stops if they are not there or if installing them
changes any package the sandboxes already use. Every arm runs with them.

Why the validity protocol is this small (Oct 12): the import check is dropped (per-repo constant); one repeat instead of
two (the second repeat found 1 flaky task in 129 at the cost of about an hour; seed repeats of the arms expose flakiness
anyway); the empty-patch pass only runs where the gold patch passes (a gold failure already excludes the task). On the
CPU B1 data this protocol reproduces every validity decision except that one flaky task.

| Step | What |
| --- | --- |
| 0 | setup: install, test dependencies, inputs, package list, scripts, submission |
| 1 | validity (skipped when an earlier version's `validity/` is attached) |
| 2 | task order (dev slice, non-slice tasks, holdout) and resume |
| 3 | serve `gemma-4-31b-it-qat-w4a16-ct` exactly as session A (0.80 / 32,768, tp = 4, sample LoRAs mounted) |
| 4 | run remaining tasks in chunks, concurrency 4, stopping before the session budget |
| 5 | merge parts into one results directory |
| 6 | scorecard 0.3.2 and summary (adoption, resolution on `sub_valid` tasks, end reasons, timing) |
| 7 | stop the server, zip |

Fixed conditions for every arm: sample prompt verbatim, sub-agent and adapter removed, temperature 0.2, top_p 0.95,
**max_output_tokens 16384**, thinking 4096, 50 tool calls / 30 min / 100 turns / 300 s command timeout, compaction
interval 15, **concurrency 4**, test dependencies installed.

**Setup:** Copy & Edit the competition's "Getting Started" notebook (Python 3.12), import this notebook. Inputs: competition
data, `metric/gemma-4-developer-agent-wheelhouse`, model `google/gemma-4` variant `gemma-4-31b-it-qat-w4a16-ct` v2;
for a resumed run or another arm, **this notebook's previous version output**. Do **not** attach the old CPU B1 output
(different image, no test dependencies; it is ignored if attached). Optional: the `graphcomp-planb` dataset for the full
scorecard. Accelerator GPU L4 × 4. **Save Version → Save & Run All.**
""")


code(r'''
# ============================== CONFIG (edit here) ==============================
from pathlib import Path

VARIANT_TAG = "@@TAG@@"           # names the run directory: runs/<VARIANT_TAG>_s<SEED>
ARM = "@@ARM@@"                    # tools: "B2" = all nine; "B3" = the three graph tools removed
SEED = @@SEED@@                      # Plan B seeds 1, 2, 3 (identical across paired arms)
TASK_SCOPE = "@@SCOPE@@"            # "all" = dev slice, then non-slice tasks, then holdout; "dev" = dev slice only
MAX_OUTPUT_TOKENS = @@MAXOUT@@        # sample value 16384 (Oct 12 decision); only the output-reserve ablation changes it
THINKING_BUDGET = @@THINK@@          # sample value 4096; only the thinking ablation changes it
PROMPT_ADDENDUM = @@ADDENDUM@@
CONCURRENCY = 4               # fixed for every arm (the 30-min per-task budget makes it part of the condition)
CHUNK = 8                     # tasks per run_arm call; the time guard is checked between chunks
SESSION_BUDGET_H = 11.0       # stop starting chunks when the next one might not finish before this (Kaggle limit is 12 h)
RETRY_INFRA_ERRORS = True     # rerun tasks whose record shows a serving / connection failure, not a model outcome
BUDGET = dict(max_tool_calls=50, max_time_minutes=30, max_turns=100, timeout_seconds=300)
COMPACTION_INTERVAL = 15      # as the sample notebook
SERVE = dict(util=0.80, max_model_len=32768)   # session A: served on the first rung; any other value is a different condition
MODEL_VARIANT = "@@MODEL@@"            # "31b" in every Plan B arm; "12b" / "e4b" / "e2b" only in the size sweep (Oct 13 announcement)
MODEL_SPECS = {   # tp: all three smaller variants fit one L4 at 4 bits and are served with tp 1 (12B at tp 4 failed, Oct 14)
    "31b": dict(slug="gemma-4-31b-it-qat-w4a16-ct", tp=4, loras=True),
    "12b": dict(slug="gemma-4-12b-it-qat-w4a16-ct", tp=1, loras=False),   # Oct 14: tp 4 failed (vLLM Transformers fallback, hidden_size 3840 vs 256)
    "e4b": dict(slug="gemma-4-e4b-it-qat-w4a16-ct", tp=1, loras=False),
    "e2b": dict(slug="gemma-4-e2b-it-qat-w4a16-ct", tp=1, loras=False),
}
MODEL_SLUG = MODEL_SPECS[MODEL_VARIANT]["slug"]
MODEL_DIR = f"/kaggle/input/models/google/gemma-4/other/{MODEL_SLUG}/2"   # any attached version of this slug is found below
MOUNT_SAMPLE_LORAS = MODEL_SPECS[MODEL_VARIANT]["loras"]   # the sample LoRAs are 31B adapters; the agent declares none either way
SERVER_STARTUP_TIMEOUT = 1800
RUN_VALIDITY = @@VALIDITY@@          # True only in notebook 01; every other notebook reads validity from 01's attached output
SHADOWED_REPOS = ("requests", "httpx")   # tests import the host copy (B1 import check, Oct 12): not scorable here
TEST_WHEELS = @@TEST_WHEELS@@   # {filename: (sha256, base64)}: inline-snapshot 0.36.0, executing 2.3.0, dirty-equals 0.11
SLICE_SEED = 20261012        # dev / holdout slices (Plan B section 2); fixed
# OFFICIAL slices (fixed Oct 12, scripts/planB/official_slices; the VM runner uses the same files)
OFFICIAL_DEV = ['fastapi_11194', 'fastapi_11355', 'fastapi_12942', 'fastapi_14258', 'fastapi_14266', 'fastapi_14356', 'fastapi_14430', 'fastapi_14485', 'fastapi_14786', 'fastapi_14794', 'fastapi_14953', 'fastapi_14962', 'fastapi_15800', 'fastapi_9425', 'rich_3052', 'rich_3468', 'rich_3480', 'rich_3518', 'rich_3521', 'rich_3535', 'rich_3905', 'rich_3935', 'rich_3944', 'rich_4075']
OFFICIAL_HOLDOUT = ['fastapi_14099', 'fastapi_14297', 'fastapi_14303', 'fastapi_14360', 'fastapi_14361', 'fastapi_14372', 'fastapi_14463', 'fastapi_14482', 'fastapi_14512', 'fastapi_14616', 'fastapi_14873', 'fastapi_14986', 'fastapi_15589', 'fastapi_9753', 'rich_2725', 'rich_3061', 'rich_3180', 'rich_3278', 'rich_3472', 'rich_3675', 'rich_3777', 'rich_3882', 'rich_3938', 'rich_4076']
ORDER_SEED = 20261012        # task order only if validity is unavailable (shuffled once, fixed)
# Session A host packages (B0_sessionA_results.md section 1); installing the test deps must not change any of them.
SESSION_A_HOST = {"pydantic": "2.12.3", "typing-inspection": "0.4.2", "fastapi": "0.136.1", "starlette": "0.52.1",
                  "requests": "2.32.4", "rich": "13.9.4", "swegemma": "0.2.11", "adk-submission": "0.2.13",
                  "adk-eval-core": "0.1.0", "google-adk": "1.36.1"}

OUT = Path("/kaggle/working/graphcomp_arm")
GC = Path("/kaggle/working/gc")
RUN_NAME = f"{VARIANT_TAG}_s{SEED}"
RUN_ROOT = OUT / "runs" / RUN_NAME
PARTS = RUN_ROOT / "parts"
OUT.mkdir(parents=True, exist_ok=True)
GC.mkdir(parents=True, exist_ok=True)
for d in ("provenance", "submissions", "scorecard", "validity"):
    (OUT / d).mkdir(exist_ok=True)
PARTS.mkdir(parents=True, exist_ok=True)
print("output folder:", OUT, "| run:", RUN_NAME)
''' + LOG + r'''
# fail fast, before the 4-minute install, if the session is not the 4 x L4 / Python 3.12 image
import sys as _sys
_gpus = [l for l in _sp.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.splitlines() if l.strip()]
plog(f"session check: Python {_sys.version.split()[0]}, {len(_gpus)} GPUs: {[g[:40] for g in _gpus]}")
if _sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python {_sys.version.split()[0]}: the wheelhouse needs 3.12 (pinned Getting Started image).")
if len(_gpus) != 4:
    raise SystemExit(f"{len(_gpus)} GPUs visible; these runs need GPU L4 x 4 (tp=4). Change the accelerator and run again.")
''')

code(helpers)

code(r'''
# ============================== locate inputs ==============================
INPUT = Path("/kaggle/input")
found = {"tasks": [], "sample_submission": [], "wheelhouse": [], "model_configs": [], "validity": [], "old_b1": [],
         "oracle_pack": [], "prior_parts": []}
for dp, dns, fns in walk(INPUT, maxdepth=9, skip=("snapshots", "wheels", "embeddings", "graphs", "__pycache__", ".git", "traces", "logs", "test_outputs", "patches")):
    if "tasks.jsonl" in fns:
        found["tasks"].append(dp / "tasks.jsonl")
    if "agent.yaml" in fns and "sample" in str(dp).lower():
        found["sample_submission"].append(dp)
    if "wheelhouse" in dp.name.lower():
        found["wheelhouse"].append(dp)
    if "config.json" in fns and MODEL_SLUG in str(dp).lower():
        found["model_configs"].append(dp)
    if "validity_env.json" in fns:
        found["validity"].append(dp)          # written by step 1 of this notebook
    if "b1_env.json" in fns:
        found["old_b1"].append(dp)            # the CPU B1 notebook's output: different image, not used
    if "oracle_pack.json.gz" in fns:
        found["oracle_pack"].append(dp / "oracle_pack.json.gz")
    elif "oracle_pack.json" in fns:          # Kaggle may unpack the .gz on dataset upload; re-packed below
        found["oracle_pack"].append(dp / "oracle_pack.json")
    if dp.parent.name == "parts" and dp.parent.parent.name == RUN_NAME and "task_results.jsonl" in fns:
        found["prior_parts"].append(dp)
for k, v in found.items():
    print(f"{k:18s}", len(v), [str(x) for x in v][:4])

if found["old_b1"]:
    plog("NOTE: an old B1 output is attached (b1_env.json). It is ignored: it was measured on the CPU image without the "
         "test dependencies. Validity comes from step 1 of this notebook.")
TASKS_JSONL = found["tasks"][0]
DATA_ROOT = TASKS_JSONL.parent
SNAPSHOTS = DATA_ROOT / "snapshots"
SAMPLE_SUB = found["sample_submission"][0] if found["sample_submission"] else None
if (MODEL_DIR is None or not Path(MODEL_DIR).exists()) and found["model_configs"]:
    MODEL_DIR = str(sorted(found["model_configs"], key=lambda p: len(str(p)))[0])
VALIDITY_ROOT = found["validity"][0] if found["validity"] else None
B1_MODE = VALIDITY_ROOT is not None and (VALIDITY_ROOT / "slices" / "dev_slice.txt").exists()
if B1_MODE:
    DEV = [l.strip() for l in open(VALIDITY_ROOT / "slices/dev_slice.txt") if l.strip()]
    HOLDOUT = [l.strip() for l in open(VALIDITY_ROOT / "slices/holdout_slice.txt") if l.strip()]
    B1_PROFILE = VALIDITY_ROOT / "slices/profile.csv"
    B1_ENV = json.load(open(VALIDITY_ROOT / "validity_env.json"))
else:
    DEV, HOLDOUT, B1_PROFILE, B1_ENV = [], [], None, None
assert TASKS_JSONL.exists() and SNAPSHOTS.exists(), "competition data not attached?"
assert SAMPLE_SUB is not None, "sample submission not found"
assert MODEL_DIR and Path(MODEL_DIR).exists(), f"model not attached (google/gemma-4, {MODEL_SLUG})"
assert MODEL_SLUG in str(MODEL_DIR).lower(), f"MODEL_DIR {MODEL_DIR} is not {MODEL_SLUG}: attach only that model"
assert (not B1_MODE) or (len(DEV) == 24 and len(HOLDOUT) == 24 and not set(DEV) & set(HOLDOUT)), "slice files look wrong"
tasks_by_id = {json.loads(l)["instance_id"]: json.loads(l) for l in open(TASKS_JSONL) if l.strip()}
ALL_IDS = sorted(tasks_by_id)
print(f"\n{len(ALL_IDS)} tasks | validity attached: {B1_MODE} ({VALIDITY_ROOT}) | dev {len(DEV)} | holdout {len(HOLDOUT)}")
print("MODEL_DIR =", MODEL_DIR, "| SAMPLE_SUB =", SAMPLE_SUB)
''')

md("""
## Install the harness and serving stack
The competition sample notebook's install cell, verbatim (as in sessions A and B1).
""")
code(install)

code(r'''
# ============================== test dependencies (embedded wheels; offline) ==============================
# The competition's wheels/ folder does not contain them (session of Oct 12, 13:58 UTC), so the three pure-Python wheels are
# embedded here, pinned to the versions Plan A's per-task fastapi environments used (planA_results/vm-a2/.../env_freeze.txt:
# inline-snapshot 0.36.0 and executing 2.3.0 in 67/67; dirty-equals 0.11 in 24/67, 0.9.0 in 43/67, one version for all).
# Installed with --no-index --no-deps: nothing is downloaded and nothing already installed may change.
import base64 as _b64, hashlib as _hl
def pip_freeze():
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    d = {}
    for l in out.splitlines():
        l = l.strip()
        if not l or l.startswith("#"):
            continue
        name, _, v = l.partition("==") if "==" in l else l.partition(" @ ")
        d[re.sub(r"[-_.]+", "-", name).lower()] = v.strip()
    return d
TW = GC / "test_wheels"
TW.mkdir(parents=True, exist_ok=True)
for fn, (sha, b64) in TEST_WHEELS.items():
    data = _b64.b64decode(b64)
    got = _hl.sha256(data).hexdigest()
    if got != sha:
        raise SystemExit(f"embedded wheel {fn}: sha256 {got} != {sha}")
    (TW / fn).write_bytes(data)
plog(f"test wheels written and verified: {sorted(TEST_WHEELS)}")
before = pip_freeze()
rc = sh(f"{sys.executable} -m pip install --no-index --no-deps " + " ".join(str(TW / f) for f in sorted(TEST_WHEELS)),
        log=OUT / "provenance/test_deps_install.log")
after = pip_freeze()
TEST_DEPS_CHANGES = {k: (before.get(k), after.get(k)) for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)}
plog(f"packages added or changed by the test-dependency install: {TEST_DEPS_CHANGES}")
(OUT / "provenance/test_deps_changes.json").write_text(json.dumps(TEST_DEPS_CHANGES, indent=1))
chk = subprocess.run([sys.executable, "-c", "import inline_snapshot, dirty_equals, executing; from inline_snapshot import snapshot; "
                      "print('test deps import ok:', inline_snapshot.__version__ if hasattr(inline_snapshot, '__version__') else '', "
                      "dirty_equals.__version__, executing.__version__)"], capture_output=True, text=True)
print(chk.stdout or chk.stderr[-1500:])
touched = {k: v for k, v in TEST_DEPS_CHANGES.items() if v[0] is not None}
if rc or chk.returncode or touched:
    raise SystemExit(f"Test-dependency install failed or changed packages that were already installed: {touched or rc or chk.stderr[-500:]}. "
                     "Nothing has run yet; see provenance/test_deps_install.log.")
''')


code(r'''
r = subprocess.run([sys.executable, "-c", "import swegemma, adk_submission, adk_eval_core, google.adk, vllm; print('stack ok')"],
                   capture_output=True, text=True)
print(r.stdout or r.stderr[-1500:])
''')

code(r'''
# ============================== provenance: GPUs, versions, full package list ==============================
import importlib.metadata as md
def ver(p):
    try:
        return md.version(p)
    except Exception:
        return None
FREEZE = pip_freeze()
env_info = {
    "time": now(), "python": platform.python_version(), "nproc": os.cpu_count(),
    "nvidia_smi": subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
                                 shell=True, capture_output=True, text=True).stdout.strip(),
    "versions": {p: ver(p) for p in list(SESSION_A_HOST) + ["vllm", "torch", "pygments", "ipython", "pytest", "inline-snapshot", "dirty-equals"]},
    "test_deps_changes": TEST_DEPS_CHANGES,
    "config": {"ARM": ARM, "SEED": SEED, "TASK_SCOPE": TASK_SCOPE, "CONCURRENCY": CONCURRENCY, "CHUNK": CHUNK,
               "BUDGET": BUDGET, "COMPACTION_INTERVAL": COMPACTION_INTERVAL, "SERVE": SERVE, "SESSION_BUDGET_H": SESSION_BUDGET_H},
}
N_GPUS = len([l for l in env_info["nvidia_smi"].splitlines() if l.strip()])
TP = min(MODEL_SPECS[MODEL_VARIANT]["tp"], N_GPUS)
env_info["tensor_parallel"] = TP
env_info["model"] = {"variant": MODEL_VARIANT, "slug": MODEL_SLUG, "dir": str(MODEL_DIR), "sample_loras_mounted": MOUNT_SAMPLE_LORAS}
(OUT / f"provenance/env_{RUN_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M')}.json").write_text(json.dumps(env_info, indent=1))
(OUT / "provenance/pip_freeze.json").write_text(json.dumps(FREEZE, indent=1, sort_keys=True))
print(json.dumps(env_info, indent=1))
if N_GPUS != 4:
    raise SystemExit(f"{N_GPUS} GPUs visible; Plan B Kaggle runs use 4 x L4 (tp=4). Set the accelerator to GPU L4 x4.")
diff_a = {p: (v, env_info["versions"].get(p)) for p, v in SESSION_A_HOST.items() if env_info["versions"].get(p) != v}
print("\nsandbox packages vs session A:", "identical" if not diff_a else f"DIFFERS {diff_a}")
if diff_a:
    raise SystemExit("The sandboxes import these packages; they must equal session A's. Use the same Kaggle image.")
import hashlib as _hlib
STACK_ID = _hlib.sha256("".join(str(FREEZE.get(k)) for k in ("vllm", "swegemma", "adk-submission", "adk-eval-core")).encode()).hexdigest()[:10]
STACK_CHANGED, _rebuilt = False, {}
if B1_MODE:
    ref = B1_ENV["freeze"]
    diff_v = {k: (ref.get(k), FREEZE.get(k)) for k in sorted(set(ref) | set(FREEZE)) if ref.get(k) != FREEZE.get(k)}
    print("full package list vs the validity session:", "identical" if not diff_v else f"DIFFERS {diff_v}")
    # Oct 13: the competition wheelhouse was re-uploaded with the SAME versions but rebuilt files (vllm, swegemma,
    # adk-submission, adk-eval-core), and old dataset versions cannot be selected. Only that kind of difference is allowed:
    # same package, same version, both installed from the wheelhouse. It makes this run a different stack build: it pairs only
    # with a B2 run on the same build (notebook 14), and validity is re-checked on the slice tasks before anything is served.
    def _whl(u):
        m = re.search(r"/([^/]+?)-([^-/]+)-[^/]*\.whl", u or "")
        return (m.group(1).lower().replace("_", "-"), m.group(2)) if m else None
    _rebuilt = {k: v for k, v in diff_v.items() if v[0] and v[1] and "/tmp/wheelhouse/" in v[0] and "/tmp/wheelhouse/" in v[1]
                and _whl(v[0]) is not None and _whl(v[0]) == _whl(v[1])}
    _other = {k: v for k, v in diff_v.items() if k not in _rebuilt}
    if _other:
        raise SystemExit("This session's packages differ from the session that measured validity, so validity would not "
                         f"carry over: {sorted(_other)}. Use the same Kaggle image (same Getting Started copy, same inputs).")
    STACK_CHANGED = bool(_rebuilt)
    if STACK_CHANGED:
        plog(f"!!! stack build {STACK_ID} differs from the validity session (same versions, rebuilt files): {sorted(_rebuilt)}. "
             "Pair this run only with B2 on the same build (notebook 14); validity is re-checked on the slice tasks first.")
(OUT / "provenance/stack.json").write_text(json.dumps({"stack_id": STACK_ID, "changed_vs_validity_session": STACK_CHANGED,
    "rebuilt": {k: list(v) for k, v in _rebuilt.items()}, "wheels": {k: FREEZE.get(k) for k in ("vllm", "swegemma", "adk-submission", "adk-eval-core")}}, indent=1))
plog(f"stack build {STACK_ID}" + (" (rebuilt wheelhouse)" if STACK_CHANGED else " (same as the validity session)" if B1_MODE else ""))
''')


md("""
## Scripts
`run_arm.py` (session A, unchanged) and `scorecard.py` 0.3.2 (v7 = TQS port, length check, `end_reason`), the same
files as `scripts/planB/`. The oracle pack is NOT embedded (Kaggle limits notebook source to 1 MB): the `graphcomp-planb`
dataset **must be attached**. The next cell stops the notebook (before anything runs on the GPU) if it is missing or lacks a task.
""")
code("%%writefile /kaggle/working/gc/run_arm.py\n" + (UP / "scripts/planB/run_arm.py").read_text())
code("%%writefile /kaggle/working/gc/scorecard.py\n" + (OUTD / "scripts/planB/scorecard.py").read_text())
code("%%writefile /kaggle/working/gc/make_slices.py\n" + (OUTD / "scripts/planB/make_slices.py").read_text())
code("%%writefile /kaggle/working/gc/profile_input.csv\n" + (UP / "planB_results/b1/profile.csv").read_text())
code('''
ORACLE_PACK = found["oracle_pack"][0] if found["oracle_pack"] else None
if ORACLE_PACK is not None and ORACLE_PACK.suffix == ".json":   # unpacked by Kaggle: scorecard.py reads the .gz
    _gz = GC / "oracle_pack.json.gz"
    with open(ORACLE_PACK, "rb") as _src, gzip.open(_gz, "wb") as _dst:
        shutil.copyfileobj(_src, _dst)
    plog(f"oracle pack found unpacked at {ORACLE_PACK}; re-packed to {_gz}")
    ORACLE_PACK = _gz
if not ORACLE_PACK:   # required (Oct 13): without it the scorecard is silently skipped after hours of GPU time
    plog("!!! STOP: oracle pack not attached")
    raise SystemExit("STOP: the oracle pack (oracle_pack.json.gz) is not attached. Add Input -> Your Work / Datasets -> "
                     "graphcomp-planb, then Save & Run All again. Nothing has run on the GPU yet.")
pk = json.load(gzip.open(ORACLE_PACK))
_pack_missing = sorted({json.loads(l)["instance_id"] for l in open(TASKS_JSONL) if l.strip()} - set(pk["tasks"]))
if _pack_missing:
    plog(f"!!! STOP: oracle pack lacks {len(_pack_missing)} tasks")
    raise SystemExit(f"STOP: oracle pack {pk['version']} lacks {len(_pack_missing)} competition tasks (e.g. {_pack_missing[:5]}); "
                     "attach the current graphcomp-planb dataset.")
plog(f"oracle pack ok: {ORACLE_PACK} | {pk['version']} | {len(pk['tasks'])} tasks | {len(pk['graphs'])} graphs")
sys.path.insert(0, str(GC))
import scorecard as sc
print("scorecard", sc.SCORECARD_VERSION)
''')


code(r'''
# ============================== submission for this arm ==============================
def write_submission(dest, tools, seed):
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "prompts").mkdir(parents=True)
    (dest / "configs").mkdir()
    prompt = (Path(SAMPLE_SUB) / "prompts/system.md").read_text()
    (dest / "prompts/system.md").write_text(prompt + (PROMPT_ADDENDUM or ""))
    (dest / "agent.yaml").write_text(
        "name: swe_planb_agent\n"
        f"model: {MODEL_SLUG}\n"
        "instruction: !include prompts/system.md\n"
        "tools:\n" + "".join(f"  - {t}\n" for t in tools) +
        "generate_content_config: !include configs/sampling.yaml\n")
    (dest / "configs/sampling.yaml").write_text(
        f"temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: {MAX_OUTPUT_TOKENS}\n"
        f"seed: {seed}\n"
        f"thinking_config:\n  thinking_budget: {THINKING_BUDGET}\n  include_thoughts: true\n")
    (dest / "eval_config.yaml").write_text(
        "evaluation:\n"
        f"  timeout_seconds: {BUDGET['timeout_seconds']}\n  max_tool_calls: {BUDGET['max_tool_calls']}\n"
        f"  max_time_minutes: {BUDGET['max_time_minutes']}\n  max_turns: {BUDGET['max_turns']}\n")
    return dest

BASE_TOOLS = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"]
GRAPH_TOOLS = ["get_code_neighbors", "search_similar_code", "get_code_subgraph"]
TOOLS = {"B2": BASE_TOOLS + GRAPH_TOOLS, "B3": BASE_TOOLS}[ARM]
SUB = write_submission(OUT / f"submissions/{RUN_NAME}", TOOLS, SEED)
shutil.copy(SUB / "prompts/system.md", OUT / "provenance/system_prompt.md")
print((SUB / "agent.yaml").read_text())
print((SUB / "configs/sampling.yaml").read_text())
plog(f"submission {RUN_NAME}: tools={TOOLS} max_output_tokens={MAX_OUTPUT_TOKENS} thinking_budget={THINKING_BUDGET} "
     f"prompt addendum={'none' if not PROMPT_ADDENDUM else repr(PROMPT_ADDENDUM[:120]) + '...'}")
''')

code(r'''
# ============================== validity in this environment (first session only) ==============================
VAL = OUT / "validity"
if B1_MODE:
    plog(f"validity: read from the attached output {VALIDITY_ROOT} (measured {B1_ENV.get('time')}); not re-run")
    shutil.copytree(VALIDITY_ROOT, VAL, dirs_exist_ok=True)   # carried forward so any notebook attaching this output finds it
    plog(f"validity: copied into this version's output ({VAL})")
    VAL_SUMMARY = json.load(open(VALIDITY_ROOT / "slices/slices_meta.json"))["sub_reason_by_repo"]
    if STACK_CHANGED:   # Oct 13: rebuilt harness wheels; check that validity still holds on the tasks this run uses
        _rc_ids = [t for t in (OFFICIAL_DEV if TASK_SCOPE == "dev" else OFFICIAL_DEV + OFFICIAL_HOLDOUT) if t.split("_")[0] not in SHADOWED_REPOS]
        _ref = {m: {r["instance_id"]: bool(r.get("resolved")) for r in read_jsonl(VALIDITY_ROOT / m / "task_results.jsonl")} for m in ("gold", "empty")}
        RC = VAL / "recheck"
        shutil.rmtree(RC, ignore_errors=True)
        RC.mkdir(parents=True)
        plog(f"validity re-check on build {STACK_ID}: gold and empty passes on {len(_rc_ids)} slice tasks (CPU, before serving)")
        for _m in ("gold", "empty"):
            with Progress(RC / _m / "task_results.jsonl", len(_rc_ids), f"re-check {_m}", every=120):
                sh(f"cd {GC} && {sys.executable} run_arm.py --mode {_m} --tasks {TASKS_JSONL} --snapshots {SNAPSHOTS} "
                   f"--submission {SUB} --results {RC / _m} --sandbox subprocess --task-ids {' '.join(_rc_ids)} "
                   f"--timeout-seconds {BUDGET['timeout_seconds']} --concurrency 1", log=RC / f"{_m}.log")
        _new = {m: {r["instance_id"]: bool(r.get("resolved")) for r in read_jsonl(RC / m / "task_results.jsonl")} for m in ("gold", "empty")}
        _dis = [t for t in _rc_ids if _new["gold"].get(t) != _ref["gold"].get(t)
                or (t in _ref["empty"] and _new["empty"].get(t) != _ref["empty"][t])]
        (RC / "recheck.json").write_text(json.dumps({"stack_id": STACK_ID, "tasks": _rc_ids, "disagree": _dis,
            "detail": {t: {"gold_01": _ref["gold"].get(t), "gold_here": _new["gold"].get(t), "empty_01": _ref["empty"].get(t),
                           "empty_here": _new["empty"].get(t)} for t in _rc_ids}}, indent=1))
        plog(f"validity re-check: {len(_rc_ids) - len(_dis)}/{len(_rc_ids)} agree with 01; disagree: {_dis or 'none'}"
             + (" -> exclude these from this run's resolution rates (validity/recheck/recheck.json)" if _dis else ""))
elif not RUN_VALIDITY:
    raise SystemExit("This notebook needs notebook 01's output attached (it holds validity/, the slices and the package list). "
                     "Add Input -> Notebook Output -> graphcomp 01 validity B2 s1, then run again.")
else:
    val_ids = [t for t in ALL_IDS if t.split("_")[0] not in SHADOWED_REPOS]
    gold_dir, empty_dir = VAL / "gold", VAL / "empty"
    for d in (gold_dir, empty_dir):
        if d.exists():
            shutil.rmtree(d)
    plog(f"validity: gold-patch pass on {len(val_ids)} tasks ({len(ALL_IDS) - len(val_ids)} {SHADOWED_REPOS} tasks not run: "
         "their tests import the host copy)")
    with Progress(gold_dir / "task_results.jsonl", len(val_ids), "validity gold", every=120):
        sh(f"cd {GC} && {sys.executable} run_arm.py --mode gold --tasks {TASKS_JSONL} --snapshots {SNAPSHOTS} "
           f"--submission {SUB} --results {gold_dir} --sandbox subprocess --task-ids {' '.join(val_ids)} "
           f"--timeout-seconds {BUDGET['timeout_seconds']} --concurrency 1", log=VAL / "gold.log")
    gold_ok = [r["instance_id"] for r in read_jsonl(gold_dir / "task_results.jsonl") if r.get("resolved")]
    plog(f"validity: gold resolves {len(gold_ok)}/{len(val_ids)}; empty-patch pass on those {len(gold_ok)}")
    with Progress(empty_dir / "task_results.jsonl", len(gold_ok), "validity empty", every=120):
        sh(f"cd {GC} && {sys.executable} run_arm.py --mode empty --tasks {TASKS_JSONL} --snapshots {SNAPSHOTS} "
           f"--submission {SUB} --results {empty_dir} --sandbox subprocess --task-ids {' '.join(gold_ok)} "
           f"--timeout-seconds {BUDGET['timeout_seconds']} --concurrency 1", log=VAL / "empty.log")
    with open(VAL / "import_verdicts.jsonl", "w") as f:
        for t in ALL_IDS:
            shadow = t.split("_")[0] in SHADOWED_REPOS
            f.write(json.dumps({"instance_id": t, "verdict": "released copy tested (workspace shadowed)" if shadow else "workspace tested",
                                "source": "per-repo verdict of the B1 import check (Oct 12): identical for all tasks of a repo"}) + "\n")
    rc = sh(f"cd {GC} && {sys.executable} make_slices.py --profile {GC / 'profile_input.csv'} --empty {empty_dir} --gold {gold_dir} "
            f"--import-check {VAL / 'import_verdicts.jsonl'} --out {VAL / 'slices'} --seed {SLICE_SEED} --size 24",
            log=VAL / "make_slices.log")
    # Official slices (fixed Oct 12; scripts/planB/official_slices): written in place of this run's own draw, which is kept
    # as *_drawn_here.txt for reference. Validity measured here decides which slice tasks count for resolution.
    (VAL / "slices").mkdir(parents=True, exist_ok=True)
    for nm, ids in (("dev", OFFICIAL_DEV), ("holdout", OFFICIAL_HOLDOUT)):
        f_ = VAL / "slices" / f"{nm}_slice.txt"
        if f_.exists():
            f_.rename(VAL / "slices" / f"{nm}_slice_drawn_here.txt")
        f_.write_text("\n".join(ids) + "\n")
    _pv = {r["instance_id"]: r for r in csv.DictReader(open(VAL / "slices" / "profile.csv"))} if (VAL / "slices" / "profile.csv").exists() else {}
    _bad = [t for t in OFFICIAL_DEV + OFFICIAL_HOLDOUT if _pv and _pv.get(t, {}).get("sub_valid") != "True"]
    plog(f"slices: OFFICIAL dev/holdout written to {VAL / 'slices'}; slice tasks not valid here: "
         f"{[(t, _pv.get(t, {}).get('sub_reason')) for t in _bad] or 'none'}")
    venv = {**env_info, "freeze": FREEZE, "time": now(),
            "protocol": "one gold pass on non-shadowed tasks; empty pass on gold-resolving tasks; requests/httpx = released copy tested"}
    (VAL / "validity_env.json").write_text(json.dumps(venv, indent=1))
    if (VAL / "slices/dev_slice.txt").exists():
        VALIDITY_ROOT, B1_MODE, B1_ENV = VAL, True, venv
        DEV = [l.strip() for l in open(VAL / "slices/dev_slice.txt") if l.strip()]
        HOLDOUT = [l.strip() for l in open(VAL / "slices/holdout_slice.txt") if l.strip()]
        B1_PROFILE = VAL / "slices/profile.csv"
    else:
        plog(f"!!! slices not written (make_slices exit {rc}; fewer than 48 eligible?). The arm still runs, in a fixed shuffled order.")
    VAL_SUMMARY = json.load(open(VAL / "slices/slices_meta.json"))["sub_reason_by_repo"] if (VAL / "slices/slices_meta.json").exists() else {}
plog(f"validity by repo: {VAL_SUMMARY}")
''')

code(r'''
# ============================== validity profile and task order ==============================
import pandas as pd, random
if B1_MODE and (DEV != OFFICIAL_DEV or HOLDOUT != OFFICIAL_HOLDOUT):
    raise SystemExit('The attached validity output holds slices that differ from the official ones (Oct 12). '
                     'Attach the output of the regenerated notebook 01.')
if B1_MODE:
    prof = pd.read_csv(B1_PROFILE)
    prof["sub_valid"] = prof["sub_valid"].astype(str) == "True"
    print(prof.groupby("repo")["sub_valid"].agg(["sum", "size"]).rename(columns={"sum": "sub_valid", "size": "n"}))
    rest = [t for t in ALL_IDS if t not in set(DEV) | set(HOLDOUT)]
    ORDER = DEV + (rest + HOLDOUT if TASK_SCOPE == "all" else [])
else:
    if TASK_SCOPE != "all":
        raise SystemExit('TASK_SCOPE="dev" needs the dev slice from step 1.')
    prof = pd.DataFrame({"instance_id": ALL_IDS, "sub_valid": float("nan")})
    ORDER = sorted(ALL_IDS)
    random.Random(ORDER_SEED).shuffle(ORDER)   # fixed shuffle: a partial session is a mix of repos, not one repo
assert len(ORDER) == len(set(ORDER)) and set(ORDER) <= set(ALL_IDS)
print(f"task order: {len(DEV)} dev + {len(ORDER) - len(DEV)} more = {len(ORDER)} ({TASK_SCOPE})")
''')

md("""
## Step 2 — resume
Parts from attached earlier versions of this notebook (same `ARM` and `SEED`) are copied in, so this version's output
holds the whole run so far. A task is done when some part has its record; records showing a serving or connection
failure are retried once (`RETRY_INFRA_ERRORS`).
""")
code(r'''
INFRA_RE = re.compile(r"APIConnectionError|ServiceUnavailable|Connection refused|ConnectError|InternalServerError|"
                      r"BadGatewayError|EngineDeadError|Server disconnected|RemoteProtocolError", re.I)
for src in found["prior_parts"]:
    dst = PARTS / src.name
    if not dst.exists():
        shutil.copytree(src, dst)
        print("copied prior part", src.name)

def part_records():
    """{instance_id: (part_dir, record)}; a later non-infra record wins over an infra-error one."""
    recs = {}
    for pdir in sorted(PARTS.iterdir()):
        for r in read_jsonl(pdir / "task_results.jsonl"):
            iid = r.get("instance_id")
            infra = bool(INFRA_RE.search(json.dumps({k: v for k, v in r.items() if isinstance(v, str)})))
            prev = recs.get(iid)
            if prev is None or (prev[2] and not infra):
                recs[iid] = (pdir, r, infra)
    return recs

recs = part_records()
done = {i for i, (_, _, infra) in recs.items() if not (infra and RETRY_INFRA_ERRORS)}
REMAINING = [t for t in ORDER if t not in done]
plog(f"resume: {len(done & set(ORDER))}/{len(ORDER)} tasks already done in {len(list(PARTS.iterdir()))} parts; "
     f"{len(REMAINING)} remaining (dev remaining: {len([t for t in REMAINING if t in DEV])})")
''')

md("""
## Step 3 — serve the model exactly as session A
`adk_submission.VllmServer`, gemma4 tool and reasoning parsers, thinking on, bf16, LoRA enabled with the sample adapters
mounted, tp = 4, `gpu_memory_utilization` 0.80, `max_model_len` 32,768. No fallback ladder: a different setting would be a
different condition, so a failure stops the notebook.
""")
code(r'''
server, SERVED = None, None
if REMAINING:
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    DTYPE = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "auto"
    adapters = discover_adapters(str(SAMPLE_SUB), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS) if MOUNT_SAMPLE_LORAS else None
    def free_gpus(timeout=120):
        subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9", shell=True)
        subprocess.run("pkill -9 -f 'VLLM::|vllm.entrypoints|EngineCore'", shell=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            used = [int(x) for x in subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                    shell=True, capture_output=True, text=True).stdout.split()]
            if used and max(used) < 1500:
                return used
            time.sleep(5)
        raise RuntimeError(f"GPU memory not released after {timeout}s: {used} MiB")
    def start_server():
        print("GPU memory before start (MiB):", free_gpus(), flush=True)
        cfg = VllmConfig(model=MODEL_DIR, port=8000, tool_call_parser="gemma4", reasoning_parser="gemma4",
                         default_chat_template_kwargs={"enable_thinking": True}, max_model_len=SERVE["max_model_len"],
                         dtype=DTYPE, gpu_memory_utilization=SERVE["util"], enable_auto_tool_choice=True, enable_lora=MOUNT_SAMPLE_LORAS,
                         max_loras=8, max_lora_rank=128, tensor_parallel_size=TP, startup_timeout=SERVER_STARTUP_TIMEOUT)
        srv = VllmServer(cfg, adapter_manifest=adapters)
        launch = shlex.join(srv.build_cmd())
        plog(f"starting vLLM: {launch[:200]} ...")
        stop_tail = threading.Event()
        def _tail():
            while not stop_tail.wait(30):
                try:
                    last = [l for l in Path(srv.log_path).read_text(errors="replace").splitlines() if l.strip()][-1][:170]
                except Exception:
                    last = "(no log yet)"
                plog(f"    vLLM starting ... {last} | {gpu_status()}")
        threading.Thread(target=_tail, daemon=True).start()
        try:
            srv.start()
        finally:
            stop_tail.set()
        return srv, {"launch": launch, "log": srv.log_path, **SERVE, "tp": TP}
    t0 = time.time()
    server, SERVED = start_server()
    logtxt = Path(SERVED["log"]).read_text(errors="replace")
    SERVED["kv"] = re.findall(r"(GPU KV cache size: [\d,]+ tokens|Maximum concurrency for [\d,]+ tokens per request: [\d.]+x)", logtxt)
    SERVED["startup_s"] = round(time.time() - t0)
    (OUT / f"provenance/serving_{RUN_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M')}.json").write_text(json.dumps(SERVED, indent=1))
    plog(f"vLLM healthy after {SERVED['startup_s']} s; {SERVED['kv']}")
else:
    plog("nothing remaining: skipping serving")

def server_alive():
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=20).read()
        return True
    except Exception:
        return False
''')

md("""
## Step 4 — run the remaining tasks
Chunks of `CHUNK` tasks through `run_arm.py --mode agent` (each chunk = one part directory). Before each chunk: the server
is checked (restarted once if dead) and the time guard estimates whether the next chunk can finish before
`SESSION_BUDGET_H`. Errors are logged, not raised, so the merge, scorecard and zip cells always run.
""")
code(r'''
import traceback
SESSION_TAG = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
chunk_secs, restarts, stop_reason = [], 0, "all remaining tasks run"
todo = list(REMAINING)
try:
    k = 0
    while todo:
        elapsed_h = (time.time() - _T0) / 3600
        est_h = (1.25 * max(chunk_secs[-3:]) if chunk_secs else 60 * 60) / 3600   # 8 tasks at concurrency 4: up to 2 x 30 min
        if elapsed_h + est_h + 0.3 > SESSION_BUDGET_H:
            stop_reason = f"time guard: {elapsed_h:.2f} h elapsed, next chunk estimated {est_h:.2f} h, budget {SESSION_BUDGET_H} h"
            break
        if not server_alive():
            if restarts >= 1:
                stop_reason = "vLLM server down twice"
                break
            plog("!!! vLLM not answering; restarting once")
            try:
                server.stop()
            except Exception:
                pass
            server, SERVED = start_server()
            restarts += 1
        chunk = todo[:CHUNK]
        part = PARTS / f"p{len(list(PARTS.iterdir())):03d}_{SESSION_TAG}_{RUN_NAME}"
        plog(f"--- chunk {k + 1}: {len(chunk)} tasks -> {part.name} | {len(todo)} remaining | {chunk}")
        t0 = time.time()
        with Progress(part / "task_results.jsonl", len(chunk), f"chunk {k + 1}", every=120):
            sh(f"cd {GC} && {sys.executable} run_arm.py --mode agent --tasks {TASKS_JSONL} --snapshots {SNAPSHOTS} "
               f"--submission {SUB} --results {part} --sandbox subprocess --task-ids {' '.join(chunk)} "
               f"--max-tool-calls {BUDGET['max_tool_calls']} --max-time-minutes {BUDGET['max_time_minutes']} "
               f"--max-turns {BUDGET['max_turns']} --timeout-seconds {BUDGET['timeout_seconds']} "
               f"--concurrency {CONCURRENCY} --compaction-interval {COMPACTION_INTERVAL} --model-path {MODEL_DIR} --model-alias {MODEL_SLUG}",
               log=part.parent / f"{part.name}.log")
        chunk_secs.append(time.time() - t0)
        _rate = sum(chunk_secs) / (CHUNK * len(chunk_secs))
        plog(f"    ETA: {len(todo) - len(chunk)} tasks left in this session's list, about "
             f"{_el(_rate * (len(todo) - len(chunk)))} at {_rate / 60:.1f} min/task (wall, concurrency {CONCURRENCY}); "
             f"session elapsed {_el(time.time() - _T0)} of {SESSION_BUDGET_H} h")
        if Path(SERVED["log"]).exists():
            shutil.copy(SERVED["log"], part / "vllm_server.log")
        # running tally over the whole run so far
        recs = part_records()
        got = [r for i, (_, r, _) in recs.items() if i in set(ORDER)]
        gcalls = ctx = 0
        for i, (pdir, r, _) in recs.items():
            tp_ = pdir / "traces" / f"trace_{i}.json"
            if tp_.exists():
                gcalls += any(c["name"] in sc.GRAPH_TOOLS for c in sc.parse_trace(tp_)["calls"])
            ctx += sc.end_reason(r, 0) == "context_exceeded"
        plog(f"    chunk {k + 1} took {chunk_secs[-1] / 60:.1f} min | run so far: {len(got)}/{len(ORDER)} tasks, "
             f"resolved {sum(bool(r.get('resolved')) for r in got)}, tasks with a graph call {gcalls}, context-exceeded {ctx}")
        done_now = {i for i, (_, _, infra) in recs.items() if not (infra and RETRY_INFRA_ERRORS)}
        failed_infra = [t for t in chunk if t not in done_now]
        todo = todo[CHUNK:]   # tasks of this chunk without a usable record are retried next session, not now
        if failed_infra:
            plog(f"    {len(failed_infra)} tasks without a usable record (will be retried next session): {failed_infra}")
        k += 1
except Exception as e:
    stop_reason = f"error: {type(e).__name__}: {e}"
    traceback.print_exc()
plog(f"run loop ended: {stop_reason}")
(RUN_ROOT / f"session_{SESSION_TAG}.json").write_text(json.dumps(
    {"session": SESSION_TAG, "stop_reason": stop_reason, "chunk_seconds": [round(x) for x in chunk_secs],
     "restarts": restarts, "concurrency": CONCURRENCY, "chunk": CHUNK}, indent=1))
''')

md("""
## Step 5 — merge the parts
One record per task (a non-infra record wins; otherwise the latest), with its trace, log, patch and test output, in
`runs/<ARM>_s<SEED>/merged_<ARM>_s<SEED>/`. That directory is the arm's results directory for the scorecard.
""")
code(r'''
MERGED = RUN_ROOT / f"merged_{RUN_NAME}"
if MERGED.exists():
    shutil.rmtree(MERGED)
for d in ("traces", "logs", "patches", "test_outputs"):
    (MERGED / d).mkdir(parents=True)
recs = part_records()
with open(MERGED / "task_results.jsonl", "w") as f:
    for iid in ORDER:
        if iid not in recs:
            continue
        pdir, r, infra = recs[iid]
        f.write(json.dumps({**r, "_part": pdir.name, "_infra_error": infra}) + "\n")
        for d in ("traces", "logs", "patches", "test_outputs"):
            for p in ([q for q in (pdir / d).iterdir() if re.search(rf"(?<![0-9A-Za-z]){re.escape(iid)}(?![0-9])", q.name)] if (pdir / d).exists() else []):
                (shutil.copytree if p.is_dir() else shutil.copy)(p, MERGED / d / p.name)
merged = read_jsonl(MERGED / "task_results.jsonl")
dev_done = sum(r["instance_id"] in DEV for r in merged)
plog(f"merged: {len(merged)}/{len(ORDER)} tasks ({dev_done}/24 dev) -> {MERGED}")
COMPLETE = len(merged) == len(ORDER) and not any(r["_infra_error"] for r in merged)
print("ARM COMPLETE" if COMPLETE else f"not complete: attach this version's output to the next version and run again")
''')

md("""
## Step 6 — scorecard and summary
Scorecard 0.3.2 on the merged run (all versions v0 to v7, `length_check.json`, `end_reason`). Resolution is reported on
`sub_valid` tasks only (B1); adoption and localization on all tasks. Every rate with numerator, denominator and Wilson interval.
""")
code(r'''
try:
    if ORACLE_PACK:
        sh(f"cd {GC} && {sys.executable} scorecard.py --results {MERGED} --oracle-pack {ORACLE_PACK} "
           + (f"--profile {B1_PROFILE} " if B1_PROFILE else "") + f"--out {OUT / 'scorecard' / RUN_NAME} --arm {ARM}", log=OUT / f"scorecard/{RUN_NAME}.log")
        T = pd.read_csv(OUT / "scorecard" / RUN_NAME / "scorecard_tasks.csv")
    else:   # oracle-free summary straight from the traces
        rows = []
        for r in merged:
            iid = r["instance_id"]
            tp_ = MERGED / "traces" / f"trace_{iid}.json"
            if not tp_.exists():
                continue
            calls = sc.parse_trace(tp_)["calls"]
            ng = sum(c["name"] in sc.GRAPH_TOOLS for c in calls)
            rows.append({"instance_id": iid, "repo": iid.split("_")[0], "n_tool_calls": len(calls), "n_graph_calls": ng,
                         "made_graph_call": int(ng > 0), "resolved": bool(r.get("resolved")), "gold_file_read": float("nan"),
                         "end_reason": sc.end_reason(r, len(calls))})
        T = pd.DataFrame(rows)
    T = T.merge(prof[["instance_id", "sub_valid"]], on="instance_id", how="left")
    T["in_dev"] = T.instance_id.isin(DEV)
    def wl(k, n):
        lo, hi = sc.wilson(int(k), int(n))
        return f"{int(k)}/{int(n)} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]" if n else "0/0"
    lines = [f"# {RUN_NAME} on Kaggle ({now()}), scorecard {sc.SCORECARD_VERSION}", "",
             f"- Tasks with results: {len(T)}/{len(ORDER)} ({int(T.in_dev.sum())}/24 dev); complete: {COMPLETE}",
             f"- Stack build: {STACK_ID}" + (" (REBUILT wheelhouse, differs from the validity session: pair only with B2 on this build)" if STACK_CHANGED else ""),
             f"- Serving: {SERVED and {k: SERVED[k] for k in ('util', 'max_model_len', 'tp', 'kv') if k in SERVED}}; concurrency {CONCURRENCY}",
             f"- This session stopped because: {stop_reason}", ""]
    for name, sub in (("all tasks", T), ("dev slice", T[T.in_dev])):
        if not len(sub):
            continue
        v = sub[sub.sub_valid == True]
        lines += [f"## {name}",
                  f"- Made any graph call (adoption): {wl(sub.made_graph_call.sum(), len(sub))}",
                  f"- Graph calls / tool calls: {int(sub.n_graph_calls.sum())}/{int(sub.n_tool_calls.sum())}",
                  (f"- Resolved, sub_valid tasks only: {wl(v.resolved.fillna(False).astype(bool).sum(), len(v))}" if B1_MODE else
                   f"- Resolved, ALL tasks (validity unavailable; includes tasks not scorable here, e.g. requests): {wl(sub.resolved.fillna(False).astype(bool).sum(), len(sub))}"),
                  (f"- Gold file read: {wl(sub.gold_file_read.sum(), len(sub))}" if ORACLE_PACK else "- Gold file read: needs the oracle pack (laptop)"),
                  f"- End reasons: {sub.end_reason.value_counts().to_dict()}",
                  "- By repo (adoption | resolved on sub_valid):"]
        for repo, g in sub.groupby("repo"):
            gv = g[g.sub_valid == True]
            lines.append(f"  - {repo}: {wl(g.made_graph_call.sum(), len(g))} | {wl(gv.resolved.fillna(False).astype(bool).sum(), len(gv))}")
        lines.append("")
    secs = [r.get("duration_seconds") for r in merged if r.get("duration_seconds")]
    if secs:
        s_ = sorted(secs)
        lines += ["## Timing", f"- Per-task wall seconds: median {s_[len(s_) // 2]:.0f}, mean {sum(s_) / len(s_):.0f}, max {s_[-1]:.0f} (n {len(s_)})"]
    sess = [json.load(open(p)) for p in sorted(RUN_ROOT.glob("session_*.json"))]
    tot_chunk_h = sum(sum(s["chunk_seconds"]) for s in sess) / 3600
    if tot_chunk_h:
        lines.append(f"- Agent time over all sessions: {tot_chunk_h:.2f} h for {len(merged)} tasks = {len(merged) / tot_chunk_h:.1f} tasks/h at concurrency {CONCURRENCY}; full 129 at this rate: {129 / (len(merged) / tot_chunk_h):.1f} h")
    lcp = OUT / "scorecard" / RUN_NAME / "length_check.json"
    lc = json.load(open(lcp)) if lcp.exists() else {"versions": {}}
    lines += ["", "## Length check (Spearman rho with n_tool_calls; AUROC resolved overall / in IQR band)" if lc["versions"] else ""]
    for vname, d in lc["versions"].items():
        lines.append(f"- {vname}: rho {d['spearman_rho_length']}, confounded {d['length_confounded']}, AUROC {d['auroc_resolved']} / {d['auroc_resolved_in_band']}")
    (RUN_ROOT / f"summary_{RUN_NAME}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
except Exception as e:
    traceback.print_exc()
    plog(f"!!! scorecard / summary failed: {type(e).__name__}: {e} (the run itself is saved; score it on the laptop)")
_sc_csv = OUT / "scorecard" / RUN_NAME / "scorecard_tasks.csv"
if not _sc_csv.exists():   # never stop here: the zip cell below must still run
    (OUT / "SCORECARD_MISSING.txt").write_text(f"scorecard_tasks.csv not written for {RUN_NAME}; rebuild it from the traces with "
                                               "scripts/planB/scorecard.py --results <merged> --oracle-pack <pack> ...\n")
    plog(f"!!! SCORECARD MISSING for {RUN_NAME}: see SCORECARD_MISSING.txt (the run itself is saved)")
else:
    plog(f"scorecard written: {_sc_csv}")
''')

code(r'''
# Stop vLLM, then zip everything (Output tab -> graphcomp_arm_<ARM>_s<SEED>.zip).
if server is not None:
    try:
        server.stop()
        plog("vLLM stopped")
    except Exception as e:
        plog(f"vLLM stop failed: {e}")
import shutil
from pathlib import Path
zp = Path(f"/kaggle/working/graphcomp_arm_{RUN_NAME}.zip")
if zp.exists():
    zp.unlink()
shutil.make_archive(str(zp.with_suffix("")), "zip", OUT)
plog(f"zip: {zp} {round(zp.stat().st_size / 1e6, 1)} MB. Unzip into planB_results/runs/ on the laptop.")
''')

LABELS = ["STEP 0/7 - setup: config and progress logging",
          "STEP 0/7 - setup: helper functions",
          "STEP 0/7 - setup: locate inputs",
          "STEP 0/7 - setup: install the harness wheelhouse (takes a few minutes)",
          "STEP 0/7 - setup: install the test dependencies (embedded wheels: inline-snapshot 0.36.0, executing 2.3.0, dirty-equals 0.11)",
          "STEP 0/7 - setup: check the serving stack imports",
          "STEP 0/7 - setup: provenance and full package list (must match the validity session)",
          None, None, None, None,
          "STEP 0/7 - setup: oracle pack and scorecard import",
          "STEP 0/7 - setup: write the arm's submission",
          "STEP 1/7 - validity: gold pass, then empty pass on gold-resolving tasks (about 40 min; skipped if attached)",
          "STEP 2/7 - task order from the validity profile",
          "STEP 2/7 - resume: collect earlier parts, list remaining tasks",
          "STEP 3/7 - serve the model (about 4 min)",
          "STEP 4/7 - run remaining tasks in chunks (hours; k/n printed every 2 min)",
          "STEP 5/7 - merge parts into one results directory",
          "STEP 6/7 - scorecard 0.3.2 and summary",
          "STEP 7/7 - stop the server and zip"]
_ci = 0
for _k, (_t, _s) in enumerate(cells):
    if _t != "code":
        continue
    lab = LABELS[_ci]; _ci += 1
    if _s.startswith("%%"):
        assert lab is None, _s[:40]
        continue
    assert lab is not None, _s[:60]
    cells[_k] = (_t, f"# {lab}\n" + _s)
assert _ci == len(LABELS), (_ci, len(LABELS))

import hashlib as _hl, base64 as _b64
_TWD = Path(__file__).with_name("test_wheels")
_TW = {f.name: (_hl.sha256(f.read_bytes()).hexdigest(), _b64.b64encode(f.read_bytes()).decode()) for f in sorted(_TWD.glob("*.whl"))}
assert len(_TW) == 3, _TW.keys()
FILL = {"@@TEST_WHEELS@@": repr(_TW), "@@TAG@@": V["tag"], "@@ARM@@": V["arm"], "@@SEED@@": str(V["seed"]), "@@SCOPE@@": V["scope"],
        "@@MAXOUT@@": str(V["max_out"]), "@@THINK@@": str(V["think"]), "@@ADDENDUM@@": repr(V["addendum"]),
        "@@VALIDITY@@": str(V["validity"]), "@@MODEL@@": V["model"]}
def fill(t):
    for k, v in FILL.items():
        t = t.replace(k, v)
    return t
cells = [(t, fill(x)) for t, x in cells]
if V["job"]:
    cells.insert(0, ("markdown", V["job"]))
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}, "cells": []}
for i, (t, s) in enumerate(cells):
    c = {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True), "id": f"arm{i:02d}"}
    if t == "code":
        c["outputs"] = []
        c["execution_count"] = None
    nb["cells"].append(c)
p = OUTD / "kaggle_notebooks" / V["file"]
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(nb, indent=1))
print(p, len(cells), "cells", round(p.stat().st_size / 1e6, 1), "MB")

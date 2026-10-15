#!/usr/bin/env python3
"""Plan B dev-slice runs on a single-L4 VM: validity, B2 baseline and the four dev ablations, one command, resumable.

    ~/gc/venv/bin/python scripts/planB/vm/dev_runs.py [--arms B2,FC,QN,T0,O8K] [--dev-slice FILE] [--out DIR]
    (run inside tmux; re-run the same command to resume: finished steps and tasks are skipped)

Steps
  0  environment: Python 3.12, one GPU, data and model found, harness imports, package list (must not change between
     runs of this script: every arm on this VM shares one environment)
  1  validity ON THIS VM: gold-patch pass over all 129 tasks, then empty-patch pass over the ones the gold patch resolves.
     The VM's Python differs from Kaggle's (e.g. its sandboxes may have pip), so which tasks can be resolved here is
     measured here, for all repos, not taken from Kaggle.
  2  dev slice: the official slice, scripts/planB/official_slices/dev_slice.txt (fixed Oct 12; the same file the Kaggle
     notebooks use), or the file given with --dev-slice. Checked against this VM's validity: a dev task that is not valid
     here still runs and is excluded from resolution rates.
  3  serve the model (gemma-4-31b-it-qat-w4a16-ct; --model 12b / e4b / e2b for the size sweep) on the single L4: ladder of (gpu_memory_utilization, max_model_len) from
     (0.90, 32768) down; the first rung that starts is recorded and reused by every later run of this script.
     Concurrency is set once from vLLM's reported KV capacity and then fixed.
  4  arms on the 24 dev tasks, in order (each a full harness run through run_arm.py, chunks of --chunk tasks):
       B2   baseline: all nine tools, sample prompt (the pair for every other arm: comparisons stay on this VM)
       FC   forced graph call every turn (Plan B 4.4 test 4)
       QN   qualified names in graph-tool arguments (Plan A fix; C2a's first variant)
       T0   thinking budget 0
       O8K  max_output_tokens 8192 (about 24k of input instead of 16k)
  5  per arm: merge parts, scorecard 0.3.2; then a paired summary (each arm vs B2 on the same tasks: resolved, adoption,
     end reasons, exact McNemar on discordant tasks) and a tarball of everything.

Logs: every line is time-stamped; one line per finished task (outcome, tool calls, graph calls, end, seconds), a heartbeat
every 2 minutes (counts, elapsed, GPU utilisation, vLLM queue), vLLM's log while it starts, an ETA after every chunk.
Everything is also appended to OUT/progress.log.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]                 # the graphComp folder
SCRIPTS = ROOT / "scripts" / "planB"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "notebook_builder"))
from gen_all import FORCED, QUALIFIED                      # noqa: E402  (identical prompt text to the Kaggle notebooks)

ARMS = {
    "B2": dict(tools="B2", addendum="", max_out=16384, think=4096, what="baseline: all nine tools, sample prompt"),
    "FC": dict(tools="B2", addendum=FORCED, max_out=16384, think=4096, what="forced graph call every turn"),
    "QN": dict(tools="B2", addendum=QUALIFIED, max_out=16384, think=4096, what="qualified names in graph-tool arguments"),
    "T0": dict(tools="B2", addendum="", max_out=16384, think=0, what="thinking budget 0"),
    "O8K": dict(tools="B2", addendum="", max_out=8192, think=4096, what="max_output_tokens 8192"),
    # Oct 14: B2 with the adapter given by --adapter NAME=DIR (agent.yaml "adapter: NAME"); pairs with B2 in the same folder,
    # where the adapter is mounted but unused, so the two arms differ only in whether the agent routes to it.
    "LORA": dict(tools="B2", addendum="", max_out=16384, think=4096, adapter=True, what="B2 + the trained adapter (--adapter)"),
}
ADAPTER = None      # (name, Path) from --adapter
BASE_TOOLS = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"]
GRAPH_TOOLS = ["get_code_neighbors", "search_similar_code", "get_code_subgraph"]
BUDGET = dict(max_tool_calls=50, max_time_minutes=30, max_turns=100, timeout_seconds=300)
COMPACTION_INTERVAL = 15
SLICE_SEED = 20261012
OFFICIAL_DEV = SCRIPTS / "official_slices" / "dev_slice.txt"
# (0.80, 32768) first = Kaggle's B2 setting (works on 4 x L4, e.g. g2-standard-48). Oct 13: 31B cannot reach 20k context on ONE
# L4 (about 20 GiB of weights on 22 GiB; Kaggle's 4 x L4 held 27,712 KV tokens in about 40 GiB): use 4 x L4, or --model e4b / 12b.
LADDER = [(0.80, 32768), (0.90, 32768), (0.95, 32768), (0.95, 28672), (0.95, 24576), (0.95, 20480)]
# Oct 13: smaller QAT w4a16 variants are allowed. 31B is the Plan B condition; the others are the size sweep (B2 only, own
# results folder ~/gc/runs_dev_<variant>, sample LoRAs not mounted because they are 31B adapters).
MODEL_SPECS = {"31b": "gemma-4-31b-it-qat-w4a16-ct", "12b": "gemma-4-12b-it-qat-w4a16-ct",
               "e4b": "gemma-4-e4b-it-qat-w4a16-ct", "e2b": "gemma-4-e2b-it-qat-w4a16-ct"}
MODEL_VARIANT = "31b"
MODEL_SLUG = MODEL_SPECS[MODEL_VARIANT]
MIN_MODEL_LEN = 20480
LEAN_MAX_SEQS = 4     # below this, max_output_tokens 16384 leaves under 4k of input: no meaningful run
INFRA_RE = re.compile(r"APIConnectionError|ServiceUnavailable|Connection refused|ConnectError|InternalServerError|"
                      r"BadGatewayError|EngineDeadError|Server disconnected|RemoteProtocolError", re.I)

# --------------------------------------------------------------------------------------------- logging
_T0 = time.time()
OUT: Path = Path(".")


def _el(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def plog(msg):
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC | +{_el(time.time() - _T0)}] {msg}"
    print(line, flush=True)
    try:
        with open(OUT / "progress.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


class step:
    """with step("STEP 1/5 - ..."): prints START and DONE / FAILED with the duration."""
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        plog(f">>> START  {self.label}")

    def __exit__(self, et, ev, tb):
        plog(f"{'!!! FAILED' if et else '<<< DONE '}  {self.label}  ({_el(time.time() - self.t)})"
             + (f"  {et.__name__}: {ev}" if et else ""))
        return False


# Progress / gpu_status / vllm_status: the same code as the Kaggle notebooks (notebook_builder/progress_block.py)
exec((SCRIPTS / "notebook_builder" / "progress_block.py").read_text(), globals())


def sh(cmd, log=None):
    """Run a shell command, stream its output, tee it to `log`; return the exit code."""
    plog(f"$ {cmd[:300]}{' ...' if len(cmd) > 300 else ''}")
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    f = open(log, "a") if log else None
    for line in p.stdout:
        if f:
            f.write(line)
        # the harness prints one "ensurepip unavailable" warning per sandbox; keep it in the log file, not on screen
        if "ensurepip unavailable" in line or "EXPERIMENTAL" in line or "check_feature_enabled" in line:
            continue
        print("    | " + line, end="", flush=True)
    p.wait()
    if f:
        f.close()
    return p.returncode


def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# --------------------------------------------------------------------------------------------- step 0
def environment(a):
    gc = Path(a.gc).expanduser()
    env = {}
    pe = gc / "paths.env"
    if pe.exists():
        env = dict(l.split("=", 1) for l in pe.read_text().splitlines() if "=" in l)
    tasks = Path(env.get("TASKS") or next(iter(sorted((gc / "data").rglob("tasks.jsonl"))), ""))
    mdir = gc / "data" / ("model" if MODEL_VARIANT == "31b" else f"model_{MODEL_VARIANT}")   # setup_vm.sh MODEL_VARIANT=...
    model = Path((env.get("MODEL") if MODEL_VARIANT == "31b" else None)
                 or next(iter(sorted(p.parent for p in mdir.rglob("config.json"))), ""))
    if not tasks.is_file():
        raise SystemExit(f"tasks.jsonl not found under {gc}/data (run setup_vm.sh first)")
    comp = tasks.parent
    sample = next((p.parent for p in sorted(comp.rglob("agent.yaml")) if "sample" in str(p).lower()), None)
    if sample is None:
        raise SystemExit(f"sample submission (agent.yaml under a 'sample' folder) not found in {comp}")
    if not (model / "config.json").exists():
        raise SystemExit(f"model {MODEL_SLUG} not found under {mdir} (run MODEL_VARIANT={MODEL_VARIANT} bash setup_vm.sh first)")
    pack = ROOT / "planB_results" / "kaggle_upload" / "graphcomp-planb" / "oracle_pack.json.gz"
    if not pack.is_file():   # required (Oct 13): otherwise every arm's scorecard is skipped after hours of GPU time
        raise SystemExit(f"oracle pack not found at {pack}: copy planB_results/kaggle_upload/graphcomp-planb to the VM "
                         "(it is in gc_vm.tgz) and run again")
    import gzip
    pk = json.load(gzip.open(pack))
    missing = sorted({json.loads(l)["instance_id"] for l in tasks.read_text().splitlines() if l.strip()} - set(pk["tasks"]))
    if missing:
        raise SystemExit(f"oracle pack {pk['version']} lacks {len(missing)} competition tasks (e.g. {missing[:5]})")
    plog(f"oracle pack ok: {pack} | {pk['version']} | {len(pk['tasks'])} tasks")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"Python {sys.version.split()[0]}: run with {gc}/venv/bin/python (3.12)")
    gpus = [l for l in subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.splitlines() if l.strip()]
    plog(f"Python {sys.version.split()[0]} | GPUs: {gpus} | tasks {tasks} | model {model} | sample submission {sample}")
    if len(gpus) != 1:
        plog(f"WARNING: {len(gpus)} GPUs visible; this runner serves with tensor_parallel_size={len(gpus) or 1}")
    r = subprocess.run([sys.executable, "-c", "import swegemma, adk_submission, adk_eval_core, google.adk, vllm, inline_snapshot, dirty_equals, executing; print('ok')"],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("harness / serving stack / test deps do not import:\n" + r.stderr[-1500:])
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (OUT / "provenance").mkdir(exist_ok=True)
    (OUT / "provenance" / "pip_freeze.txt").write_text(freeze)
    ref = OUT / "provenance" / "pip_freeze_reference.txt"
    if ref.exists():
        a_, b_ = set(ref.read_text().splitlines()), set(freeze.splitlines())
        if a_ != b_:
            raise SystemExit("The package list changed since the first run of this script, so later arms would not be "
                             f"comparable. Removed: {sorted(a_ - b_)[:20]} Added: {sorted(b_ - a_)[:20]}")
        plog("package list identical to the first run of this script")
    else:
        ref.write_text(freeze)
        plog(f"package list recorded as the reference for every arm on this VM ({len(freeze.splitlines())} packages)")
    nvs = subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip()
    (OUT / "provenance" / f"env_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M')}.json").write_text(
        json.dumps({"time": now(), "python": sys.version, "nvidia_smi": nvs, "tasks": str(tasks), "model": str(model),
                    "sample": str(sample), "argv": sys.argv}, indent=1))
    return tasks, comp / "snapshots", sample, model, max(1, len(gpus))


def write_submission(dest, arm, seed, sample):
    cfg = ARMS[arm]
    tools = BASE_TOOLS + (GRAPH_TOOLS if cfg["tools"] == "B2" else [])
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "prompts").mkdir(parents=True)
    (dest / "configs").mkdir()
    (dest / "prompts/system.md").write_text((Path(sample) / "prompts/system.md").read_text() + cfg["addendum"])
    adapter_line = ""
    if ADAPTER is not None:           # the adapter is part of every submission in an adapter run (mounted for all arms)
        shutil.copytree(ADAPTER[1], dest / "adapters" / ADAPTER[0])
        if cfg.get("adapter"):
            adapter_line = f"adapter: {ADAPTER[0]}\n"
    elif cfg.get("adapter"):
        raise SystemExit("arm LORA needs --adapter NAME=DIR")
    (dest / "agent.yaml").write_text(
        f"name: swe_planb_agent\nmodel: {MODEL_SLUG}\n{adapter_line}instruction: !include prompts/system.md\n"
        "tools:\n" + "".join(f"  - {t}\n" for t in tools) + "generate_content_config: !include configs/sampling.yaml\n")
    (dest / "configs/sampling.yaml").write_text(
        f"temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: {cfg['max_out']}\nseed: {seed}\n"
        f"thinking_config:\n  thinking_budget: {cfg['think']}\n  include_thoughts: true\n")
    (dest / "eval_config.yaml").write_text(
        "evaluation:\n" + "".join(f"  {k}: {v}\n" for k, v in BUDGET.items()))
    return dest


def run_arm_cmd(mode, tasks, snaps, sub, results, ids, conc, model=None):
    cmd = (f"{shlex.quote(sys.executable)} {shlex.quote(str(SCRIPTS / 'run_arm.py'))} --mode {mode} "
           f"--tasks {tasks} --snapshots {snaps} --submission {sub} --results {results} --sandbox subprocess "
           f"--task-ids {' '.join(ids)} --timeout-seconds {BUDGET['timeout_seconds']} --concurrency {conc}")
    if mode == "agent":
        cmd += (f" --max-tool-calls {BUDGET['max_tool_calls']} --max-time-minutes {BUDGET['max_time_minutes']} "
                f"--max-turns {BUDGET['max_turns']} --compaction-interval {COMPACTION_INTERVAL} --model-path {model} --model-alias {MODEL_SLUG}")
    return cmd


# --------------------------------------------------------------------------------------------- steps 1 and 2
def validity_and_slice(a, tasks, snaps, sample):
    V = OUT / "validity"
    V.mkdir(exist_ok=True)
    all_ids = sorted(json.loads(l)["instance_id"] for l in open(tasks) if l.strip())
    sub = write_submission(V / "submission_B2", "B2", 1, sample)
    gold_dir, empty_dir = V / "gold", V / "empty"
    with step("STEP 1/5 - validity on this VM: gold pass on all tasks, then empty pass on gold-resolving tasks"):
        if (V / "done.json").exists():
            plog(f"validity: already measured ({json.load(open(V / 'done.json'))['time']}); reusing")
        else:
            for d in (gold_dir, empty_dir):
                shutil.rmtree(d, ignore_errors=True)
            plog(f"validity: gold-patch pass on {len(all_ids)} tasks (concurrency 1; about 15 s per task)")
            with Progress(gold_dir / "task_results.jsonl", len(all_ids), "validity gold", every=120):
                sh(run_arm_cmd("gold", tasks, snaps, sub, gold_dir, all_ids, 1), log=V / "gold.log")
            gold_ok = [r["instance_id"] for r in read_jsonl(gold_dir / "task_results.jsonl") if r.get("resolved")]
            plog(f"validity: gold resolves {len(gold_ok)}/{len(all_ids)}; empty-patch pass on those {len(gold_ok)}")
            with Progress(empty_dir / "task_results.jsonl", len(gold_ok), "validity empty", every=120):
                sh(run_arm_cmd("empty", tasks, snaps, sub, empty_dir, gold_ok, 1), log=V / "empty.log")
            (V / "done.json").write_text(json.dumps({"time": now(), "n_tasks": len(all_ids), "gold_resolved": len(gold_ok)}))
        prof_in = ROOT / "planB_results" / "b1" / "profile.csv"
        rc = sh(f"{shlex.quote(sys.executable)} {SCRIPTS / 'make_slices.py'} --profile {prof_in} --empty {empty_dir} "
                f"--gold {gold_dir} --out {V / 'slices'} --seed {SLICE_SEED} --size 24", log=V / "make_slices.log")
        meta_p = V / "slices" / "slices_meta.json"
        reasons = json.load(open(meta_p))["sub_reason_by_repo"] if meta_p.exists() else {}
        plog(f"validity by repo (this VM): {reasons}")
    with step("STEP 2/5 - dev slice"):
        prof = {r["instance_id"]: r for r in csv.DictReader(open(V / "slices" / "profile.csv"))} if (V / "slices" / "profile.csv").exists() else {}
        slice_file = Path(a.dev_slice) if a.dev_slice else OFFICIAL_DEV
        if not slice_file.exists():
            raise SystemExit(f"dev slice file {slice_file} not found (copy scripts/planB/official_slices to the VM)")
        dev = [l.strip() for l in open(slice_file) if l.strip()]
        src = "OFFICIAL slice (scripts/planB/official_slices, fixed Oct 12)" if slice_file == OFFICIAL_DEV else f"given: {slice_file}"
        if not prof:
            raise SystemExit(f"validity profile missing (make_slices exit {rc}); see {V / 'make_slices.log'}")
        bad = [t for t in dev if prof and prof.get(t, {}).get("sub_valid") != "True"]
        plog(f"dev slice ({len(dev)} tasks, {src}): {dev}")
        if bad:
            plog(f"WARNING: {len(bad)} dev tasks are not resolvable on this VM ({[(t, prof.get(t, {}).get('sub_reason')) for t in bad]}); "
                 "they still run, and are excluded from resolution rates")
        (OUT / "dev_slice_used.txt").write_text("\n".join(dev) + "\n")
    return dev, prof


# --------------------------------------------------------------------------------------------- step 3
class Server:
    def __init__(self, model, sample, tp, loras, lean=True):
        self.model, self.sample, self.tp, self.loras, self.lean = model, sample, tp, loras, lean
        self.srv, self.info = None, None

    @staticmethod
    def free_gpus(timeout=180):
        subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9", shell=True)
        subprocess.run("pkill -9 -f 'VLLM::|vllm.entrypoints|EngineCore'", shell=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            used = [int(x) for x in subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                    shell=True, capture_output=True, text=True).stdout.split()]
            if used and max(used) < 1500:
                return used
            time.sleep(5)
        raise RuntimeError(f"GPU memory not released after {timeout}s")

    def try_start(self, util, L):
        import torch
        from adk_submission import VllmConfig, VllmServer, discover_adapters
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
        plog(f"vLLM: try gpu_memory_utilization={util} max_model_len={L} | GPU memory before (MiB): {self.free_gpus()}")
        adapters = discover_adapters(str(self.sample), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS) if self.loras else None
        extra = {ADAPTER[0]: str(ADAPTER[1])} if ADAPTER is not None else None
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "auto"
        cfg = VllmConfig(model=str(self.model), port=8000, tool_call_parser="gemma4", reasoning_parser="gemma4",
                         default_chat_template_kwargs={"enable_thinking": True}, max_model_len=L, dtype=dtype,
                         gpu_memory_utilization=util, enable_auto_tool_choice=True, enable_lora=bool(self.loras or ADAPTER),
                         max_loras=8, max_lora_rank=128, tensor_parallel_size=self.tp, startup_timeout=1800,
                         # lean (Oct 13; 31B did not fit one L4: a fixed 2.62 GiB allocation failed at every context length):
                         # text only (no image / audio encoder budget) and at most LEAN_MAX_SEQS sequences in the scheduler.
                         # The agent sends text only and the VM's concurrency is below that, so the agent's condition is unchanged.
                         limit_mm_per_prompt=("image=0,audio=0" if self.lean else None),
                         extra_args=(["--max-num-seqs", str(LEAN_MAX_SEQS)] if self.lean else []))
        srv = VllmServer(cfg, adapter_manifest=adapters, lora_modules=extra)
        launch = shlex.join(srv.build_cmd())
        stop = threading.Event()

        def tail():
            while not stop.wait(30):
                try:
                    last = [l for l in Path(srv.log_path).read_text(errors="replace").splitlines() if l.strip()][-1][:170]
                except Exception:
                    last = "(no log yet)"
                plog(f"    vLLM starting ... {last} | {gpu_status()}")
        threading.Thread(target=tail, daemon=True).start()
        t0 = time.time()
        try:
            srv.start()
            ok, err = True, ""
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {str(e)[-600:]}"
            try:
                srv.stop()
            except Exception:
                pass
        finally:
            stop.set()
        logtxt = Path(srv.log_path).read_text(errors="replace") if Path(srv.log_path).exists() else ""
        shutil.copy(srv.log_path, OUT / "provenance" / f"vllm_u{util}_L{L}.log") if Path(srv.log_path).exists() else None
        kv = re.findall(r"(GPU KV cache size: [\d,]+ tokens|Maximum concurrency for [\d,]+ tokens per request: [\d.]+x)", logtxt)
        info = {"util": util, "max_model_len": L, "ok": ok, "seconds": round(time.time() - t0), "kv": kv, "launch": launch,
                "error": err, "loras": bool(self.loras), "tp": self.tp, "lean": self.lean}
        plog(f"vLLM: {'STARTED' if ok else 'FAILED'} util={util} L={L} in {info['seconds']} s | {kv} {err[-300:]}")
        if not ok:
            for l in [l for l in logtxt.splitlines() if re.search(r"(?i)out of memory|No available memory|KV cache|Error", l)][-6:]:
                plog(f"    vLLM log: {l.strip()[:200]}")
        if ok:
            self.srv, self.info = srv, info
        return info

    def start(self, settings):
        if "serve" in settings:     # the condition is fixed by the first run of this script
            s = settings["serve"]
            info = self.try_start(s["util"], s["max_model_len"])
            if not info["ok"]:
                raise SystemExit(f"vLLM no longer starts at the recorded setting {s}; see provenance/. Not changing it silently.")
            return info
        attempts = []
        for util, L in LADDER:
            info = self.try_start(util, L)
            attempts.append(info)
            if info["ok"]:
                break
        (OUT / "provenance" / "serve_attempts.json").write_text(json.dumps(attempts, indent=1))
        if not self.info:
            raise SystemExit(f"vLLM did not start on any rung down to max_model_len {LADDER[-1][1]}; see provenance/serve_attempts.json")
        if self.info["max_model_len"] < MIN_MODEL_LEN:
            raise SystemExit("max_model_len below the minimum for max_output_tokens 16384")
        return self.info

    def alive(self):
        import urllib.request
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=20).read()
            return True
        except Exception:
            return False

    def stop(self):
        if self.srv:
            try:
                self.srv.stop()
            except Exception:
                pass


def concurrency_from(info):
    m = [float(x) for x in re.findall(r"per request: ([\d.]+)x", " ".join(info.get("kv", [])))]
    return max(1, min(4, int(math.floor(m[0])))) if m else 1


# --------------------------------------------------------------------------------------------- step 4
def part_records(parts):
    recs = {}
    for pdir in sorted(parts.iterdir()) if parts.exists() else []:
        for r in read_jsonl(pdir / "task_results.jsonl"):
            iid = r.get("instance_id")
            infra = bool(INFRA_RE.search(json.dumps({k: v for k, v in r.items() if isinstance(v, str)})))
            prev = recs.get(iid)
            if prev is None or (prev[2] and not infra):
                recs[iid] = (pdir, r, infra)
    return recs


def run_arm_on_dev(arm, seed, dev, a, tasks, snaps, sample, model, server, conc, tag=""):
    name = f"{arm}_s{seed}{tag}"
    root = OUT / "runs" / name
    parts = root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    sub = write_submission(OUT / "submissions" / name, arm, seed, sample)
    cfg = ARMS[arm]
    plog(f"arm {name}: {cfg['what']} | max_output_tokens={cfg['max_out']} thinking={cfg['think']} | concurrency {conc} "
         f"| prompt addendum: {'none' if not cfg['addendum'] else repr(cfg['addendum'][:100]) + '...'}")
    recs = part_records(parts)
    todo = [t for t in dev if not (t in recs and not recs[t][2])]
    plog(f"arm {name}: {len(dev) - len(todo)}/{len(dev)} tasks already done; {len(todo)} to run")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    secs, k = [], 0
    while todo:
        chunk, todo = todo[:a.chunk], todo[a.chunk:]
        if not server.alive():
            plog("!!! vLLM not answering: restarting once at the recorded setting")
            server.stop()
            server.start({"serve": server.info})
            if not server.alive():
                raise SystemExit("vLLM down after a restart")
        k += 1
        part = parts / f"p{len(list(parts.iterdir())):03d}_{stamp}"
        plog(f"--- {name} chunk {k}: {chunk}")
        t0 = time.time()
        with Progress(part / "task_results.jsonl", len(chunk), f"{name} chunk {k}", every=120):
            sh(run_arm_cmd("agent", tasks, snaps, sub, part, chunk, conc, model), log=root / f"{part.name}.log")
        secs.append(time.time() - t0)
        if server.srv and Path(server.srv.log_path).exists():
            shutil.copy(server.srv.log_path, part / "vllm_server.log")
        rate = sum(secs) / sum(min(a.chunk, len(dev)) for _ in secs)
        done_n = len([1 for t, v in part_records(parts).items() if not v[2]])
        plog(f"    {name}: {done_n}/{len(dev)} done | chunk took {_el(secs[-1])} | ETA for this arm {_el(rate * len(todo))} "
             f"| total elapsed {_el(time.time() - _T0)}")
    # merge
    merged = root / f"merged_{name}"
    shutil.rmtree(merged, ignore_errors=True)
    for d in ("traces", "logs", "patches", "test_outputs"):
        (merged / d).mkdir(parents=True)
    recs = part_records(parts)
    with open(merged / "task_results.jsonl", "w") as f:
        for iid in dev:
            if iid not in recs:
                continue
            pdir, r, infra = recs[iid]
            f.write(json.dumps({**r, "_part": pdir.name, "_infra_error": infra}) + "\n")
            for d in ("traces", "logs", "patches", "test_outputs"):
                if (pdir / d).exists():
                    for q in (pdir / d).iterdir():
                        if re.search(rf"(?<![0-9A-Za-z]){re.escape(iid)}(?![0-9])", q.name):
                            (shutil.copytree if q.is_dir() else shutil.copy)(q, merged / d / q.name)
    n = len(read_jsonl(merged / "task_results.jsonl"))
    plog(f"arm {name}: merged {n}/{len(dev)} tasks -> {merged}")
    pack = ROOT / "planB_results" / "kaggle_upload" / "graphcomp-planb" / "oracle_pack.json.gz"
    if pack.exists():
        sh(f"{shlex.quote(sys.executable)} {SCRIPTS / 'scorecard.py'} --results {merged} --oracle-pack {pack} "
           f"--profile {OUT / 'validity' / 'slices' / 'profile.csv'} --out {OUT / 'scorecard' / name} --arm {arm}",
           log=OUT / "scorecard" / f"{name}.log")
    else:
        plog(f"!!! oracle pack not at {pack}: scorecard skipped (copy planB_results/kaggle_upload/graphcomp-planb to the VM)")
    if not (OUT / "scorecard" / name / "scorecard_tasks.csv").exists():
        (OUT / "scorecard").mkdir(exist_ok=True)
        (OUT / "scorecard" / f"SCORECARD_MISSING_{name}.txt").write_text(f"scorecard_tasks.csv not written for {name}; see {name}.log\n")
        plog(f"!!! SCORECARD MISSING for {name} (the run itself is saved; rebuild it from the traces)")
    return merged


# --------------------------------------------------------------------------------------------- step 5
def wilson(k, n, z=1.959964):
    if not n:
        return "0/0"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} = {p:.0%} [{max(0, c - h):.0%}, {min(1, c + h):.0%}]"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summary(arms_done, seed, dev, prof, settings):
    valid = {t for t in dev if prof.get(t, {}).get("sub_valid") == "True"} if prof else set(dev)
    res, rows = {}, []
    for arm in arms_done:
        name = f"{arm}_s{seed}"
        recs = {r["instance_id"]: r for r in read_jsonl(OUT / "runs" / name / f"merged_{name}" / "task_results.jsonl")}
        tc = OUT / "scorecard" / name / "scorecard_tasks.csv"
        sct = {r["instance_id"]: r for r in csv.DictReader(open(tc))} if tc.exists() else {}
        res[arm] = {t: bool(recs[t].get("resolved")) for t in recs}
        adopt = sum(1 for t in recs if sct.get(t, {}).get("made_graph_call") in ("1", "1.0"))
        gc_ = sum(float(sct[t]["n_graph_calls"]) for t in sct if sct[t].get("n_graph_calls"))
        tcalls = sum(float(sct[t]["n_tool_calls"]) for t in sct if sct[t].get("n_tool_calls"))
        ends = {}
        for t in sct:
            ends[sct[t].get("end_reason", "?")] = ends.get(sct[t].get("end_reason", "?"), 0) + 1
        secs = sorted(r.get("duration_seconds") or 0 for r in recs.values())
        med = f"{secs[len(secs) // 2]:.0f}" if secs else "n/a"
        rv = [t for t in recs if t in valid]
        line = (f"| {arm} | {ARMS[arm]['what']} | {len(recs)}/{len(dev)} | {wilson(sum(res[arm][t] for t in rv), len(rv))} | "
                f"{wilson(adopt, len(sct)) if sct else 'n/a'} | {int(gc_)}/{int(tcalls)} | {ends or 'n/a'} | {med}")
        if arm != "B2" and "B2" in res:
            both = [t for t in rv if t in res["B2"]]
            b = sum(res[arm][t] and not res["B2"][t] for t in both)
            c = sum(res["B2"][t] and not res[arm][t] for t in both)
            line += f" | {b} vs {c}, p = {mcnemar_exact(b, c):.3f} |"
        else:
            line += " | (baseline) |"
        rows.append(line)
    md = [f"# Dev-slice runs on a single L4 ({now()})", "",
          f"- Serving: {settings.get('serve')}; concurrency {settings.get('concurrency')} (fixed for every arm)",
          f"- Dev slice: {len(dev)} tasks, {len(valid)} resolvable on this VM (validity measured here)",
          "- Comparisons are paired on this VM only (B2 here is the baseline; Kaggle runs are a different machine type).", "",
          "| Arm | What | Tasks run | Resolved (valid tasks) | Made a graph call | Graph / tool calls | End reasons | Median s | Arm-only vs B2-only resolved, exact McNemar |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"] + rows
    (OUT / "summary_dev.md").write_text("\n".join(md) + "\n")
    for l in md:
        plog(l)


# --------------------------------------------------------------------------------------------- main
def main():
    global OUT, MODEL_VARIANT, MODEL_SLUG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gc", default="~/gc", help="setup_vm.sh folder (data, venv, paths.env)")
    ap.add_argument("--out", help="results folder (re-use it to resume); default ~/gc/runs_dev (31b) or ~/gc/runs_dev_<model>")
    ap.add_argument("--model", default="31b", choices=list(MODEL_SPECS),
                    help="31b = the Plan B condition; 12b / e4b / e2b = size sweep (run --arms B2; sample LoRAs not mounted)")
    ap.add_argument("--arms", default="B2,FC,QN,T0,O8K")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", help="comma list, e.g. 1,2,3 (overrides --seed); used for data generation")
    ap.add_argument("--task-set", default="dev", choices=["dev", "rest"],
                    help="dev = the official dev slice (evaluation); rest = valid tasks in neither slice (LoRA training data only)")
    ap.add_argument("--adapter", help="NAME=DIR: serve this LoRA and enable arm LORA (results in ~/gc/runs_dev_<model>_<NAME>)")
    ap.add_argument("--dev-slice", help="use this dev slice file instead of the official one (scripts/planB/official_slices/dev_slice.txt)")
    ap.add_argument("--chunk", type=int, default=4, help="tasks per run_arm call (resume granularity)")
    ap.add_argument("--concurrency", type=int, help="override the KV-derived concurrency (only on the first run)")
    ap.add_argument("--no-loras", action="store_true", help="do not mount the sample LoRAs (frees memory; recorded)")
    ap.add_argument("--no-lean", action="store_true", help="serve without the lean settings (text only, max 4 sequences)")
    a = ap.parse_args()
    MODEL_VARIANT, MODEL_SLUG = a.model, MODEL_SPECS[a.model]
    global ADAPTER
    if a.adapter:
        _n, _d = a.adapter.split("=", 1)
        ADAPTER = (_n, Path(_d).expanduser().resolve())
        if not (ADAPTER[1] / "adapter_config.json").exists():
            raise SystemExit(f"--adapter: {ADAPTER[1]} has no adapter_config.json")
    if a.model != "31b":
        a.no_loras = True      # the sample LoRAs are 31B adapters; the agent declares no adapter in any arm
    _default = "~/gc/runs_dev" if a.model == "31b" else f"~/gc/runs_dev_{a.model}"
    if ADAPTER is not None:
        _default += f"_{ADAPTER[0]}"
    OUT = Path(a.out or _default).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)
    shared_val = Path("~/gc/runs_dev/validity").expanduser()
    if not (OUT / "validity").exists():               # validity does not depend on the model or adapter: reuse this VM's own
        for _cand in [shared_val] + sorted(Path("~/gc").expanduser().glob("runs_dev*/validity")):
            if (_cand / "slices").exists() and _cand != OUT / "validity":
                shutil.copytree(_cand, OUT / "validity")
                break
    for d in ("provenance", "runs", "submissions", "scorecard"):
        (OUT / d).mkdir(exist_ok=True)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; choose from {list(ARMS)}")
    if arms[0] != "B2" and "B2" not in [p.name.split("_")[0] for p in (OUT / "runs").iterdir()]:
        plog("WARNING: B2 (the baseline every comparison needs) is not in this run and has not been run before")
    sfile = OUT / "settings.json"
    settings = json.load(open(sfile)) if sfile.exists() else {}
    _ad = None if ADAPTER is None else {"name": ADAPTER[0], "dir": str(ADAPTER[1]),
           "sha256": __import__("hashlib").sha256(b"".join(f.read_bytes() for f in sorted(ADAPTER[1].glob("*")) if f.is_file())).hexdigest()[:16]}
    if settings and settings.get("adapter") != _ad:
        raise SystemExit(f"{OUT} was run with adapter {settings.get('adapter')}, not {_ad}: use another --out")
    if settings.get("model", "gemma-4-31b-it-qat-w4a16-ct") != MODEL_SLUG and settings:
        raise SystemExit(f"{OUT} was run with {settings.get('model')}, not {MODEL_SLUG}: use another --out")
    plog(f"=== Plan B dev-slice runs on a single L4 | out {OUT} | arms {arms} | seed {a.seed} | settings {settings or '(first run)'}")
    server = None
    try:
        with step("STEP 0/5 - environment (Python, GPU, data, model, harness, package list)"):
            tasks, snaps, sample, model, tp = environment(a)
        dev, prof = validity_and_slice(a, tasks, snaps, sample)
        seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [a.seed]
        tag = ""
        if a.task_set == "rest":    # LoRA training data: valid tasks in neither official slice (never dev / holdout)
            hold = {l.strip() for l in open(SCRIPTS / "official_slices" / "holdout_slice.txt") if l.strip()}
            dev = sorted(t for t, r in prof.items() if r.get("sub_valid") == "True" and t not in set(dev) | hold)
            tag = "_rest"
            plog(f"task set REST (training data only): {len(dev)} valid tasks in neither slice; seeds {seeds}")
            (OUT / "rest_tasks_used.txt").write_text("\n".join(dev) + "\n")
        with step("STEP 3/5 - serve the model on the L4 (ladder on the first run; recorded setting afterwards)"):
            if "loras" in settings and settings["loras"] == a.no_loras:
                raise SystemExit(f"--no-loras differs from the first run (loras={settings['loras']}); keep it the same")
            if "lean" in settings and settings["lean"] == a.no_lean:
                raise SystemExit(f"--no-lean differs from the first run (lean={settings['lean']}); keep it the same")
            server = Server(model, sample, tp, loras=not a.no_loras, lean=not a.no_lean)
            info = server.start(settings)
            if "serve" not in settings:
                settings["serve"] = {k: info[k] for k in ("util", "max_model_len", "kv", "launch", "tp")}
                settings["loras"] = not a.no_loras
                settings["lean"] = not a.no_lean
                settings["model"] = MODEL_SLUG
                settings["adapter"] = _ad
                settings["concurrency"] = a.concurrency or (min(concurrency_from(info), LEAN_MAX_SEQS) if not a.no_lean else concurrency_from(info))
                settings["first_run"] = now()
                sfile.write_text(json.dumps(settings, indent=1))
            plog(f"serving fixed at {settings['serve']['util']} / {settings['serve']['max_model_len']}; concurrency {settings['concurrency']}")
        done = []
        with step(f"STEP 4/5 - arms {arms} on the {len(dev)} dev tasks"):
            for i, arm in enumerate(arms, 1):
                with step(f"STEP 4/5 - arm {i}/{len(arms)}: {arm} ({ARMS[arm]['what']})"):
                    for _sd in seeds:
                        run_arm_on_dev(arm, _sd, dev, a, tasks, snaps, sample, model, server, settings["concurrency"], tag=tag)
                    done.append(arm)
    finally:
        if server:
            server.stop()
            plog("vLLM stopped")
    with step("STEP 5/5 - paired summary and tarball"):
        prev = sorted({p.name.rsplit("_s", 1)[0] for p in (OUT / "runs").iterdir() if p.name.endswith(f"_s{a.seed}")})
        if a.task_set == "dev":
            summary([x for x in ARMS if x in prev], a.seed, dev, prof, settings)
        else:
            plog("task set REST: no paired summary (training data); trajectories in runs/<ARM>_s<seed>_rest/merged_*")
        tgz = OUT.parent / f"{OUT.name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M')}.tar.gz"
        with tarfile.open(tgz, "w:gz") as t:
            t.add(OUT, arcname=OUT.name)
        plog(f"tarball {tgz} ({tgz.stat().st_size / 1e6:.1f} MB). Copy it to planB_results/vm/ on the laptop (unpack into vm/<model>_<scope>_<arms>/).")


if __name__ == "__main__":
    main()

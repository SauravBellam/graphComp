#!/usr/bin/env python3
"""Plan A orchestrator: runs the babystep chain with every independent step in parallel.

Dependency DAG (HANDOFF A.4 / section 7):

    A1 ──┬──────────────► A5 ──┐
         │                     ├──► A6
    A2 (shards in parallel) ───┤
         └──► A2-analyze ──────┘
    A3 (independent; needs HF access)
    A4 (needs A1 only for gold co-change pairs; runs after A1)

Stage 1 (parallel): A1 shards, A3, A2 shards.   Stage 2 (parallel, after A1): A4, A5.
Stage 3: A2-analyze (after A2).   Stage 4: A6 (after A1, A2-analyze, A5).

Usage:
  python planA/run_all.py --data /path/to/archive --out results --jobs 4 \
      [--a1-shards 4] [--a2-shards 4] [--a2-backend venv|docker] [--stages a1,a3,a4,a2,a5,a6] \
      [--skip a3] [--encoder <model id from A3>]

Each step is also runnable on its own (see each script's docstring); for multi-machine runs give
each VM a disjoint ``--shard i/N`` for A1/A2/A4/A6 and merge with ``a1_census.py merge <out/a1>``.
A failed step does not stop independent steps; the final report lists what ran.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(name: str, cmd: list[str], log_dir: Path) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{name}.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    rec = {"name": name, "exit": p.returncode, "seconds": round(time.time() - t0, 1), "log": str(log)}
    print(f"[run_all] {name}: exit={p.returncode} ({rec['seconds']}s) log={log}", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 2)))
    ap.add_argument("--a1-shards", type=int, default=1)
    ap.add_argument("--a2-shards", type=int, default=1)
    ap.add_argument("--a2-jobs", type=int, default=1, help="parallel tasks inside each A2 shard")
    ap.add_argument("--a2-backend", default="venv", choices=["venv", "docker"])
    ap.add_argument("--a2-timeout", type=int, default=1800)
    ap.add_argument("--stages", default="a1,a2,a3,a4,a5,a6")
    ap.add_argument("--skip", default="")
    ap.add_argument("--encoder", default=None, help="pass the encoder A3 identified to A5 mode (c)")
    ap.add_argument("--a3-models", nargs="*", default=None)
    ap.add_argument("--task-ids", nargs="*", default=None)
    args = ap.parse_args(argv)

    out = args.out
    cache = args.cache or out / "cache"
    stages = {s.strip() for s in args.stages.split(",") if s.strip()} - {s.strip() for s in args.skip.split(",") if s.strip()}
    logs = out / "logs"
    tids = (["--task-ids", *args.task_ids] if args.task_ids else [])
    results: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=args.jobs)

    def submit(name, cmd):
        return pool.submit(run, name, cmd, logs)

    # ---- stage 1: A1 shards, A3, A2 shards (all independent)
    futs_a1, futs_a2, fut_a3 = [], [], None
    if "a1" in stages:
        for i in range(args.a1_shards):
            cmd = [PY, str(HERE / "a1_census.py"), "--data", str(args.data), "--out", str(out / "a1"), "--cache", str(cache), *tids]
            if args.a1_shards > 1:
                cmd += ["--shard", f"{i}/{args.a1_shards}"]
            futs_a1.append(submit(f"a1_shard{i}", cmd))
    if "a3" in stages:
        cmd = [PY, str(HERE / "a3_encoder.py"), "--data", str(args.data), "--out", str(out / "a3")]
        if args.a3_models:
            cmd += ["--models", *args.a3_models]
        fut_a3 = submit("a3", cmd)
    if "a2" in stages:
        for i in range(args.a2_shards):
            cmd = [PY, str(HERE / "a2" / "run_task.py"), "--data", str(args.data), "--out", str(out / "a2_runs"), "--cache", str(cache),
                   "--backend", args.a2_backend, "--jobs", str(args.a2_jobs), "--timeout", str(args.a2_timeout), *tids]
            if args.a2_shards > 1:
                cmd += ["--shard", f"{i}/{args.a2_shards}"]
            futs_a2.append(submit(f"a2_shard{i}", cmd))

    # ---- stage 2: after A1 -> merge, A4, A5
    wait(futs_a1)
    results += [f.result() for f in futs_a1]
    if "a1" in stages and args.a1_shards > 1:
        results.append(run("a1_merge", [PY, str(HERE / "a1_census.py"), "merge", str(out / "a1")], logs))
    futs_s2 = []
    a1_ok = (out / "a1" / "a1_gold_symbols.jsonl").exists()
    if "a4" in stages:
        cmd = [PY, str(HERE / "a4_probe.py"), "--data", str(args.data), "--out", str(out / "a4")]
        if a1_ok:
            cmd += ["--a1", str(out / "a1")]
        futs_s2.append(submit("a4", cmd))
    if "a5" in stages and a1_ok:
        cmd = [PY, str(HERE / "a5_retrieval.py"), "--data", str(args.data), "--a1", str(out / "a1"), "--out", str(out / "a5"), *tids]
        if args.encoder:
            cmd += ["--encoder", args.encoder]
        futs_s2.append(submit("a5", cmd))
    elif "a5" in stages:
        print("[run_all] A5 skipped: A1 outputs missing", flush=True)

    # ---- stage 3: A2 analysis after A2 shards
    wait(futs_a2)
    results += [f.result() for f in futs_a2]
    if "a2" in stages and a1_ok:
        results.append(run("a2_analyze", [PY, str(HERE / "a2" / "analyze.py"), "--data", str(args.data), "--runs", str(out / "a2_runs"),
                                          "--a1", str(out / "a1"), "--out", str(out / "a2")], logs))
    wait(futs_s2)
    results += [f.result() for f in futs_s2]
    if fut_a3 is not None:
        wait([fut_a3])
        results.append(fut_a3.result())

    # ---- stage 4: A6
    if "a6" in stages and a1_ok:
        cmd = [PY, str(HERE / "a6_release.py"), "--data", str(args.data), "--a1", str(out / "a1"), "--out", str(out / "a6"), "--cache", str(cache)]
        if (out / "a2_runs").exists():
            cmd += ["--a2-runs", str(out / "a2_runs")]
        if (out / "a2" / "a2_tasks.csv").exists():
            cmd += ["--a2-analysis", str(out / "a2")]
        results.append(run("a6", cmd, logs))
    pool.shutdown(wait=True)

    report = {"data": str(args.data), "out": str(out), "stages": sorted(stages), "steps": results,
              "all_ok": all(r["exit"] == 0 for r in results)}
    (out / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

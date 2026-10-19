"""Which code do the tests import in the scoring sandbox? (B0.md section 3.3, run here with SubprocessManager.)

For each task: build a Phase-2 sandbox with the same calls, in the same order, as swegemma 0.2.11
verify_task (wheels, snapshot, synthetic hardware, git exclude, editable install, test deps, workspace
test config, baseline commit), then import the package under test (a) from plain python with the
harness's flags and (b) from inside a pytest run in the task's tests/ directory, and report the file.

    python import_check.py --tasks tasks.jsonl --snapshots DIR --submission SUBDIR --out out.jsonl --task-ids ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
from pathlib import Path

PKG = {"requests": "requests", "fastapi": "fastapi", "rich": "rich", "httpx": "httpx"}
PROBE = '''def test_zz_probe_import():
    import sys, importlib
    m = importlib.import_module("{pkg}")
    print("PYTEST_FILE", m.__file__)
    print("PYTEST_VERSION", getattr(m, "__version__", None))
    print("PYTEST_EXE", sys.executable)
    print("PYTEST_PATH0", sys.path[:4])
'''


def run(mgr, sid, cmd, timeout=300):
    r = mgr.exec(sid, cmd, timeout=timeout)
    return {"exit": r.exit_code, "out": (r.stdout or "")[-3000:], "err": (r.stderr or "")[-1500:]}


def grab(text, key):
    for line in text.splitlines():
        if line.strip().startswith(key):
            return line.strip()[len(key):].strip()
    return None


async def check(task, a, cfg, mgr):
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.harness.container_setup import (extract_snapshot, install_editable_package,
        install_test_dependencies, setup_baseline_commit, setup_container_wheels, setup_git_exclude,
        setup_synthetic_hardware, setup_workspace_test_config)
    from swegemma.sandbox import sandbox_start

    pkg = PKG.get(task.instance_id.split("_")[0], task.instance_id.split("_")[0])
    snap, base, patch = resolve_task_snapshot_paths(a.snapshots, task.instance_id, task.repo)
    sid = await sandbox_start(mgr)
    rec = {"instance_id": task.instance_id, "pkg": pkg, "sandbox": "subprocess"}
    try:
        setup_container_wheels(mgr, sid, cfg)
        extract_snapshot(mgr, sid, snap, patch_path=patch, base_snapshot_path=base)
        setup_synthetic_hardware(mgr, sid)
        setup_git_exclude(mgr, sid)
        install_editable_package(mgr, sid)
        install_test_dependencies(mgr, sid, task.repo, fast_path=True, config=cfg)
        setup_workspace_test_config(mgr, sid, repo=task.repo)
        setup_baseline_commit(mgr, sid, "eval_baseline")
        env = "PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1"
        plain = run(mgr, sid, f'cd /workspace && {env} python3 -s -c "import {pkg},sys; print(\'PLAIN_FILE\', {pkg}.__file__); print(\'PLAIN_VERSION\', getattr({pkg},\'__version__\',None)); print(\'PLAIN_EXE\', sys.executable)"')
        rec["plain_file"] = grab(plain["out"], "PLAIN_FILE")
        rec["plain_version"] = grab(plain["out"], "PLAIN_VERSION")
        rec["plain_exe"] = grab(plain["out"], "PLAIN_EXE")
        if rec["plain_file"] is None:
            rec["plain_error"] = (plain["err"] or plain["out"])[-600:]
        tdir = run(mgr, sid, "cd /workspace && (test -d tests && echo tests) || echo .")["out"].strip().splitlines()[-1]
        run(mgr, sid, f"cd /workspace && cat > {tdir}/test_zz_probe_import.py <<'PYEOF'\n{PROBE.format(pkg=pkg)}\nPYEOF")
        pt = run(mgr, sid, f"cd /workspace && {env} python3 -s -m pytest {tdir}/test_zz_probe_import.py -q -s -p no:anyio -o timeout=0 2>&1 | tail -40")
        rec["pytest_file"] = grab(pt["out"], "PYTEST_FILE")
        rec["pytest_version"] = grab(pt["out"], "PYTEST_VERSION")
        rec["pytest_exe"] = grab(pt["out"], "PYTEST_EXE")
        if rec["pytest_file"] is None:
            rec["pytest_tail"] = pt["out"][-1200:]
        show = run(mgr, sid, f"python3 -m pip show {pkg} 2>/dev/null | grep -E '^(Version|Location|Editable project location):'")
        rec["pip_show"] = show["out"].strip()
        snapver = run(mgr, sid, f"cd /workspace && grep -rhoE '__version__ *= *[\"\\x27][^\"\\x27]+' --include=__init__.py --include=__version__.py . 2>/dev/null | head -3")
        rec["snapshot_version_strings"] = snapver["out"].strip()
        ws = run(mgr, sid, "cd /workspace && pwd -P")["out"].strip()
        rec["workspace_real"] = ws
        f = rec.get("pytest_file") or ""
        rec["verdict"] = ("workspace tested" if ws and f.startswith(ws) else
                          "import failed" if not f else "released copy tested (workspace shadowed)")
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            mgr.stop(sid)
        except Exception:
            pass
    return rec


async def amain(a):
    from swegemma.config import EvalConfig
    from swegemma.cli import _build_cli_model_registry
    from swegemma.models.task import load_tasks
    from swegemma.sandbox import SubprocessManager

    cfg = EvalConfig(tasks_path=Path(a.tasks), snapshots_dir=Path(a.snapshots), results_dir=Path(a.out).parent,
                     submission_dir=Path(a.submission), models=_build_cli_model_registry(a.submission), sandbox="subprocess")
    mgr = SubprocessManager(timeout_seconds=300)
    tasks = [t for t in load_tasks(Path(a.tasks)) if t.instance_id in set(a.task_ids)]
    with open(a.out, "w") as f:
        for t in tasks:
            rec = await check(t, a, cfg, mgr)
            rec["host_python"] = platform.python_version()
            print(json.dumps(rec)[:600])
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-ids", nargs="+", required=True)
    asyncio.run(amain(ap.parse_args()))

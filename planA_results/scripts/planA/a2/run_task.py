#!/usr/bin/env python3
"""A2 runner: execute one task's target tests under the tracer, pre-fix and post-fix.

Mirrors the harness's Phase-2 setup (HARNESS_README section 4.2 / swegemma verification.py):
extract snapshot, `pip install -e . --no-deps`, pytest.ini (``-p no:anyio``, norecursedirs,
python_classes) + conftest hook, baseline commit, apply ``test_patch``, run pytest on the files the
test patch touches. Differences, on purpose (this is an audit, not the competition):
  * network allowed; the repo's own test requirements are installed best-effort and the result
    logged per task (``deps.json``), because the official cached test site-packages are not in
    the public archive (HANDOFF section 6)
  * two runs: ``pre`` (base + test_patch) and ``post`` (+ gold ``patch``)
  * ``-p trace_plugin`` with sys.setprofile; junit outcomes recorded by the plugin

Backends:
  venv   : a fresh venv per task on the host (Python >= 3.11 for co_qualname; 3.13 matches the harness)
  docker : runs this same script inside a container built from ``planA/a2/Dockerfile.audit``
           (2 vCPU / 4 GiB like the harness); the host side only orchestrates

Usage (host):
  python planA/a2/run_task.py --data <archive> --out results/a2 --cache cache --backend venv \
      --shard 0/4 --jobs 4 [--task-ids ...] [--timeout 1800] [--wheels <archive>/wheels]

Per task output in <out>/<instance_id>/: trace_pre.json, trace_post.json, pytest_pre.log,
pytest_post.log, deps.json, status.json.

VERIFY-ON-DATA: whether the snapshot .tgz has a single top-level dir; whether `pip install -e`
works for each repo at each commit with the wheels dir (fastapi uses pdm-backend, rich uses
poetry-core, requests setuptools); which test requirement files each repo uses at each commit.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import venv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import Snapshot, Task, find_snapshot, load_tasks, patch_files, shard  # noqa: E402

PYTEST_INI = """[pytest]
addopts = -p no:anyio
norecursedirs = .* build dist venv
python_classes = Test* *Test
python_files = test_*.py *_test.py
filterwarnings =
    ignore::DeprecationWarning
    ignore::UserWarning
"""
CONFTEST_HEADER = "# Hermetic test discovery hook for SWE-gemma\n"

# requirement files the repos used at various points in time (best effort, all optional)
REQ_FILES = [
    "requirements-tests.txt", "requirements-test.txt", "requirements-dev.txt", "requirements_dev.txt",
    "requirements-docs-tests.txt", "test-requirements.txt", "tests/requirements.txt", "requirements.txt",
    "requirements-dev-lock.txt",
]
# extras to try from pyproject [project.optional-dependencies]
EXTRAS = ["test", "tests", "testing", "dev", "all", "standard"]


def sh(cmd: str, cwd: Path | None = None, env: dict | None = None, timeout: int | None = None, log: Path | None = None) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if log is not None:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n$ {cmd}\n[exit {res.returncode}]\n{res.stdout}\n{res.stderr}\n")
    return res


def git(ws: Path, args: str, env=None) -> subprocess.CompletedProcess:
    return sh(f"git {args}", cwd=ws, env=env)


def make_env(venv_dir: Path, ws: Path, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{venv_dir / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{HERE}:{ws}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "audit"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "audit@local"
    env.pop("PYTEST_ADDOPTS", None)
    if extra:
        env.update(extra)
    return env


def write_test_config(ws: Path) -> None:
    (ws / "pytest.ini").write_text(PYTEST_INI)
    cf = ws / "conftest.py"
    existing = cf.read_text(encoding="utf-8", errors="replace") if cf.exists() else ""
    if CONFTEST_HEADER.strip() not in existing:
        cf.write_text(CONFTEST_HEADER + ("\n" + existing if existing.strip() else "") + "\n")


def install_deps(py: str, ws: Path, wheels: Path | None, log: Path, timeout: int) -> dict:
    """Editable install + best-effort test requirements. Returns a record of what succeeded."""
    rec: dict = {"steps": []}
    base = f"{shlex.quote(py)} -m pip install -q --disable-pip-version-check"
    find = f"--find-links={shlex.quote(str(wheels))}" if wheels else ""

    def step(name, cmd):
        r = sh(cmd, cwd=ws, timeout=timeout, log=log)
        rec["steps"].append({"name": name, "cmd": cmd, "exit": r.returncode, "stderr_tail": r.stderr[-800:]})
        return r.returncode == 0

    step("pip_upgrade", f"{base} 'pip>=24' 'setuptools' 'wheel' pytest pytest-timeout")
    # editable install, no deps (what the harness does), then deps from PyPI (wheels first)
    ok = step("editable_no_deps", f"{base} {find} --no-build-isolation --no-deps -e .")
    if not ok:
        ok = step("editable_no_deps_isolated", f"{base} {find} --no-deps -e .")
    rec["editable_ok"] = ok
    # runtime deps of the package
    step("runtime_deps", f"{base} {find} -e .")
    # extras
    pyproject = ws / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
            data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="replace"))
            opt = data.get("project", {}).get("optional-dependencies", {}) or {}
            for ex in EXTRAS:
                if ex in opt:
                    step(f"extra_{ex}", f"{base} {find} -e '.[{ex}]'")
            # poetry dev groups
            tool = data.get("tool", {}).get("poetry", {})
            groups = {}
            for gname, g in (tool.get("group", {}) or {}).items():
                groups[gname] = list((g.get("dependencies") or {}).keys())
            if tool.get("dev-dependencies"):
                groups["dev"] = list(tool["dev-dependencies"].keys())
            for gname, deps in groups.items():
                deps = [d for d in deps if d.lower() != "python"]
                if deps:
                    step(f"poetry_group_{gname}", f"{base} {find} {' '.join(shlex.quote(d) for d in deps)}")
        except Exception as e:  # noqa: BLE001
            rec["pyproject_error"] = repr(e)
    for rf in REQ_FILES:
        p = ws / rf
        if p.exists():
            step(f"req_{rf}", f"{base} {find} -r {shlex.quote(rf)}")
    # repo-specific known needs (VERIFY-ON-DATA)
    if (ws / "src" / "requests").exists() or (ws / "requests").exists():
        step("requests_httpbin", f"{base} {find} pytest-httpbin httpbin trustme pysocks")
    if (ws / "fastapi").exists():
        step("fastapi_tests", f"{base} {find} httpx 'python-multipart' email-validator pyyaml orjson ujson "
                              f"'sqlmodel' 'dirty-equals' 'pytest-asyncio' 'flask' 'passlib[bcrypt]' 'pyjwt' 'inline-snapshot'")
    if (ws / "rich").exists():
        step("rich_tests", f"{base} {find} pytest-cov attrs ipywidgets")
    if (ws / "httpx").exists():
        step("httpx_tests", f"{base} {find} trio anyio 'pytest-asyncio' trustme uvicorn chardet brotli zstandard socksio h2")
    r = sh(f"{shlex.quote(py)} -c \"import importlib,sys; print(sys.version)\"", cwd=ws, log=log)
    rec["python"] = r.stdout.strip()
    fr = sh(f"{shlex.quote(py)} -m pip freeze --all", cwd=ws)
    rec["freeze"] = fr.stdout.splitlines()
    return rec


def apply_patch(ws: Path, patch_text: str, name: str, env: dict, log: Path) -> str | None:
    """Apply like the harness (4 fallbacks). Returns the strategy that worked ('exact' when plain
    git apply succeeded), or None. Anything but 'exact' means the patch did not match the snapshot
    cleanly and should be looked at."""
    p = ws.parent / f"{name}.patch"
    p.write_text(patch_text if patch_text.endswith("\n") else patch_text + "\n", encoding="utf-8")
    for label, flags in (("exact", ""), ("3way", "-3"), ("ignore_whitespace", "--ignore-space-change --ignore-whitespace"),
                         ("recount", "--recount")):
        r = sh(f"git apply --unsafe-paths {flags} {shlex.quote(str(p))}", cwd=ws, env=env, log=log)
        if r.returncode == 0:
            return label
    r = sh(f"patch -p1 --batch --forward < {shlex.quote(str(p))}", cwd=ws, env=env, log=log)
    return "gnu_patch" if r.returncode == 0 else None


def run_pytest(py: str, ws: Path, targets: list[str], env: dict, out_json: Path, phase: str, log: Path, timeout: int,
               test_timeout: int = 0) -> dict:
    env = dict(env)
    env.update({"GC_TRACE_ROOT": str(ws), "GC_TRACE_OUT": str(out_json), "GC_TRACE_PHASE": phase,
                "PYTHONSAFEPATH": "1", "PYTHONNOUSERSITE": "1"})
    tgt = " ".join(shlex.quote(t) for t in targets) or "."
    cmd = (f"{shlex.quote(py)} -s -m pytest {tgt} -p trace_plugin -p no:anyio -p no:cacheprovider "
           f"-o addopts= -o timeout={int(test_timeout)} -o python_classes='Test* *Test' -q -x --no-header -rN "
           f"--junitxml={shlex.quote(str(out_json.with_suffix('.xml')))}")
    cmd = cmd.replace(" -x ", " ")  # run all tests, do not stop at first failure
    t0 = time.time()
    try:
        r = sh(cmd, cwd=ws, env=env, timeout=timeout, log=log)
        return {"exit": r.returncode, "seconds": round(time.time() - t0, 1), "timeout": False,
                **classify_pytest(r.returncode, r.stdout + "\n" + r.stderr),
                "stdout_tail": r.stdout[-3000:], "stderr_tail": r.stderr[-2000:]}
    except subprocess.TimeoutExpired as e:
        return {"exit": None, "seconds": round(time.time() - t0, 1), "timeout": True,
                "stdout_tail": (e.stdout or "")[-3000:] if isinstance(e.stdout, str) else "",
                "stderr_tail": (e.stderr or "")[-2000:] if isinstance(e.stderr, str) else ""}


PYTEST_EXIT = {0: "all_passed", 1: "some_failed", 2: "interrupted", 3: "internal_error", 4: "usage_error", 5: "no_tests"}
_MISSING_MOD = re.compile(r"(?:ModuleNotFoundError|ImportError): No module named '([^']+)'")
_COLLECT_ERR = re.compile(r"ERROR collecting (\S+)")
_SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed|deselected)")


def classify_pytest(code: int | None, text: str) -> dict:
    """Classify a pytest run so environment failures are not mistaken for results.

    run_class: clean (tests collected and ran; failures allowed), collection_error, no_tests,
    import_error (missing module before/while collecting), crashed (exit 3/4), timeout.
    """
    counts = {}
    for n, k in _SUMMARY.findall(text or ""):
        k = "error" if k == "errors" else k
        counts[k] = counts.get(k, 0) + int(n)
    missing = sorted(set(_MISSING_MOD.findall(text or "")))
    coll = sorted(set(_COLLECT_ERR.findall(text or "")))
    if code is None:
        cls = "timeout"
    elif code < 0:
        cls = "killed"          # terminated by a signal (e.g. -9 SIGKILL); no trace is written
    elif code == 5 or (code in (0, 1) and sum(counts.get(k, 0) for k in ("passed", "failed")) == 0 and not counts.get("error")):
        cls = "no_tests"
    elif coll or code == 2:
        cls = "import_error" if missing else "collection_error"
    elif code in (3, 4):
        cls = "crashed"
    else:
        cls = "clean"
    return {"exit_meaning": PYTEST_EXIT.get(code, "timeout" if code is None else f"exit_{code}"),
            "run_class": cls, "counts": counts, "missing_modules": missing, "collection_errors": coll[:20]}


def run_one_venv(task: Task, data: Path, out: Path, cache: Path, wheels: Path | None, timeout: int, keep: bool,
                 test_timeout: int = 0) -> dict:
    tdir = out / task.instance_id
    tdir.mkdir(parents=True, exist_ok=True)
    status_p = tdir / "status.json"
    if status_p.exists():
        try:
            st = json.loads(status_p.read_text())
            if st.get("done"):
                return st
        except Exception:
            pass
    log = tdir / "setup.log"
    log.write_text("")
    status: dict = {"instance_id": task.instance_id, "repo": task.repo, "backend": "venv", "done": False,
                    "test_timeout": test_timeout}
    t0 = time.time()
    sp = find_snapshot(data, task)
    if sp is None:
        status["error"] = "snapshot missing"
        status_p.write_text(json.dumps(status, indent=2))
        return status
    work = cache / "a2" / task.instance_id
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    ws = work / "workspace"
    try:
        snap = Snapshot(sp, work, "workspace")
        snap.extract(force=True)
        venv_dir = work / "venv"
        venv.create(venv_dir, with_pip=True, clear=True)
        py = str(venv_dir / "bin" / "python")
        env = make_env(venv_dir, ws)
        if not (ws / ".git").exists():
            git(ws, "init -q", env)
        git(ws, "config user.email audit@local", env)
        git(ws, "config user.name audit", env)
        (ws / ".git" / "info").mkdir(parents=True, exist_ok=True)
        with open(ws / ".git" / "info" / "exclude", "a") as f:
            f.write("__pycache__/\n*.pyc\n.pytest_cache/\n*.egg-info/\nbuild/\ndist/\n.coverage\n")
        status["deps"] = install_deps(py, ws, wheels, log, timeout)
        (tdir / "env_freeze.txt").write_text("\n".join(status["deps"].pop("freeze", [])) + "\n")
        (tdir / "deps.json").write_text(json.dumps(status["deps"], indent=2))
        write_test_config(ws)
        git(ws, "add -A", env)
        git(ws, "commit -q -m baseline --allow-empty", env)
        targets = [p for p in patch_files(task.test_patch) if p.endswith(".py") and Path(p).name != "conftest.py"]
        status["targets"] = targets
        status["test_patch_strategy"] = apply_patch(ws, task.test_patch, "test", env, log) if task.test_patch.strip() else "empty"
        status["test_patch_applied"] = status["test_patch_strategy"] is not None
        write_test_config(ws)
        status["pre"] = run_pytest(py, ws, targets, env, tdir / "trace_pre.json", "pre", tdir / "pytest_pre.log", timeout, test_timeout)
        status["gold_patch_strategy"] = apply_patch(ws, task.patch, "gold", env, log) if task.patch.strip() else "empty"
        status["gold_patch_applied"] = status["gold_patch_strategy"] is not None
        status["post"] = run_pytest(py, ws, targets, env, tdir / "trace_post.json", "post", tdir / "pytest_post.log", timeout, test_timeout)
        status["trace_pre_exists"] = (tdir / "trace_pre.json").exists()
        status["trace_post_exists"] = (tdir / "trace_post.json").exists()
        status["done"] = True
    except Exception as e:  # noqa: BLE001
        status["error"] = repr(e)
    finally:
        status["seconds"] = round(time.time() - t0, 1)
        status_p.write_text(json.dumps(status, indent=2))
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
    return status


def run_one_docker(task: Task, data: Path, out: Path, cache: Path, wheels: Path | None, timeout: int, image: str, cpus: str, mem: str,
                   test_timeout: int = 0) -> dict:
    """Run this script in a container for a single task. The container sees the archive read-only."""
    tdir = out / task.instance_id
    tdir.mkdir(parents=True, exist_ok=True)
    status_p = tdir / "status.json"
    if status_p.exists():
        try:
            st = json.loads(status_p.read_text())
            if st.get("done"):
                return st
        except Exception:
            pass
    mounts = [f"-v {shlex.quote(str(data.resolve()))}:/data:ro", f"-v {shlex.quote(str(out.resolve()))}:/out",
              f"-v {shlex.quote(str(HERE.parent.resolve()))}:/audit/planA:ro"]
    wh = "--wheels /data/wheels" if (wheels or (data / 'wheels').exists()) else ""
    cmd = (f"docker run --rm --cpus={cpus} --memory={mem} {' '.join(mounts)} {image} "
           f"python3 /audit/planA/a2/run_task.py --data /data --out /out --cache /tmp/cache --backend venv "
           f"--task-ids {shlex.quote(task.instance_id)} --timeout {timeout} --test-timeout {int(test_timeout)} {wh} --jobs 1")
    r = sh(cmd, timeout=timeout * 3 + 600, log=tdir / "docker.log")
    if status_p.exists():
        st = json.loads(status_p.read_text())
        st["backend"] = "docker"
        status_p.write_text(json.dumps(st, indent=2))
        return st
    st = {"instance_id": task.instance_id, "done": False, "error": f"docker exit {r.returncode}", "stderr": r.stderr[-1000:]}
    status_p.write_text(json.dumps(st, indent=2))
    return st


def _run_task(t: Task, args, wheels):
    """Module-level so ProcessPoolExecutor can pickle it (a closure inside main() cannot be)."""
    if args.backend == "docker":
        return run_one_docker(t, args.data, args.out, args.cache, wheels, args.timeout, args.image, args.cpus, args.mem, args.test_timeout)
    return run_one_venv(t, args.data, args.out, args.cache, wheels, args.timeout, args.keep, args.test_timeout)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--backend", choices=["venv", "docker"], default="venv")
    ap.add_argument("--image", default="graphcomp-audit:latest")
    ap.add_argument("--cpus", default="2")
    ap.add_argument("--mem", default="4g")
    ap.add_argument("--wheels", type=Path, default=None, help="offline wheel dir (used as --find-links, PyPI still allowed)")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--task-ids", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per pytest run / pip step")
    ap.add_argument("--keep", action="store_true", help="keep extracted workspace + venv")
    ap.add_argument("--test-timeout", type=int, default=0,
                    help="per-test pytest-timeout seconds (0 = off, as the harness). A hanging test then fails alone "
                         "and the session still finishes and writes its trace")
    args = ap.parse_args(argv)

    if sys.version_info < (3, 11):
        sys.exit("A2 needs Python >= 3.11 (co_qualname); the harness uses 3.13")
    tasks = shard(load_tasks(args.data, args.task_ids), args.shard)
    args.out.mkdir(parents=True, exist_ok=True)
    wheels = args.wheels if args.wheels and args.wheels.exists() else ((args.data / "wheels") if (args.data / "wheels").exists() else None)
    print(f"[a2] {len(tasks)} tasks backend={args.backend} jobs={args.jobs} shard={args.shard} wheels={wheels}", flush=True)

    _run = functools.partial(_run_task, args=args, wheels=wheels)

    results = []
    if args.jobs <= 1:
        for i, t in enumerate(tasks):
            st = _run(t)
            results.append(st)
            print(f"[a2] {i + 1}/{len(tasks)} {t.instance_id}: done={st.get('done')} pre_exit={st.get('pre', {}).get('exit')} "
                  f"post_exit={st.get('post', {}).get('exit')} err={st.get('error')} {st.get('seconds')}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_run, t): t for t in tasks}
            for i, f in enumerate(as_completed(futs)):
                t = futs[f]
                try:
                    st = f.result()
                except Exception as e:  # noqa: BLE001
                    st = {"instance_id": t.instance_id, "done": False, "error": repr(e)}
                results.append(st)
                print(f"[a2] {i + 1}/{len(tasks)} {t.instance_id}: done={st.get('done')} pre_exit={st.get('pre', {}).get('exit')} "
                      f"post_exit={st.get('post', {}).get('exit')} err={st.get('error')} {st.get('seconds')}s", flush=True)
    summ = {"n": len(results), "done": sum(1 for r in results if r.get("done")),
            "with_pre_trace": sum(1 for r in results if r.get("trace_pre_exists")),
            "with_post_trace": sum(1 for r in results if r.get("trace_post_exists")),
            "errors": {r["instance_id"]: r.get("error") for r in results if r.get("error")}}
    suffix = f"_shard{args.shard.replace('/', 'of')}" if args.shard else ""
    (args.out / f"a2_run_summary{suffix}.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()

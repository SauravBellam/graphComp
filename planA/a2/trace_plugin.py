"""pytest plugin: dynamic call-graph tracer for the A2 oracle.

Loaded with ``-p trace_plugin`` (PYTHONPATH must include this directory). Configuration via env:

  GC_TRACE_ROOT   absolute path of the workspace (only frames whose file is under it are kept;
                  anything under a venv/site-packages inside it is dropped)
  GC_TRACE_OUT    output JSON path
  GC_TRACE_PHASE  free text label (pre|post), stored in the output

Records, per test nodeid:
  * executed function ids (module.qualname, module derived from the file path relative to root)
  * outcome (passed/failed/error/skipped) from pytest's report hooks
Globally:
  * direct edges   (caller frame in repo -> callee frame in repo, caller is the immediate parent)
  * via-lib edges  (nearest repo ancestor -> callee, when the immediate parent is library code)
  * per-edge provenance: set of test nodeids in which it was observed (compact: edge index -> list)

Coroutines/generators: a 'call' event fires on every resume; we keep only the first entry of a
frame object, detected as f_lasti <= the offset observed on first entry of a reference generator /
coroutine (computed at import time for this interpreter).

Threads: uses threading.setprofile so fastapi's TestClient (anyio worker thread) is traced.
Uses sys.setprofile rather than settrace (no per-line overhead).

Function id = <module from path>.<co_qualname>. The module is the dotted relative path; the
analysis step applies the same prefix rules as the static side (drop src/docs_src) so both sides
agree. Nested functions keep '<locals>' and are classified separately by the analysis.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

ROOT = os.path.abspath(os.environ.get("GC_TRACE_ROOT", os.getcwd()))
OUT = os.environ.get("GC_TRACE_OUT", "trace.json")
PHASE = os.environ.get("GC_TRACE_PHASE", "")
_ROOT_SLASH = ROOT.rstrip("/") + "/"
_SKIP_PARTS = ("/site-packages/", "/.venv/", "/venv/", "/dist-packages/", "/node_modules/")

CO_GEN_FLAGS = 0x20 | 0x80 | 0x200  # CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR


def _first_lasti(kind: str) -> int:
    """Measure f_lasti at the first 'call' event of a generator/coroutine on this interpreter."""
    seen = {}

    def prof(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "_probe":
            seen.setdefault("v", frame.f_lasti)

    if kind == "gen":
        def _probe():
            yield 1
            yield 2
        sys.setprofile(prof)
        try:
            g = _probe()
            next(g)
            next(g)
        finally:
            sys.setprofile(None)
    else:
        async def _probe():
            return 1
        sys.setprofile(prof)
        try:
            c = _probe()
            try:
                c.send(None)
            except StopIteration:
                pass
        finally:
            sys.setprofile(None)
    return seen.get("v", 0)


FIRST_LASTI = max(_first_lasti("gen"), _first_lasti("coro"))


class Tracer:
    def __init__(self):
        self.lock = threading.Lock()
        self.func_ids: dict[tuple, int] = {}      # (filename, qualname) -> idx
        self.func_names: list[str] = []
        self.func_files: list[str] = []
        self.edges_direct: dict[tuple[int, int], set[str]] = {}
        self.edges_vialib: dict[tuple[int, int], set[str]] = {}
        self.per_test_exec: dict[str, set[int]] = {}
        self.outcomes: dict[str, str] = {}
        self.current: str = "<no-test>"
        self.n_events = 0
        self.installed = False

    # --- helpers
    def _in_repo(self, fn: str) -> bool:
        if not fn.startswith(_ROOT_SLASH):
            return False
        for s in _SKIP_PARTS:
            if s in fn:
                return False
        return True

    def _fid(self, frame) -> int:
        code = frame.f_code
        key = (code.co_filename, code.co_qualname if hasattr(code, "co_qualname") else code.co_name)
        idx = self.func_ids.get(key)
        if idx is None:
            rel = os.path.relpath(code.co_filename, ROOT)
            mod = rel[:-3] if rel.endswith(".py") else rel
            mod = mod.replace(os.sep, ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            idx = len(self.func_names)
            self.func_ids[key] = idx
            self.func_names.append(f"{mod}.{key[1]}")
            self.func_files.append(rel)
        return idx

    # --- profile hook
    def __call__(self, frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        fn = code.co_filename
        if not self._in_repo(fn):
            return
        if (code.co_flags & CO_GEN_FLAGS) and frame.f_lasti > FIRST_LASTI:
            return  # resume of a generator/coroutine, not a new call
        with self.lock:
            self.n_events += 1
            callee = self._fid(frame)
            self.per_test_exec.setdefault(self.current, set()).add(callee)
            parent = frame.f_back
            if parent is None:
                return
            if self._in_repo(parent.f_code.co_filename):
                caller = self._fid(parent)
                self.edges_direct.setdefault((caller, callee), set()).add(self.current)
            else:
                anc = parent.f_back
                depth = 0
                while anc is not None and depth < 200:
                    if self._in_repo(anc.f_code.co_filename):
                        caller = self._fid(anc)
                        self.edges_vialib.setdefault((caller, callee), set()).add(self.current)
                        break
                    anc = anc.f_back
                    depth += 1

    def install(self):
        if not self.installed:
            sys.setprofile(self)
            threading.setprofile(self)
            self.installed = True

    def uninstall(self):
        if self.installed:
            sys.setprofile(None)
            threading.setprofile(None)
            self.installed = False

    def dump(self):
        tests = sorted((set(self.per_test_exec) | set(self.outcomes)) - {"<no-test>", "<between-tests>"})
        # keep execution outside tests (collection/import time) under its own key
        for k in ("<no-test>", "<between-tests>"):
            if k in self.per_test_exec:
                tests.append(k)
        tid = {t: i for i, t in enumerate(tests)}

        def enc(edges):
            return [[c, e, sorted(tid[t] for t in ts if t in tid)] for (c, e), ts in sorted(edges.items())]

        data = {
            "phase": PHASE, "root": ROOT, "python": sys.version, "first_lasti": FIRST_LASTI,
            "n_events": self.n_events, "written_at": time.time(),
            "functions": self.func_names, "files": self.func_files,
            "tests": tests,
            "outcomes": {t: self.outcomes.get(t, "unknown") for t in tests if not t.startswith("<")},
            "per_test_exec": {t: sorted(s) for t, s in self.per_test_exec.items()},
            "edges_direct": enc(self.edges_direct),
            "edges_vialib": enc(self.edges_vialib),
        }
        os.makedirs(os.path.dirname(os.path.abspath(OUT)) or ".", exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(data, f)


TRACER = Tracer()


def pytest_sessionstart(session):
    TRACER.install()


def pytest_runtest_logstart(nodeid, location):
    TRACER.current = nodeid


def pytest_runtest_logreport(report):
    # setup/call/teardown: failures anywhere make the test failed/error
    nid = report.nodeid
    prev = TRACER.outcomes.get(nid)
    if report.when == "call":
        out = "passed" if report.passed else ("skipped" if report.skipped else "failed")
    elif report.when == "setup":
        out = "skipped" if report.skipped else ("error" if report.failed else None)
    else:
        out = "error" if report.failed else None
    if out is None:
        return
    if prev in ("failed", "error") and out == "passed":
        return
    TRACER.outcomes[nid] = out


def pytest_runtest_logfinish(nodeid, location):
    TRACER.current = "<between-tests>"


def pytest_sessionfinish(session, exitstatus):
    TRACER.uninstall()
    TRACER.dump()


def pytest_unconfigure(config):
    if TRACER.installed:
        TRACER.uninstall()
        TRACER.dump()

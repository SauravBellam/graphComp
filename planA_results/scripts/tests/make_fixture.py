#!/usr/bin/env python3
"""Build a tiny synthetic competition archive so the Plan A pipeline can be tested end to end
without the (non-redistributable) Kaggle data.

Layout produced under <dest>/: tasks.jsonl, graphs/, embeddings/, snapshots/, wheels/ (empty).
The fake repo ``mini`` mimics the traits the audit targets:
  * a sync function, an async function, a class with __init__ and a method, a generator,
    a nested function, a decorator-registered handler, a ``docs_src``-style directory and a
    ``src/``-style second package (``src/minisrc``)
  * the provided graph OMITS the async function (finding 5.1), lists the nested function under
    a flattened id (``mini.core.inner``, as the real generator does), contains a containment
    pseudo-edge (class -> __init__), only ``calls`` edges, test nodes, and drops the ``docs_src.``
    prefix in ids
  * two tasks on the same base commit (shared graph) + one on a second commit
  * gold patches that change a sync function (task 1), the async function (task 2), and a
    docs_src file + a src/ file (task 3, multi-file); test patches adding failing tests
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

REPO_FILES_V1 = {
    "pyproject.toml": """[project]
name = "mini"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = []
[project.optional-dependencies]
test = ["pytest"]
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
[tool.setuptools]
packages = ["mini"]
""",
    "mini/__init__.py": "from .core import Engine, run, compute\n",
    "mini/util.py": '''"""utilities"""

def helper(x):
    return x + 1


def unused_helper(x):
    return x * 2


def gen_values(n):
    for i in range(n):
        yield helper(i)
''',
    "mini/core.py": '''"""core module"""
from .util import helper, gen_values
import asyncio

REGISTRY = {}


def register(fn):
    REGISTRY[fn.__name__] = fn
    return fn


class Engine:
    def __init__(self, factor):
        self.factor = factor

    def run(self, x):
        return helper(x) * self.factor

    def total(self, n):
        return sum(gen_values(n))


def compute(x):
    eng = Engine(2)
    return eng.run(x)


async def acompute(x):
    await asyncio.sleep(0)
    return helper(x) - 1


def run(x):
    def inner(y):
        return y + 100
    return inner(compute(x))


@register
def handler(x):
    return compute(x) + 1


async def total(n):
    return Engine(1).total(n)
''',
    "mini/aio.py": '''async def fetch(x):
    return x
''',
    "docs_src/tutorial/tutorial001.py": '''from mini import compute


def read_items():
    return compute(3)
''',
    "src/minisrc/extra.py": '''def extra_fn(x):
    return x - 3
''',
    "tests/__init__.py": "",
    "tests/test_core.py": '''import asyncio
from mini.core import Engine, compute, run, acompute, handler


def test_compute():
    assert compute(1) == 4


def test_run():
    assert run(1) == 104


class TestEngine:
    def test_total(self):
        assert Engine(1).total(3) == 6
''',
}

# gold patch for task 1: compute should add 1 (bug fix in a sync function)
PATCH_1 = """diff --git a/mini/core.py b/mini/core.py
--- a/mini/core.py
+++ b/mini/core.py
@@ -23,7 +23,7 @@ class Engine:

 def compute(x):
     eng = Engine(2)
-    return eng.run(x)
+    return eng.run(x) + 1


 async def acompute(x):
"""
TEST_PATCH_1 = """diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -3,10 +3,10 @@ from mini.core import Engine, compute, run, acompute, handler
 
 
 def test_compute():
-    assert compute(1) == 4
+    assert compute(1) == 5
 
 
 def test_run():
-    assert run(1) == 104
+    assert run(1) == 105
 
 
"""
# task 2: async function fix
PATCH_2 = """diff --git a/mini/core.py b/mini/core.py
--- a/mini/core.py
+++ b/mini/core.py
@@ -28,7 +28,7 @@ def compute(x):

 async def acompute(x):
     await asyncio.sleep(0)
-    return helper(x) - 1
+    return helper(x) + 1


 def run(x):
"""
TEST_PATCH_2 = """diff --git a/tests/test_async.py b/tests/test_async.py
new file mode 100644
--- /dev/null
+++ b/tests/test_async.py
@@ -0,0 +1,6 @@
+import asyncio
+from mini.core import acompute
+
+
+def test_acompute():
+    assert asyncio.run(acompute(1)) == 3
"""
# task 3 (second commit): docs_src + src multi-file change, plus a brand-new method
PATCH_3 = """diff --git a/docs_src/tutorial/tutorial001.py b/docs_src/tutorial/tutorial001.py
--- a/docs_src/tutorial/tutorial001.py
+++ b/docs_src/tutorial/tutorial001.py
@@ -2,4 +2,4 @@ from mini import compute


 def read_items():
-    return compute(3)
+    return compute(4)
diff --git a/src/minisrc/extra.py b/src/minisrc/extra.py
--- a/src/minisrc/extra.py
+++ b/src/minisrc/extra.py
@@ -1,2 +1,6 @@
 def extra_fn(x):
-    return x - 3
+    return x - 4
+
+
+def extra_new(x):
+    return x
diff --git a/mini/core.py b/mini/core.py
--- a/mini/core.py
+++ b/mini/core.py
@@ -20,6 +20,9 @@ class Engine:
     def total(self, n):
         return sum(gen_values(n))

+    def double(self, x):
+        return 2 * self.run(x)
+

 def compute(x):
     eng = Engine(2)
"""
TEST_PATCH_3 = """diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -13,3 +13,7 @@ def test_run():
 class TestEngine:
     def test_total(self):
         assert Engine(1).total(3) == 6
+
+
+def test_double():
+    assert Engine(1).double(1) == 4
"""


def sh(cmd, cwd):
    subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True)


def make_repo(root: Path, files: dict) -> str:
    root.mkdir(parents=True, exist_ok=True)
    for rel, txt in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
    sh("git init -q -b main && git config user.email f@x && git config user.name f && git add -A && git commit -q -m base", root)
    return subprocess.run("git rev-parse HEAD", shell=True, cwd=root, capture_output=True, text=True).stdout.strip()


def tgz(root: Path, dest: Path, prefix: str = "") -> None:
    with tarfile.open(dest, "w:gz") as tf:
        for p in sorted(root.rglob("*")):
            arc = (prefix + str(p.relative_to(root))) if prefix else str(p.relative_to(root))
            tf.add(p, arcname=arc, recursive=False)


def node(nid, text):
    return {"id": nid, "name": nid.rsplit(".", 1)[-1], "text": text}


def make_graph(files: dict, with_double: bool = False) -> dict:
    core = files["mini/core.py"]
    util = files["mini/util.py"]

    def seg(src, start, end):
        lines = src.splitlines()
        return "\n".join(lines[start - 1:end])

    nodes = [
        node("mini.core", ""), node("mini.util", ""), node("mini", ""),
        node("mini.util.helper", seg(util, 3, 4)), node("mini.util.unused_helper", seg(util, 7, 8)),
        node("mini.util.gen_values", seg(util, 11, 13)),
        node("mini.core.register", seg(core, 8, 10)), node("mini.core.Engine", seg(core, 13, 21)),
        node("mini.core.Engine.__init__", seg(core, 14, 15)), node("mini.core.Engine.run", seg(core, 17, 18)),
        node("mini.core.Engine.total", seg(core, 20, 21)), node("mini.core.compute", seg(core, 24, 26)),
        # NOTE: mini.core.acompute (async def) deliberately omitted (finding 5.1).
        # run.<locals>.inner IS a node, under the generator's flattened id (enclosing function dropped).
        node("mini.core.run", seg(core, 34, 37)), node("mini.core.inner", seg(core, 35, 36)),
        node("mini.core.handler", seg(core, 40, 42)),
        node("tutorial.tutorial001.read_items", "def read_items():\n    return compute(3)"),   # docs_src. dropped
        node("minisrc.extra.extra_fn", "def extra_fn(x):\n    return x - 3"),                 # src. dropped
        node("tests.test_core.test_compute", "def test_compute():\n    assert compute(1) == 4"),
        node("tests.test_core.test_run", "def test_run():\n    assert run(1) == 104"),
        node("tests.test_core.TestEngine", "class TestEngine:"),
        node("tests.test_core.TestEngine.test_total", "def test_total(self):\n    assert Engine(1).total(3) == 6"),
        node("tests.test_core", ""), node("mini.core.orphan", "def orphan():\n    pass"),
    ]
    edges = [
        ("mini.core.Engine", "mini.core.Engine.__init__"),      # containment pseudo-edge
        ("mini.core.Engine.run", "mini.util.helper"),
        ("mini.core.Engine.total", "mini.util.gen_values"),
        ("mini.core.compute", "mini.core.Engine"),
        ("mini.core.compute", "mini.core.Engine.run"),
        ("mini.core.run", "mini.core.compute"),
        ("mini.core.run", "mini.core.inner"),
        ("mini.core.handler", "mini.core.compute"),
        ("mini.core.handler", "mini.util.unused_helper"),       # spurious static edge (precision test)
        ("tests.test_core.test_compute", "mini.core.compute"),
        ("tests.test_core.test_run", "mini.core.run"),
        ("tests.test_core.TestEngine.test_total", "mini.core.Engine.total"),
        ("tutorial.tutorial001.read_items", "mini.core.compute"),
    ]
    return {"directed": True, "multigraph": True, "graph": {}, "nodes": nodes,
            "edges": [{"source": s, "target": d, "type": "calls", "key": 0} for s, d in edges]}


def make_embeddings(graph: dict, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    base = rng.normal(size=256).astype(np.float32)
    for n in graph["nodes"]:
        v = 0.6 * base + rng.normal(size=256).astype(np.float32)   # one dominant direction, like the real ones
        out[n["id"]] = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def save_npz(path: Path, emb: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k, arr in emb.items():
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            zf.writestr(f"{k}.npy", buf.getvalue())


def main(dest: Path) -> None:
    dest = Path(dest)
    for d in ("graphs", "embeddings", "snapshots", "wheels", "sandbox"):
        (dest / d).mkdir(parents=True, exist_ok=True)
    work = dest / "_build"
    repo1 = work / "repo1"
    c1 = make_repo(repo1, REPO_FILES_V1)
    files2 = dict(REPO_FILES_V1)
    files2["mini/util.py"] = REPO_FILES_V1["mini/util.py"] + "\n\ndef late_addition():\n    return 42\n"
    repo2 = work / "repo2"
    c2 = make_repo(repo2, files2)
    tasks = [
        {"instance_id": "mini_1", "repo": "example/mini", "base_commit": c1, "problem_statement":
            "`compute()` returns a value that is one too small. Calling `mini.core.compute(1)` gives 4 but should give 5. "
            "See Engine.run and the helper.", "patch": PATCH_1, "test_patch": TEST_PATCH_1, "hints_text": "",
            "created_at": "2026-01-01T00:00:00Z", "FAIL_TO_PASS": ["tests/test_core.py::test_compute"], "PASS_TO_PASS": []},
        {"instance_id": "mini_2", "repo": "example/mini", "base_commit": c1, "problem_statement":
            "Add support for async compute: acompute should add one instead of subtracting.", "patch": PATCH_2,
            "test_patch": TEST_PATCH_2, "hints_text": "", "created_at": "2026-01-02T00:00:00Z",
            "FAIL_TO_PASS": ["tests/test_async.py::test_acompute"], "PASS_TO_PASS": []},
        {"instance_id": "mini_3", "repo": "example/mini", "base_commit": c2, "problem_statement":
            "Tutorial read_items uses the wrong constant and extra_fn in src/minisrc/extra.py is off by one; "
            "also add Engine.double.", "patch": PATCH_3, "test_patch": TEST_PATCH_3, "hints_text": "",
            "created_at": "2026-01-03T00:00:00Z", "FAIL_TO_PASS": ["tests/test_core.py::test_double"], "PASS_TO_PASS": []},
    ]
    with open(dest / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    g1 = make_graph(REPO_FILES_V1)
    g2 = make_graph(files2)
    g2["nodes"].append(node("mini.util.late_addition", "def late_addition():\n    return 42"))
    (dest / "graphs" / f"mini_{c1}.json").write_text(json.dumps(g1, indent=1))
    (dest / "graphs" / f"mini_{c2}.json").write_text(json.dumps(g2, indent=1))
    (dest / "graphs" / "mini_1.json").write_text("")  # empty hard-link twin, like the real archive
    save_npz(dest / "embeddings" / f"mini_{c1}.npz", make_embeddings(g1, 1))
    save_npz(dest / "embeddings" / f"mini_{c2}.npz", make_embeddings(g2, 2))
    tgz(repo1, dest / "snapshots" / "mini_1.tgz")
    tgz(repo1, dest / "snapshots" / "mini_2.tgz", prefix="workspace/")   # single top-level dir variant
    tgz(repo2, dest / "snapshots" / "mini_3.tgz")
    (dest / "sandbox" / "setup.py").write_text("# placeholder\n")
    print(json.dumps({"dest": str(dest), "commits": [c1, c2], "tasks": [t["instance_id"] for t in tasks]}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "fixture"))

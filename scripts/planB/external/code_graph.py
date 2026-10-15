#!/usr/bin/env python3
"""Build a code graph in the competition's format from a repository checkout.

    python code_graph.py REPO_DIR OUT.json

Format, measured on the competition graphs (Oct 15; validate_graph.py reports the overlap on the same snapshots):
- NetworkX node-link directed multigraph; edges all have type "calls" and are SYMMETRIC (every relation is stored in both
  directions). Relations: class <-> its methods, caller <-> callee, base class <-> subclass.
- Nodes: every class, function and method. id = name = dotted path from the repo root (leading `src.` / `docs_src.` dropped)
  + class chain + name; functions nested in functions hang off the enclosing class or module.
- Text: functions = their source; classes = an `ast.unparse` skeleton: `class X(bases):`, a `None` line, then each method's
  decorators and signature with a `...` body.
Calls are resolved statically: local and enclosing-scope names, `self.` / `cls.` / `super().` methods (with repo base
classes), imported names and module aliases, and `Class.method`; instantiating a class links the class node; `obj.method()` on other objects is not linked.
`async def` functions get no node (as in the competition graphs); functions nested inside them do.
"""
import ast
import copy
import json
import os
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "site-packages", "build", "dist", ".tox", ".eggs"}
STRIP_PREFIX = ("src", "docs_src")


def module_name(root, path):
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if len(parts) > 1 and parts[0] in STRIP_PREFIX:
        parts = parts[1:]
    return ".".join(parts)


def class_skeleton(n):
    c = copy.deepcopy(n)
    body = [ast.Expr(ast.Constant(None))]
    for s in n.body:
        if isinstance(s, ast.FunctionDef):
            f = copy.deepcopy(s)
            f.body = [ast.Expr(ast.Constant(Ellipsis))]
            body.append(f)
    c.body = body
    return ast.unparse(c)


class Collector(ast.NodeVisitor):
    def __init__(self, mod):
        self.mod, self.stack, self.defs, self.imports, self.bases = mod, [], {}, {}, {}

    def visit_Import(self, n):
        for a in n.names:
            self.imports[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]

    def visit_ImportFrom(self, n):
        base = n.module or ""
        if n.level:
            pkg = self.mod.split(".")[:-n.level] if self.mod else []
            base = ".".join(pkg + ([base] if base else []))
        for a in n.names:
            self.imports[a.asname or a.name] = f"{base}.{a.name}" if base else a.name

    def _def(self, n, is_class, add=True):
        # qualified name = owner's qualified name (or the module) + own name, as in the competition graphs:
        # requests.auth.HTTPDigestAuth.md5_utf8 (function in a method), tests.test_requests.BadFileObj (class in a method)
        if is_class:      # a class: owned by the enclosing class only if no function lies between them
            owner = self.stack[-1][0] if self.stack and self.stack[-1][1] == "class" else None
        elif isinstance(getattr(n, "_parent", None), (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = None      # a function directly in a function body sits at module level (tests.test_requests.hook)
        else:             # otherwise the nearest enclosing class, through any blocks (HTTPDigestAuth.md5_utf8 inside an `if`)
            owner = next((q for q, k in reversed(self.stack) if k == "class"), None)
        q = (f"{owner}.{n.name}" if owner else ".".join(p for p in (self.mod, n.name) if p))
        if add:
            self.defs.setdefault(q, (n, owner))
            if is_class:
                self.bases[q] = [ast.unparse(b) for b in n.bases]
        self.stack.append((q, "class" if is_class else "function"))
        self.generic_visit(n)
        self.stack.pop()

    def visit_ClassDef(self, n):
        self._def(n, True)

    def visit_FunctionDef(self, n):
        self._def(n, False)

    def visit_AsyncFunctionDef(self, n):
        self._def(n, False, add=False)   # no node for an async function (as in the competition graphs); its nested defs count


def build(root):
    root = Path(root).resolve()
    mods, defs, bases = {}, {}, {}
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            p = Path(dp) / fn
            try:
                src = p.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(src)
            except (SyntaxError, ValueError):
                continue
            mod = module_name(root, p)
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child._parent = node
            c = Collector(mod)
            c.visit(tree)
            mods[mod] = c
            for q, (n, cls) in c.defs.items():
                defs.setdefault(q, (n, cls, mod, src))
            bases.update(c.bases)
    short = {}
    for q in defs:
        short.setdefault(q.rsplit(".", 1)[-1], []).append(q)

    def resolve(dotted, mod):
        c = mods.get(mod)
        head, _, rest = dotted.partition(".")
        cands = [f"{mod}.{dotted}" if mod else dotted]
        if c is not None and head in c.imports:
            full = c.imports[head] + (f".{rest}" if rest else "")
            cands.append(full)
            tail = full.rsplit(".", 1)[-1]
            if full not in defs and len(short.get(tail, [])) == 1:      # re-exported through a package __init__
                cands.append(short[tail][0])
        cands.append(dotted)
        return next((x for x in cands if x in defs), None)

    def method(cls, name, seen=()):
        if f"{cls}.{name}" in defs:
            return f"{cls}.{name}"
        for b in bases.get(cls, []):
            bq = resolve(b, defs[cls][2]) if cls in defs else None
            if bq and bq not in seen and bq in bases:
                r = method(bq, name, seen + (bq,))
                if r:
                    return r
        return None

    rel = set()
    for q, (n, cls, mod, _src) in defs.items():
        parent = q.rsplit(".", 1)[0]
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and parent in defs and isinstance(defs[parent][0], ast.ClassDef):
            rel.add((parent, q))                                            # class <-> method
        if isinstance(n, ast.ClassDef):
            for b in bases.get(q, []):
                bq = resolve(b, mod)
                if bq and bq != q:
                    rel.add((bq, q))                                        # base <-> subclass
            continue
        scope = parent
        for sub in ast.walk(n):
            if not isinstance(sub, ast.Call):
                continue
            f, tgt = sub.func, None
            if isinstance(f, ast.Name):
                tgt = (f"{scope}.{f.id}" if f"{scope}.{f.id}" in defs else None) or resolve(f.id, mod)   # a class: links the class node
            elif isinstance(f, ast.Attribute):
                v = f.value
                if isinstance(v, ast.Name) and v.id in ("self", "cls") and cls:
                    tgt = method(cls, f.attr)
                elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super" and cls:
                    for b in bases.get(cls, []):
                        bq = resolve(b, mod)
                        tgt = tgt or (method(bq, f.attr) if bq else None)
                else:
                    try:
                        tgt = resolve(ast.unparse(f), mod)
                    except Exception:
                        tgt = None
            if tgt and tgt != q:
                rel.add((q, tgt))
    edges, k = [], {}
    for a, b in sorted(rel | {(b, a) for a, b in rel}):
        edges.append({"type": "calls", "source": a, "target": b, "key": k.get((a, b), 0)})
        k[(a, b)] = k.get((a, b), 0) + 1
    nodes = [{"id": q, "name": q, "text": class_skeleton(n) if isinstance(n, ast.ClassDef) else (ast.get_source_segment(src, n) or "")}
             for q, (n, _c, _m, src) in defs.items()]
    return {"directed": True, "multigraph": True, "graph": {}, "nodes": nodes, "edges": edges}


def selftest():
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "src/pkg").mkdir(parents=True)
    (d / "src/pkg/__init__.py").write_text("from .core import solve\n")
    (d / "src/pkg/core.py").write_text(
        "class Base:\n    def run(self):\n        return 0\n"
        "class A(Base):\n    x = 1\n    def __init__(self):\n        self.go()\n    def go(self):\n        if True:\n            def inner():\n                return helper()\n        def hook():\n            return 0\n        return inner()\n"
        "def helper():\n    return 1\ndef solve():\n    return A()\n")
    (d / "docs_src/ex").mkdir(parents=True)
    (d / "docs_src/ex/tut.py").write_text("from pkg import solve\ndef main():\n    solve()\n")
    g = build(d)
    ids = {n["id"] for n in g["nodes"]}
    assert {"pkg.core.A", "pkg.core.A.__init__", "pkg.core.A.go", "pkg.core.A.inner", "pkg.core.hook", "pkg.core.helper", "ex.tut.main"} <= ids, ids
    e = {(x["source"], x["target"]) for x in g["edges"]}
    for a, b in [("pkg.core.A.__init__", "pkg.core.A.go"), ("pkg.core.A.inner", "pkg.core.helper"), ("pkg.core.solve", "pkg.core.A"),
                 ("ex.tut.main", "pkg.core.solve"), ("pkg.core.A", "pkg.core.A.go"), ("pkg.core.Base", "pkg.core.A")]:
        assert (a, b) in e and (b, a) in e, (a, b)
    txt = {n["id"]: n["text"] for n in g["nodes"]}["pkg.core.A"]
    assert txt == "class A(Base):\n    None\n\n    def __init__(self):\n        ...\n\n    def go(self):\n        ...", repr(txt)
    print("selftest ok:", len(ids), "nodes", len(e), "edges (symmetric)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        json.dump(build(sys.argv[1]), open(sys.argv[2], "w"))

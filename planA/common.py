"""Shared helpers for the Plan A audit (graphComp).

Everything here is deterministic and CPU-only. Data layout expected under ``--data``
(the competition archive as downloaded from Kaggle)::

    <data>/tasks.jsonl
    <data>/graphs/<repo_short>_<base_commit>.json      (networkx node_link JSON; nodes: id,name,text; edges: source,target,type)
    <data>/embeddings/<repo_short>_<base_commit>.npz   (one float32 vector per node id)
    <data>/snapshots/<instance_id>.tgz                 (repo working tree + .git at base_commit)
    <data>/wheels/*.whl                                (optional, offline deps)
    <data>/sandbox/setup.py                            (optional)

Some archives also carry task-named hard-link copies (``graphs/<instance_id>.json``) and one
member of each pair may be empty; ``find_graph_path`` picks the non-empty one.

Node id conventions observed so far (HANDOFF.md section 5; verify on the data):
  * ``pkg.module.Class.method`` dotted ids, no file/class/module node types
  * ``docs_src/...`` files lose the leading ``docs_src.`` segment
  * ``src/`` layouts (requests) drop the ``src.`` segment
  * nested functions (``f.<locals>.g``) are not nodes
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

# ----------------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------------

TASK_FIELDS = (
    "instance_id", "repo", "base_commit", "problem_statement", "patch", "test_patch",
    "hints_text", "created_at", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit",
)


@dataclass
class Task:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str = ""
    patch: str = ""
    test_patch: str = ""
    hints_text: str = ""
    created_at: str = ""
    FAIL_TO_PASS: tuple = ()
    PASS_TO_PASS: tuple = ()
    environment_setup_commit: str = ""

    @property
    def repo_short(self) -> str:
        return self.repo.rsplit("/", 1)[-1]

    @property
    def graph_key(self) -> str:
        return f"{self.repo_short}_{self.base_commit}"

    @property
    def test_files(self) -> list[str]:
        return patch_files(self.test_patch)

    @property
    def gold_files(self) -> list[str]:
        return patch_files(self.patch)


def _norm_tests(val) -> tuple:
    if not val:
        return ()
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("["):
            try:
                return tuple(str(x) for x in json.loads(s))
            except Exception:
                pass
        return tuple(x.strip() for x in s.split(",") if x.strip())
    return tuple(str(x) for x in val)


def load_tasks(data_dir: Path, instance_ids: Iterable[str] | None = None) -> list[Task]:
    path = Path(data_dir) / "tasks.jsonl"
    if not path.exists():
        hits = sorted(Path(data_dir).rglob("tasks.jsonl"))
        if not hits:
            raise FileNotFoundError(f"tasks.jsonl not found under {data_dir}")
        path = hits[0]
    wanted = set(instance_ids) if instance_ids else None
    out: list[Task] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            iid = d.get("instance_id")
            if not iid or iid in seen:
                continue
            if wanted and iid not in wanted:
                continue
            seen.add(iid)
            kw = {k: d[k] for k in TASK_FIELDS if k in d}
            kw["FAIL_TO_PASS"] = _norm_tests(kw.get("FAIL_TO_PASS"))
            kw["PASS_TO_PASS"] = _norm_tests(kw.get("PASS_TO_PASS"))
            out.append(Task(**kw))
    return out


def shard(items: list, spec: str | None) -> list:
    """``--shard i/N`` -> every item with index % N == i (0-based, same rule as swegemma)."""
    if not spec:
        return items
    i, n = spec.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise ValueError(f"bad shard spec {spec}")
    return [x for k, x in enumerate(items) if k % n == i]


# ----------------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------------

@dataclass
class Graph:
    key: str
    path: Path
    nodes: dict[str, dict]                       # id -> {name, text}
    out_edges: dict[str, list[tuple[str, str]]]   # src -> [(dst, type)]
    in_edges: dict[str, list[tuple[str, str]]]    # dst -> [(src, type)]
    edge_types: Counter = field(default_factory=Counter)
    n_edges: int = 0
    raw_meta: dict = field(default_factory=dict)

    def edges(self) -> Iterator[tuple[str, str, str]]:
        for s, lst in self.out_edges.items():
            for d, t in lst:
                yield s, d, t

    def edge_set(self, types: set[str] | None = None) -> set[tuple[str, str]]:
        return {(s, d) for s, d, t in self.edges() if types is None or t in types}

    def neighbors(self, n: str) -> set[str]:
        return {d for d, _ in self.out_edges.get(n, ())} | {s for s, _ in self.in_edges.get(n, ())}

    def successors(self, n: str) -> set[str]:
        return {d for d, _ in self.out_edges.get(n, ())}

    def __contains__(self, n: str) -> bool:
        return n in self.nodes

    def is_test_node(self, n: str) -> bool:
        return is_test_module_id(n)


def is_test_module_id(node_id: str) -> bool:
    parts = node_id.split(".")
    return any(p in ("tests", "test", "testing") or p.startswith("test_") or p.endswith("_test") for p in parts)


def find_graph_path(data_dir: Path, task: Task, kind: str = "graphs") -> Path | None:
    """Pick the first non-empty candidate among commit-named and task-named files."""
    ext = ".json" if kind == "graphs" else ".npz"
    d = Path(data_dir) / kind
    if not d.exists():
        hits = [p for p in Path(data_dir).rglob(kind) if p.is_dir()]
        if not hits:
            return None
        d = hits[0]
    repo_slug = task.repo.replace("/", "_")
    cands = [
        d / f"{task.graph_key}{ext}",
        d / f"{repo_slug}_{task.base_commit}{ext}",
        d / f"{task.instance_id}{ext}",
        d / f"{repo_slug}{ext}",
        d / f"{task.repo_short}{ext}",
    ]
    best = None
    for c in cands:
        if c.exists():
            if c.stat().st_size > 100:
                return c
            best = best or c
    return best


def load_graph(path: Path) -> Graph:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    nodes: dict[str, dict] = {}
    for n in data.get("nodes", []):
        nid = str(n["id"])
        nodes[nid] = {"name": str(n.get("name", nid)), "text": n.get("text") or ""}
    edge_key = "edges" if "edges" in data else "links"
    out_edges: dict[str, list] = defaultdict(list)
    in_edges: dict[str, list] = defaultdict(list)
    et = Counter()
    n_edges = 0
    for e in data.get(edge_key, []):
        s, d = str(e["source"]), str(e["target"])
        t = str(e.get("type") or e.get("relation") or "unknown")
        out_edges[s].append((d, t))
        in_edges[d].append((s, t))
        et[t] += 1
        n_edges += 1
    meta = {k: v for k, v in data.items() if k not in ("nodes", edge_key)}
    return Graph(key=path.stem, path=path, nodes=nodes, out_edges=dict(out_edges),
                 in_edges=dict(in_edges), edge_types=et, n_edges=n_edges, raw_meta=meta)


_GRAPH_CACHE: dict[Path, Graph] = {}


def get_graph(data_dir: Path, task: Task) -> Graph | None:
    p = find_graph_path(data_dir, task, "graphs")
    if p is None or p.stat().st_size <= 100:
        return None
    if p not in _GRAPH_CACHE:
        if len(_GRAPH_CACHE) > 8:
            _GRAPH_CACHE.clear()
        _GRAPH_CACHE[p] = load_graph(p)
    return _GRAPH_CACHE[p]


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            out[str(k)] = np.asarray(z[k], dtype=np.float32).reshape(-1)
    return out


def get_embeddings(data_dir: Path, task: Task) -> dict[str, np.ndarray] | None:
    p = find_graph_path(data_dir, task, "embeddings")
    if p is None or p.stat().st_size <= 100:
        return None
    return load_embeddings(p)


def containment_edge(src: str, dst: str) -> bool:
    """A 'calls' edge from a symbol to its own member (class -> class.__init__)."""
    return dst.startswith(src + ".")


# ----------------------------------------------------------------------------
# Harness-identical name resolution (swegemma.graph.graph_utils.resolve_node_name)
# ----------------------------------------------------------------------------

def resolve_node_name(query: str, node_ids: Iterable[str]) -> tuple[str | None, str]:
    """Returns (resolved id, tier) with tier in exact|suffix|ci|substring|none.

    Mirrors swegemma 0.2.7 exactly, including the tie-break (fewest dots, then shortest).
    """
    ids = list(node_ids) if not isinstance(node_ids, (list, set, dict)) else node_ids
    if query in ids:
        return query, "exact"
    key = lambda x: (len(x.split(".")), len(x))  # noqa: E731
    cands = [n for n in ids if n.endswith(f".{query}") or n.endswith(f";{query}")]
    if cands:
        return min(cands, key=key), "suffix"
    lq = query.lower()
    cands = [n for n in ids if n.lower() == lq or n.lower().endswith(f".{lq}") or n.lower().endswith(f";{lq}")]
    if cands:
        return min(cands, key=key), "ci"
    cands = [n for n in ids if query in n or lq in n.lower()]
    if cands:
        return min(cands, key=key), "substring"
    return None, "none"


# ----------------------------------------------------------------------------
# Unified diffs
# ----------------------------------------------------------------------------

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class FileDiff:
    path: str                      # new path (b/...) or old path for deletions
    old_path: str | None
    is_new: bool = False
    is_deleted: bool = False
    hunks: list[tuple[int, int, list[str]]] = field(default_factory=list)  # (old_start, new_start, lines)


def _unquote(p: str) -> str:
    p = p.strip().split("\t", 1)[0].strip()
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1].encode("utf-8").decode("unicode_escape")
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def parse_patch(patch: str) -> list[FileDiff]:
    out: list[FileDiff] = []
    cur: FileDiff | None = None
    old_p = None
    for line in (patch or "").splitlines():
        if line.startswith("diff --git"):
            cur = None
            old_p = None
            continue
        if line.startswith("--- "):
            old_p = _unquote(line[4:])
            continue
        if line.startswith("+++ "):
            new_p = _unquote(line[4:])
            is_new = old_p in (None, "/dev/null")
            is_del = new_p == "/dev/null"
            cur = FileDiff(path=old_p if is_del else new_p, old_path=None if is_new else old_p,
                           is_new=is_new, is_deleted=is_del)
            out.append(cur)
            continue
        m = HUNK_RE.match(line)
        if m and cur is not None:
            cur.hunks.append((int(m.group(1)), int(m.group(3)), []))
            continue
        if cur is not None and cur.hunks and (line[:1] in ("+", "-", " ") or line == ""):
            if line.startswith("\\ No newline"):
                continue
            cur.hunks[-1][2].append(line if line else " ")
    return out


def patch_files(patch: str) -> list[str]:
    seen = []
    for fd in parse_patch(patch):
        if fd.path not in seen and fd.path != "/dev/null":
            seen.append(fd.path)
    return seen


def apply_filediff(base_lines: list[str], fd: FileDiff) -> tuple[list[str], set[int], set[int]]:
    """Apply one file's hunks. Returns (new_lines, removed_old_linenos, added_new_linenos), 1-based."""
    new: list[str] = []
    removed: set[int] = set()
    added: set[int] = set()
    pos = 1
    for old_start, _new_start, lines in fd.hunks:
        new.extend(base_lines[pos - 1: old_start - 1])
        pos = old_start
        for ln in lines:
            tag, body = ln[0], ln[1:]
            if tag == " ":
                new.append(base_lines[pos - 1] if pos - 1 < len(base_lines) else body)
                pos += 1
            elif tag == "-":
                removed.add(pos)
                pos += 1
            elif tag == "+":
                new.append(body)
                added.add(len(new))
    new.extend(base_lines[pos - 1:])
    return new, removed, added


# ----------------------------------------------------------------------------
# Snapshots
# ----------------------------------------------------------------------------

def find_snapshot(data_dir: Path, task: Task) -> Path | None:
    d = Path(data_dir) / "snapshots"
    if not d.exists():
        hits = [p for p in Path(data_dir).rglob("snapshots") if p.is_dir()]
        if not hits:
            return None
        d = hits[0]
    for name in (f"{task.instance_id}.tgz", f"{task.instance_id}.tar.gz", f"{task.graph_key}.tgz"):
        p = d / name
        if p.exists() and p.stat().st_size > 100:
            return p
    return None


class Snapshot:
    """Access to a task's repository tree. Extracts the .tgz once into ``cache_dir/<instance_id>``.

    Use ``read(relpath)`` for single files (lazy extraction of just those members when the
    full tree has not been extracted) or ``extract()`` for the whole tree (needed by A2).
    """

    def __init__(self, tgz: Path, cache_dir: Path, instance_id: str):
        self.tgz = Path(tgz)
        self.root = Path(cache_dir) / instance_id
        self._members: dict[str, str] | None = None   # relpath -> member name
        self._prefix: str | None = None

    def _scan(self) -> None:
        if self._members is not None:
            return
        self._members = {}
        with tarfile.open(self.tgz, "r:*") as tf:
            names = [m.name for m in tf.getmembers() if m.isfile()]
        # detect a single top-level directory prefix (e.g. "workspace/" or "<repo>/")
        tops = {n.split("/", 1)[0] for n in names if "/" in n}
        has_root_files = any("/" not in n for n in names)
        if len(tops) == 1 and not has_root_files:
            self._prefix = next(iter(tops)) + "/"
        else:
            self._prefix = ""
        for n in names:
            rel = n[len(self._prefix):] if self._prefix and n.startswith(self._prefix) else n
            if rel.startswith("./"):
                rel = rel[2:]
            self._members[rel] = n

    def extract(self, force: bool = False) -> Path:
        marker = self.root / ".extracted"
        if marker.exists() and not force:
            return self.root
        self._scan()
        self.root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(self.tgz, "r:*") as tf:
            kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            if self._prefix:
                members = []
                for m in tf.getmembers():
                    if m.name.startswith(self._prefix) and m.name != self._prefix.rstrip("/"):
                        m.name = m.name[len(self._prefix):]
                        if m.name:
                            members.append(m)
                tf.extractall(self.root, members=members, **kw)
            else:
                tf.extractall(self.root, **kw)
        marker.write_text("ok\n")
        return self.root

    def exists(self, rel: str) -> bool:
        if (self.root / ".extracted").exists():
            return (self.root / rel).exists()
        self._scan()
        return rel in self._members

    def read(self, rel: str) -> str | None:
        if (self.root / ".extracted").exists():
            p = self.root / rel
            return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        self._scan()
        m = self._members.get(rel)
        if m is None:
            return None
        with tarfile.open(self.tgz, "r:*") as tf:
            f = tf.extractfile(m)
            return f.read().decode("utf-8", errors="replace") if f else None

    def read_many(self, rels: list[str]) -> dict[str, str | None]:
        if (self.root / ".extracted").exists():
            return {r: self.read(r) for r in rels}
        self._scan()
        want = {self._members[r]: r for r in rels if r in self._members}
        out: dict[str, str | None] = {r: None for r in rels}
        if want:
            with tarfile.open(self.tgz, "r:*") as tf:
                for m in tf:
                    if m.name in want and m.isfile():
                        f = tf.extractfile(m)
                        out[want[m.name]] = f.read().decode("utf-8", errors="replace") if f else None
        return out

    def list_files(self, suffix: str = ".py") -> list[str]:
        if (self.root / ".extracted").exists():
            return sorted(str(p.relative_to(self.root)) for p in self.root.rglob(f"*{suffix}")
                          if ".git" not in p.parts)
        self._scan()
        return sorted(r for r in self._members if r.endswith(suffix) and not r.startswith(".git/"))


# ----------------------------------------------------------------------------
# Path <-> module id
# ----------------------------------------------------------------------------

DROP_PREFIXES = ("src", "docs_src", "lib")


def path_to_module_candidates(rel_path: str) -> list[tuple[str, str]]:
    """Candidate dotted module ids for a file path, most specific first: [(module_id, rule)].

    Rules: 'full' (plain dotted path), 'drop:<seg>' (leading segment such as src/docs_src removed).
    """
    p = rel_path.replace("\\", "/")
    if p.endswith(".py"):
        p = p[:-3]
    parts = [x for x in p.split("/") if x and x != "."]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return []
    out = [(".".join(parts), "full")]
    if len(parts) > 1:
        out.append((".".join(parts[1:]), f"drop:{parts[0]}"))
    if len(parts) > 2:
        out.append((".".join(parts[2:]), f"drop:{parts[0]}/{parts[1]}"))
    return out


class ModuleIndex:
    """Maps file paths to module ids present in a graph, learning the prefix rule from the data."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.module_ids: set[str] = set()
        for nid in graph.nodes:
            parts = nid.split(".")
            for k in range(1, len(parts) + 1):
                self.module_ids.add(".".join(parts[:k]))

    def module_for(self, rel_path: str) -> tuple[str | None, str]:
        """Returns (module id, rule) choosing the first candidate that is a prefix of some node id."""
        for mod, rule in path_to_module_candidates(rel_path):
            if mod in self.module_ids:
                return mod, rule
        return None, "none"


# ----------------------------------------------------------------------------
# AST definitions and gold-function mapping
# ----------------------------------------------------------------------------

@dataclass
class Def:
    qualname: str          # Class.method, f.<locals>.g for nested
    kind: str              # function | async_function | class
    lineno: int            # first line incl. decorators
    end_lineno: int
    nested: bool
    decorators: list[str]


def iter_defs(src: str) -> list[Def]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[Def] = []

    def dec_names(n):
        names = []
        for d in getattr(n, "decorator_list", []):
            try:
                names.append(ast.unparse(d))
            except Exception:
                names.append("?")
        return names

    def walk(node, prefix: str, in_func: bool):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(ch, ast.ClassDef) else (
                    "async_function" if isinstance(ch, ast.AsyncFunctionDef) else "function")
                qn = f"{prefix}{ch.name}"
                start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                out.append(Def(qn, kind, start, ch.end_lineno or ch.lineno, in_func, dec_names(ch)))
                if kind == "class":
                    walk(ch, qn + ".", in_func)
                else:
                    walk(ch, qn + ".<locals>.", True)
            else:
                walk(ch, prefix, in_func)

    walk(tree, "", False)
    return out


def innermost_def(defs: list[Def], lineno: int) -> Def | None:
    inner = [d for d in defs if d.lineno <= lineno <= d.end_lineno]
    return max(inner, key=lambda d: d.lineno) if inner else None


@dataclass
class GoldSymbol:
    file: str
    qualname: str | None        # None -> module-level change
    kind: str                   # function|async_function|class|module
    nested: bool
    new_symbol: bool            # did not exist in base
    source: str                 # removed|added


def gold_symbols(task: Task, snap: Snapshot, include_tests: bool = False) -> list[GoldSymbol]:
    """Map every changed line of the gold patch to its innermost enclosing definition.

    Removed lines use base coordinates; added lines use post-patch coordinates, and a brand-new
    definition is reported with ``new_symbol=True`` and ALSO climbs to the nearest enclosing symbol
    that exists in the base (class, else module) -- that is where an agent must edit.
    """
    out: list[GoldSymbol] = []
    seen: set[tuple] = set()

    def add(gs: GoldSymbol):
        k = (gs.file, gs.qualname, gs.new_symbol)
        if k not in seen:
            seen.add(k)
            out.append(gs)

    for fd in parse_patch(task.patch):
        if not fd.path.endswith(".py"):
            continue
        if not include_tests and is_test_path(fd.path):
            continue
        base_src = None if fd.is_new else snap.read(fd.old_path or fd.path)
        if base_src is None:
            add(GoldSymbol(fd.path, None, "module", False, True, "added"))
            continue
        base_lines = base_src.splitlines()
        new_lines, removed, added = apply_filediff(base_lines, fd)
        base_defs = iter_defs(base_src)
        base_q = {d.qualname: d for d in base_defs}
        new_defs = iter_defs("\n".join(new_lines) + "\n")
        for ln in sorted(removed):
            d = innermost_def(base_defs, ln)
            if d is None:
                add(GoldSymbol(fd.path, None, "module", False, False, "removed"))
            else:
                add(GoldSymbol(fd.path, d.qualname, d.kind, d.nested, False, "removed"))
        for ln in sorted(added):
            if not new_lines[ln - 1].strip():
                continue
            d = innermost_def(new_defs, ln)
            if d is None:
                add(GoldSymbol(fd.path, None, "module", False, False, "added"))
                continue
            if d.qualname in base_q:
                b = base_q[d.qualname]
                add(GoldSymbol(fd.path, d.qualname, b.kind, b.nested, False, "added"))
            else:
                add(GoldSymbol(fd.path, d.qualname, d.kind, d.nested, True, "added"))
                parent = d.qualname
                while "." in parent:
                    parent = parent.rsplit(".", 1)[0]
                    if parent.endswith("<locals>"):
                        continue
                    if parent in base_q:
                        b = base_q[parent]
                        add(GoldSymbol(fd.path, parent, b.kind, b.nested, False, "added"))
                        break
                else:
                    add(GoldSymbol(fd.path, None, "module", False, False, "added"))
    return out


def is_test_path(rel: str) -> bool:
    p = Path(rel)
    name = p.name
    if (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py") or name == "conftest.py":
        return True
    return bool({"tests", "test", "testing"} & {x.lower() for x in p.parts[:-1]})


def qualname_to_node_id(module: str, qualname: str | None) -> str:
    return module if not qualname else f"{module}.{qualname}"


# ----------------------------------------------------------------------------
# Issue identifier extraction
# ----------------------------------------------------------------------------

_STOP = {
    "the", "and", "for", "with", "this", "that", "from", "into", "when", "then", "than",
    "true", "false", "none", "null", "python", "fastapi", "requests", "rich", "httpx",
    "http", "https", "json", "error", "errors", "bug", "issue", "feature", "request",
    "response", "example", "code", "file", "files", "test", "tests", "version", "import",
    "def", "class", "return", "self", "print", "str", "int", "list", "dict", "type",
}


def extract_identifiers(text: str) -> dict[str, list[str]]:
    """Deterministic identifier extraction (no LLM). Returns candidates by kind.

    kinds: backtick (``code`` spans, split on whitespace), dotted (a.b.c), call (name(...)),
    snake (has underscore), camel (CamelCase), path (x/y.py -> dotted module).
    """
    t = text or ""
    out: dict[str, list[str]] = defaultdict(list)

    def push(kind, tok):
        tok = tok.strip("`'\"()[]{}<>,.;: \n\t")
        if not tok or len(tok) < 3 or tok.lower() in _STOP:
            return
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_./]*$", tok):
            return
        if tok not in out[kind]:
            out[kind].append(tok)

    for m in re.finditer(r"```.*?```|`([^`\n]+)`", t, flags=re.S):
        span = m.group(1)
        if span:
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", span):
                push("backtick", tok)
    for tok in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", t):
        if not re.match(r"^\d", tok):
            push("dotted", tok)
    for tok in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", t):
        push("call", tok)
    for tok in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", t):
        push("snake", tok)
    for tok in re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b", t):
        push("camel", tok)
    for tok in re.findall(r"\b[\w/]+?\.py\b", t):
        mod = tok[:-3].replace("/", ".").strip(".")
        push("path", mod)
    return dict(out)


def flat_identifiers(ids: dict[str, list[str]], kinds: Iterable[str] | None = None) -> list[str]:
    seen = []
    for k, lst in ids.items():
        if kinds and k not in kinds:
            continue
        for x in lst:
            if x not in seen:
                seen.append(x)
    return seen


# ----------------------------------------------------------------------------
# BM25
# ----------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN_RE.findall(text or ""):
        # split snake_case and CamelCase, keep the original too
        toks.append(t.lower())
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", t)
        if len(parts) > 1:
            toks.extend(p.lower() for p in parts)
        if "_" in t:
            toks.extend(p.lower() for p in t.split("_") if p)
    return toks


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tf: list[Counter] = []
        self.dl: list[int] = []
        df: Counter = Counter()
        for i in self.ids:
            toks = tokenize(docs[i])
            c = Counter(toks)
            self.tf.append(c)
            self.dl.append(len(toks))
            df.update(c.keys())
        self.n = len(self.ids)
        self.avgdl = (sum(self.dl) / self.n) if self.n else 0.0
        self.idf = {t: math.log(1 + (self.n - n_t + 0.5) / (n_t + 0.5)) for t, n_t in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = Counter(tokenize(query))
        s = np.zeros(self.n, dtype=np.float64)
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1))
                s[i] += idf * f * (self.k1 + 1) / denom
        return s

    def rank(self, query: str) -> list[str]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")
        return [self.ids[i] for i in order if s[i] > 0]


# ----------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------

def bfs_distance(graph: Graph, sources: Iterable[str], directed: bool = False, max_hops: int = 50) -> dict[str, int]:
    dist: dict[str, int] = {}
    frontier = [s for s in sources if s in graph.nodes]
    for s in frontier:
        dist[s] = 0
    hop = 0
    while frontier and hop < max_hops:
        hop += 1
        nxt = []
        for u in frontier:
            nbrs = graph.successors(u) if directed else graph.neighbors(u)
            for v in nbrs:
                if v not in dist:
                    dist[v] = hop
                    nxt.append(v)
        frontier = nxt
    return dist


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("")
        return
    cols = columns or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else (json.dumps(v) if isinstance(v, (list, dict)) else v))
                        for k, v in r.items()})


def pct(a: float, b: float) -> float | None:
    return round(100.0 * a / b, 2) if b else None

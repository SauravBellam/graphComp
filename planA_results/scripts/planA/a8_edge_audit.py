#!/usr/bin/env python3
"""A8: characterise the provided `calls` edges against the source (closes the "hand-check" gap).

A2 found that, inside executed code, only ~21% of static edges were observed at runtime and ~38% of
runtime calls between existing nodes are in the graph. The analysis inferred a name-matching
generator but could not state it. This step reads the caller's source for every edge and records how
the callee appears there, for three edge populations taken from the A2 analysis outputs:

  unconfirmed : static edge, both endpoints executed, call never observed (a2_static_unconfirmed.jsonl)
  missed      : runtime call between two existing nodes that the graph lacks
                (a2_dyn_edges.jsonl, cause missed_between_nodes)
  captured    : runtime call that the graph has (a2_dyn_edges.jsonl, cause captured): the baseline

Per edge, from the caller's AST (its own body, not nested defs/classes, which are separate nodes;
for a class caller: __init__ plus class-level statements):

  how      : call_direct (callee()), call_self (self./cls.callee()), call_super (super().callee()),
             call_attr_other (obj.callee()), instantiation (Class() for a class or collapsed __init__),
             call_in_nested_scope (explicit call, but inside a def/class nested in the caller),
             referenced (name used without a call: callback, decorator, passed as argument),
             text_only (name appears only in a string/comment), implicit_dunder (callee is a dunder such
             as __call__, __enter__, __iter__ and the caller has no explicit call), no_reference
  channel  : for runtime edges, direct (caller frame calls callee) or vialib (through library frames)
  ambiguity: n_same_name = number of provided nodes sharing the callee's short name; a name-matching
             generator should over-produce edges whose callee name is shared (compare with captured)

Outputs (``--out``): a8_edges.csv (one row per distinct (graph, src, dst, population), with the number
of tasks it occurs in and the first matching source line), a8_sample.csv (stratified random sample of
non-test edges with a caller snippet and empty human_label/notes columns: 50 unconfirmed, 30 missed,
by default), a8_summary.json (category shares per population, unique edges and task-weighted, per repo,
and the ambiguity comparison).

Usage: a8_edge_audit.py --data <archive> --a2 <a2 analysis dir> --out <dir> [--cache <dir>]
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ModuleIndex, Snapshot, find_snapshot, get_graph, is_test_module_id, load_tasks, provided_node_id, read_jsonl,
    write_csv,
)
from stats import wilson  # noqa: E402

DUNDER_IMPLICIT = {"__call__", "__enter__", "__exit__", "__aenter__", "__aexit__", "__iter__", "__next__",
                   "__aiter__", "__anext__", "__getattr__", "__getattribute__", "__setattr__", "__get__", "__set__",
                   "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__hash__", "__len__", "__bool__",
                   "__str__", "__repr__", "__format__", "__getitem__", "__setitem__", "__delitem__", "__contains__",
                   "__init_subclass__", "__new__", "__del__", "__add__", "__radd__", "__iadd__", "__or__", "__and__",
                   "__rich__", "__rich_console__", "__rich_measure__", "__rich_repr__", "__post_init__", "__class_getitem__",
                   "__missing__", "__reduce__", "__copy__", "__deepcopy__", "__set_name__", "__await__", "__fspath__"}
HOW_ORDER = ["call_direct", "call_self", "call_super", "call_attr_other", "instantiation"]


# --- source index -------------------------------------------------------------------------------

def index_defs(tree: ast.AST) -> dict[str, ast.AST]:
    """qualname (with <locals>) -> def/class node, same scheme as common.iter_defs."""
    out: dict[str, ast.AST] = {}

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + ch.name
                out.setdefault(q, ch)
                walk(ch, q + ".<locals>.")
            elif isinstance(ch, ast.ClassDef):
                q = prefix + ch.name
                out.setdefault(q, ch)
                walk(ch, q + ".")
            else:
                walk(ch, prefix)
    walk(tree, "")
    return out


class SourceIndex:
    """Provided-scheme node id -> (file, def node, source lines) for one snapshot."""

    def __init__(self, g, snap: Snapshot):
        mi = ModuleIndex(g)
        self.by_id: dict[str, tuple[str, ast.AST, list[str]]] = {}
        files = snap.list_files(".py")
        targets = []
        for f in files:
            mod, _ = mi.module_for(f, infer=True)
            if mod is not None:
                targets.append((f, mod))
        texts = snap.read_many([f for f, _ in targets])
        for f, mod in targets:
            src = texts.get(f) or ""
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            lines = src.splitlines()
            for q, node in index_defs(tree).items():
                nid = provided_node_id(mod, q)
                self.by_id.setdefault(nid, (f, node, lines))
        # short-name multiplicity over the provided graph (what a name matcher would see)
        self.same_name = Counter(n.rsplit(".", 1)[-1] for n in g.nodes)


def own_statements(node: ast.AST) -> list[ast.AST]:
    """The caller's own code: function body, or __init__ + class-level statements for a class."""
    if isinstance(node, ast.ClassDef):
        stmts = [s for s in node.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        init = next((s for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == "__init__"), None)
        if init is not None:
            stmts += init.body + init.args.defaults + [d for d in init.decorator_list]
        return stmts
    return list(node.body) + list(getattr(node, "decorator_list", [])) + list(node.args.defaults) if hasattr(node, "args") else list(node.body)


def iter_own(nodes: list[ast.AST]):
    """Walk statements without entering nested defs / classes / lambdas' own scopes (lambdas kept)."""
    stack = list(nodes)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.extend(n.decorator_list)        # a nested def/class statement: only its decorators are ours
            continue
        yield n
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stack.extend(ch.decorator_list)   # decorators run in the enclosing scope
                continue
            stack.append(ch)


def classify(caller: ast.AST | None, lines: list[str], callee_short: str, callee_is_class: bool) -> tuple[str, int | None]:
    """How does callee_short appear in the caller's own code? Returns (how, line number of evidence)."""
    if caller is None:
        return "caller_not_found", None
    found: dict[str, int] = {}
    referenced_line = None
    for n in iter_own(own_statements(caller)):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id == callee_short:
                found.setdefault("instantiation" if callee_is_class else "call_direct", n.lineno)
            elif isinstance(f, ast.Attribute) and f.attr == callee_short:
                v = f.value
                if isinstance(v, ast.Name) and v.id in ("self", "cls"):
                    found.setdefault("call_self", n.lineno)
                elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super":
                    found.setdefault("call_super", n.lineno)
                else:
                    found.setdefault("instantiation" if callee_is_class else "call_attr_other", n.lineno)
        elif isinstance(n, ast.Name) and n.id == callee_short and referenced_line is None:
            referenced_line = n.lineno
        elif isinstance(n, ast.Attribute) and n.attr == callee_short and referenced_line is None:
            referenced_line = n.lineno
    for how in HOW_ORDER:
        if how in found:
            return how, found[how]
    # explicit call inside a nested def/class of the caller (a generator may credit it to the outer def)
    for n in ast.walk(caller):
        if n is not caller and isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Name) and f.id == callee_short) or (isinstance(f, ast.Attribute) and f.attr == callee_short):
                return "call_in_nested_scope", n.lineno
    if referenced_line is not None:
        return "referenced", referenced_line
    if callee_short in DUNDER_IMPLICIT:
        return "implicit_dunder", None
    start = getattr(caller, "lineno", 1)
    end = getattr(caller, "end_lineno", start)
    for i in range(start - 1, min(end, len(lines))):
        if callee_short in lines[i]:
            return "text_only", i + 1
    return "no_reference", None


def callee_info(idx: SourceIndex, dst: str) -> tuple[str, bool]:
    short = dst.rsplit(".", 1)[-1]
    ent = idx.by_id.get(dst)
    is_class = bool(ent and isinstance(ent[1], ast.ClassDef))
    return short, is_class


# --- main ----------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--a2", required=True, type=Path, help="A2 analysis dir (a2_dyn_edges.jsonl, a2_static_unconfirmed.jsonl)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--sample-unconfirmed", type=int, default=50)
    ap.add_argument("--sample-missed", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.cache or args.out.parent / "cache"

    tasks = {t.instance_id: t for t in load_tasks(args.data)}
    # distinct (graph, src, dst, population) -> tasks, channels, nontest
    edges: dict[tuple[str, str, str, str], dict] = {}

    def add(pop, iid, src, dst, nontest, channel=None):
        t = tasks.get(iid)
        if t is None:
            return
        k = (t.graph_key, src, dst, pop)
        e = edges.setdefault(k, {"graph_key": t.graph_key, "repo": t.repo, "population": pop, "src": src, "dst": dst,
                                 "nontest": nontest, "tasks": set(), "channels": set()})
        e["tasks"].add(iid)
        if channel:
            e["channels"].add(channel)

    for r in read_jsonl(args.a2 / "a2_static_unconfirmed.jsonl"):
        add("unconfirmed", r["instance_id"], r["src"], r["dst"], bool(r.get("nontest")))
    for r in read_jsonl(args.a2 / "a2_dyn_edges.jsonl"):
        if r.get("cause") == "missed_between_nodes":
            add("missed", r["instance_id"], r["src"], r["dst"], bool(r.get("nontest")), r.get("channel"))
        elif r.get("cause") == "captured":
            add("captured", r["instance_id"], r["src"], r["dst"], bool(r.get("nontest")), r.get("channel"))
    print(f"[a8] distinct edges: {Counter(e['population'] for e in edges.values())}", flush=True)

    by_graph: dict[str, list[dict]] = defaultdict(list)
    for e in edges.values():
        by_graph[e["graph_key"]].append(e)
    first_task = {}
    for iid, t in tasks.items():
        first_task.setdefault(t.graph_key, t)

    rows = []
    for gi, (key, es) in enumerate(sorted(by_graph.items())):
        t = first_task[key]
        g = get_graph(args.data, t)
        sp = find_snapshot(args.data, t)
        if g is None or sp is None:
            continue
        idx = SourceIndex(g, Snapshot(sp, cache, t.instance_id))
        for e in es:
            ent = idx.by_id.get(e["src"])
            caller = ent[1] if ent else None
            lines = ent[2] if ent else []
            short, is_class = callee_info(idx, e["dst"])
            how, line = classify(caller, lines, short, is_class)
            # a vialib runtime edge goes through library frames: an explicit call is not expected
            channel = ("direct" if "direct" in e["channels"] else "vialib") if e["channels"] else ""
            rows.append({"graph_key": key, "repo": e["repo"], "population": e["population"], "src": e["src"],
                         "dst": e["dst"], "nontest": e["nontest"], "n_tasks": len(e["tasks"]), "channel": channel,
                         "how": how, "evidence_line": line,
                         "evidence": (lines[line - 1].strip()[:200] if line and lines and line <= len(lines) else ""),
                         "caller_file": ent[0] if ent else "", "caller_lineno": getattr(caller, "lineno", None),
                         "callee_short": short, "callee_is_class": is_class,
                         "n_same_name": idx.same_name.get(short, 0),
                         "callee_in_snapshot": e["dst"] in idx.by_id})
        print(f"[a8] {gi + 1}/{len(by_graph)} {key}: {len(es)} edges", flush=True)

    rows.sort(key=lambda r: (r["population"], r["graph_key"], r["src"], r["dst"]))
    write_csv(args.out / "a8_edges.csv", rows)

    # ---- summary
    def shares(rs, weight):
        tot = sum(weight(r) for r in rs)
        c = Counter()
        for r in rs:
            c[r["how"]] += weight(r)
        return {"n": tot, "how": {k: {"count": v, "share": round(v / tot, 4) if tot else None} for k, v in c.most_common()}}

    def block(rs):
        out = {}
        for pop in ("unconfirmed", "missed", "captured"):
            p = [r for r in rs if r["population"] == pop]
            if not p:
                continue
            b = {"unique_edges": shares(p, lambda r: 1), "task_weighted": shares(p, lambda r: r["n_tasks"])}
            ex = [r for r in p if r["how"] in ("call_direct", "call_self", "call_super", "call_attr_other", "instantiation")]
            b["explicit_callsite_share"] = wilson(len(ex), len(p))
            amb = [r for r in p if r["n_same_name"] >= 2]
            b["callee_name_shared_share"] = wilson(len(amb), len(p))
            b["median_n_same_name"] = sorted(r["n_same_name"] for r in p)[len(p) // 2]
            if pop != "unconfirmed":
                b["channel"] = dict(Counter(r["channel"] for r in p))
                d = [r for r in p if r["channel"] == "direct"]
                b["direct_channel_how"] = shares(d, lambda r: 1) if d else None
            out[pop] = b
        return out

    nontest = [r for r in rows if r["nontest"]]
    summary = {"all_edges": block(rows), "nontest_edges": block(nontest),
               "per_repo_nontest": {repo: block([r for r in nontest if r["repo"] == repo])
                                    for repo in sorted({r["repo"] for r in rows})},
               "how_definitions": {
                   "call_direct": "callee() by bare name", "call_self": "self.callee() / cls.callee()",
                   "call_super": "super().callee()", "call_attr_other": "obj.callee() on any other object",
                   "instantiation": "Callee() where the callee is a class (or a collapsed __init__)",
                   "call_in_nested_scope": "explicit call, but inside a def/class nested in the caller",
                   "referenced": "name used without a call (callback, decorator, passed, assigned)",
                   "implicit_dunder": "callee is a dunder method and nothing in the caller names it",
                   "text_only": "name appears only in a string or comment of the caller",
                   "no_reference": "the caller's own code does not mention the callee at all",
                   "caller_not_found": "caller node id not located in the snapshot"},
               "notes": "Unconfirmed edges with an explicit callsite are plausible calls that the tests did not run "
                        "(or a name match to a same-named method of another class). no_reference / text_only "
                        "unconfirmed edges are spurious. Compare callee_name_shared_share across populations: "
                        "a name-matching generator inflates it for unconfirmed edges relative to captured ones."}
    (args.out / "a8_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- stratified sample for hand review (non-test edges)
    rng = random.Random(args.seed)
    sample = []
    for pop, n in (("unconfirmed", args.sample_unconfirmed), ("missed", args.sample_missed)):
        pool = [r for r in nontest if r["population"] == pop]
        by_how: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            by_how[r["how"]].append(r)
        # proportional allocation with at least 2 per non-empty category
        alloc = {h: max(2, round(n * len(v) / max(1, len(pool)))) for h, v in by_how.items()}
        while sum(alloc.values()) > n and any(a > 2 for a in alloc.values()):
            h = max(alloc, key=lambda k: alloc[k])
            alloc[h] -= 1
        for h, v in by_how.items():
            for r in rng.sample(v, min(alloc[h], len(v))):
                sample.append({**r, "caller_snippet": "", "human_label": "", "notes": ""})
    # caller snippet (first 30 lines of the caller) for the sampled rows
    need = defaultdict(list)
    for s in sample:
        need[s["graph_key"]].append(s)
    for key, ss in need.items():
        t = first_task[key]
        g = get_graph(args.data, t)
        idx = SourceIndex(g, Snapshot(find_snapshot(args.data, t), cache, t.instance_id))
        for s in ss:
            ent = idx.by_id.get(s["src"])
            if ent:
                node, lines = ent[1], ent[2]
                a, b = node.lineno - 1, min(getattr(node, "end_lineno", node.lineno), node.lineno + 29)
                s["caller_snippet"] = "\n".join(lines[a:b])
    write_csv(args.out / "a8_sample.csv", sample)
    print(json.dumps({k: {p: v.get("unique_edges", {}).get("n") for p, v in summary["nontest_edges"].items()}
                      for k in ["nontest"]}, indent=2))
    print(f"[a8] wrote {len(rows)} edge rows, {len(sample)} sample rows")


if __name__ == "__main__":
    main()

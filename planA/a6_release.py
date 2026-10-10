#!/usr/bin/env python3
"""A6: release artefacts.

1. Typed graph v2 per base commit, in the harness's node_link JSON format (loadable by
   swegemma.graph.load_graph_from_json), built from the snapshot with `ast`:
     nodes : every module, class, function and async function (ids as in the provided graph;
             the prefix rule learned per file by ModuleIndex when the provided graph is available),
             attributes: id, name, text, kind, file, lineno, end_lineno, is_test, is_async,
             in_provided_graph
     edges : contains (module->class->def), imports (module->module, resolved inside the repo),
             inherits (class->class, resolved by name inside the repo), calls_static (the provided
             `calls` edges, containment pseudo-edges removed), calls_dynamic (from A2 traces, with
             attribute tests=<n fail-to-pass tests observing it>, phase=pre|post), tests
             (test function -> executed non-test function, from A2 per-test execution sets)
2. Per-task profile CSV for Plan B: repo, d (strict/any, undirected), v_file_rank, v_high,
   multi_file, n_gold_src_files, gold nodes, gold-node coverage, A2 recall/precision if available,
   fail-to-pass count, gold reachable (static / dynamic).

Inputs: archive, A1 out dir, optional A2 runs dir + A2 analysis dir.
Outputs: <out>/graphs_v2/<graph_key>.json, <out>/task_profile.csv, <out>/a6_summary.json.
Shardable by graph (``--shard i/N``) for the graph build.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2.analyze import Mapper, load_trace  # noqa: E402
from common import (  # noqa: E402
    ModuleIndex, Snapshot, containment_edge, find_snapshot, get_graph, is_test_path, iter_defs, load_tasks,
    path_to_module_candidates, read_jsonl, shard, write_csv,
)


def build_typed_graph(task, g, snap: Snapshot) -> dict:
    mi = ModuleIndex(g) if g else None
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    files = snap.list_files(".py")
    texts = snap.read_many(files)
    mod_of_file: dict[str, str] = {}
    class_short: dict[str, list[str]] = defaultdict(list)
    for f in files:
        mod, rule = (mi.module_for(f) if mi else (None, "none"))
        if mod is None:
            cands = path_to_module_candidates(f)
            if not cands:
                continue
            # prefer dropping src/docs_src when the provided graph cannot tell us
            mod = next((m for m, r in cands if r in ("drop:src", "drop:docs_src")), cands[0][0])
        mod_of_file[f] = mod
        src = texts.get(f) or ""
        nodes.setdefault(mod, {"id": mod, "name": mod.rsplit(".", 1)[-1], "text": "", "kind": "module", "file": f,
                               "lineno": 1, "end_lineno": src.count("\n") + 1, "is_test": is_test_path(f),
                               "is_async": False, "in_provided_graph": bool(g and mod in g.nodes)})
        lines = src.splitlines()
        for d in iter_defs(src):
            if d.nested:
                continue
            nid = f"{mod}.{d.qualname}"
            nodes[nid] = {"id": nid, "name": d.qualname.rsplit(".", 1)[-1], "text": "\n".join(lines[d.lineno - 1: d.end_lineno]),
                          "kind": d.kind, "file": f, "lineno": d.lineno, "end_lineno": d.end_lineno,
                          "is_test": is_test_path(f), "is_async": d.kind == "async_function",
                          "in_provided_graph": bool(g and nid in g.nodes)}
            parent = f"{mod}.{d.qualname.rsplit('.', 1)[0]}" if "." in d.qualname else mod
            edges.append({"source": parent, "target": nid, "type": "contains"})
            if d.kind == "class":
                class_short[d.qualname.rsplit(".", 1)[-1]].append(nid)
    # imports + inherits need a second pass with ast
    all_mods = set(mod_of_file.values())
    for f, mod in mod_of_file.items():
        src = texts.get(f) or ""
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        pkg = mod.rsplit(".", 1)[0] if "." in mod else ""
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    tgt = _resolve_mod(a.name, all_mods)
                    if tgt:
                        edges.append({"source": mod, "target": tgt, "type": "imports"})
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                if n.level:
                    up = pkg.split(".") if pkg else []
                    up = up[: max(0, len(up) - (n.level - 1))]
                    base = ".".join(up + ([base] if base else []))
                for a in n.names:
                    tgt = _resolve_mod(f"{base}.{a.name}" if base else a.name, all_mods) or _resolve_mod(base, all_mods)
                    if tgt:
                        edges.append({"source": mod, "target": tgt, "type": "imports"})
            elif isinstance(n, ast.ClassDef):
                for b in n.bases:
                    try:
                        bn = ast.unparse(b).rsplit(".", 1)[-1]
                    except Exception:
                        continue
                    cands = class_short.get(bn, [])
                    src_id = next((nid for nid, a in nodes.items() if a["file"] == f and a["kind"] == "class" and a["lineno"] <= n.lineno <= a["end_lineno"] and a["name"] == n.name), None)
                    if src_id and len(cands) == 1:
                        edges.append({"source": src_id, "target": cands[0], "type": "inherits"})
                    elif src_id and cands:
                        same = [c for c in cands if c.startswith(mod + ".")]
                        edges.append({"source": src_id, "target": (same or cands)[0], "type": "inherits", "ambiguous": len(cands)})
    if g is not None:
        for s, d, t in g.edges():
            if t == "calls" and s != d and not containment_edge(s, d):
                edges.append({"source": s, "target": d, "type": "calls_static"})
                for x in (s, d):
                    if x not in nodes:
                        nodes[x] = {"id": x, "name": g.nodes.get(x, {}).get("name", x), "text": g.nodes.get(x, {}).get("text", ""),
                                    "kind": "provided_only", "file": None, "is_test": is_test_path(x.replace(".", "/")),
                                    "is_async": False, "in_provided_graph": True}
    return {"directed": True, "multigraph": True, "graph": {"repo_name": task.repo, "base_commit": task.base_commit, "version": "graphcomp-v2"},
            "nodes": list(nodes.values()), "edges": edges}


def _resolve_mod(name: str, mods: set[str]) -> str | None:
    if not name:
        return None
    if name in mods:
        return name
    parts = name.split(".")
    for k in range(len(parts) - 1, 0, -1):
        cand = ".".join(parts[:k])
        if cand in mods:
            return cand
    return None


def add_dynamic_edges(gv2: dict, g, tasks_for_key: list, runs: Path, a2_rows: dict) -> Counter:
    cnt = Counter()
    node_ids = {n["id"] for n in gv2["nodes"]}
    seen_dyn: dict[tuple, dict] = {}
    seen_test: set[tuple] = set()
    for t in tasks_for_key:
        for phase in ("pre", "post"):
            tr = load_trace(runs / t.instance_id / f"trace_{phase}.json")
            if tr is None:
                continue
            m = Mapper(g) if g else None
            mapped = []
            for fn, rel in zip(tr["functions"], tr["files"]):
                modpath = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel.replace("/", ".")
                if modpath.endswith(".__init__"):
                    modpath = modpath[: -len(".__init__")]
                qual = fn[len(modpath) + 1:] if fn.startswith(modpath + ".") else fn
                if m:
                    nid, st = m.map(rel, qual)
                else:
                    nid, st = f"{modpath}.{qual}", "node"
                if st == "missing" and nid in node_ids:
                    st = "node"  # present in the typed graph though not in the provided one
                mapped.append((nid, st))
            f2p = set(a2_rows.get(t.instance_id, {}).get("fail_to_pass", []))
            tests = tr.get("tests", [])
            for c, e, tl in tr["edges_direct"]:
                (cn, cs), (en, es) = mapped[c], mapped[e]
                if cs in ("node", "collapsed_init") and es in ("node", "collapsed_init") and cn != en:
                    key = (cn, en, phase)
                    rec = seen_dyn.setdefault(key, {"source": cn, "target": en, "type": "calls_dynamic", "phase": phase,
                                                    "tasks": set(), "n_tests": 0, "n_f2p_tests": 0})
                    rec["tasks"].add(t.instance_id)
                    rec["n_tests"] += len(tl)
                    rec["n_f2p_tests"] += sum(1 for i in tl if i < len(tests) and tests[i] in f2p)
            for test, execd in tr.get("per_test_exec", {}).items():
                if test.startswith("<"):
                    continue
                from a2.analyze import test_node_id
                tn = test_node_id(test)
                for i in execd:
                    nid, st = mapped[i]
                    if st in ("node", "collapsed_init") and not is_test_path(nid.replace(".", "/")) and (tn, nid) not in seen_test:
                        seen_test.add((tn, nid))
                        gv2["edges"].append({"source": tn, "target": nid, "type": "tests", "task": t.instance_id})
                        cnt["tests"] += 1
    for rec in seen_dyn.values():
        rec["tasks"] = sorted(rec["tasks"])
        gv2["edges"].append(rec)
        cnt["calls_dynamic"] += 1
    return cnt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--a1", required=True, type=Path)
    ap.add_argument("--a2-runs", type=Path, default=None)
    ap.add_argument("--a2-analysis", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--shard", default=None)
    ap.add_argument("--skip-graphs", action="store_true")
    args = ap.parse_args(argv)
    (args.out / "graphs_v2").mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.data)
    by_key = defaultdict(list)
    for t in tasks:
        by_key[t.graph_key].append(t)
    a2_rows: dict[str, dict] = {}
    if args.a2_analysis and (args.a2_analysis / "a2_tasks.csv").exists():
        with open(args.a2_analysis / "a2_tasks.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k in ("fail_to_pass", "gold_static_reachable_directed", "gold_executed_by_f2p_pre", "gold_executed_by_f2p_post"):
                    try:
                        r[k] = json.loads(r[k]) if r.get(k) else []
                    except Exception:
                        r[k] = []
                a2_rows[r["instance_id"]] = r

    summary = {"graphs": {}}
    if not args.skip_graphs:
        for key in shard(sorted(by_key), args.shard):
            t0 = by_key[key][0]
            g = get_graph(args.data, t0)
            sp = find_snapshot(args.data, t0)
            if sp is None:
                summary["graphs"][key] = {"error": "snapshot missing"}
                continue
            snap = Snapshot(sp, args.cache, t0.instance_id)
            gv2 = build_typed_graph(t0, g, snap)
            cnt = Counter(e["type"] for e in gv2["edges"])
            if args.a2_runs:
                cnt.update(add_dynamic_edges(gv2, g, by_key[key], args.a2_runs, a2_rows))
            (args.out / "graphs_v2" / f"{key}.json").write_text(json.dumps(gv2))
            kinds = Counter(n["kind"] for n in gv2["nodes"])
            summary["graphs"][key] = {"n_nodes": len(gv2["nodes"]), "node_kinds": dict(kinds), "edge_types": dict(cnt),
                                      "async_defs": sum(1 for n in gv2["nodes"] if n.get("is_async")),
                                      "nodes_not_in_provided": sum(1 for n in gv2["nodes"] if not n.get("in_provided_graph"))}
            print(f"[a6] {key}: nodes={len(gv2['nodes'])} edges={dict(cnt)}", flush=True)

    # task profile
    prof = []
    a1_tasks = {}
    with open(args.a1 / "a1_tasks.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a1_tasks[r["instance_id"]] = r
    for t in tasks:
        a = a1_tasks.get(t.instance_id, {})
        b = a2_rows.get(t.instance_id, {})
        prof.append({
            "instance_id": t.instance_id, "repo": t.repo, "graph_key": t.graph_key,
            "d_strict": a.get("d_strict_undirected") or None, "d_any": a.get("d_any_undirected") or None,
            "d_directed_any": a.get("d_any_directed") or None,
            "v_file_rank": a.get("v_file_rank") or None, "v_node_rank": a.get("v_node_rank") or None, "v_high": a.get("v_high"),
            "multi_file": a.get("multi_file"), "n_gold_src_files": a.get("n_gold_src_files"),
            "n_gold_symbols": a.get("n_gold_symbols"), "n_gold_symbols_present": a.get("n_gold_symbols_present"),
            "gold_nodes": a.get("gold_nodes"), "n_identifiers": a.get("n_identifiers"), "n_anchors": a.get("n_resolved_any"),
            "a2_recall_direct": b.get("recall_direct"), "a2_observed_precision": b.get("observed_precision"),
            "a2_pct_nontest_nodes_executed": b.get("pct_nontest_nodes_executed"),
            "a2_n_fail_to_pass": b.get("n_fail_to_pass"), "a2_gold_static_reachable": b.get("gold_any_static_reachable"),
            "a2_gold_dyn_reachable": b.get("gold_any_dyn_reachable"),
            "cell": _cell(a),
        })
    write_csv(args.out / "task_profile.csv", prof)
    summary["task_profile_cells"] = dict(Counter(p["cell"] for p in prof))
    (args.out / "a6_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "graphs"}, indent=2))


def _cell(a: dict) -> str:
    """Plan B payoff-surface cell: d in {0,1,>=2,undef} x v in {high (<=5), low, undef}."""
    d = a.get("d_any_undirected")
    v = a.get("v_file_rank")
    dc = "d_undef" if d in (None, "") else ("d0" if int(float(d)) == 0 else ("d1" if int(float(d)) == 1 else "d2+"))
    vc = "v_undef" if v in (None, "") else ("v_high" if int(float(v)) <= 5 else "v_low")
    return f"{dc}|{vc}"


if __name__ == "__main__":
    main()

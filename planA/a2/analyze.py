#!/usr/bin/env python3
"""A2 analysis: static `calls` edges vs the dynamic oracle, and gold reachability.

Inputs: A2 run directory (<out>/<instance_id>/trace_{pre,post}.json, status.json), the archive
(graphs, tasks), and A1's gold symbols (a1_gold_symbols.jsonl) for gold node ids.

Normalisation (HANDOFF section 6):
  * dynamic function ids are built from file paths; the same prefix rules as the static side are
    applied by trying path_to_module_candidates until the module is a prefix of some node id
  * nested functions ('<locals>') are counted separately as "not representable"
  * static containment pseudo-edges (target startswith source + '.') are dropped
  * 'Class.__init__' collapses to 'Class' on both sides when 'Class' is a node (calls to a class
    appear dynamically as __init__ frames)
  * test code: edges from test modules are reported separately

Metrics per task (pre-fix trace = oracle for the base-commit graph):
  * recall_direct      = |static ∩ dyn_direct| / |dyn_direct|   (over dynamic edges whose both
                         endpoints are graph nodes; edges with a non-node endpoint are reported
                         as `dyn_unrepresentable`)
  * recall_vialib      = same with dynamic edges via library frames added
  * observed_precision = |static ∩ dyn_any| / |static edges with both endpoints executed|
                         (precision outside executed code is unknown and said so)
  * same metrics restricted to edges incident to gold nodes
  * fail-to-pass tests (pre: failed/error, post: passed); for each, whether a gold node is
    (a) reachable from the test node by static BFS on directed calls, (b) executed dynamically
  * executed-code coverage: executed nodes / graph nodes (non-test)

Outputs: a2_tasks.csv, a2_edges_sample.jsonl (missing dynamic edges, for inspection), a2_summary.json.

VERIFY-ON-DATA: test node id mapping from pytest nodeids ("tests/test_x.py::TestC::test_y[param]"
-> "tests.test_x.TestC.test_y"); whether test modules are graph nodes at all in each repo.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    Graph, bfs_distance, containment_edge, get_graph, is_test_module_id, load_tasks, path_to_module_candidates, pct,
    read_jsonl, write_csv, write_jsonl,
)


def load_trace(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class Mapper:
    """Dynamic function id -> graph node id."""

    def __init__(self, g: Graph):
        self.g = g
        self.prefixes: set[str] = set()
        for nid in g.nodes:
            parts = nid.split(".")
            for k in range(1, len(parts) + 1):
                self.prefixes.add(".".join(parts[:k]))
        self.cache: dict[tuple[str, str], tuple[str | None, str]] = {}

    def map(self, rel_file: str, qualname: str) -> tuple[str | None, str]:
        """Returns (node id or best-effort id, status) status in node|collapsed_init|nested|missing|unmapped_module."""
        key = (rel_file, qualname)
        if key in self.cache:
            return self.cache[key]
        mod = None
        for cand, _rule in path_to_module_candidates(rel_file):
            if cand in self.prefixes:
                mod = cand
                break
        if mod is None:
            res = (None, "unmapped_module")
        elif "<locals>" in qualname:
            res = (f"{mod}.{qualname}", "nested")
        else:
            nid = f"{mod}.{qualname}" if qualname != "<module>" else mod
            if qualname.endswith(".__init__") and f"{mod}.{qualname[:-9]}" in self.g.nodes:
                res = (f"{mod}.{qualname[:-9]}", "collapsed_init")   # Class.__init__ -> Class (both sides)
            elif nid in self.g.nodes:
                res = (nid, "node")
            else:
                res = (nid, "missing")
        self.cache[key] = res
        return res


def test_node_id(nodeid: str) -> str:
    path, _, rest = nodeid.partition("::")
    mod = path[:-3] if path.endswith(".py") else path
    mod = mod.replace("/", ".")
    rest = re.sub(r"\[.*\]$", "", rest).replace("::", ".")
    return f"{mod}.{rest}" if rest else mod


def analyze_task(task, g: Graph, tr_pre: dict, tr_post: dict | None, gold_nodes: set[str]) -> tuple[dict, list[dict]]:
    m = Mapper(g)
    funcs = tr_pre["functions"]
    files = tr_pre["files"]
    mapped = []
    status_count = Counter()
    for fn, rel in zip(funcs, files):
        qual = fn.split(".", 1)[1] if "." in fn else fn
        # fn = "<dotted path>.<qualname>"; recover qualname by removing the module part derived from the file
        modpath = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel.replace("/", ".")
        if modpath.endswith(".__init__"):
            modpath = modpath[: -len(".__init__")]
        qual = fn[len(modpath) + 1:] if fn.startswith(modpath + ".") else qual
        nid, st = m.map(rel, qual)
        mapped.append((nid, st))
        status_count[st] += 1

    def collapse(n: str) -> str:
        return n[:-9] if n.endswith(".__init__") and n[:-9] in g.nodes else n

    static_calls = {(collapse(s), collapse(d)) for s, d, t in g.edges() if t == "calls"}
    static_norm = {(s, d) for s, d in static_calls if s != d and not containment_edge(s, d)}
    module_level = [0]

    def dyn_edges(key: str) -> tuple[set, int, int, set]:
        """Returns (edges between graph nodes, n_unrepresentable, n_nested, test-origin edges)."""
        ok: set = set()
        unrep = nested = 0
        for c, e, _tests in tr_pre[key]:
            (cn, cs), (en, es) = mapped[c], mapped[e]
            if funcs[c].endswith(".<module>") or funcs[e].endswith(".<module>"):
                module_level[0] += 1   # import-time execution, not a call between functions
                continue
            if cs == "nested" or es == "nested":
                nested += 1
                continue
            if cs in ("node", "collapsed_init") and es in ("node", "collapsed_init"):
                if cn != en and not containment_edge(cn, en):
                    ok.add((cn, en))
            else:
                unrep += 1
        tests_origin = {(c, e) for c, e in ok if is_test_module_id(c)}
        return ok, unrep, nested, tests_origin

    dyn_direct, unrep_d, nested_d, _ = dyn_edges("edges_direct")
    dyn_vialib, unrep_v, nested_v, _ = dyn_edges("edges_vialib")
    dyn_any = dyn_direct | dyn_vialib
    executed_nodes = {mapped[i][0] for i in range(len(mapped)) if mapped[i][1] in ("node", "collapsed_init")}
    static_observed = {(s, d) for s, d in static_norm if s in executed_nodes and d in executed_nodes}

    def rec(dyn: set) -> float | None:
        return pct(len(static_norm & dyn), len(dyn)) if dyn else None

    def rec_nontest(dyn: set) -> float | None:
        dd = {(s, d) for s, d in dyn if not is_test_module_id(s) and not is_test_module_id(d)}
        return pct(len(static_norm & dd), len(dd)) if dd else None

    gold_inc_dyn = {(s, d) for s, d in dyn_any if s in gold_nodes or d in gold_nodes}
    gold_inc_static_obs = {(s, d) for s, d in static_observed if s in gold_nodes or d in gold_nodes}

    # fail-to-pass tests
    pre_out = tr_pre.get("outcomes", {})
    post_out = (tr_post or {}).get("outcomes", {})
    f2p = [t for t, o in pre_out.items() if o in ("failed", "error") and post_out.get(t) == "passed"]
    f2p_nodes = {test_node_id(t) for t in f2p}
    # Tests added by test_patch are not nodes of the base-commit graph: fall back to the enclosing
    # test class, then the test module, and record which level was used.
    f2p_nodes_in_graph: set[str] = set()
    f2p_source_levels = Counter()
    for tn in f2p_nodes:
        cur, level = tn, "function"
        while cur and cur not in g.nodes and "." in cur:
            cur = cur.rsplit(".", 1)[0]
            level = "class_or_module"
        if cur in g.nodes:
            f2p_nodes_in_graph.add(cur)
            f2p_source_levels["function" if cur == tn else ("module" if is_test_module_id(cur) and cur.split(".")[-1].startswith("test_") else "class")] += 1
        else:
            f2p_source_levels["none"] += 1
    exec_idx = tr_pre.get("per_test_exec", {})
    gold_exec_by_f2p = set()
    for t in f2p:
        for i in exec_idx.get(t, []):
            nid, st = mapped[i]
            if st in ("node", "collapsed_init") and nid in gold_nodes:
                gold_exec_by_f2p.add(nid)
    # also check post-fix trace (gold symbol may only be executed once the fix exists)
    gold_exec_post = set()
    if tr_post:
        m_post = Mapper(g)
        mp = []
        for fn, rel in zip(tr_post["functions"], tr_post["files"]):
            modpath = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel.replace("/", ".")
            if modpath.endswith(".__init__"):
                modpath = modpath[: -len(".__init__")]
            qual = fn[len(modpath) + 1:] if fn.startswith(modpath + ".") else fn
            mp.append(m_post.map(rel, qual))
        for t in f2p:
            for i in tr_post.get("per_test_exec", {}).get(t, []):
                nid, st = mp[i]
                if st in ("node", "collapsed_init") and nid in gold_nodes:
                    gold_exec_post.add(nid)
    static_reach = set()
    if f2p_nodes_in_graph:
        dist = bfs_distance(g, f2p_nodes_in_graph, directed=True)
        static_reach = {x for x in gold_nodes if x in dist}
    static_reach_und = set()
    if f2p_nodes_in_graph:
        dist = bfs_distance(g, f2p_nodes_in_graph, directed=False)
        static_reach_und = {x for x in gold_nodes if x in dist}

    non_test_nodes = [n for n in g.nodes if not is_test_module_id(n)]
    row = {
        "instance_id": task.instance_id, "repo": task.repo, "graph_key": g.key,
        "n_dyn_events": tr_pre.get("n_events"), "n_dyn_functions": len(funcs),
        "map_status": dict(status_count),
        "n_executed_nodes": len(executed_nodes),
        "pct_nontest_nodes_executed": pct(len([n for n in executed_nodes if not is_test_module_id(n)]), len(non_test_nodes)),
        "n_static_calls": len(static_norm), "n_static_observed": len(static_observed),
        "n_dyn_direct": len(dyn_direct), "n_dyn_vialib_only": len(dyn_vialib - dyn_direct), "n_dyn_any": len(dyn_any),
        "dyn_unrepresentable_direct": unrep_d, "dyn_nested_direct": nested_d, "dyn_module_level_edges": module_level[0],
        "recall_direct": rec(dyn_direct), "recall_any": rec(dyn_any),
        "recall_direct_nontest": rec_nontest(dyn_direct), "recall_any_nontest": rec_nontest(dyn_any),
        "observed_precision": pct(len(static_observed & dyn_any), len(static_observed)) if static_observed else None,
        "n_gold_nodes": len(gold_nodes),
        "gold_incident_dyn_edges": len(gold_inc_dyn),
        "gold_recall_any": pct(len(gold_inc_dyn & static_norm), len(gold_inc_dyn)) if gold_inc_dyn else None,
        "gold_observed_precision": pct(len(gold_inc_static_obs & dyn_any), len(gold_inc_static_obs)) if gold_inc_static_obs else None,
        "n_tests_pre": len(pre_out), "n_tests_post": len(post_out),
        "pre_outcomes": dict(Counter(pre_out.values())), "post_outcomes": dict(Counter(post_out.values())),
        "n_fail_to_pass": len(f2p), "fail_to_pass": f2p[:20],
        "n_f2p_test_nodes_in_graph": len(f2p_nodes_in_graph), "f2p_source_levels": dict(f2p_source_levels),
        "gold_static_reachable_directed": sorted(static_reach), "gold_static_reachable_undirected": sorted(static_reach_und),
        "gold_executed_by_f2p_pre": sorted(gold_exec_by_f2p), "gold_executed_by_f2p_post": sorted(gold_exec_post),
        "gold_dyn_reachable_not_static": sorted((gold_exec_by_f2p | gold_exec_post) - static_reach),
        "gold_any_dyn_reachable": bool(gold_exec_by_f2p | gold_exec_post),
        "gold_any_static_reachable": bool(static_reach),
    }
    missing = [{"instance_id": task.instance_id, "src": s, "dst": d, "kind": "dyn_not_static",
                "gold_incident": s in gold_nodes or d in gold_nodes}
               for s, d in sorted(dyn_direct - static_norm)[:200]]
    missing += [{"instance_id": task.instance_id, "src": s, "dst": d, "kind": "static_observed_not_dyn",
                 "gold_incident": s in gold_nodes or d in gold_nodes}
                for s, d in sorted(static_observed - dyn_any)[:200]]
    return row, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path, help="A2 run dir (<out> of run_task.py)")
    ap.add_argument("--a1", required=True, type=Path, help="A1 output dir (for a1_gold_symbols.jsonl)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    gold_p = args.a1 / "a1_gold_symbols.jsonl"
    gold_by_task: dict[str, set[str]] = defaultdict(set)
    if gold_p.exists():
        for r in read_jsonl(gold_p):
            if r.get("present_in_graph") and r.get("node_id") and not r.get("new_symbol"):
                gold_by_task[r["instance_id"]].add(r["node_id"])
    else:
        print(f"[a2-analyze] WARNING {gold_p} missing; run A1 first (gold metrics will be empty)")

    rows, samples, run_status = [], [], []
    for t in load_tasks(args.data):
        tdir = args.runs / t.instance_id
        st_p = tdir / "status.json"
        st = json.loads(st_p.read_text()) if st_p.exists() else {"done": False, "error": "not run"}
        run_status.append({"instance_id": t.instance_id, "repo": t.repo, "done": st.get("done"), "error": st.get("error"),
                           "editable_ok": (st.get("deps") or {}).get("editable_ok"),
                           "pre_exit": (st.get("pre") or {}).get("exit"), "post_exit": (st.get("post") or {}).get("exit"),
                           "pre_timeout": (st.get("pre") or {}).get("timeout"), "seconds": st.get("seconds")})
        g = get_graph(args.data, t)
        tr_pre = load_trace(tdir / "trace_pre.json")
        if g is None or tr_pre is None:
            continue
        tr_post = load_trace(tdir / "trace_post.json")
        row, miss = analyze_task(t, g, tr_pre, tr_post, gold_by_task.get(t.instance_id, set()))
        rows.append(row)
        samples.extend(miss)
        print(f"[a2-analyze] {t.instance_id}: recall_direct={row['recall_direct']} obs_prec={row['observed_precision']} "
              f"f2p={row['n_fail_to_pass']} gold_dyn={row['gold_any_dyn_reachable']} gold_static={row['gold_any_static_reachable']}", flush=True)

    write_csv(args.out / "a2_tasks.csv", rows)
    write_csv(args.out / "a2_run_status.csv", run_status)
    write_jsonl(args.out / "a2_edges_sample.jsonl", samples)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    per_repo = defaultdict(list)
    for r in rows:
        per_repo[r["repo"]].append(r)
    summ = {
        "n_tasks_total": len(run_status), "n_tasks_run_done": sum(1 for r in run_status if r["done"]),
        "n_tasks_analyzed": len(rows),
        "n_tasks_with_fail_to_pass": sum(1 for r in rows if r["n_fail_to_pass"] > 0),
        "recall_direct_mean": mean(r["recall_direct"] for r in rows),
        "recall_any_mean": mean(r["recall_any"] for r in rows),
        "recall_direct_nontest_mean": mean(r["recall_direct_nontest"] for r in rows),
        "observed_precision_mean": mean(r["observed_precision"] for r in rows),
        "gold_recall_any_mean": mean(r["gold_recall_any"] for r in rows),
        "pooled_recall_direct": pct(sum(round((r["recall_direct"] or 0) * r["n_dyn_direct"] / 100) for r in rows),
                                    sum(r["n_dyn_direct"] for r in rows)),
        "tasks_gold_dyn_reachable": sum(1 for r in rows if r["gold_any_dyn_reachable"]),
        "tasks_gold_static_reachable": sum(1 for r in rows if r["gold_any_static_reachable"]),
        "tasks_gold_dyn_not_static": sum(1 for r in rows if r["gold_dyn_reachable_not_static"]),
        "per_repo": {
            repo: {"n": len(rs), "recall_direct_mean": mean(r["recall_direct"] for r in rs),
                   "recall_any_mean": mean(r["recall_any"] for r in rs),
                   "observed_precision_mean": mean(r["observed_precision"] for r in rs),
                   "pct_nontest_nodes_executed_mean": mean(r["pct_nontest_nodes_executed"] for r in rs),
                   "tasks_with_f2p": sum(1 for r in rs if r["n_fail_to_pass"] > 0),
                   "gold_dyn_not_static": sum(1 for r in rs if r["gold_dyn_reachable_not_static"])}
            for repo, rs in per_repo.items()},
        "run_errors": {r["instance_id"]: r["error"] for r in run_status if r.get("error")},
    }
    (args.out / "a2_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()

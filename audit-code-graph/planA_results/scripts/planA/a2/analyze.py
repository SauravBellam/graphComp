#!/usr/bin/env python3
"""A2 analysis: the provided `calls` graph against the dynamic oracle, with every miss attributed.

Inputs: A2 run directory (<runs>/<instance_id>/trace_{pre,post}.json, status.json, pytest_*.log),
the archive (graphs, tasks.jsonl), and A1 outputs (a1_gold_symbols.jsonl; a1_async_audit.csv as a
fallback source of async kinds for traces written before the tracer recorded kinds).

1. Run validity, per task (``a2_run_status.csv``)
   run_class: setup_failed | timeout | import_error | collection_error | no_tests | crashed | clean
   (pytest exit code, collection errors and missing modules parsed from the logs), plus a
   fail-to-pass cross-check against tasks.jsonl FAIL_TO_PASS (or, when empty, the test functions the
   test_patch adds, as the harness does). Each expected test is confirmed (pre fail -> post pass),
   pre_passed, post_failed or not_collected. A task is ``usable`` when both runs are clean and at
   least one expected test is confirmed. Pooled numbers are reported for usable tasks and for all.

2. Edge attribution (``a2_dyn_edges.jsonl``, every dynamic edge, uncapped)
   Each observed caller->callee pair between repo functions (module-level/import-time edges and
   self/containment pairs excluded and counted) gets one cause:
     captured                 both endpoints are nodes and the static graph has the edge
     missed_between_nodes     both endpoints are nodes, edge absent (subcause sync / async_node)
     absent_async             an endpoint is an async def that is not a node
     absent_nested            an endpoint is a nested function (not representable by design)
     absent_sync              an endpoint is a sync def / class that is not a node
     unmapped_module          an endpoint's file maps to no module of the graph
   recall_representable = captured / (captured + missed_between_nodes)
   recall_agent_facing  = captured / all attributed edges except unmapped_module
   Both are stored as raw numerators and denominators, for direct edges (immediate parent frame in
   the repo) and any edges (direct + via library frames), overall and non-test only.

3. Static precision (``a2_static_unconfirmed.jsonl``)
   observed static edges = static edges whose both endpoints executed; confirmed = also observed
   dynamically. Precision outside executed code is unknown and not estimated.

4. Gold functions (``a2_gold.jsonl``), four-way, from A1's gold symbols (absent ones included):
   in_graph_executed | in_graph_not_executed | absent_executed | absent_not_executed
   where executed = run by a confirmed fail-to-pass test before or after the fix; plus static
   reachability (directed BFS from the f2p test node, or its class/module when the new test is not a
   base-graph node). ``absent_executed`` is the case the graph cannot represent at all.

Outputs also: a2_tasks.csv (one row per analysed task, raw counts), a2_summary.json (pooled ratios
with task-cluster bootstrap intervals, Wilson intervals for task-level rates, per repo).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    Graph, bfs_distance, containment_edge, get_graph, is_test_module_id, load_tasks, path_to_module_candidates,
    read_jsonl, write_csv, write_jsonl,
)
from stats import mean_ci, ratio_ci, wilson  # noqa: E402

CAUSES = ["captured", "missed_between_nodes", "absent_async", "absent_nested", "absent_sync", "unmapped_module"]


def load_trace(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class Mapper:
    """Dynamic function (file, qualname) -> graph node id."""

    def __init__(self, g: Graph):
        self.g = g
        self.prefixes: set[str] = set()
        for nid in g.nodes:
            parts = nid.split(".")
            for k in range(1, len(parts) + 1):
                self.prefixes.add(".".join(parts[:k]))
        self.cache: dict[tuple[str, str], tuple[str | None, str]] = {}

    def module(self, rel_file: str) -> str | None:
        cands = path_to_module_candidates(rel_file)
        for cand, _rule in cands:
            if cand in self.prefixes:
                return cand
        for cand, _rule in cands:   # module with no nodes at all (e.g. only async defs): use parent package
            parent = cand.rsplit(".", 1)[0] if "." in cand else None
            if parent and parent in self.prefixes:
                return cand
        return None

    def map(self, rel_file: str, qualname: str) -> tuple[str | None, str]:
        """Returns (node id or would-be id, status) with status node|collapsed_init|nested|missing|unmapped_module."""
        key = (rel_file, qualname)
        if key in self.cache:
            return self.cache[key]
        mod = self.module(rel_file)
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


def _qualname(fn: str, rel: str) -> str:
    modpath = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel.replace("/", ".")
    if modpath.endswith(".__init__"):
        modpath = modpath[: -len(".__init__")]
    return fn[len(modpath) + 1:] if fn.startswith(modpath + ".") else fn.split(".")[-1]


def map_trace(tr: dict, m: Mapper, kind_fallback: dict[str, str]) -> list[dict]:
    kinds = tr.get("kinds")
    out = []
    for i, (fn, rel) in enumerate(zip(tr["functions"], tr["files"])):
        qual = _qualname(fn, rel)
        nid, st = m.map(rel, qual)
        raw = None
        mod = m.module(rel)
        if mod is not None:
            raw = mod if qual == "<module>" else f"{mod}.{qual}"
        k = kinds[i] if kinds else kind_fallback.get(raw or "", "unknown")
        out.append({"fn": fn, "nid": nid, "raw": raw, "status": st, "kind": k,
                    "module_level": qual == "<module>" or fn.endswith(".<module>")})
    return out


def test_node_id(nodeid: str) -> str:
    path, _, rest = nodeid.partition("::")
    mod = path[:-3] if path.endswith(".py") else path
    mod = mod.replace("/", ".")
    rest = re.sub(r"\[.*\]$", "", rest).replace("::", ".")
    return f"{mod}.{rest}" if rest else mod


# --- fail-to-pass expectations (same matching rules as swegemma verification) -------------------

def expected_f2p(task) -> tuple[list[str], str]:
    if task.FAIL_TO_PASS:
        return list(task.FAIL_TO_PASS), "tasks.jsonl"
    added = re.findall(r"^\+\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(", task.test_patch or "", re.M)
    return list(dict.fromkeys(added)), "test_patch_added_defs"


def _matches(nodeid: str, req: str) -> bool:
    """Does a collected pytest nodeid satisfy a FAIL_TO_PASS requirement?

    A requirement with '::' must match the file path and the test name (parametrised variants of
    the same base name count); a bare name matches the test name or its unparametrised base."""
    name = nodeid.split("::")[-1]
    base = name.split("[", 1)[0]
    req_short = req.split("::")[-1]
    req_base = req_short.split("[", 1)[0]
    if nodeid == req:
        return True
    if "::" in req:
        same_path = nodeid.split("::")[0] == req.split("::")[0]
        same_cls = nodeid.split("::")[1:-1] == req.split("::")[1:-1] or len(req.split("::")) == 2
        return same_path and same_cls and (name == req_short or (req_short == req_base and base == req_base))
    return name == req_short or base == req_base


def check_f2p(task, pre_out: dict, post_out: dict) -> dict:
    reqs, source = expected_f2p(task)
    statuses = []
    for req in reqs:
        hits = [t for t in set(pre_out) | set(post_out) if _matches(t, req)]
        if not hits:
            statuses.append("not_collected")
            continue
        pre_bad = any(pre_out.get(t) in ("failed", "error") for t in hits)
        post_ok = all(post_out.get(t) == "passed" for t in hits)
        statuses.append("confirmed" if pre_bad and post_ok else ("pre_passed" if not pre_bad else "post_failed"))
    c = Counter(statuses)
    observed = [t for t, o in pre_out.items() if o in ("failed", "error") and post_out.get(t) == "passed"]
    extra = [t for t in observed if not any(_matches(t, r) for r in reqs)]
    return {"f2p_expected_source": source, "n_f2p_expected": len(reqs), "n_f2p_confirmed": c["confirmed"],
            "n_f2p_pre_passed": c["pre_passed"], "n_f2p_post_failed": c["post_failed"],
            "n_f2p_not_collected": c["not_collected"], "n_f2p_observed": len(observed), "n_f2p_extra": len(extra),
            "f2p_all_confirmed": bool(reqs) and c["confirmed"] == len(reqs),
            "f2p_observed": observed}


# --- run validity ----------------------------------------------------------------------------

def run_validity(tdir: Path, st: dict) -> dict:
    from run_task import classify_pytest
    out = {"done": bool(st.get("done")), "error": st.get("error"),
           "editable_ok": (st.get("deps") or {}).get("editable_ok"),
           "test_patch_applied": st.get("test_patch_applied"), "gold_patch_applied": st.get("gold_patch_applied"),
           "test_patch_strategy": st.get("test_patch_strategy"), "gold_patch_strategy": st.get("gold_patch_strategy"),
           "seconds": st.get("seconds")}
    for ph in ("pre", "post"):
        r = st.get(ph) or {}
        if "run_class" not in r:   # older status files: classify from the log
            log = tdir / f"pytest_{ph}.log"
            text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else r.get("stdout_tail", "")
            r = {**r, **classify_pytest(r.get("exit") if not r.get("timeout") else None, text)} if (r or log.exists()) else {}
        out[f"{ph}_exit"] = r.get("exit")
        out[f"{ph}_run_class"] = r.get("run_class", "not_run")
        out[f"{ph}_missing_modules"] = r.get("missing_modules", [])
        out[f"{ph}_counts"] = r.get("counts", {})
    if not out["done"] or out["editable_ok"] is False or out["test_patch_applied"] is False or out["gold_patch_applied"] is False:
        out["run_class"] = "setup_failed"
    elif out["pre_run_class"] != "clean":
        out["run_class"] = out["pre_run_class"]
    elif out["post_run_class"] != "clean":
        out["run_class"] = "post_" + out["post_run_class"]
    else:
        out["run_class"] = "clean"
    return out


# --- per task analysis -----------------------------------------------------------------------

def analyze_task(task, g: Graph, tr_pre: dict, tr_post: dict | None, gold_rows: list[dict],
                 kind_fallback: dict[str, str]) -> tuple[dict, list[dict], list[dict], list[dict]]:
    m = Mapper(g)
    mp = map_trace(tr_pre, m, kind_fallback)
    mq = map_trace(tr_post, m, kind_fallback) if tr_post else []

    def collapse(n: str) -> str:
        return n[:-9] if n.endswith(".__init__") and n[:-9] in g.nodes else n

    static_norm = {(collapse(s), collapse(d)) for s, d, t in g.edges() if t == "calls"}
    static_norm = {(s, d) for s, d in static_norm if s != d and not containment_edge(s, d)}
    gold_ids = {r["node_id"] for r in gold_rows if r.get("node_id") and r.get("qualname") and not r.get("new_symbol")}

    pre_out = tr_pre.get("outcomes", {})
    post_out = (tr_post or {}).get("outcomes", {})
    f2p = check_f2p(task, pre_out, post_out)
    f2p_tests = f2p["f2p_observed"]

    # ---- edge attribution
    tests = tr_pre.get("tests", [])
    edges: dict[tuple[int, int], dict] = {}
    for channel in ("edges_direct", "edges_vialib"):
        for c, e, tl in tr_pre.get(channel, []):
            k = (c, e)
            if k in edges:
                edges[k]["n_tests"] = max(edges[k]["n_tests"], len(tl))
                continue
            edges[k] = {"c": c, "e": e, "channel": "direct" if channel == "edges_direct" else "vialib",
                        "n_tests": len(tl), "f2p": any(tests[i] in f2p_tests for i in tl if i < len(tests))}
    n_module_level = n_self = 0
    dyn_rows = []
    counts = Counter()
    captured_pairs, dyn_node_pairs = set(), set()
    for rec in edges.values():
        a, b = mp[rec["c"]], mp[rec["e"]]
        if a["module_level"] or b["module_level"]:
            n_module_level += 1
            continue

        def absent_reason(x):
            if x["status"] in ("node", "collapsed_init"):
                return None
            if x["status"] == "unmapped_module":
                return "unmapped_module"
            if x["status"] == "nested":
                return "absent_nested"
            return "absent_async" if x["kind"] in ("async_function", "async_generator") else "absent_sync"

        ra, rb = absent_reason(a), absent_reason(b)
        sub = ""
        if ra is None and rb is None:
            if a["nid"] == b["nid"] or containment_edge(a["nid"], b["nid"]):
                n_self += 1
                continue
            dyn_node_pairs.add((a["nid"], b["nid"]))
            if (a["nid"], b["nid"]) in static_norm:
                cause = "captured"
                captured_pairs.add((a["nid"], b["nid"]))
            else:
                cause = "missed_between_nodes"
                sub = "async_node" if {a["kind"], b["kind"]} & {"async_function", "async_generator"} else "sync"
        else:
            reasons = {r for r in (ra, rb) if r}
            cause = next(r for r in ("unmapped_module", "absent_async", "absent_nested", "absent_sync") if r in reasons)
            sub = "+".join(sorted(reasons))
        nontest = not (is_test_module_id(a["nid"] or "tests") or is_test_module_id(b["nid"] or "tests"))
        gold_inc = (a["nid"] in gold_ids) or (b["nid"] in gold_ids) or (a["raw"] in gold_ids) or (b["raw"] in gold_ids)
        for scope, ok in (("all", True), ("nontest", nontest), ("gold", gold_inc)):
            if not ok:
                continue
            counts[(scope, "any", cause)] += 1
            if rec["channel"] == "direct":
                counts[(scope, "direct", cause)] += 1
        dyn_rows.append({"instance_id": task.instance_id, "src": a["nid"] or a["fn"], "dst": b["nid"] or b["fn"],
                         "src_kind": a["kind"], "dst_kind": b["kind"], "channel": rec["channel"], "cause": cause,
                         "subcause": sub, "nontest": nontest, "gold_incident": gold_inc, "n_tests": rec["n_tests"],
                         "seen_in_f2p_test": rec["f2p"]})

    executed = {x["nid"] for x in mp if x["status"] in ("node", "collapsed_init")}
    static_observed = {(s, d) for s, d in static_norm if s in executed and d in executed}
    static_confirmed = static_observed & dyn_node_pairs
    unconfirmed = [{"instance_id": task.instance_id, "src": s, "dst": d,
                    "gold_incident": s in gold_ids or d in gold_ids, "nontest": not (is_test_module_id(s) or is_test_module_id(d))}
                   for s, d in sorted(static_observed - static_confirmed)]

    # ---- gold four-way status
    def executed_by(trace_map, trace, test_ids) -> set[str]:
        out = set()
        pte = (trace or {}).get("per_test_exec", {})
        for t in test_ids:
            for i in pte.get(t, []):
                x = trace_map[i]
                for v in (x["nid"], x["raw"]):
                    if v:
                        out.add(v)
        return out

    all_tests_pre = [t for t in tr_pre.get("per_test_exec", {}) if not t.startswith("<")]
    exec_f2p = executed_by(mp, tr_pre, f2p_tests) | executed_by(mq, tr_post, f2p_tests)
    exec_any = executed_by(mp, tr_pre, all_tests_pre) | (executed_by(mq, tr_post, [t for t in (tr_post or {}).get("per_test_exec", {}) if not t.startswith("<")]) if tr_post else set())
    sources, levels = set(), Counter()
    for t in f2p_tests:
        tn = test_node_id(t)
        cur = tn
        while cur not in g.nodes and "." in cur:
            cur = cur.rsplit(".", 1)[0]
        if cur in g.nodes:
            sources.add(cur)
            levels["function" if cur == tn else "class_or_module"] += 1
        else:
            levels["none"] += 1
    reach_dir = bfs_distance(g, sources, directed=True) if sources else {}
    reach_und = bfs_distance(g, sources, directed=False) if sources else {}
    gold_out = []
    gc = Counter()
    for r in gold_rows:
        if not r.get("qualname"):
            continue
        nid = r.get("node_id")
        if r.get("new_symbol"):
            st = "new_symbol_executed_post" if nid and nid in executed_by(mq, tr_post, f2p_tests) else "new_symbol_not_executed"
        else:
            inside = bool(r.get("present_in_graph"))
            ran = bool(nid and nid in exec_f2p)
            st = ("in_graph" if inside else "absent") + ("_executed" if ran else "_not_executed")
        gc[st] += 1
        gold_out.append({"instance_id": task.instance_id, "node_id": nid, "kind": r.get("kind"), "nested": r.get("nested"),
                         "status": st, "executed_by_any_test": bool(nid and nid in exec_any),
                         "static_reachable_directed": bool(nid and nid in reach_dir),
                         "static_hops_directed": reach_dir.get(nid) if nid else None,
                         "static_reachable_undirected": bool(nid and nid in reach_und),
                         "agent_query_outcome": r.get("agent_query_outcome")})

    def cnt(scope, ch, cause):
        return counts[(scope, ch, cause)]

    row = {"instance_id": task.instance_id, "repo": task.repo, "graph_key": g.key,
           "n_dyn_functions": len(mp), "map_status": dict(Counter(x["status"] for x in mp)),
           "n_dyn_async_functions": sum(1 for x in mp if x["kind"] in ("async_function", "async_generator")),
           "n_executed_nodes": len(executed),
           "n_nontest_nodes": sum(1 for n in g.nodes if not is_test_module_id(n)),
           "n_nontest_nodes_executed": sum(1 for n in executed if not is_test_module_id(n)),
           "n_dyn_module_level_edges": n_module_level, "n_dyn_self_or_containment": n_self,
           "n_static_calls": len(static_norm), "n_static_observed": len(static_observed),
           "n_static_confirmed": len(static_confirmed),
           **f2p, "f2p_source_levels": dict(levels)}
    row["f2p_observed"] = row["f2p_observed"][:30]
    for scope in ("all", "nontest", "gold"):
        for ch in ("direct", "any"):
            for cause in CAUSES:
                row[f"{scope}_{ch}_{cause}"] = cnt(scope, ch, cause)
    for st in ("in_graph_executed", "in_graph_not_executed", "absent_executed", "absent_not_executed",
               "new_symbol_executed_post", "new_symbol_not_executed"):
        row[f"gold_{st}"] = gc[st]
    row["gold_any_absent_executed"] = gc["absent_executed"] > 0
    row["gold_any_in_graph_executed"] = gc["in_graph_executed"] > 0
    row["gold_any_static_reachable"] = any(x["static_reachable_directed"] for x in gold_out if x["status"].startswith("in_graph"))
    row["gold_executed_not_static"] = sum(1 for x in gold_out if x["status"] == "in_graph_executed" and not x["static_reachable_directed"])
    return row, dyn_rows, unconfirmed, gold_out


# --- summary ----------------------------------------------------------------------------------

def summarize(rows: list[dict]) -> dict:
    def pooled(rs, scope, ch):
        cap = [r[f"{scope}_{ch}_captured"] for r in rs]
        miss_nodes = [r[f"{scope}_{ch}_missed_between_nodes"] for r in rs]
        agent_den = [sum(r[f"{scope}_{ch}_{c}"] for c in CAUSES if c != "unmapped_module") for r in rs]
        out = {"recall_representable": ratio_ci(cap, [a + b for a, b in zip(cap, miss_nodes)]),
               "recall_agent_facing": ratio_ci(cap, agent_den)}
        misses = [d - c for d, c in zip(agent_den, cap)]
        out["miss_share_by_cause"] = {c: ratio_ci([r[f"{scope}_{ch}_{c}"] for r in rs], misses)
                                      for c in ("missed_between_nodes", "absent_async", "absent_nested", "absent_sync")}
        out["unmapped_module_edges"] = sum(r[f"{scope}_{ch}_unmapped_module"] for r in rs)
        return out

    def block(rs):
        if not rs:
            return {"n_tasks": 0}
        b = {"n_tasks": len(rs)}
        for scope in ("all", "nontest", "gold"):
            for ch in ("direct", "any"):
                b[f"{scope}_{ch}"] = pooled(rs, scope, ch)
        b["observed_precision"] = ratio_ci([r["n_static_confirmed"] for r in rs], [r["n_static_observed"] for r in rs])
        b["nontest_node_coverage"] = ratio_ci([r["n_nontest_nodes_executed"] for r in rs], [r["n_nontest_nodes"] for r in rs])
        b["tasks_gold_absent_but_executed"] = wilson(sum(1 for r in rs if r["gold_any_absent_executed"]), len(rs))
        b["tasks_gold_in_graph_executed"] = wilson(sum(1 for r in rs if r["gold_any_in_graph_executed"]), len(rs))
        b["tasks_gold_static_reachable"] = wilson(sum(1 for r in rs if r["gold_any_static_reachable"]), len(rs))
        b["tasks_gold_executed_not_static"] = wilson(sum(1 for r in rs if r["gold_executed_not_static"] > 0), len(rs))
        b["gold_status_counts"] = {k: sum(r[f"gold_{k}"] for r in rs) for k in
                                   ("in_graph_executed", "in_graph_not_executed", "absent_executed", "absent_not_executed",
                                    "new_symbol_executed_post", "new_symbol_not_executed")}
        b["per_task_recall_representable_any"] = mean_ci(
            [r["all_any_captured"] / (r["all_any_captured"] + r["all_any_missed_between_nodes"])
             if (r["all_any_captured"] + r["all_any_missed_between_nodes"]) else None for r in rs])
        return b

    usable = [r for r in rows if r.get("usable")]
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)
    return {"usable_tasks": block(usable), "all_analyzed_tasks": block(rows),
            "per_repo_usable": {k: block([r for r in v if r.get("usable")]) for k, v in by_repo.items()},
            "per_repo_all": {k: block(v) for k, v in by_repo.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path, help="A2 run dir (<out> of run_task.py)")
    ap.add_argument("--a1", required=True, type=Path, help="A1 output dir")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    gold_by_task: dict[str, list[dict]] = defaultdict(list)
    gp = args.a1 / "a1_gold_symbols.jsonl"
    if gp.exists():
        for r in read_jsonl(gp):
            gold_by_task[r["instance_id"]].append(r)
    else:
        print(f"[a2-analyze] WARNING {gp} missing; run A1 first (gold metrics will be empty)")
    kind_fallback: dict[str, str] = {}
    ap_csv = args.a1 / "a1_async_audit.csv"
    if ap_csv.exists():
        with open(ap_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                kind_fallback[r["node_id"]] = r["kind"]

    rows, dyn_all, unconf_all, gold_all, status_rows = [], [], [], [], []
    for t in load_tasks(args.data):
        tdir = args.runs / t.instance_id
        st = json.loads((tdir / "status.json").read_text()) if (tdir / "status.json").exists() else {}
        val = run_validity(tdir, st) if st else {"done": False, "run_class": "not_run"}
        g = get_graph(args.data, t)
        tr_pre = load_trace(tdir / "trace_pre.json")
        tr_post = load_trace(tdir / "trace_post.json")
        srow = {"instance_id": t.instance_id, "repo": t.repo, **val}
        if g is None or tr_pre is None:
            srow["usable"] = False
            status_rows.append(srow)
            continue
        row, dyn, unconf, gold = analyze_task(t, g, tr_pre, tr_post, gold_by_task.get(t.instance_id, []), kind_fallback)
        row["run_class"] = val["run_class"]
        row["usable"] = val["run_class"] == "clean" and row["n_f2p_confirmed"] > 0
        srow.update({k: row[k] for k in ("n_f2p_expected", "f2p_expected_source", "n_f2p_confirmed", "n_f2p_pre_passed",
                                         "n_f2p_post_failed", "n_f2p_not_collected", "n_f2p_extra", "usable")})
        status_rows.append(srow)
        rows.append(row)
        dyn_all += dyn
        unconf_all += unconf
        gold_all += gold
        print(f"[a2-analyze] {t.instance_id}: class={row['run_class']} usable={row['usable']} "
              f"captured={row['all_any_captured']} missed_nodes={row['all_any_missed_between_nodes']} "
              f"absent_async={row['all_any_absent_async']} gold={dict((k, row['gold_' + k]) for k in ('in_graph_executed', 'absent_executed'))}",
              flush=True)

    write_csv(args.out / "a2_tasks.csv", rows)
    write_csv(args.out / "a2_run_status.csv", status_rows)
    write_jsonl(args.out / "a2_dyn_edges.jsonl", dyn_all)
    write_jsonl(args.out / "a2_static_unconfirmed.jsonl", unconf_all)
    write_jsonl(args.out / "a2_gold.jsonl", gold_all)

    coverage = {"n_tasks": len(status_rows),
                "run_class": dict(Counter(r["run_class"] for r in status_rows)),
                "usable": wilson(sum(1 for r in status_rows if r.get("usable")), len(status_rows)),
                "per_repo": {repo: {"n": len(rs), "usable": sum(1 for r in rs if r.get("usable")),
                                    "run_class": dict(Counter(r["run_class"] for r in rs))}
                             for repo, rs in _group(status_rows).items()},
                "patch_strategies": {"test": dict(Counter(str(r.get("test_patch_strategy")) for r in status_rows)),
                                     "gold": dict(Counter(str(r.get("gold_patch_strategy")) for r in status_rows))},
                "top_missing_modules": Counter(mm for r in status_rows for mm in (r.get("pre_missing_modules") or [])).most_common(25),
                "f2p_check": {"tasks_all_expected_confirmed": sum(1 for r in rows if r.get("f2p_all_confirmed")),
                              "expected": sum(r.get("n_f2p_expected", 0) for r in rows),
                              "confirmed": sum(r.get("n_f2p_confirmed", 0) for r in rows),
                              "pre_passed": sum(r.get("n_f2p_pre_passed", 0) for r in rows),
                              "post_failed": sum(r.get("n_f2p_post_failed", 0) for r in rows),
                              "not_collected": sum(r.get("n_f2p_not_collected", 0) for r in rows)}}
    summ = {"coverage": coverage, **summarize(rows)}
    (args.out / "a2_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({"coverage": coverage, "usable_tasks": summ["usable_tasks"].get("all_any")}, indent=2))


def _group(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["repo"]].append(r)
    return out


if __name__ == "__main__":
    main()

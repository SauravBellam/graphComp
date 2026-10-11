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
   reachability. Two source sets are used for the BFS (both reported):
     test_node : the f2p test's own node (or its class/module) when it exists in the base graph.
                 Fail-to-pass tests are usually *new* in test_patch, so this set is empty for most
                 tasks (Oct 10 run: 78 of 90 usable tasks) and reachability is then undefined.
     entry     : the repo functions where the f2p test enters the library: executed non-test nodes
                 whose observed callers in that test are all test code (the roots of the test's
                 dynamic call forest; robust to the test client running the app in another thread).
                 This is what an agent could reach by reading the failing test and following
                 `calls` edges.
   ``absent_executed`` is the case the graph cannot represent at all.

Nested definitions: the provided graph lists them under a flattened id (enclosing functions and
``<locals>`` dropped, classes kept; see common.flatten_qualname). The mapper tries that id first;
``absent_nested`` now means "nested and not present even under the flattened id".

Fail-to-pass statuses: confirmed (pre fail/error -> post pass), uncollected_pre (not collected in
the pre run, passed post: counts as fail-to-pass under the harness, which only runs post-fix and
needs exit 0), pre_passed (collected and passed pre), post_failed, not_collected. ``usable`` keeps
its definition (clean runs + >= 1 confirmed); ``strict`` adds every expected test confirmed;
``harness_resolved_post`` = post run exit 0 and every expected test passed, the harness's criterion.

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
    Graph, bfs_distance, containment_edge, flatten_qualname, get_graph, is_test_module_id, load_tasks,
    path_to_module_candidates, read_jsonl, write_csv, write_jsonl,
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
            flat = f"{mod}.{flatten_qualname(qualname)}"
            if flat.endswith(".__init__") and flat[:-9] in self.g.nodes:
                res = (flat[:-9], "collapsed_init")
            elif flat in self.g.nodes:
                res = (flat, "node")          # nested def present under the provided graph's flattened id
            else:
                res = (flat, "nested")        # nested and absent even when flattened
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

_ADDED_TEST_DEF = re.compile(r"^\+\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(", re.M)
_REMOVED_TEST_DEF = re.compile(r"^-\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(", re.M)


def expected_f2p(task) -> tuple[list[str], str]:
    """The harness's rule: FAIL_TO_PASS when given, else every `def test_*` on a '+' line of test_patch."""
    if task.FAIL_TO_PASS:
        return list(task.FAIL_TO_PASS), "tasks.jsonl"
    added = _ADDED_TEST_DEF.findall(task.test_patch or "")
    return list(dict.fromkeys(added)), "test_patch_added_defs"


def expected_f2p_truly_added(task) -> tuple[list[str], str]:
    """Diff-aware variant: drop test names that also appear on a '-' def line of the patch, i.e. tests
    the patch moves or rewrites rather than adds (their pre-fix version existed and could pass). Used as
    a sensitivity set next to the harness rule; FAIL_TO_PASS, when given, is used unchanged."""
    if task.FAIL_TO_PASS:
        return list(task.FAIL_TO_PASS), "tasks.jsonl"
    patch = task.test_patch or ""
    removed = set(_REMOVED_TEST_DEF.findall(patch))
    added = [n for n in _ADDED_TEST_DEF.findall(patch) if n not in removed]
    return list(dict.fromkeys(added)), "test_patch_truly_added_defs"


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


def _f2p_statuses(reqs: list[str], pre_out: dict, post_out: dict) -> tuple[Counter, list[str]]:
    statuses = []
    uncollected_pre_tests: list[str] = []
    for req in reqs:
        hits = [t for t in set(pre_out) | set(post_out) if _matches(t, req)]
        if not hits:
            statuses.append("not_collected")
            continue
        post_ok = all(post_out.get(t) == "passed" for t in hits)
        pre_bad = any(pre_out.get(t) in ("failed", "error") for t in hits)
        pre_missing = all(t not in pre_out for t in hits)
        pre_skipped = all(pre_out.get(t) == "skipped" for t in hits if t in pre_out) and not pre_missing
        if pre_bad and post_ok:
            statuses.append("confirmed")
        elif pre_missing and post_ok:
            # the test existed only after the fix (pre run could not collect its module): the harness
            # would count the pre state as failing (non-zero exit); keep it apart from a real pre pass
            statuses.append("uncollected_pre")
            uncollected_pre_tests += [t for t in hits if post_out.get(t) == "passed"]
        elif pre_skipped and post_ok:
            statuses.append("pre_skipped")    # skipped before the fix, passed after: not a pre pass
        elif not pre_bad:
            statuses.append("pre_passed")
        else:
            statuses.append("post_failed")
    return Counter(statuses), uncollected_pre_tests


def check_f2p(task, pre_out: dict, post_out: dict) -> dict:
    reqs, source = expected_f2p(task)
    c, uncollected_pre_tests = _f2p_statuses(reqs, pre_out, post_out)
    observed = [t for t, o in pre_out.items() if o in ("failed", "error") and post_out.get(t) == "passed"]
    extra = [t for t in observed if not any(_matches(t, r) for r in reqs)]
    post_counts = Counter(post_out.values())
    passes_post = ("confirmed", "uncollected_pre", "pre_skipped", "pre_passed")
    all_post_pass = bool(reqs) and sum(c[k] for k in passes_post) == len(reqs)
    # diff-aware sensitivity set: only tests the patch truly adds (not moved / rewritten ones)
    reqs_add, source_add = expected_f2p_truly_added(task)
    ca, _ = _f2p_statuses(reqs_add, pre_out, post_out)
    added = {"f2p_added_expected_source": source_add, "n_f2p_added_expected": len(reqs_add),
             "n_f2p_added_confirmed": ca["confirmed"], "n_f2p_added_uncollected_pre": ca["uncollected_pre"],
             "n_f2p_added_pre_skipped": ca["pre_skipped"], "n_f2p_added_pre_passed": ca["pre_passed"],
             "n_f2p_added_post_failed": ca["post_failed"], "n_f2p_added_not_collected": ca["not_collected"],
             "f2p_added_all_confirmed": bool(reqs_add) and ca["confirmed"] == len(reqs_add),
             "f2p_added_all_confirmed_or_uncollected": bool(reqs_add) and ca["confirmed"] + ca["uncollected_pre"] == len(reqs_add)}
    return {"f2p_expected_source": source, "n_f2p_expected": len(reqs), "n_f2p_confirmed": c["confirmed"],
            "n_f2p_uncollected_pre": c["uncollected_pre"], "n_f2p_pre_skipped": c["pre_skipped"],
            "n_f2p_pre_passed": c["pre_passed"], "n_f2p_post_failed": c["post_failed"],
            "n_f2p_not_collected": c["not_collected"], "n_f2p_observed": len(observed), "n_f2p_extra": len(extra),
            **added,
            "f2p_all_confirmed": bool(reqs) and c["confirmed"] == len(reqs),
            "f2p_all_confirmed_or_uncollected": bool(reqs) and c["confirmed"] + c["uncollected_pre"] == len(reqs),
            # the harness's own criterion needs only the post run: exit 0 and every required test passed
            "harness_all_expected_pass_post": all_post_pass and post_counts.get("failed", 0) == 0 and post_counts.get("error", 0) == 0,
            "f2p_observed": observed, "f2p_uncollected_pre": sorted(set(uncollected_pre_tests))}


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

def pre_trace_missing(tr_pre: dict | None) -> bool:
    """No usable pre-fix trace: absent (run killed) or no test executed (collection/import error)."""
    if not tr_pre:
        return True
    return not any(not t.startswith("<") for t in tr_pre.get("per_test_exec", {}))


def analyze_task(task, g: Graph, tr_pre: dict | None, tr_post: dict | None, gold_rows: list[dict],
                 kind_fallback: dict[str, str], edges_from_post: bool = False
                 ) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """edges_from_post: attribute edges (recall / precision) from the post-fix trace instead of the
    pre-fix one. Only used, opt-in, when the pre-fix trace is missing or empty; edges incident to a
    gold symbol are then dropped from the counts because the patch changed those functions."""
    m = Mapper(g)
    mp = map_trace(tr_pre, m, kind_fallback) if tr_pre else []
    mq = map_trace(tr_post, m, kind_fallback) if tr_post else []
    tr_edges, m_edges = (tr_post, mq) if edges_from_post else (tr_pre or {}, mp)

    def collapse(n: str) -> str:
        return n[:-9] if n.endswith(".__init__") and n[:-9] in g.nodes else n

    static_norm = {(collapse(s), collapse(d)) for s, d, t in g.edges() if t == "calls"}
    static_norm = {(s, d) for s, d in static_norm if s != d and not containment_edge(s, d)}
    gold_ids = {r["node_id"] for r in gold_rows if r.get("node_id") and r.get("qualname") and not r.get("new_symbol")}

    pre_out = (tr_pre or {}).get("outcomes", {})
    post_out = (tr_post or {}).get("outcomes", {})
    f2p = check_f2p(task, pre_out, post_out)
    f2p_tests = f2p["f2p_observed"]

    # ---- edge attribution (pre-fix trace; post-fix trace only under edges_from_post)
    tests = tr_edges.get("tests", [])
    edges: dict[tuple[int, int], dict] = {}
    n_gold_edges_dropped = 0
    for channel in ("edges_direct", "edges_vialib"):
        for c, e, tl in tr_edges.get(channel, []):
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
        a, b = m_edges[rec["c"]], m_edges[rec["e"]]
        if a["module_level"] or b["module_level"]:
            n_module_level += 1
            continue
        if edges_from_post and ({a["nid"], a["raw"], b["nid"], b["raw"]} & gold_ids):
            n_gold_edges_dropped += 1     # post-fix code of a patched function: not the base graph's call
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

    executed = {x["nid"] for x in m_edges if x["status"] in ("node", "collapsed_init")}
    if edges_from_post:
        executed -= gold_ids
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

    all_tests_pre = [t for t in (tr_pre or {}).get("per_test_exec", {}) if not t.startswith("<")]
    exec_f2p = executed_by(mp, tr_pre, f2p_tests) | executed_by(mq, tr_post, f2p_tests)
    exec_any = executed_by(mp, tr_pre, all_tests_pre) | (executed_by(mq, tr_post, [t for t in (tr_post or {}).get("per_test_exec", {}) if not t.startswith("<")]) if tr_post else set())
    # (a) sources = the f2p test's own node when the base graph has it (rare: f2p tests are new)
    f2p_all = list(dict.fromkeys(f2p_tests + f2p.get("f2p_uncollected_pre", [])))
    sources, levels = set(), Counter()
    for t in f2p_all:
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
    # (b) sources = the nodes where the f2p test enters the library (roots of its dynamic call forest)
    entry = entry_nodes(f2p_all, tr_pre, mp) | entry_nodes(f2p_all, tr_post, mq)
    entry_src = {n for n in entry if not is_test_module_id(n)} or entry
    reach_entry_dir = bfs_distance(g, entry_src, directed=True) if entry_src else {}
    reach_entry_und = bfs_distance(g, entry_src, directed=False) if entry_src else {}
    levels["entry_nodes"] = len(entry_src)
    gold_out = []
    gc = Counter()
    for r in gold_rows:
        if not r.get("qualname"):
            continue
        nid = r.get("node_id")
        if nid and nid not in g.nodes and flatten_qualname(nid) in g.nodes:
            nid = flatten_qualname(nid)       # A1 outputs from before the flattened-id fix
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
                         "static_sources_defined": bool(sources),
                         "entry_sources_defined": bool(entry_src),
                         "is_entry": bool(nid and nid in entry_src),
                         "static_reachable_from_entry_directed": bool(nid and nid in reach_entry_dir),
                         "static_hops_from_entry_directed": reach_entry_dir.get(nid) if nid else None,
                         "static_reachable_from_entry_undirected": bool(nid and nid in reach_entry_und),
                         "static_hops_from_entry_undirected": reach_entry_und.get(nid) if nid else None,
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
           "edges_from": "post" if edges_from_post else "pre", "n_gold_edges_dropped": n_gold_edges_dropped,
           **f2p, "f2p_source_levels": dict(levels)}
    row["f2p_observed"] = row["f2p_observed"][:30]
    row["f2p_uncollected_pre"] = row["f2p_uncollected_pre"][:30]
    for scope in ("all", "nontest", "gold"):
        for ch in ("direct", "any"):
            for cause in CAUSES:
                row[f"{scope}_{ch}_{cause}"] = cnt(scope, ch, cause)
    for st in ("in_graph_executed", "in_graph_not_executed", "absent_executed", "absent_not_executed",
               "new_symbol_executed_post", "new_symbol_not_executed"):
        row[f"gold_{st}"] = gc[st]
    row["gold_any_absent_executed"] = gc["absent_executed"] > 0
    row["gold_any_in_graph_executed"] = gc["in_graph_executed"] > 0
    row["static_sources_defined"] = bool(sources)
    row["n_entry_sources"] = len(entry_src)
    row["gold_any_static_reachable"] = any(x["static_reachable_directed"] for x in gold_out if x["status"].startswith("in_graph"))
    row["gold_executed_not_static"] = sum(1 for x in gold_out if x["status"] == "in_graph_executed" and not x["static_reachable_directed"])
    ing_exec = [x for x in gold_out if x["status"] == "in_graph_executed"]
    row["gold_any_static_reachable_entry"] = any(x["static_reachable_from_entry_directed"] for x in gold_out if x["status"].startswith("in_graph"))
    row["gold_all_executed_static_reachable_entry"] = bool(ing_exec) and all(x["static_reachable_from_entry_directed"] for x in ing_exec)
    row["gold_executed_not_static_entry"] = sum(1 for x in ing_exec if not x["static_reachable_from_entry_directed"])
    row["gold_executed_is_entry"] = sum(1 for x in ing_exec if x["is_entry"])
    row["gold_executed_static_reachable_entry_undirected"] = sum(1 for x in ing_exec if x["static_reachable_from_entry_undirected"])
    row["gold_min_hops_from_entry_directed"] = min((x["static_hops_from_entry_directed"] for x in ing_exec
                                                   if x["static_hops_from_entry_directed"] is not None), default=None)
    return row, dyn_rows, unconfirmed, gold_out


def entry_nodes(f2p_tests: list[str], tr: dict | None, mapped: list[dict]) -> set[str]:
    """Base-graph nodes where a fail-to-pass test enters the library: the roots of the test's dynamic
    call forest, i.e. executed non-test nodes none of whose caller chains (direct or via library
    frames, walked through frames that are not nodes) reaches another non-test node before test code.

    Defined on the per-test execution set rather than on "caller is the test frame" so it also works
    when the library runs in another thread (fastapi's TestClient runs the app through an anyio worker
    thread, where no ancestor frame is the test)."""
    if not tr or not f2p_tests:
        return set()
    tests = tr.get("tests", [])
    tid = {t: i for i, t in enumerate(tests)}
    pte = tr.get("per_test_exec", {})
    fns = tr.get("functions", [])

    def is_test_frame(i: int) -> bool:
        x = mapped[i]
        return bool(x["module_level"]) or is_test_module_id(x["nid"] or x["raw"] or fns[i])

    out = set()
    for t in f2p_tests:
        if t not in pte or t not in tid:
            continue
        ti = tid[t]
        execd = set(pte[t])
        callers: dict[int, set[int]] = defaultdict(set)
        for ch in ("edges_direct", "edges_vialib"):
            for c, e, tl in tr.get(ch, []):
                if ti in tl:
                    callers[e].add(c)
        def blocked(i: int) -> bool:
            """Does some caller chain of frame i reach another non-test *node* before reaching test code?
            Frames that are not nodes (absent async defs such as ``FastAPI.__call__``, nested or unmapped
            functions) are walked through, so a node whose only repo ancestors are absent defs is a root."""
            nid = mapped[i]["nid"]
            stack = [c for c in callers.get(i, ()) if c != i]
            seen: set[int] = set()
            while stack:
                c = stack.pop()
                if c in seen:
                    continue
                seen.add(c)
                if is_test_frame(c):
                    continue
                xc = mapped[c]
                if xc["status"] in ("node", "collapsed_init"):
                    if xc["nid"] != nid:
                        return True
                    continue
                stack.extend(cc for cc in callers.get(c, ()) if cc != c)
            return False

        for i in execd:
            x = mapped[i]
            if x["status"] not in ("node", "collapsed_init") or x["module_level"] or is_test_module_id(x["nid"]):
                continue
            if not blocked(i):
                out.add(x["nid"])
    return out


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
        # (a) test-node sources: only meaningful where the f2p test exists in the base graph
        with_src = [r for r in rs if r.get("static_sources_defined")]
        b["tasks_static_sources_defined"] = wilson(len(with_src), len(rs))
        b["tasks_gold_static_reachable"] = wilson(sum(1 for r in rs if r["gold_any_static_reachable"]), len(rs))
        b["tasks_gold_static_reachable_given_sources"] = wilson(sum(1 for r in with_src if r["gold_any_static_reachable"]), len(with_src))
        b["tasks_gold_executed_not_static"] = wilson(sum(1 for r in rs if r["gold_executed_not_static"] > 0), len(rs))
        # (b) entry sources: where the f2p test enters the library (defined wherever the test ran library code)
        with_entry = [r for r in rs if r.get("n_entry_sources", 0) > 0]
        exec_entry = [r for r in with_entry if r.get("gold_any_in_graph_executed")]
        b["tasks_entry_sources_defined"] = wilson(len(with_entry), len(rs))
        b["tasks_gold_static_reachable_entry"] = wilson(sum(1 for r in exec_entry if r["gold_any_static_reachable_entry"]), len(exec_entry))
        b["tasks_gold_all_executed_reachable_entry"] = wilson(sum(1 for r in exec_entry if r["gold_all_executed_static_reachable_entry"]), len(exec_entry))
        b["tasks_gold_executed_not_static_entry"] = wilson(sum(1 for r in exec_entry if r["gold_executed_not_static_entry"] > 0), len(exec_entry))
        b["tasks_gold_executed_is_entry"] = wilson(sum(1 for r in exec_entry if r["gold_executed_is_entry"] > 0), len(exec_entry))
        b["tasks_gold_reachable_entry_undirected_only"] = wilson(
            sum(1 for r in exec_entry if not r["gold_any_static_reachable_entry"] and r["gold_executed_static_reachable_entry_undirected"] > 0), len(exec_entry))
        b["gold_min_hops_from_entry_hist"] = dict(Counter(str(r.get("gold_min_hops_from_entry_directed")) for r in exec_entry))
        b["f2p_statuses"] = {k: sum(r.get(f"n_f2p_{k}", 0) for r in rs)
                             for k in ("expected", "confirmed", "uncollected_pre", "pre_skipped", "pre_passed", "post_failed", "not_collected")}
        b["f2p_statuses_truly_added"] = {k: sum(r.get(f"n_f2p_added_{k}", 0) for r in rs)
                                         for k in ("expected", "confirmed", "uncollected_pre", "pre_skipped", "pre_passed", "post_failed", "not_collected")}
        b["tasks_harness_resolved_post"] = wilson(sum(1 for r in rs if r.get("harness_all_expected_pass_post")), len(rs))
        b["gold_status_counts"] = {k: sum(r[f"gold_{k}"] for r in rs) for k in
                                   ("in_graph_executed", "in_graph_not_executed", "absent_executed", "absent_not_executed",
                                    "new_symbol_executed_post", "new_symbol_not_executed")}
        b["per_task_recall_representable_any"] = mean_ci(
            [r["all_any_captured"] / (r["all_any_captured"] + r["all_any_missed_between_nodes"])
             if (r["all_any_captured"] + r["all_any_missed_between_nodes"]) else None for r in rs])
        return b

    usable = [r for r in rows if r.get("usable")]
    strict = [r for r in usable if r.get("f2p_all_confirmed")]
    strict_added = [r for r in usable if r.get("f2p_added_all_confirmed")]
    # edge-only extension: usable tasks plus tasks whose edges came from the post-fix trace (pre trace
    # missing/empty) and whose post run passed every expected test. Use for recall/precision only.
    post_edges = [r for r in rows if r.get("edges_from") == "post" and r.get("post_run_class") == "clean"
                  and r.get("harness_all_expected_pass_post")]
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)
    out = {"usable_tasks": block(usable), "strict_tasks": block(strict),
           "strict_truly_added_tasks": block(strict_added)}
    if post_edges:
        out["usable_plus_post_edge_tasks"] = block(usable + [r for r in post_edges if not r.get("usable")])
        out["post_edge_tasks_only"] = block(post_edges)
    return {**out, "all_analyzed_tasks": block(rows),
            "per_repo_usable": {k: block([r for r in v if r.get("usable")]) for k, v in by_repo.items()},
            "per_repo_all": {k: block(v) for k, v in by_repo.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path, help="A2 run dir (<out> of run_task.py)")
    ap.add_argument("--a1", required=True, type=Path, help="A1 output dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--edges-from-post-when-pre-missing", action="store_true",
                    help="for tasks whose pre-fix trace is missing or empty (killed run, collection/import "
                         "error), attribute edges from the post-fix trace (gold-incident edges dropped); "
                         "reported in separate summary blocks, 'usable' is unchanged")
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
        use_post = bool(args.edges_from_post_when_pre_missing and tr_post and pre_trace_missing(tr_pre))
        if g is None or (tr_pre is None and not use_post):
            srow["usable"] = False
            status_rows.append(srow)
            continue
        row, dyn, unconf, gold = analyze_task(t, g, tr_pre, tr_post, gold_by_task.get(t.instance_id, []), kind_fallback,
                                              edges_from_post=use_post)
        row["run_class"] = val["run_class"]
        row["post_run_class"] = val.get("post_run_class")
        row["usable"] = val["run_class"] == "clean" and row["n_f2p_confirmed"] > 0
        row["strict"] = bool(row["usable"] and row["f2p_all_confirmed"])
        row["strict_truly_added"] = bool(row["usable"] and row["f2p_added_all_confirmed"])
        srow.update({k: row[k] for k in ("n_f2p_expected", "f2p_expected_source", "n_f2p_confirmed", "n_f2p_uncollected_pre",
                                         "n_f2p_pre_skipped", "n_f2p_pre_passed", "n_f2p_post_failed", "n_f2p_not_collected",
                                         "n_f2p_extra", "n_f2p_added_expected", "n_f2p_added_confirmed",
                                         "harness_all_expected_pass_post", "usable", "strict", "strict_truly_added",
                                         "edges_from")})
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
                              "tasks_all_expected_confirmed_or_uncollected": sum(1 for r in rows if r.get("f2p_all_confirmed_or_uncollected")),
                              "tasks_harness_resolved_post": sum(1 for r in rows if r.get("harness_all_expected_pass_post")),
                              "tasks_strict": sum(1 for r in rows if r.get("strict")),
                              "tasks_strict_truly_added": sum(1 for r in rows if r.get("strict_truly_added")),
                              "pre_skipped": sum(r.get("n_f2p_pre_skipped", 0) for r in rows),
                              "truly_added_expected": sum(r.get("n_f2p_added_expected", 0) for r in rows),
                              "truly_added_confirmed": sum(r.get("n_f2p_added_confirmed", 0) for r in rows),
                              "tasks_edges_from_post": sum(1 for r in rows if r.get("edges_from") == "post"),
                              "expected": sum(r.get("n_f2p_expected", 0) for r in rows),
                              "confirmed": sum(r.get("n_f2p_confirmed", 0) for r in rows),
                              "uncollected_pre": sum(r.get("n_f2p_uncollected_pre", 0) for r in rows),
                              "pre_passed": sum(r.get("n_f2p_pre_passed", 0) for r in rows),
                              "post_failed": sum(r.get("n_f2p_post_failed", 0) for r in rows),
                              "not_collected": sum(r.get("n_f2p_not_collected", 0) for r in rows)},
                "static_reachability_sources": {
                    "tasks_with_test_node_source": sum(1 for r in rows if r.get("static_sources_defined")),
                    "tasks_with_entry_source": sum(1 for r in rows if r.get("n_entry_sources", 0) > 0)}}
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

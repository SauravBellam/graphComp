"""Build oracle_pack.json.gz: the Plan A ground truth the Plan B scorecard needs, slimmed for Kaggle.

Run on the laptop from the graphComp folder:
    python3 scripts/planB/build_oracle_pack.py --out planB_results/kaggle_upload/oracle_pack.json.gz
    python3 scripts/planB/build_oracle_pack.py --out x.json.gz --tasks rich_3518 rich_3718   # subset

Inputs (paper repository copies on the laptop):
  planA_results/vm-a2/results/a6_v3/task_profile.csv       cell, graph_key, gold_nodes
  planA_results/vm-a2/results/a6_v3/graphs_v2/<key>.json   v2 graph: file, is_test, is_async, in_provided_graph
  planA_results/vm-a2/results/a1_v2/a1_gold_symbols.jsonl  gold symbols and files
  planA_results/vm-a2/results/a2_v3/a2_tasks.csv            f2p_observed (fail-to-pass test ids)
  planA_results/vm-a2/results/a2_runs/<task>/trace_{pre,post}.json   executed functions per test
  archive/graphs/<key>.json                                 provided graph (in-degree for hubs)

Per graph key: for every node of the PROVIDED graph, its file, is_test and calls in-degree
(provided-graph `calls` edges); the 90th/95th in-degree percentiles.
Per task: gold files (non-test), gold node ids, executed-by-fail-to-pass node ids (R2 = gold + executed),
1-hop expansion over v2 calls_static/calls_dynamic/inherits among non-test nodes (R3), the subset of
gold/R2 nodes that are async, and which gold/R2 nodes are missing from the provided graph.

Executed sets: A2's per_test_exec indices into trace['functions'] for the f2p tests, mapped to v2 node
ids by exact id, '<module>' -> module id, then by dropping leading components until a v2 id matches.
This is a simplified re-mapping, not A2's map_trace; it is recorded in the pack's `notes`.
"""
from __future__ import annotations

import argparse, csv, gzip, json, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "planA_results/vm-a2/results"
CALL_TYPES_V2 = {"calls_static", "calls_dynamic", "inherits"}


def pct(vals, q):
    if not vals:
        return 0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(q / 100 * (len(s) - 1)))))
    return s[k]


def map_fn(name, v2ids):
    if not name:
        return None
    if name.endswith(".<module>"):
        name = name[: -len(".<module>")]
    if name in v2ids:
        return name
    parts = name.split(".")
    for i in range(1, len(parts) - 1):
        cand = ".".join(parts[i:])
        if cand in v2ids:
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*")
    a = ap.parse_args()

    prof = {r["instance_id"]: r for r in csv.DictReader(open(RES / "a6_v3/task_profile.csv"))}
    a2 = {r["instance_id"]: r for r in csv.DictReader(open(RES / "a2_v3/a2_tasks.csv"))}
    gold_rows = defaultdict(list)
    for line in open(RES / "a1_v2/a1_gold_symbols.jsonl"):
        r = json.loads(line)
        gold_rows[r["instance_id"]].append(r)

    tasks = a.tasks or sorted(prof)
    keys = sorted({prof[t]["graph_key"] for t in tasks if prof[t].get("graph_key")})
    graphs, tasks_out = {}, {}
    v2cache = {}

    for key in keys:
        v2p = RES / "a6_v3/graphs_v2" / f"{key}.json"
        pp = ROOT / "archive/graphs" / f"{key}.json"
        if not v2p.exists() or not pp.exists():
            print("missing graph", key)
            continue
        v2 = json.load(open(v2p))
        prov = json.load(open(pp))
        v2nodes = {n["id"]: n for n in v2["nodes"]}
        indeg = defaultdict(int)
        for e in prov["edges"]:
            indeg[e["target"]] += 1
        files, findex, nodes = [], {}, {}
        for n in prov["nodes"]:
            vn = v2nodes.get(n["id"], {})
            f = vn.get("file") or ""
            if f not in findex:
                findex[f] = len(files)
                files.append(f)
            nodes[n["id"]] = [findex[f], int(bool(vn.get("is_test"))), indeg.get(n["id"], 0)]
        degs = [v[2] for v in nodes.values()]
        graphs[key] = {"files": files, "nodes": nodes, "indeg_p90": pct(degs, 90), "indeg_p95": pct(degs, 95)}
        adj = defaultdict(set)
        for e in v2["edges"]:
            if e["type"] in CALL_TYPES_V2:
                adj[e["source"]].add(e["target"])
                adj[e["target"]].add(e["source"])
        v2cache[key] = (v2nodes, adj, set(nodes))

    for t in tasks:
        p = prof.get(t)
        if not p or p.get("graph_key") not in v2cache:
            continue
        key = p["graph_key"]
        v2nodes, adj, provset = v2cache[key]
        v2ids = set(v2nodes)
        gr = gold_rows.get(t, [])
        gold_files = sorted({r["file"] for r in gr if r.get("file") and not _is_test_path(r["file"])})
        gold_nodes = sorted({r["node_id"] for r in gr if r.get("node_id")})
        executed = set()
        f2p = []
        try:
            f2p = json.loads(a2.get(t, {}).get("f2p_observed") or "[]")
        except Exception:
            f2p = []
        for ph in ("pre", "post"):
            tp = RES / "a2_runs" / t / f"trace_{ph}.json"
            if not tp.exists():
                continue
            tr = json.load(open(tp))
            fns = tr.get("functions", [])
            pte = tr.get("per_test_exec", {})
            for test in f2p:
                for i in pte.get(test, []):
                    if 0 <= i < len(fns):
                        nid = map_fn(fns[i], v2ids)
                        if nid and not v2nodes[nid].get("is_test"):
                            executed.add(nid)
        r1 = set(gold_nodes)
        r2 = r1 | executed
        r3 = set(r2)
        for n in r2:
            for m in adj.get(n, ()):
                if m in v2nodes and not v2nodes[m].get("is_test"):
                    r3.add(m)
        asyncs = sorted(n for n in r2 if v2nodes.get(n, {}).get("kind") in ("async_function",) or v2nodes.get(n, {}).get("is_async"))
        missing = sorted(n for n in r2 if n not in provset)
        tasks_out[t] = {
            "repo": t.split("_")[0], "graph_key": key, "cell": p.get("cell"),
            "multi_file": p.get("multi_file") == "True", "a2_usable": p.get("a2_usable") == "True",
            "gold_files": gold_files, "R1": sorted(r1), "R2": sorted(r2), "R3": sorted(r3),
            "n_executed_f2p": len(executed), "f2p_tests": f2p,
            "async_R2": asyncs, "R2_missing_from_provided": missing,
        }

    pack = {"version": "oracle-pack-1", "graphs": graphs, "tasks": tasks_out,
            "notes": "executed = f2p per_test_exec mapped by exact id / <module> / suffix drop (not A2 map_trace); "
                     "indeg = provided-graph calls in-degree; file/is_test from graphs_v2"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(pack, f, separators=(",", ":"))
    ne = [v["n_executed_f2p"] for v in tasks_out.values()]
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB): {len(graphs)} graphs, {len(tasks_out)} tasks; "
          f"tasks with executed>0: {sum(x>0 for x in ne)}; median executed {statistics.median(ne) if ne else 0}")


def _is_test_path(p):
    q = "/" + p
    return "/tests/" in q or "/test/" in q or p.split("/")[-1].startswith("test_") or p.endswith("_test.py") or p.endswith("conftest.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A1 (static census) + B1 (task profile: structural distance d, lexical visibility v).

Per graph (127 distinct base commits):
  * node/edge counts, edge types, containment pseudo-edges inside `calls`, test-node share,
    isolated nodes, degree hubs
  * async omission test (HANDOFF finding 5.1): parse every .py module of the snapshot that
    has at least one node in the graph, enumerate defs with `ast`, and record presence in the
    graph split by sync / async / class / nested
Per task:
  * gold-patch symbols (from `patch`, mapped with `ast` on the base snapshot) and whether each
    exists as a node; prefix rule that matched (full / drop:docs_src / drop:src)
  * issue identifiers (deterministic extraction) and how they resolve under the harness's
    4-tier `resolve_node_name` (exact / suffix / case-insensitive / substring)
  * d: GraphLocator-style distance = max over gold nodes of the shortest undirected hop
    distance from any resolved issue node (also the directed variant); undefined when no
    identifier resolves or no gold symbol is a node (reported, never imputed)
  * v: best BM25 rank of a gold file (file-level, over node text grouped by module) and of a
    gold node (node-level), using the problem statement as query
  * multi-file flag, number of gold source files, repo

Outputs (``--out``): a1_graphs.csv, a1_tasks.csv, a1_gold_symbols.jsonl, a1_identifiers.jsonl,
a1_async_audit.csv, a1_summary.json.  Shardable with ``--shard i/N`` (per task); graph-level
stats are computed for the graphs touched by the shard and merged by ``merge_shards``.

VERIFY-ON-DATA: node-id prefix rules (docs_src / src), that edge types are only `calls`, that
graph files of hard-link pairs can be empty, `edge_type` filter case (data says `calls`, README
says `CALLS`; swegemma compares with == so `CALLS` returns nothing - confirm by data).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BM25, Graph, ModuleIndex, Snapshot, Task, bfs_distance, containment_edge, extract_identifiers,
    find_snapshot, flat_identifiers, get_graph, gold_symbols, is_test_path, iter_defs, load_tasks,
    pct, qualname_to_node_id, resolve_node_name, shard, write_csv, write_jsonl,
)


def graph_stats(g: Graph) -> dict:
    n = len(g.nodes)
    test_nodes = [x for x in g.nodes if g.is_test_node(x)]
    deg = Counter()
    for s, d, _ in g.edges():
        deg[s] += 1
        deg[d] += 1
    isolated = [x for x in g.nodes if deg[x] == 0]
    containment = sum(1 for s, d, t in g.edges() if t == "calls" and containment_edge(s, d))
    from_test = sum(1 for s, d, t in g.edges() if g.is_test_node(s))
    self_loops = sum(1 for s, d, t in g.edges() if s == d)
    dangling = sum(1 for s, d, t in g.edges() if s not in g.nodes or d not in g.nodes)
    async_text = sum(1 for x in g.nodes.values() if x["text"].lstrip().startswith("async def"))
    async_any = sum(1 for x in g.nodes.values() if "\nasync def " in x["text"] or x["text"].lstrip().startswith("async def"))
    in_deg = Counter(d for s, d, _ in g.edges())
    hubs = in_deg.most_common(5)
    return {
        "graph_key": g.key, "n_nodes": n, "n_edges": g.n_edges,
        "edge_types": json.dumps(dict(g.edge_types)),
        "pct_containment_in_calls": pct(containment, g.edge_types.get("calls", 0)),
        "pct_test_nodes": pct(len(test_nodes), n),
        "pct_edges_from_test": pct(from_test, g.n_edges),
        "n_isolated": len(isolated), "pct_isolated": pct(len(isolated), n),
        "n_isolated_test": sum(1 for x in isolated if g.is_test_node(x)),
        "self_loops": self_loops, "dangling_edges": dangling,
        "nodes_text_async_def": async_text, "nodes_text_contains_async_def": async_any,
        "top_in_degree": json.dumps(hubs),
        "meta_keys": json.dumps(sorted(g.raw_meta.keys())),
    }


def async_audit(g: Graph, snap: Snapshot, max_files: int | None = None) -> tuple[list[dict], dict]:
    """For every .py file whose module has >=1 node, list defs and whether they are graph nodes."""
    mi = ModuleIndex(g)
    rows = []
    agg = Counter()
    files = snap.list_files(".py")
    # only files that map to a module with at least one node (so missing defs are real omissions)
    targets = []
    for f in files:
        mod, rule = mi.module_for(f)
        if mod is None:
            agg["files_unmapped"] += 1
            continue
        if not any(nid == mod or nid.startswith(mod + ".") for nid in g.nodes):
            agg["files_module_without_nodes"] += 1
            continue
        targets.append((f, mod, rule))
    if max_files:
        targets = targets[:max_files]
    texts = snap.read_many([f for f, _, _ in targets])
    for f, mod, rule in targets:
        src = texts.get(f)
        if src is None:
            continue
        for d in iter_defs(src):
            nid = qualname_to_node_id(mod, d.qualname)
            present = nid in g.nodes
            cat = "nested" if d.nested else d.kind
            agg[f"{cat}_total"] += 1
            agg[f"{cat}_present"] += int(present)
            rows.append({"graph_key": g.key, "file": f, "module": mod, "rule": rule, "node_id": nid,
                         "kind": d.kind, "nested": d.nested, "present": present,
                         "decorators": ";".join(d.decorators)[:200], "is_test": is_test_path(f)})
    agg["files_checked"] = len(targets)
    return rows, dict(agg)


def task_profile(task: Task, g: Graph | None, snap: Snapshot | None, bm25_cache: dict) -> tuple[dict, list[dict], list[dict]]:
    row = {"instance_id": task.instance_id, "repo": task.repo, "base_commit": task.base_commit,
           "graph_key": task.graph_key, "graph_available": g is not None, "snapshot_available": snap is not None,
           "problem_chars": len(task.problem_statement or ""), "hints_chars": len(task.hints_text or "")}
    gold_rows: list[dict] = []
    id_rows: list[dict] = []
    gold_files = [f for f in task.gold_files if f.endswith(".py") and not is_test_path(f)]
    row["n_gold_src_files"] = len(gold_files)
    row["multi_file"] = len(gold_files) > 1
    row["gold_files"] = gold_files
    if snap is None:
        return row, gold_rows, id_rows
    syms = gold_symbols(task, snap)
    mi = ModuleIndex(g) if g else None
    gold_nodes: list[str] = []
    n_sym = n_present = n_nested = n_new = n_async = n_async_present = 0
    for s in syms:
        mod, rule = (mi.module_for(s.file) if mi else (None, "none"))
        nid = qualname_to_node_id(mod, s.qualname) if mod else None
        present = bool(g and nid and nid in g.nodes)
        if s.qualname is not None:
            n_sym += 1
            n_present += int(present)
            n_nested += int(s.nested)
            n_new += int(s.new_symbol)
            if s.kind == "async_function":
                n_async += 1
                n_async_present += int(present)
        if present and not s.new_symbol:
            gold_nodes.append(nid)
        gold_rows.append({"instance_id": task.instance_id, "file": s.file, "qualname": s.qualname, "kind": s.kind,
                          "nested": s.nested, "new_symbol": s.new_symbol, "source": s.source,
                          "module": mod, "prefix_rule": rule, "node_id": nid, "present_in_graph": present})
    row.update({"n_gold_symbols": n_sym, "n_gold_symbols_present": n_present, "n_gold_nested": n_nested,
                "n_gold_new_symbols": n_new, "n_gold_async": n_async, "n_gold_async_present": n_async_present,
                "gold_nodes": sorted(set(gold_nodes)), "module_level_only": n_sym == 0})
    if g is None:
        return row, gold_rows, id_rows

    # identifiers -> nodes (harness tiers)
    ids = extract_identifiers(task.problem_statement)
    node_list = list(g.nodes)
    resolved_strict: list[str] = []
    resolved_any: list[str] = []
    tier_count = Counter()
    for kind, toks in ids.items():
        for tok in toks:
            nid, tier = resolve_node_name(tok, node_list)
            tier_count[tier] += 1
            id_rows.append({"instance_id": task.instance_id, "kind": kind, "token": tok, "resolved": nid, "tier": tier,
                            "resolved_is_test": bool(nid and g.is_test_node(nid)),
                            "resolved_is_gold": bool(nid and nid in gold_nodes)})
            if nid:
                resolved_any.append(nid)
                if tier in ("exact", "suffix"):
                    resolved_strict.append(nid)
    row["n_identifiers"] = sum(len(v) for v in ids.values())
    row["identifier_tiers"] = dict(tier_count)
    row["n_resolved_strict"] = len(set(resolved_strict))
    row["n_resolved_any"] = len(set(resolved_any))
    row["anchors_hit_gold"] = len(set(resolved_any) & set(gold_nodes))

    # structural distance d (GraphLocator): max over gold nodes of min hop distance from any anchor
    for label, anchors in (("strict", set(resolved_strict)), ("any", set(resolved_any))):
        for mode, directed in (("undirected", False), ("directed", True)):
            key = f"d_{label}_{mode}"
            if not anchors or not gold_nodes:
                row[key] = None
                continue
            dist = bfs_distance(g, anchors, directed=directed)
            ds = [dist.get(x) for x in set(gold_nodes)]
            row[key] = None if any(x is None for x in ds) else max(ds)
            row[key + "_min"] = None if all(x is None for x in ds) else min(x for x in ds if x is not None)
    row["d_defined"] = row.get("d_any_undirected") is not None

    # lexical visibility v: BM25 over node text (node level) and grouped by module (file level)
    if g.key not in bm25_cache:
        node_docs = {nid: f"{nid}\n{a['text']}" for nid, a in g.nodes.items()}
        mod_docs: dict[str, list[str]] = defaultdict(list)
        for nid, a in g.nodes.items():
            mod = module_of_node(nid, g)
            mod_docs[mod].append(f"{nid}\n{a['text']}")
        bm25_cache[g.key] = (BM25(node_docs), BM25({m: "\n".join(v) for m, v in mod_docs.items()}))
    bm_node, bm_mod = bm25_cache[g.key]
    q = task.problem_statement or ""
    node_rank = bm_node.rank(q)
    gold_set = set(gold_nodes)
    row["v_node_rank"] = next((i + 1 for i, x in enumerate(node_rank) if x in gold_set), None) if gold_set else None
    mod_rank = bm_mod.rank(q)
    gold_mods = set()
    for f in gold_files:
        m, _ = mi.module_for(f)
        if m:
            gold_mods.add(m)
    row["gold_modules"] = sorted(gold_mods)
    row["v_file_rank"] = next((i + 1 for i, x in enumerate(mod_rank) if x in gold_mods), None) if gold_mods else None
    row["v_file_ranks_all"] = [next((i + 1 for i, x in enumerate(mod_rank) if x == m), None) for m in sorted(gold_mods)]
    row["v_defined"] = row["v_file_rank"] is not None
    row["v_high"] = bool(row["v_file_rank"] is not None and row["v_file_rank"] <= 5)
    return row, gold_rows, id_rows


_MODULE_OF: dict[str, dict[str, str]] = {}


def module_of_node(nid: str, g: Graph) -> str:
    """Heuristic module id of a node: longest prefix that is not itself a def-like node name.

    VERIFY-ON-DATA: nodes carry no file attribute. We use: strip trailing segments that start with
    an uppercase letter (classes) or are known defs (other node ids with text starting with def/class).
    """
    cache = _MODULE_OF.setdefault(g.key, {})
    if nid in cache:
        return cache[nid]
    parts = nid.split(".")
    # walk from the right while the prefix is a node whose text starts with def/class/async
    k = len(parts)
    while k > 1:
        prefix = ".".join(parts[:k])
        text = g.nodes.get(prefix, {}).get("text", "").lstrip()
        if prefix in g.nodes and text.startswith(("def ", "async def ", "class ", "@")):
            k -= 1
            continue
        break
    mod = ".".join(parts[:k])
    cache[nid] = mod
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=None, help="snapshot extraction cache (default <out>/../cache)")
    ap.add_argument("--shard", default=None, help="i/N")
    ap.add_argument("--task-ids", nargs="*", default=None)
    ap.add_argument("--no-async-audit", action="store_true")
    ap.add_argument("--async-max-files", type=int, default=None)
    args = ap.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache or out.parent / "cache"
    tasks = shard(load_tasks(args.data, args.task_ids), args.shard)
    print(f"[a1] {len(tasks)} tasks (shard={args.shard})", flush=True)

    graph_rows, task_rows, gold_rows, id_rows, async_rows = [], [], [], [], []
    async_agg: dict[str, dict] = {}
    seen_graphs: set[str] = set()
    bm25_cache: dict = {}
    for i, t in enumerate(tasks):
        g = get_graph(args.data, t)
        sp = find_snapshot(args.data, t)
        snap = Snapshot(sp, cache, t.instance_id) if sp else None
        if g is not None and g.key not in seen_graphs:
            seen_graphs.add(g.key)
            gs = graph_stats(g)
            gs["example_instance"] = t.instance_id
            gs["repo"] = t.repo
            if snap is not None and not args.no_async_audit:
                rows, agg = async_audit(g, snap, args.async_max_files)
                async_rows.extend(rows)
                async_agg[g.key] = agg
                gs.update({f"audit_{k}": v for k, v in agg.items()})
            graph_rows.append(gs)
        row, grows, irows = task_profile(t, g, snap, bm25_cache)
        task_rows.append(row)
        gold_rows.extend(grows)
        id_rows.extend(irows)
        print(f"[a1] {i + 1}/{len(tasks)} {t.instance_id}: graph={'y' if g else 'n'} snap={'y' if snap else 'n'} "
              f"gold={row.get('n_gold_symbols')} present={row.get('n_gold_symbols_present')} "
              f"d={row.get('d_any_undirected')} v_file={row.get('v_file_rank')}", flush=True)

    suffix = f"_shard{args.shard.replace('/', 'of')}" if args.shard else ""
    write_csv(out / f"a1_graphs{suffix}.csv", graph_rows)
    write_csv(out / f"a1_tasks{suffix}.csv", task_rows)
    write_jsonl(out / f"a1_gold_symbols{suffix}.jsonl", gold_rows)
    write_jsonl(out / f"a1_identifiers{suffix}.jsonl", id_rows)
    write_csv(out / f"a1_async_audit{suffix}.csv", async_rows)
    summary = summarize(task_rows, graph_rows, async_rows, id_rows)
    (out / f"a1_summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def summarize(task_rows, graph_rows, async_rows, id_rows) -> dict:
    n = len(task_rows)
    by_repo = Counter(r["repo"] for r in task_rows)
    s = {
        "n_tasks": n, "per_repo": dict(by_repo), "n_graphs": len(graph_rows),
        "tasks_graph_missing": sum(1 for r in task_rows if not r.get("graph_available")),
        "tasks_snapshot_missing": sum(1 for r in task_rows if not r.get("snapshot_available")),
        "gold_symbols": sum(r.get("n_gold_symbols", 0) or 0 for r in task_rows),
        "gold_symbols_present": sum(r.get("n_gold_symbols_present", 0) or 0 for r in task_rows),
        "gold_async": sum(r.get("n_gold_async", 0) or 0 for r in task_rows),
        "gold_async_present": sum(r.get("n_gold_async_present", 0) or 0 for r in task_rows),
        "gold_nested": sum(r.get("n_gold_nested", 0) or 0 for r in task_rows),
        "tasks_module_level_only": sum(1 for r in task_rows if r.get("module_level_only")),
        "tasks_multi_file": sum(1 for r in task_rows if r.get("multi_file")),
        "tasks_d_defined": sum(1 for r in task_rows if r.get("d_defined")),
        "tasks_v_defined": sum(1 for r in task_rows if r.get("v_defined")),
        "tasks_zero_identifiers": sum(1 for r in task_rows if r.get("n_identifiers") == 0),
        "d_any_undirected_hist": dict(Counter(str(r.get("d_any_undirected")) for r in task_rows)),
        "d_strict_undirected_hist": dict(Counter(str(r.get("d_strict_undirected")) for r in task_rows)),
        "v_file_rank_le5": sum(1 for r in task_rows if r.get("v_high")),
        "identifier_tier_hist": dict(Counter(r["tier"] for r in id_rows)),
        "identifier_resolved_to_test_node": sum(1 for r in id_rows if r.get("resolved_is_test")),
        "identifier_resolved_to_gold": sum(1 for r in id_rows if r.get("resolved_is_gold")),
        "edge_types_union": dict(sum((Counter(json.loads(r["edge_types"])) for r in graph_rows), Counter())),
    }
    if async_rows:
        agg = Counter()
        for r in async_rows:
            cat = "nested" if r["nested"] else r["kind"]
            agg[f"{cat}_total"] += 1
            agg[f"{cat}_present"] += int(r["present"])
        s["async_audit"] = {
            k: v for k, v in agg.items()
        }
        for cat in ("function", "async_function", "class", "nested"):
            s["async_audit"][f"{cat}_present_pct"] = pct(agg[f"{cat}_present"], agg[f"{cat}_total"])
        per_repo = defaultdict(Counter)
        key_repo = {g["graph_key"]: g["repo"] for g in graph_rows}
        for r in async_rows:
            cat = "nested" if r["nested"] else r["kind"]
            c = per_repo[key_repo.get(r["graph_key"], "?")]
            c[f"{cat}_total"] += 1
            c[f"{cat}_present"] += int(r["present"])
        s["async_audit_per_repo"] = {k: dict(v) for k, v in per_repo.items()}
    return s


def merge_shards(out: Path) -> None:
    """Concatenate a1_*_shardXofN.* files into unsharded files (dedupe graphs by key)."""
    import csv
    import glob
    for stem in ("a1_graphs", "a1_tasks", "a1_async_audit"):
        rows, cols = [], []
        for p in sorted(glob.glob(str(out / f"{stem}_shard*.csv"))):
            with open(p, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for c in (r.fieldnames or []):
                    if c not in cols:
                        cols.append(c)
                rows.extend(r)
        if stem in ("a1_graphs",):
            seen, ded = set(), []
            for r in rows:
                if r["graph_key"] not in seen:
                    seen.add(r["graph_key"])
                    ded.append(r)
            rows = ded
        if rows:
            with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols, restval="")
                w.writeheader()
                w.writerows(rows)
    for stem in ("a1_gold_symbols", "a1_identifiers"):
        lines = []
        for p in sorted(glob.glob(str(out / f"{stem}_shard*.jsonl"))):
            lines.extend(Path(p).read_text(encoding="utf-8").splitlines())
        if lines:
            (out / f"{stem}.jsonl").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge_shards(Path(sys.argv[2]))
    else:
        main()

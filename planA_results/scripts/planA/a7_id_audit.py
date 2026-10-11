#!/usr/bin/env python3
"""A7: reconcile the provided graph's node ids with the source tree (closes the two open data gaps).

Oct 10 run left two unexplained numbers: requests "missing" 19.7% of its sync functions (A1), and
A6 found 43 to 139 provided nodes per graph that the source parse could not locate
(``provided_only``). Inspection of the v2 graphs (Oct 11) showed both are id-scheme differences,
not missing code:

  * nested definitions are nodes, under a flattened id: enclosing *functions* and ``<locals>``
    dropped, classes kept (``Response.iter_content.<locals>.generate`` -> ``Response.generate``,
    ``test_x.<locals>.Thing._repr_html_`` -> ``Thing._repr_html_``)  [common.flatten_qualname]
  * in requests, a subset of ``TestRequests`` methods appear at module level
    (``tests.test_requests.test_cookie_parameters``) with no class segment

This script measures both exactly, per graph and per repo, against the real snapshot:

  provided nodes  -> matched_def | matched_nested_flat | class_dropped_method | package_or_module
                     | unmatched (text head kept so the residue can be read)
  parsed defs     -> present | present_nested_flat | present_class_dropped | absent_async
                     | absent_nested | absent_other

and, for class-dropped methods, dumps the features an explanation would need (line numbers,
decorators, preceding sibling, nested defs/classes in the body, whether the provided node's text
equals the method's source) so the generator's rule can be found or the residue reported as such.

Outputs (``--out``): a7_provided_nodes.csv (one row per provided node that is not a plain match),
a7_defs_absent.csv (one row per parsed def that is absent under every id), a7_class_dropped.csv,
a7_summary.json. Shardable by graph with ``--shard i/N``; merge with ``a7_id_audit.py merge <out>``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ModuleIndex, Snapshot, find_snapshot, flatten_qualname, get_graph, is_test_path, iter_defs, load_tasks, shard,
    write_csv,
)


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def audit_graph(g, snap: Snapshot) -> tuple[dict, list[dict], list[dict], list[dict]]:
    mi = ModuleIndex(g)
    files = snap.list_files(".py")
    targets = []
    for f in files:
        mod, rule = mi.module_for(f, infer=True)
        if mod is not None:
            targets.append((f, mod, rule))
    texts = snap.read_many([f for f, _, _ in targets])

    defs: list[dict] = []
    by_flat: dict[str, list[dict]] = defaultdict(list)
    by_short: dict[str, list[dict]] = defaultdict(list)     # (module, bare name) -> defs
    modules: set[str] = set()
    for f, mod, rule in targets:
        src = texts.get(f)
        if src is None:
            continue
        modules.add(mod)
        lines = src.splitlines()
        file_defs = iter_defs(src)
        for i, d in enumerate(file_defs):
            flat_q = flatten_qualname(d.qualname)
            rec = {"file": f, "module": mod, "rule": rule, "qualname": d.qualname, "flat_qualname": flat_q,
                   "id_flat": f"{mod}.{flat_q}", "id_locals": f"{mod}.{d.qualname}",
                   "kind": d.kind, "nested": d.nested, "lineno": d.lineno, "end_lineno": d.end_lineno,
                   "decorators": ";".join(d.decorators)[:200], "is_test": is_test_path(f),
                   "text": "\n".join(lines[d.lineno - 1: d.end_lineno]), "_idx": i, "_file_defs": file_defs}
            defs.append(rec)
            by_flat[rec["id_flat"]].append(rec)
            by_short[(mod, flat_q.rsplit(".", 1)[-1])].append(rec)

    # ---- provided nodes
    pkg_prefixes: set[str] = set()
    for m in modules:
        parts = m.split(".")
        for k in range(1, len(parts) + 1):
            pkg_prefixes.add(".".join(parts[:k]))
    prov_rows: list[dict] = []
    prov_counts = Counter()
    for nid, attrs in g.nodes.items():
        cands = by_flat.get(nid, [])
        if cands:
            st = "matched_nested_flat" if all(c["nested"] for c in cands) else "matched_def"
            prov_counts[st] += 1
            if st == "matched_def":
                continue
            row = {"node_id": nid, "status": st, "n_defs_with_this_id": len(cands),
                   "def_qualnames": ";".join(c["qualname"] for c in cands)[:300]}
        elif nid in pkg_prefixes:
            st = "package_or_module"
            prov_counts[st] += 1
            row = {"node_id": nid, "status": st}
        else:
            mod, _, name = nid.rpartition(".")
            dropped = [c for c in by_short.get((mod, name), []) if c["id_flat"] != nid and "." in c["flat_qualname"]]
            if dropped:
                st = "class_dropped_method"
                row = {"node_id": nid, "status": st, "n_candidates": len(dropped),
                       "candidates": ";".join(c["id_flat"] for c in dropped)[:300],
                       "text_matches_candidate": any(norm_text(c["text"]) == norm_text(attrs.get("text")) for c in dropped)}
            else:
                st = "unmatched"
                row = {"node_id": nid, "status": st, "text_head": (attrs.get("text") or "")[:120].replace("\n", " | ")}
            prov_counts[st] += 1
        row["graph_key"] = g.key
        row["is_test"] = g.is_test_node(nid)
        prov_rows.append(row)

    # ---- parsed defs
    def_counts = Counter()
    absent_rows: list[dict] = []
    dropped_rows: list[dict] = []
    for d in defs:
        if d["id_flat"] in g.nodes:
            st = "present_nested_flat" if d["nested"] else "present"
        else:
            short = f"{d['module']}.{d['flat_qualname'].rsplit('.', 1)[-1]}"
            if "." in d["flat_qualname"] and short in g.nodes and not by_flat.get(short):
                st = "present_class_dropped"
            elif d["kind"] == "async_function":
                st = "absent_async"
            elif d["nested"]:
                st = "absent_nested"
            else:
                st = "absent_other"
        def_counts[(d["kind"] if not d["nested"] else "nested", st)] += 1
        base = {k: v for k, v in d.items() if not k.startswith("_") and k != "text"}
        base["graph_key"] = g.key
        if st.startswith("absent"):
            absent_rows.append({**base, "status": st})
        if st == "present_class_dropped":
            fd = d["_file_defs"]
            prev = fd[d["_idx"] - 1] if d["_idx"] > 0 else None
            body = d["text"]
            prov_text = g.nodes.get(short, {}).get("text", "")
            dropped_rows.append({**base, "status": st, "provided_id": short,
                                 "provided_text_equal": norm_text(prov_text) == norm_text(body),
                                 "n_lines": d["end_lineno"] - d["lineno"] + 1,
                                 "has_nested_def": bool(re.search(r"\n\s+def ", body)),
                                 "has_nested_class": bool(re.search(r"\n\s+class ", body)),
                                 "has_decorator": bool(d["decorators"]),
                                 "prev_qualname": prev.qualname if prev else "", "prev_kind": prev.kind if prev else "",
                                 "prev_nested": prev.nested if prev else "",
                                 "prev_end_lineno": prev.end_lineno if prev else "",
                                 "gap_lines_from_prev": (d["lineno"] - prev.end_lineno - 1) if prev else ""})
    summ = {"graph_key": g.key, "n_provided": len(g.nodes), "provided": dict(prov_counts),
            "defs": {f"{k[0]}|{k[1]}": v for k, v in sorted(def_counts.items())},
            "files_checked": len(targets)}
    return summ, prov_rows, absent_rows, dropped_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--shard", default=None, help="i/N over distinct graphs")
    ap.add_argument("--task-ids", nargs="*", default=None)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.cache or args.out.parent / "cache"

    tasks = load_tasks(args.data, args.task_ids)
    by_key: dict[str, list] = defaultdict(list)
    for t in tasks:
        by_key[t.graph_key].append(t)
    keys = shard(sorted(by_key), args.shard)
    summaries, prov_all, absent_all, dropped_all = [], [], [], []
    for i, key in enumerate(keys):
        t = by_key[key][0]
        g = get_graph(args.data, t)
        sp = find_snapshot(args.data, t)
        if g is None or sp is None:
            summaries.append({"graph_key": key, "error": "graph or snapshot missing"})
            continue
        snap = Snapshot(sp, cache, t.instance_id)
        summ, prov, absent, dropped = audit_graph(g, snap)
        summ["repo"] = t.repo
        summaries.append(summ)
        prov_all += prov
        absent_all += absent
        dropped_all += dropped
        print(f"[a7] {i + 1}/{len(keys)} {key}: provided={summ['provided']} absent_defs="
              f"{sum(v for k, v in summ['defs'].items() if '|absent' in k)}", flush=True)
    suffix = f"_shard{args.shard.replace('/', 'of')}" if args.shard else ""
    write_csv(args.out / f"a7_provided_nodes{suffix}.csv", prov_all)
    write_csv(args.out / f"a7_defs_absent{suffix}.csv", absent_all)
    write_csv(args.out / f"a7_class_dropped{suffix}.csv", dropped_all)
    (args.out / f"a7_summary{suffix}.json").write_text(json.dumps(summarize(summaries), indent=2))
    print(json.dumps(summarize(summaries)["per_repo"], indent=2))


def summarize(summaries: list[dict]) -> dict:
    per_repo: dict[str, dict] = {}
    for s in summaries:
        if "error" in s:
            continue
        r = per_repo.setdefault(s["repo"], {"n_graphs": 0, "provided": Counter(), "defs": Counter()})
        r["n_graphs"] += 1
        r["provided"].update(s["provided"])
        r["defs"].update(s["defs"])
    out = {"per_repo": {k: {"n_graphs": v["n_graphs"], "provided": dict(v["provided"]), "defs": dict(v["defs"])}
                        for k, v in per_repo.items()},
           "graphs": summaries}
    return out


def merge_shards(out: Path) -> None:
    import glob
    for stem in ("a7_provided_nodes", "a7_defs_absent", "a7_class_dropped"):
        rows, cols = [], []
        for p in sorted(glob.glob(str(out / f"{stem}_shard*.csv"))):
            with open(p, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for c in (r.fieldnames or []):
                    if c not in cols:
                        cols.append(c)
                rows.extend(r)
        if rows:
            with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols, restval="")
                w.writeheader()
                w.writerows(rows)
    graphs = []
    for p in sorted(glob.glob(str(out / "a7_summary_shard*.json"))):
        graphs += json.loads(Path(p).read_text())["graphs"]
    (out / "a7_summary.json").write_text(json.dumps({**summarize(graphs), "merged_from_shards": True}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge_shards(Path(sys.argv[2]))
    else:
        main()

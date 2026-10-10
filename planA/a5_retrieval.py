#!/usr/bin/env python3
"""A5: retrieval-mode audit. How often does each retrieval mode surface the gold functions / files?

Modes (per task, query = problem statement):
  (a) harness   : issue identifiers (deterministic extraction) -> swegemma's 4-tier resolve_node_name
                  -> cosine ranking against all nodes (exactly search_similar_code), union over
                  identifiers ranked by best similarity; the resolved anchors themselves count as rank 0
  (a0) anchors  : the resolved anchors alone (what an agent sees before any similarity call)
  (b) bm25      : BM25 over node text (id + text), free-text issue query
  (c) freetext  : issue text embedded with the encoder identified in A3 (``--encoder``), cosine
                  against the provided vectors.  Skipped when no encoder is given.
  (d) harness+1hop : mode (a) plus 1-hop call neighbours of the anchors and of the top-10 hits
Metrics: gold-node Hit@10 (any gold node in top 10) and gold-file Hit@5 (file = module of a gold
node; a node's module via the same heuristic as A1), MRR, per repo; share of tasks where the mode
returns nothing at all (no identifier resolves).

Inputs: archive + A1 outputs (a1_gold_symbols.jsonl for gold node ids; a1_tasks.csv not required).
Outputs: a5_tasks.csv, a5_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_census import module_of_node  # noqa: E402
from common import (  # noqa: E402
    BM25, ModuleIndex, extract_identifiers, flat_identifiers, get_embeddings, get_graph, is_test_path, load_tasks,
    pct, read_jsonl, resolve_node_name, write_csv,
)


def rank_metrics(ranked: list[str], gold_nodes: set[str], gold_mods: set[str], mod_of) -> dict:
    hit_node = next((i + 1 for i, n in enumerate(ranked) if n in gold_nodes), None)
    files_seen, hit_file = [], None
    for n in ranked:
        m = mod_of(n)
        if m not in files_seen:
            files_seen.append(m)
            if m in gold_mods and hit_file is None:
                hit_file = len(files_seen)
    return {"node_rank": hit_node, "file_rank": hit_file, "hit10_node": bool(hit_node and hit_node <= 10),
            "hit5_file": bool(hit_file and hit_file <= 5), "n_returned": len(ranked),
            "rr_node": (1 / hit_node) if hit_node else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--a1", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--encoder", default=None, help="sentence-transformers model id identified by A3 (mode c)")
    ap.add_argument("--encoder-prompt", default="task: code retrieval | query: ")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--task-ids", nargs="*", default=None)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    gold_by_task: dict[str, set[str]] = defaultdict(set)
    gold_files_by_task: dict[str, set[str]] = defaultdict(set)
    for r in read_jsonl(args.a1 / "a1_gold_symbols.jsonl"):
        if r.get("node_id") and r.get("present_in_graph") and not r.get("new_symbol"):
            gold_by_task[r["instance_id"]].add(r["node_id"])
        if r.get("module"):
            gold_files_by_task[r["instance_id"]].add(r["module"])

    encoder = None
    if args.encoder:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(args.encoder, device="cpu")

    rows = []
    bm_cache: dict[str, BM25] = {}
    for t in load_tasks(args.data, args.task_ids):
        g = get_graph(args.data, t)
        E = get_embeddings(args.data, t)
        gold_nodes = gold_by_task.get(t.instance_id, set())
        if g is None or E is None:
            rows.append({"instance_id": t.instance_id, "repo": t.repo, "skipped": "graph/embeddings missing"})
            continue
        mi = ModuleIndex(g)
        gold_mods = set(gold_files_by_task.get(t.instance_id, set()))
        for f in t.gold_files:
            if f.endswith(".py") and not is_test_path(f):
                m, _ = mi.module_for(f)
                if m:
                    gold_mods.add(m)
        mod_of = lambda n: module_of_node(n, g)  # noqa: E731
        ids = [n for n in g.nodes if n in E]
        idx = {n: i for i, n in enumerate(ids)}
        M = np.stack([E[n] for n in ids]).astype(np.float32)
        M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        base = {"instance_id": t.instance_id, "repo": t.repo, "n_gold_nodes": len(gold_nodes), "n_gold_modules": len(gold_mods),
                "gold_in_embeddings": sum(1 for x in gold_nodes if x in idx)}

        # (a) harness mode
        toks = flat_identifiers(extract_identifiers(t.problem_statement))
        anchors = []
        for tok in toks:
            nid, tier = resolve_node_name(tok, g.nodes)
            if nid and nid not in anchors:
                anchors.append(nid)
        base["n_identifiers"] = len(toks)
        base["n_anchors"] = len(anchors)
        best: dict[str, float] = {}
        for a in anchors:
            if a not in idx:
                continue
            sims = M @ M[idx[a]]
            order = np.argsort(-sims)
            cnt = 0
            for j in order:
                n = ids[j]
                if n == a:
                    continue
                best[n] = max(best.get(n, -2), float(sims[j]))
                cnt += 1
                if cnt >= args.k:
                    break
        harness_rank = anchors + [n for n, _ in sorted(best.items(), key=lambda kv: -kv[1]) if n not in anchors]
        for name, ranked in (("a0_anchors", anchors), ("a_harness", harness_rank)):
            for k, v in rank_metrics(ranked, gold_nodes, gold_mods, mod_of).items():
                base[f"{name}_{k}"] = v
        # (d) harness + 1 hop
        hop = list(harness_rank)
        for a in anchors + harness_rank[: args.k]:
            for n in sorted(g.neighbors(a)):
                if n not in hop:
                    hop.append(n)
        for k, v in rank_metrics(hop, gold_nodes, gold_mods, mod_of).items():
            base[f"d_harness_1hop_{k}"] = v
        # (b) BM25
        if g.key not in bm_cache:
            bm_cache[g.key] = BM25({n: f"{n}\n{a['text']}" for n, a in g.nodes.items()})
        bm_rank = bm_cache[g.key].rank(t.problem_statement)
        for k, v in rank_metrics(bm_rank, gold_nodes, gold_mods, mod_of).items():
            base[f"b_bm25_{k}"] = v
        # (c) free text embedding
        if encoder is not None:
            q = encoder.encode([args.encoder_prompt + (t.problem_statement or "")], convert_to_numpy=True)[0][: M.shape[1]]
            q = q / (np.linalg.norm(q) or 1)
            order = np.argsort(-(M @ q))
            ft_rank = [ids[j] for j in order]
            for k, v in rank_metrics(ft_rank, gold_nodes, gold_mods, mod_of).items():
                base[f"c_freetext_{k}"] = v
        rows.append(base)
        print(f"[a5] {t.instance_id}: anchors={len(anchors)} a_hit10={base['a_harness_hit10_node']} "
              f"bm25_hit10={base['b_bm25_hit10_node']} 1hop_hit10={base['d_harness_1hop_hit10_node']}", flush=True)

    write_csv(args.out / "a5_tasks.csv", rows)
    modes = ["a0_anchors", "a_harness", "d_harness_1hop", "b_bm25"] + (["c_freetext"] if encoder else [])
    valid = [r for r in rows if "skipped" not in r]
    with_gold = [r for r in valid if r["n_gold_nodes"] > 0]

    def summ(rs):
        return {m: {"hit10_node_pct": pct(sum(1 for r in rs if r.get(f"{m}_hit10_node")), len(rs)),
                    "hit5_file_pct": pct(sum(1 for r in rs if r.get(f"{m}_hit5_file")), len(rs)),
                    "mrr_node": round(sum(r.get(f"{m}_rr_node", 0) for r in rs) / len(rs), 4) if rs else None,
                    "empty_pct": pct(sum(1 for r in rs if r.get(f"{m}_n_returned") == 0), len(rs))} for m in modes}

    per_repo = defaultdict(list)
    for r in with_gold:
        per_repo[r["repo"]].append(r)
    summary = {"n_tasks": len(rows), "n_valid": len(valid), "n_with_gold_nodes": len(with_gold),
               "tasks_zero_anchors": sum(1 for r in valid if r["n_anchors"] == 0),
               "overall_tasks_with_gold_nodes": summ(with_gold),
               "per_repo": {k: summ(v) for k, v in per_repo.items()}}
    (args.out / "a5_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

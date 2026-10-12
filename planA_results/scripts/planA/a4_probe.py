#!/usr/bin/env python3
"""A4: what does vector similarity track?

For each graph (or a sample of graphs) compare the cosine similarity of node pairs that are
  * call-adjacent (a `calls` edge, containment pseudo-edges excluded)
  * in the same module (file)
  * name-overlapping (Jaccard of split identifier tokens >= 0.5, different modules)
  * gold co-change pairs (two gold nodes of the same task, from A1's a1_gold_symbols.jsonl)
against random pairs. Reports mean cosine per relation, AUROC of cosine for relation-vs-random
(Mann-Whitney), and the nearest-neighbour composition: among each node's top-10 cosine neighbours,
the share that are call neighbours / same module / name-overlap / test nodes.

Outputs: a4_relations.csv (per graph x relation), a4_knn.csv, a4_summary.json.
Shardable by graph with ``--shard i/N``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_census import module_of_node  # noqa: E402
from common import containment_edge, get_embeddings, get_graph, load_tasks, read_jsonl, shard, tokenize, write_csv  # noqa: E402


def auroc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if len(pos) == 0 or len(neg) == 0:
        return None
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def name_tokens(nid: str) -> set[str]:
    return set(tokenize(nid.rsplit(".", 1)[-1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--a1", type=Path, default=None, help="A1 output dir (gold co-change pairs)")
    ap.add_argument("--max-graphs", type=int, default=None)
    ap.add_argument("--random-pairs", type=int, default=20000)
    ap.add_argument("--knn-sample", type=int, default=500)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    tasks = load_tasks(args.data)
    by_key: dict[str, list] = defaultdict(list)
    for t in tasks:
        by_key[t.graph_key].append(t)
    keys = shard(sorted(by_key), args.shard)
    if args.max_graphs:
        keys = keys[: args.max_graphs]
    gold_by_task: dict[str, set[str]] = defaultdict(set)
    if args.a1 and (args.a1 / "a1_gold_symbols.jsonl").exists():
        for r in read_jsonl(args.a1 / "a1_gold_symbols.jsonl"):
            if r.get("present_in_graph") and r.get("node_id"):
                gold_by_task[r["instance_id"]].add(r["node_id"])

    rel_rows, knn_rows = [], []
    for key in keys:
        t0 = by_key[key][0]
        g = get_graph(args.data, t0)
        E = get_embeddings(args.data, t0)
        if g is None or E is None:
            continue
        ids = [n for n in g.nodes if n in E]
        if len(ids) < 10:
            continue
        idx = {n: i for i, n in enumerate(ids)}
        M = np.stack([E[n] for n in ids]).astype(np.float32)
        M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        mod = {n: module_of_node(n, g) for n in ids}

        def cos(pairs):
            if not pairs:
                return np.zeros(0)
            a = np.array([idx[x] for x, _ in pairs]); b = np.array([idx[y] for _, y in pairs])
            return (M[a] * M[b]).sum(axis=1)

        call_pairs = list({(s, d) for s, d, ty in g.edges() if ty == "calls" and s != d and not containment_edge(s, d)
                           and s in idx and d in idx})
        contain_pairs = list({(s, d) for s, d, ty in g.edges() if containment_edge(s, d) and s in idx and d in idx})
        same_mod_pairs, name_pairs = [], []
        by_mod: dict[str, list[str]] = defaultdict(list)
        for n in ids:
            by_mod[mod[n]].append(n)
        for m, lst in by_mod.items():
            if len(lst) < 2:
                continue
            for _ in range(min(len(lst) * 2, 400)):
                a, b = rng.sample(lst, 2)
                same_mod_pairs.append((a, b))
        toks = {n: name_tokens(n) for n in ids}
        for _ in range(min(len(ids) * 5, 100000)):
            a, b = rng.sample(ids, 2)
            if mod[a] != mod[b] and toks[a] and toks[b]:
                j = len(toks[a] & toks[b]) / len(toks[a] | toks[b])
                if j >= 0.5:
                    name_pairs.append((a, b))
        rand_pairs = [tuple(rng.sample(ids, 2)) for _ in range(args.random_pairs)]
        gold_pairs = []
        for t in by_key[key]:
            gn = sorted(x for x in gold_by_task.get(t.instance_id, ()) if x in idx)
            gold_pairs += [(gn[i], gn[j]) for i in range(len(gn)) for j in range(i + 1, len(gn))]
        neg = cos(rand_pairs)
        for rel, pairs in (("calls", call_pairs), ("containment", contain_pairs), ("same_module", same_mod_pairs),
                           ("name_overlap", name_pairs), ("gold_cochange", gold_pairs), ("random", rand_pairs)):
            c = cos(pairs)
            rel_rows.append({"graph_key": key, "repo": t0.repo, "relation": rel, "n_pairs": len(pairs),
                             "cos_mean": float(c.mean()) if len(c) else None, "cos_median": float(np.median(c)) if len(c) else None,
                             "auroc_vs_random": auroc(c, neg) if rel != "random" else None})
        # kNN composition
        samp = rng.sample(ids, min(args.knn_sample, len(ids)))
        S = M[[idx[n] for n in samp]] @ M.T
        comp = defaultdict(float)
        for i, n in enumerate(samp):
            order = np.argsort(-S[i])
            top = [ids[j] for j in order if ids[j] != n][:10]
            nb = g.neighbors(n)
            comp["call_neighbor"] += sum(1 for x in top if x in nb) / 10
            comp["same_module"] += sum(1 for x in top if mod[x] == mod[n]) / 10
            comp["name_overlap"] += sum(1 for x in top if toks[x] and toks[n] and len(toks[x] & toks[n]) / len(toks[x] | toks[n]) >= 0.5) / 10
            comp["test_node"] += sum(1 for x in top if g.is_test_node(x)) / 10
            comp["top1_cos"] += float(S[i][order[1] if ids[order[0]] == n else order[0]])
        knn_rows.append({"graph_key": key, "repo": t0.repo, "n_sampled": len(samp),
                         **{f"top10_share_{k}": round(v / len(samp), 4) for k, v in comp.items() if k != "top1_cos"},
                         "top1_cos_mean": round(comp["top1_cos"] / len(samp), 4)})
        print(f"[a4] {key}: " + ", ".join(f"{r['relation']}={r['cos_mean']:.3f}" for r in rel_rows[-6:] if r["cos_mean"] is not None), flush=True)

    suffix = f"_shard{args.shard.replace('/', 'of')}" if args.shard else ""
    write_csv(args.out / f"a4_relations{suffix}.csv", rel_rows)
    write_csv(args.out / f"a4_knn{suffix}.csv", knn_rows)
    summ = defaultdict(list)
    for r in rel_rows:
        if r["cos_mean"] is not None:
            summ[r["relation"]].append((r["cos_mean"], r["auroc_vs_random"]))
    summary = {rel: {"n_graphs": len(v), "cos_mean_avg": round(sum(x for x, _ in v) / len(v), 4),
                     "auroc_avg": (round(sum(a for _, a in v if a is not None) / max(1, sum(1 for _, a in v if a is not None)), 4)
                                   if any(a is not None for _, a in v) else None)}
               for rel, v in summ.items()}
    (args.out / f"a4_summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A3: which encoder produced the 256-d node embeddings?

For a sample of nodes per graph, embed each node's ``text`` (and, as controls, its ``name`` / id)
with candidate public encoders and compare with the provided vector:
  * cosine(provided, candidate) per node; mean and share >= 0.99  (identified if mean >= 0.99)
  * rank agreement: Spearman correlation between the provided-vector similarity ranking of the
    sample and the candidate's ranking (robust to normalisation / prefix differences)
Variants per model: raw text, documented document prompt, name-only, name+text; truncated to the
provided dimension (Matryoshka: first 256 dims, re-normalised).

Candidates (``--models``), all via sentence-transformers:
  google/embeddinggemma-300m   (v1, gated: needs HF token with accepted licence)
  google/embeddinggemma-2      (v2 name per HANDOFF; VERIFY the exact hub id)
  sentence-transformers/all-MiniLM-L6-v2 (negative control, 384-d -> truncated)
Also checks trivially whether provided vectors are unit-norm and their dimension, and the
"one dominant direction" claim (mean cosine to the mean vector, top PC explained variance).

Runs on CPU (slow but fine for a few hundred nodes per graph); ``--sample`` controls size.
Needs network access to huggingface.co (blocked in some sandboxes) and ``pip install
sentence-transformers``. If no candidate reaches 0.99, the paper reports the nearest candidate
and A5 uses BM25 as the free-text stand-in.

Outputs: a3_per_model.csv, a3_vectors_stats.csv, a3_summary.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_embeddings, get_graph, load_tasks, write_csv  # noqa: E402

DOC_PROMPTS = {
    # documented EmbeddingGemma prompts (model card); VERIFY against the card for v2
    "none": "",
    "document": "title: none | text: ",
    "code_retrieval_query": "task: code retrieval | query: ",
    "search_result": "task: search result | query: ",
}


def vec_stats(E: np.ndarray) -> dict:
    norms = np.linalg.norm(E, axis=1)
    mean = E.mean(axis=0)
    mu = mean / (np.linalg.norm(mean) or 1)
    En = E / np.maximum(norms[:, None], 1e-9)
    cos_mean = En @ mu
    # top principal component share
    C = En - En.mean(axis=0)
    try:
        s = np.linalg.svd(C, compute_uv=False)
        pc1 = float(s[0] ** 2 / (s ** 2).sum())
    except Exception:
        pc1 = None
    pair = En[: min(len(En), 500)]
    sims = pair @ pair.T
    iu = np.triu_indices(len(pair), 1)
    return {"n": len(E), "dim": E.shape[1], "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
            "unit_norm": bool(abs(norms.mean() - 1) < 1e-3 and norms.std() < 1e-3),
            "cos_to_mean_vector_mean": float(cos_mean.mean()), "pc1_explained": pc1,
            "pairwise_cos_mean": float(sims[iu].mean()) if len(iu[0]) else None,
            "pairwise_cos_p05": float(np.percentile(sims[iu], 5)) if len(iu[0]) else None}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if r == r else 0.0


def encode(model, texts: list[str], prompt: str, dim: int, batch: int) -> np.ndarray:
    X = model.encode([prompt + t for t in texts], batch_size=batch, convert_to_numpy=True, normalize_embeddings=False,
                     show_progress_bar=False)
    X = np.asarray(X, dtype=np.float32)[:, :dim]
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return X


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", nargs="*", default=["google/embeddinggemma-300m", "google/embeddinggemma-2",
                                                    "sentence-transformers/all-MiniLM-L6-v2"])
    ap.add_argument("--graphs", type=int, default=3, help="number of distinct graphs to sample (one per repo first)")
    ap.add_argument("--sample", type=int, default=200, help="nodes per graph")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-only", action="store_true", help="only vector statistics, no encoders")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    tasks = load_tasks(args.data)
    # one graph per repo first, then more
    chosen, seen_repo, seen_key = [], set(), set()
    for t in tasks:
        if t.repo not in seen_repo and t.graph_key not in seen_key:
            chosen.append(t); seen_repo.add(t.repo); seen_key.add(t.graph_key)
    for t in tasks:
        if len(chosen) >= args.graphs:
            break
        if t.graph_key not in seen_key:
            chosen.append(t); seen_key.add(t.graph_key)
    chosen = chosen[: args.graphs]

    stats_rows, model_rows = [], []
    samples = []
    for t in chosen:
        g = get_graph(args.data, t)
        E = get_embeddings(args.data, t)
        if g is None or E is None:
            print(f"[a3] {t.graph_key}: graph or embeddings missing", flush=True)
            continue
        keys = [k for k in E if k in g.nodes]
        extra = [k for k in E if k not in g.nodes]
        M = np.stack([E[k] for k in keys]) if keys else np.zeros((0, 1))
        st = vec_stats(M) if len(keys) else {}
        st.update({"graph_key": g.key, "repo": t.repo, "n_graph_nodes": len(g.nodes), "n_vectors": len(E),
                   "n_vectors_not_nodes": len(extra), "n_nodes_without_vector": sum(1 for n in g.nodes if n not in E),
                   "example_extra_keys": extra[:5]})
        stats_rows.append(st)
        rng.shuffle(keys)
        samp = [k for k in keys if g.nodes[k]["text"].strip()][: args.sample]
        samples.append((t, g, samp, np.stack([E[k] for k in samp]) if samp else None))
        print(f"[a3] {g.key}: dim={st.get('dim')} unit_norm={st.get('unit_norm')} pc1={st.get('pc1_explained')} "
              f"cos_to_mean={st.get('cos_to_mean_vector_mean')}", flush=True)
    write_csv(args.out / "a3_vectors_stats.csv", stats_rows)
    summary = {"vector_stats": stats_rows, "models": {}}
    if args.stats_only or not samples:
        (args.out / "a3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("pip install sentence-transformers (and set HF_TOKEN for gated models)")

    for name in args.models:
        try:
            model = SentenceTransformer(name, device="cpu")
        except Exception as e:  # noqa: BLE001
            print(f"[a3] cannot load {name}: {e}", flush=True)
            summary["models"][name] = {"error": str(e)[:500]}
            continue
        best = None
        for t, g, samp, P in samples:
            if P is None:
                continue
            dim = P.shape[1]
            Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
            ref_sims = (Pn @ Pn.T)[np.triu_indices(len(samp), 1)]
            variants = {
                "text": [g.nodes[k]["text"][: args.max_chars] for k in samp],
                "name": [g.nodes[k]["name"] for k in samp],
                "id": list(samp),
                "name_text": [f"{g.nodes[k]['name']}\n{g.nodes[k]['text'][: args.max_chars]}" for k in samp],
            }
            for vname, texts in variants.items():
                for pname, prompt in DOC_PROMPTS.items():
                    if vname != "text" and pname not in ("none", "document"):
                        continue
                    try:
                        X = encode(model, texts, prompt, dim, args.batch)
                    except Exception as e:  # noqa: BLE001
                        model_rows.append({"model": name, "graph_key": g.key, "variant": vname, "prompt": pname, "error": str(e)[:300]})
                        continue
                    cos = (X * Pn).sum(axis=1)
                    cand_sims = (X @ X.T)[np.triu_indices(len(samp), 1)]
                    row = {"model": name, "graph_key": g.key, "variant": vname, "prompt": pname, "n": len(samp),
                           "cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)),
                           "share_ge_0.99": float((cos >= 0.99).mean()), "share_ge_0.95": float((cos >= 0.95).mean()),
                           "spearman_pairwise": spearman(ref_sims, cand_sims), "model_dim": int(X.shape[1])}
                    model_rows.append(row)
                    print(f"[a3] {name} {g.key} {vname}/{pname}: cos_mean={row['cos_mean']:.4f} "
                          f">=.99={row['share_ge_0.99']:.2f} spearman={row['spearman_pairwise']:.3f}", flush=True)
                    if best is None or row["cos_mean"] > best["cos_mean"]:
                        best = row
        summary["models"][name] = {"best": best, "identified": bool(best and best["cos_mean"] >= 0.99)}
        write_csv(args.out / "a3_per_model.csv", model_rows)
    ident = [m for m, v in summary["models"].items() if v.get("identified")]
    summary["identified_encoder"] = ident[0] if ident else None
    (args.out / "a3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "vector_stats"}, indent=2, default=str))


if __name__ == "__main__":
    main()

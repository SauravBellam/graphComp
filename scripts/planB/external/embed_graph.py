#!/usr/bin/env python3
"""Node embeddings for a code graph, in the competition's .npz layout (one float32 vector of length 256 per node id).

    python embed_graph.py GRAPH.json OUT.npz [--model Qwen/Qwen3-Embedding-0.6B] [--hops 2] [--cache DIR]

Why this recipe (Oct 15, see the experiment log):
- The competition vectors are mostly structural: call-linked nodes have cosine 0.985-0.99 vs 0.76-0.90 for random pairs, and
  their similarity structure does not match CodeT5+ 110M text embeddings (Spearman 0.26). `search_similar_code` resolves the
  query to a node and ranks the other nodes by cosine (HARNESS_README section 6), so only node-to-node geometry matters.
- Text embedding: Qwen3-Embedding-0.6B (Apache-2.0, Matryoshka: truncated to 256 dims), a strong small code retriever
  (jina-code-embeddings, arXiv 2508.21290, Table 2; jina-code-embeddings-0.5b scores higher but is CC-BY-NC, selectable).
- Structure: k rounds of mean aggregation over the symmetric call graph, X <- D^-1 (A + I) X, to give the call-graph
  neighbourhood similarity the deployed tool returns; k is chosen with validate_graph.py --hops (Spearman against the
  competition vectors on the same snapshots). Vectors are rescaled to the competition's typical norm (~4.4).
Embeddings are cached by text hash (--cache), since most nodes are unchanged between commits of a repository.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

DIM, TARGET_NORM = 256, 4.4


def load(name="Qwen/Qwen3-Embedding-0.6B"):
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    return {"name": name, "tok": tok, "model": model, "dev": dev}


def embed_texts(m, texts, batch=16, max_len=512):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            t = m["tok"](texts[i:i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(m["dev"])
            h = m["model"](**t).last_hidden_state[:, -1, :].float()       # last-token pooling (left padding)
            h = h[:, :DIM]
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, DIM), np.float32)


def embed(graph, m, hops=2, cache=None):
    ids = [n["id"] for n in graph["nodes"]]
    texts = [n["text"] or n["name"] for n in graph["nodes"]]
    keys = [hashlib.sha1(f"{m['name']}\n{t}".encode()).hexdigest() for t in texts]
    have = {}
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        for k in set(keys):
            p = Path(cache) / f"{k}.npy"
            if p.exists():
                have[k] = np.load(p)
    todo = sorted({k: t for k, t in zip(keys, texts) if k not in have}.items())
    if todo:
        V = embed_texts(m, [t for _, t in todo])
        for (k, _), v in zip(todo, V):
            have[k] = v
            if cache:
                np.save(Path(cache) / f"{k}.npy", v)
    X = np.stack([have[k] for k in keys]).astype(np.float64)
    idx = {i: j for j, i in enumerate(ids)}
    nbrs = [[] for _ in ids]
    for e in graph["edges"]:
        a, b = idx.get(e["source"]), idx.get(e["target"])
        if a is not None and b is not None and a != b:
            nbrs[a].append(b)
    for _ in range(hops):
        X = np.stack([(X[j] + X[nb].sum(0)) / (1 + len(nb)) if nb else X[j] for j, nb in enumerate(nbrs)])
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * TARGET_NORM
    return {i: X[j].astype(np.float32) for i, j in idx.items()}


def save_npz(vecs, out):
    np.savez(out, **vecs)


if __name__ == "__main__":
    a = sys.argv
    g = json.load(open(a[1]))
    model = a[a.index("--model") + 1] if "--model" in a else "Qwen/Qwen3-Embedding-0.6B"
    hops = int(a[a.index("--hops") + 1]) if "--hops" in a else 2
    cache = a[a.index("--cache") + 1] if "--cache" in a else None
    save_npz(embed(g, load(model), hops=hops, cache=cache), a[2])

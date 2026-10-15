#!/usr/bin/env python3
"""Compare code_graph.py (+ optional embed_graph.py) with the competition graphs on the same snapshots.

    python validate_graph.py COMP_DATA_DIR N_PER_REPO [--embed MODEL --dims 256 --hops 0,1,2,3]

Per snapshot: node recall/precision against the competition node ids, `calls` edge recall/precision, share of shared
nodes with identical text; with --embed, the Spearman correlation between our and the competition's pairwise node cosines
for each smoothing depth (hops), so the embedding recipe can be matched to what the deployed tools return.
"""
import json, sys, tarfile, tempfile, glob, random
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import code_graph


def spearman(a, b):
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    data, n = Path(sys.argv[1]), int(sys.argv[2])
    embed = sys.argv[sys.argv.index("--embed") + 1] if "--embed" in sys.argv else None
    hops = [int(x) for x in sys.argv[sys.argv.index("--hops") + 1].split(",")] if "--hops" in sys.argv else [0]
    tasks = [json.loads(l) for l in open(data / "tasks.jsonl")]
    rnd = random.Random(0)
    picks = []
    for repo in ("fastapi", "rich", "requests", "httpx"):
        c = [t for t in tasks if t["instance_id"].startswith(repo + "_")]
        picks += rnd.sample(c, min(n, len(c)))
    if embed:
        import embed_graph
        model = embed_graph.load(embed)
    for t in picks:
        short = t["instance_id"].split("_")[0]
        gp = data / "graphs" / f"{short}_{t['base_commit']}.json"
        if not gp.exists():
            print(t["instance_id"], "no competition graph"); continue
        comp = json.load(open(gp))
        with tempfile.TemporaryDirectory() as d:
            tarfile.open(data / "snapshots" / f"{t['instance_id']}.tgz").extractall(d)
            ours = code_graph.build(d)
        cn = {x["id"]: x["text"] for x in comp["nodes"]}; on = {x["id"]: x["text"] for x in ours["nodes"]}
        ce = {(e["source"], e["target"]) for e in comp["edges"]}; oe = {(e["source"], e["target"]) for e in ours["edges"]}
        both = set(cn) & set(on)
        line = (f"{t['instance_id']:16s} nodes comp {len(cn)} ours {len(on)} recall {len(both) / max(1, len(cn)):.3f} "
                f"precision {len(both) / max(1, len(on)):.3f} | edges comp {len(ce)} ours {len(oe)} recall {len(ce & oe) / max(1, len(ce)):.3f} "
                f"precision {len(ce & oe) / max(1, len(oe)):.3f} | same text {sum(cn[i] == on[i] for i in both) / max(1, len(both)):.3f}")
        if embed:
            z = np.load(str(gp).replace("/graphs/", "/embeddings/").replace(".json", ".npz"))
            ids = sorted(both)[:400]
            X = np.stack([z[i] for i in ids]).astype(np.float64)
            norm = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
            iu = np.triu_indices(len(ids), 1); ref = (norm(X) @ norm(X).T)[iu]
            for h in hops:
                E = embed_graph.embed(ours, model, hops=h)
                Y = np.stack([E[i] for i in ids]).astype(np.float64)
                line += f" | hops {h}: spearman {spearman((norm(Y) @ norm(Y).T)[iu], ref):.3f}"
        print(line, flush=True)
        missing = sorted(set(cn) - set(on))[:4]; extra = sorted(set(on) - set(cn))[:4]
        if missing or extra:
            print("   e.g. missing", missing, "| extra", extra)


if __name__ == "__main__":
    main()

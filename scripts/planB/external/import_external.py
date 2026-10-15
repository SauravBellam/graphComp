#!/usr/bin/env python3
"""Import external SWE tasks from the four competition repositories into the competition's data layout.

    python import_external.py --out DIR --comp-tasks COMP/tasks.jsonl [--hops K] [--limit N]

Sources (Hugging Face, Oct 15 scan of the `repo` column): princeton-nlp/SWE-bench test (psf/requests 44) and train
(tiangolo/fastapi 28); nebius/SWE-rebench-V2 (Textualize/rich 28, encode/httpx 5); nebius/SWE-rebench filtered + test
(encode/httpx 25 + 62); nebius/SWE-bench-extra (encode/httpx 16). Only the matching row groups are read (HTTP ranges).

Steps: normalize to the competition schema (instance_id = <repo_short>_<PR number>); DROP every task whose id equals a
competition task (same pull request: no holdout / dev leakage) and duplicates across sources; fetch each base commit into
a self-contained checkout and write snapshots/<id>.tgz (rooted at ./, with .git, like the competition snapshots); build
graphs/<repo_short>_<commit>.json (code_graph.py) and embeddings/<same>.npz (embed_graph.py, cached); write tasks.jsonl,
extra_fields.jsonl (FAIL_TO_PASS / PASS_TO_PASS kept aside) and graphcomp_external.json. Validity (gold passes, empty
fails) is run afterwards with run_arm.py and recorded in graphcomp_external.json by --record-validity.
"""
import argparse, io, json, re, subprocess, sys, tarfile, tempfile, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import code_graph

REPOS = {"psf/requests": "requests", "tiangolo/fastapi": "fastapi", "fastapi/fastapi": "fastapi", "Textualize/rich": "rich",
         "encode/httpx": "httpx"}
SOURCES = [("nebius/SWE-rebench-V2", "train"), ("nebius/SWE-rebench", "filtered"), ("nebius/SWE-rebench", "test"),
           ("nebius/SWE-bench-extra", "train"), ("princeton-nlp/SWE-bench", "test"), ("princeton-nlp/SWE-bench", "train")]
FIELDS = ["instance_id", "repo", "base_commit", "problem_statement", "hints_text", "patch", "test_patch", "created_at"]
EXTRA = ["FAIL_TO_PASS", "PASS_TO_PASS"]


class HTTPFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, o, w=0):
        self.pos = o if w == 0 else self.pos + o if w == 1 else self.size + o
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.size, self.pos + n) - 1
        b = urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"}), timeout=120).read()
        self.pos += len(b)
        return b
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def rows_for_repos(ds, split):
    import pyarrow.parquet as pq
    urls = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{ds}/parquet/default/{split}", timeout=60))
    out = []
    for u in urls:
        pf = pq.ParquetFile(HTTPFile(u))
        cols = [c for c in FIELDS + EXTRA if c in pf.schema_arrow.names]
        for g in range(pf.num_row_groups):
            repos = pf.read_row_group(g, columns=["repo"]).column("repo").to_pylist()
            if any(r in REPOS for r in repos):
                t = pf.read_row_group(g, columns=cols).to_pylist()
                out += [r for r in t if r["repo"] in REPOS]
    return out


def normalize(r, ds):
    short = REPOS[r["repo"]]
    pr = re.search(r"(\d+)$", r["instance_id"]).group(1)
    row = {k: r.get(k) or "" for k in FIELDS}
    row["instance_id"], row["source_instance_id"], row["source"] = f"{short}_{pr}", r["instance_id"], ds
    extra = {k: (json.loads(r[k]) if isinstance(r.get(k), str) and r[k].startswith("[") else r.get(k) or []) for k in EXTRA}
    return row, extra


def checkout(repo, commit, cache, dest):
    bare = cache / repo.replace("/", "__")
    if not bare.exists():
        subprocess.run(["git", "clone", "--bare", "--quiet", f"https://github.com/{repo}.git", str(bare)], check=True)
    subprocess.run(["git", "init", "--quiet", str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "--depth", "1", f"file://{bare}", commit], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", "FETCH_HEAD"], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comp-tasks", required=True)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--record-validity", nargs=2, metavar=("GOLD_DIR", "EMPTY_DIR"))
    a = ap.parse_args()
    out = Path(a.out)
    if a.record_validity:
        rd = lambda p: {json.loads(l)["instance_id"]: json.loads(l) for l in open(Path(p) / "task_results.jsonl") if l.strip()} if (Path(p) / "task_results.jsonl").exists() else {}
        g, e = rd(a.record_validity[0]), rd(a.record_validity[1])
        meta = json.loads((out / "graphcomp_external.json").read_text())
        meta["valid"] = sorted(t for t, r in g.items() if r.get("resolved") and t in e and not e[t].get("resolved"))
        meta["gold_resolved"], meta["checked"] = sum(bool(r.get("resolved")) for r in g.values()), len(g)
        (out / "graphcomp_external.json").write_text(json.dumps(meta, indent=1))
        print(f"valid: {len(meta['valid'])} of {len(g)} checked (gold resolves {meta['gold_resolved']})")
        return
    comp = {json.loads(l)["instance_id"] for l in open(a.comp_tasks) if l.strip()}
    for d in ("snapshots", "graphs", "embeddings"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows, extras, dropped = {}, {}, {"competition_pr": [], "duplicate": []}
    for ds, split in SOURCES:
        got = rows_for_repos(ds, split)
        print(f"{ds}/{split}: {len(got)} rows for our repos", flush=True)
        for r in got:
            row, extra = normalize(r, f"{ds}/{split}")
            if row["instance_id"] in comp:
                dropped["competition_pr"].append(row["instance_id"]); continue
            if row["instance_id"] in rows:
                dropped["duplicate"].append(row["instance_id"]); continue
            rows[row["instance_id"]], extras[row["instance_id"]] = row, extra
    ids = sorted(rows)[: a.limit or None]
    print(f"kept {len(ids)} tasks; dropped same-PR-as-competition {len(dropped['competition_pr'])} {sorted(set(dropped['competition_pr']))}, "
          f"duplicates {len(dropped['duplicate'])}", flush=True)
    import embed_graph
    model = embed_graph.load(a.model)
    cache, done = out.parent / "git_cache", []
    cache.mkdir(exist_ok=True)
    for i, iid in enumerate(ids):
        r = rows[iid]
        short = iid.split("_")[0]
        gname = f"{short}_{r['base_commit']}"
        try:
            with tempfile.TemporaryDirectory() as d:
                checkout(r["repo"], r["base_commit"], cache, Path(d) / "repo")
                with tarfile.open(out / "snapshots" / f"{iid}.tgz", "w:gz") as tf:
                    tf.add(Path(d) / "repo", arcname=".")
                if not (out / "graphs" / f"{gname}.json").exists():
                    g = code_graph.build(Path(d) / "repo")
                    json.dump(g, open(out / "graphs" / f"{gname}.json", "w"))
                    embed_graph.save_npz(embed_graph.embed(g, model, hops=a.hops, cache=out.parent / "emb_cache"), out / "embeddings" / f"{gname}.npz")
            done.append(iid)
            print(f"[{i + 1}/{len(ids)}] {iid} ({r['source']}) ok", flush=True)
        except Exception as e:
            print(f"[{i + 1}/{len(ids)}] {iid} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    with open(out / "tasks.jsonl", "w") as f, open(out / "extra_fields.jsonl", "w") as g_:
        for iid in done:
            f.write(json.dumps({k: rows[iid][k] for k in FIELDS}) + "\n")
            g_.write(json.dumps({"instance_id": iid, "source": rows[iid]["source"], "source_instance_id": rows[iid]["source_instance_id"], **extras[iid]}) + "\n")
    (out / "graphcomp_external.json").write_text(json.dumps({
        "built": done, "valid": [], "dropped": dropped, "hops": a.hops, "embedding_model": a.model,
        "sources": [f"{d}/{s}" for d, s in SOURCES], "note": "valid is filled by --record-validity (gold passes, empty fails)"}, indent=1))
    print(f"built {len(done)} tasks into {out}")


if __name__ == "__main__":
    main()

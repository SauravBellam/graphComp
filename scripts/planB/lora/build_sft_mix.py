#!/usr/bin/env python3
"""Merge trajectory sets into one SFT mix, without duplicate trajectories (Oct 15).

    python build_sft_mix.py --out DIR --external CONVERTED_DIR [...] [--rft train.jsonl ...] [--extra traj.jsonl ...]
                            [--external-tasks IMPORTED/tasks.jsonl] [--near-dup 0.9] [--fit-tokens 32768]
                            [--drop-labels network_or_install_cmds]

Inputs, all in build_rft_data.py's row format ({instance_id, repo, source, messages, ...}):
  --external  convert_trajs.py output folders (train/<source>/*.jsonl.gz; Qwen3-Coder-480B trajectories replayed into the harness)
  --rft       our own verified RFT trajectories (build_rft_data.py train.jsonl)
  --extra     any other trajectory jsonl in the same format (e.g. later teacher runs)

Duplicates (the same problem solved again by a different route is kept):
  task key   instance_id without convert_trajs' "@base" suffix, so the CoderForge and nebius copies of one GitHub issue meet
  exact      same task and the same sequence of tool calls (name + arguments): one copy kept
  near       same task and route similarity >= --near-dup (build_rft_data.route: (tool, target) sequence, difflib ratio):
             the trajectory with fewer tool calls is kept
Official dev / holdout tasks are kept (user, Oct 15) unless --exclude-dev / --exclude-holdout (scripts/planB/official_slices;
refused at the end if any remain). Excluded: and rows carrying any
--drop-labels label (default: package installs / downloads, which break the offline deployment rule).

Imported external tasks (--external-tasks, import_external.py's competition-layout tasks.jsonl): each is matched to the
trajectory tasks by (repo, pull request number) and listed as covered or not covered; the not-covered list is the queue for
new teacher trajectories.

Output (DIR): train_all.jsonl.gz (every kept row + est_tokens), train_fit.jsonl (rows with est_tokens <= --fit-tokens: the
whole trajectory fits the 31B training window, so its late turns are trained too), external_tasks_covered.jsonl,
external_tasks_uncovered.jsonl, manifest.json. est_tokens = characters / 3.6 (an estimate; train_lora.py renders with the
real tokenizer and drops what does not fit).
"""
import argparse, glob, gzip, hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import difflib
from build_rft_data import route        # one definition of a trajectory's route for every builder

CHARS_PER_TOKEN = 3.6


def rows_of(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def task_key(r):
    return r["instance_id"].split("@")[0]


def actions_hash(msgs):
    calls = [json.dumps(tc["function"], sort_keys=True) for m in msgs for tc in (m.get("tool_calls") or [])]
    return hashlib.sha1("\n".join(calls).encode()).hexdigest()


def n_calls(msgs):
    return sum(1 for m in msgs if m.get("tool_calls"))


def pr_key(repo, iid):
    """(owner/repo lowercased, PR number) for a SWE-rebench / SWE-bench style id ('owner__repo-123', 'httpx_123')."""
    m = re.search(r"(\d+)$", iid.split("@")[0])
    return (repo.lower(), m.group(1)) if m and repo else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--external", nargs="*", default=[])
    ap.add_argument("--rft", nargs="*", default=[])
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--external-tasks")
    ap.add_argument("--slices", default=str(HERE.parent / "official_slices"))
    ap.add_argument("--exclude-dev", action="store_true")
    ap.add_argument("--exclude-holdout", action="store_true")
    ap.add_argument("--near-dup", type=float, default=0.9)
    ap.add_argument("--fit-tokens", type=int, default=32768, help="the deployed server's max_model_len: longer cannot happen at deploy")
    ap.add_argument("--drop-labels", default="network_or_install_cmds")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # dev and holdout tasks may be trained on (user, Oct 15); --exclude-holdout / --exclude-dev keep them out
    banned = {l.strip() for f in (("holdout_slice.txt",) if a.exclude_holdout else ()) + (("dev_slice.txt",) if a.exclude_dev else ())
              for l in open(Path(a.slices) / f) if l.strip()}
    drop_labels = [x for x in a.drop_labels.split(",") if x]

    files = [(p, "external") for d in a.external for p in sorted(glob.glob(f"{d}/train/*/*.jsonl.gz"))] + \
            [(p, "rft") for p in a.rft] + [(p, "extra") for p in a.extra]
    stats, by_src = Counter(), Counter()
    by_task = defaultdict(list)          # task -> [(n_calls, row_id, route, est_tokens)]; rows themselves stay on disk
    seen_exact = set()
    kept_ids = []                         # (file index, line index) of rows that survive the exact pass
    for fi, (p, kind) in enumerate(files):
        for li, r in enumerate(rows_of(p)):
            stats["read"] += 1
            k = task_key(r)
            if k in banned:
                stats["dropped_holdout_task"] += 1; continue
            lab = r.get("labels") or {}
            if any(lab.get(x) for x in drop_labels):
                stats["dropped_label:" + "+".join(x for x in drop_labels if lab.get(x))] += 1; continue
            h = (k, actions_hash(r["messages"]))
            if h in seen_exact:
                stats["dropped_exact_duplicate"] += 1; continue
            seen_exact.add(h)
            est = int(sum(len(json.dumps(m)) for m in r["messages"]) / CHARS_PER_TOKEN)
            by_task[k].append((n_calls(r["messages"]), (fi, li), route(r["messages"]), est))   # the route only: RAM
            kept_ids.append((fi, li))
        print(f"read {p}: {stats['read']} rows so far", flush=True)

    keep = {}                             # (fi, li) -> est_tokens
    for k, cands in by_task.items():
        chosen = []
        for c in sorted(cands, key=lambda c: c[0]):           # fewest tool calls first
            if any(difflib.SequenceMatcher(None, c[2], o[2]).ratio() >= a.near_dup for o in chosen):   # = route_similarity
                stats["dropped_near_duplicate"] += 1; continue
            chosen.append(c)
        for c in chosen:
            keep[c[1]] = c[3]
    del by_task

    n_all = n_fit = 0
    tasks_all, tasks_fit, fit_src = set(), set(), Counter()
    with gzip.open(out / "train_all.jsonl.gz", "wt") as fa, open(out / "train_fit.jsonl", "w") as ff:
        for fi, (p, kind) in enumerate(files):
            for li, r in enumerate(rows_of(p)):
                if (fi, li) not in keep:
                    continue
                r["est_tokens"] = keep[(fi, li)]
                r["mix_source"] = kind
                if task_key(r) in banned:
                    raise SystemExit(f"STOP: holdout task {r['instance_id']} reached the output")
                line = json.dumps(r) + "\n"
                fa.write(line); n_all += 1; tasks_all.add(task_key(r)); by_src[r.get("source", kind)] += 1
                if r["est_tokens"] <= a.fit_tokens:
                    ff.write(line); n_fit += 1; tasks_fit.add(task_key(r)); fit_src[r.get("source", kind)] += 1

    ext = {}
    if a.external_tasks:
        traj_prs = set()
        for fi, (p, kind) in enumerate(files):
            for r in rows_of(p):
                k = pr_key(r.get("repo", ""), r["instance_id"])
                if k:
                    traj_prs.add(k)
        cov = unc = 0
        with open(out / "external_tasks_covered.jsonl", "w") as fc, open(out / "external_tasks_uncovered.jsonl", "w") as fu:
            for t in rows_of(a.external_tasks):
                k = pr_key(t.get("repo", ""), t["instance_id"])
                (fc if k in traj_prs else fu).write(json.dumps({"instance_id": t["instance_id"], "repo": t.get("repo")}) + "\n")
                cov += k in traj_prs; unc += k not in traj_prs
        ext = {"imported_tasks": cov + unc, "covered_by_a_trajectory": cov, "uncovered": unc}

    m = {"inputs": [p for p, _ in files], "drop_labels": drop_labels, "near_dup_threshold": a.near_dup,
         "fit_tokens": a.fit_tokens, "counts": dict(stats), "kept_all": n_all, "tasks_all": len(tasks_all),
         "kept_fit": n_fit, "tasks_fit": len(tasks_fit), "kept_all_by_source": dict(by_src), "kept_fit_by_source": dict(fit_src),
         "external_tasks": ext}
    (out / "manifest.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()

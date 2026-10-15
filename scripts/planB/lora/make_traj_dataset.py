#!/usr/bin/env python3
"""Build the trajectories dataset from notebook 01 + notebook 19 outputs (every attempt kept; training views derived).

    python make_traj_dataset.py --nb19 NB19_OUTPUT_DIR --nb01-run MERGED_B2_S1_DIR --tasks tasks.jsonl --out OUT_DIR

OUT_DIR/
  attempts.jsonl             one row per attempt: source, seed, instance_id, split (pool / dev / holdout / extra_holdout /
                             coverage), valid, resolved, seconds, tool_calls, graph_calls, error, trace path
  traces/<source>/           every trace (resolved or not), as recorded by the harness
  train/verified_all.jsonl   every resolved trajectory on a valid pool task (never dev / holdout / extra holdout), fastest first
  train/verified_accurate.jsonl   the same, keeping only accurate tool use (tool_trace: every source edit on gold code and a
                             reproduction or test after the last edit)
  train/nb19_train.jsonl     exactly what notebook 19 trained on (its rft/train.jsonl), for reproducibility
  tool_trace/                tool_trace.py report over all attempts on valid tasks
  manifest.json, README.md
"""
import argparse, csv, json, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRA_HOLDOUT = ["fastapi_15661", "rich_3471", "rich_3718"]      # notebook 19 step 2: never trained on
COVERAGE = ["requests_6589", "httpx_3672"]                      # notebook 19 holdout budget checks


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nb19", required=True)
    ap.add_argument("--nb01-run", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--slices", default=str(HERE.parent / "official_slices"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nb19, out = Path(a.nb19), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = {l.strip() for l in open(Path(a.slices) / "dev_slice.txt") if l.strip()}
    hold = {l.strip() for l in open(Path(a.slices) / "holdout_slice.txt") if l.strip()}
    prof_path = nb19 / "validity/slices/profile.csv"
    prof = {r["instance_id"]: r for r in csv.DictReader(open(prof_path))}
    split = lambda t: ("dev" if t in dev else "holdout" if t in hold else "extra_holdout" if t in EXTRA_HOLDOUT
                       else "coverage" if t in COVERAGE else "pool")
    sources = [("nb01_B2_s1", Path(a.nb01_run), 1)] + [(f"nb19_s{s}", nb19 / f"runs/NB19/merged_s{s}", s) for s in (11, 12, 13)]
    sources += [("nb19_holdout_BASE31", nb19 / "runs/NB19/holdout_BASE31", 1)]
    rows = []
    for name, d, seed in sources:
        if not (d / "task_results.jsonl").exists():
            print(f"missing {d}", file=sys.stderr); continue
        (out / "traces" / name).mkdir(parents=True, exist_ok=True)
        for r in read_jsonl(d / "task_results.jsonl"):
            iid = r["instance_id"]
            tp = d / "traces" / f"trace_{iid}.json"
            names = []
            if tp.exists():
                shutil.copy(tp, out / "traces" / name / tp.name)
                names = [tc.get("function_name") for s in json.load(open(tp)).get("steps", []) for tc in s.get("tool_calls") or []]
            rows.append({"source": name, "seed": seed, "instance_id": iid, "split": split(iid),
                         "valid": prof.get(iid, {}).get("sub_valid") == "True", "resolved": bool(r.get("resolved")),
                         "seconds": r.get("duration_seconds"), "tool_calls": len(names),
                         "graph_calls": sum(n in ("get_code_neighbors", "search_similar_code", "get_code_subgraph") for n in names),
                         "error": (r.get("error") or "")[:200], "trace": f"traces/{name}/{tp.name}" if tp.exists() else None})
    with open(out / "attempts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # training views (build_rft_data.py next to this script; official slices + extra holdout never trained on)
    (out / "train").mkdir(exist_ok=True)
    (out / "never_train.txt").write_text("\n".join(EXTRA_HOLDOUT) + "\n")
    runs = [str(d) for _, d, _ in sources[:4]]
    base = [sys.executable, str(HERE / "build_rft_data.py"), "--runs", *runs, "--profile", str(prof_path), "--max-per-task", "1000",
            "--order", "time", "--exclude", str(out / "never_train.txt")]
    for view, extra in (("verified_all", []), ("verified_accurate", ["--accurate-tools", a.tasks])):
        tmp = out / f"_build_{view}"
        subprocess.run(base + ["--out", str(tmp)] + extra, check=True, stdout=subprocess.DEVNULL)
        shutil.move(str(tmp / "train.jsonl"), out / "train" / f"{view}.jsonl")
        shutil.move(str(tmp / "manifest.json"), out / "train" / f"{view}_manifest.json")
        shutil.move(str(tmp / "excluded.csv"), out / "train" / f"{view}_excluded.csv")
        shutil.rmtree(tmp)
    shutil.copy(nb19 / "rft/train.jsonl", out / "train/nb19_train.jsonl")
    shutil.copy(nb19 / "rft/manifest.json", out / "train/nb19_train_manifest.json")
    # tool tracing over every attempt on valid tasks
    sys.path.insert(0, str(HERE.parent))
    import tool_trace
    gold = {t["instance_id"]: {"patch": t["patch"], "problem": t.get("problem_statement", "") + "\n" + (t.get("hints_text") or "")}
            for t in map(json.loads, open(a.tasks)) if t.get("patch")}
    valid = {t for t, r in prof.items() if r.get("sub_valid") == "True"}
    calls, runs_ = tool_trace.analyze([d for _, d, _ in sources if (d / "task_results.jsonl").exists()], gold, valid)
    tool_trace.write(out / "tool_trace", calls, runs_)
    count = lambda p: sum(1 for l in open(p) if l.strip())
    man = {"attempts": len(rows), "by_source": {s: sum(r["source"] == s for r in rows) for s, _, _ in sources},
           "resolved_valid_pool": sum(r["resolved"] and r["valid"] and r["split"] == "pool" for r in rows),
           "tasks_ever_solved_pool": len({r["instance_id"] for r in rows if r["resolved"] and r["valid"] and r["split"] == "pool"}),
           "train_views": {v: count(out / "train" / f"{v}.jsonl") for v in ("verified_all", "verified_accurate", "nb19_train")},
           "holdout_BASE31": json.load(open(nb19 / "NB19_STATE.json")).get("holdout", {}).get("BASE31", {}).get("resolved"),
           "never_trained": {"dev": len(dev), "holdout": len(hold), "extra_holdout": EXTRA_HOLDOUT}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()

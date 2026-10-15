#!/usr/bin/env python3
"""Search for the best forced graph-call rate: 0% (B2) and 100% (FC) first, then midpoints toward the best rate.

    ~/gc/venv/bin/python scripts/planB/vm/fc_search.py --out ~/gc/runs_fcsearch_12b -- --model 12b --no-lean \\
        --auto-concurrency --dev-slice SLICE.txt

Each probe is one dev_runs.py call with one arm (B2, FC, or FCnnn = forced on nnn% of tool calls; see fc_rate_arm in
dev_runs.py), so every probe starts its own vLLM server: same weights, empty cache, nothing carried over between probes.
Everything after `--` is passed to dev_runs.py; --out is shared, so validity and serving settings are measured once and
every probe is paired with B2 on the same tasks.

Next rate: the midpoint of the widest gap next to the current best (ties -> the lower rate), until that gap is at most
--min-gap points or --max-probes probes have run. Resumable: a rate with merged results is read, not re-run.
After every probe, in --out (everything a write-up needs):
  fc_search.md        table: rate, arm, measured graph-call share, resolved, McNemar vs B2, wall time, why the rate was probed
  fc_search.json      the same per probe, plus solved task lists, timestamps, tool-call totals, serving settings
  fc_search_tasks.csv one row per (probe, task): valid, resolved, tool calls, graph calls, LLM calls, seconds, error
  fc_search_prompts/  the exact system prompt each probe ran with
  fc_search_by_category.csv / the "By problem type" section of fc_search.md: per rate, resolved within each task group
                      (repo, graph-distance x visibility cell, multi-file, async in gold, gold is entry point), from the
                      validity profile, to see which kinds of problem forcing helps or hurts
  fc_search_tools.md  per probe: each tool's use vs outcome (resolved when used / not used, calls per resolved and
                      unresolved task, share of calls), from scripts/planB/tool_usage.py; per-tool call counts are also
                      columns of fc_search_tasks.csv
"""
import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tool_usage  # noqa: E402
GRAPH = {"get_code_neighbors", "search_similar_code", "get_code_subgraph"}
CATEGORIES = ("repo", "cell", "multi_file", "gold_has_async", "gold_is_entry_point")   # validity-profile columns
PROFILE = {}        # instance_id -> profile row (filled in main)


def arm(r):
    return "B2" if r == 0 else "FC" if r == 100 else f"FC{r:03d}"


def mcnemar(b, c):
    """Exact two-sided McNemar p on b, c discordant pairs."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def read(out, r, valid):
    name = arm(r)
    m = out / "runs" / f"{name}_s1" / f"merged_{name}_s1"
    if not (m / "task_results.jsonl").exists():
        return None
    rows = [json.loads(l) for l in open(m / "task_results.jsonl") if l.strip()]
    solved = sorted(x["instance_id"] for x in rows if x.get("resolved") and x["instance_id"] in valid)
    tasks = []
    for x in rows:
        t = m / "traces" / f"trace_{x['instance_id']}.json"
        names = [tc.get("function_name") for s in (json.load(open(t)).get("steps", []) if t.exists() else [])
                 for tc in s.get("tool_calls") or []]
        tasks.append({"rate": r, "arm": name, "instance_id": x["instance_id"], "valid": x["instance_id"] in valid,
                      **{c: PROFILE.get(x["instance_id"], {}).get(c, "") for c in CATEGORIES},
                      **{f"calls_{k}": names.count(k) for k in tool_usage.TOOLS},
                      "resolved": bool(x.get("resolved")), "tool_calls": len(names), "graph_calls": sum(n in GRAPH for n in names),
                      "llm_calls": x.get("total_llm_calls"), "seconds": x.get("duration_seconds"),
                      "error": (x.get("error") or "")[:120]})
    calls, graph = sum(t["tool_calls"] for t in tasks), sum(t["graph_calls"] for t in tasks)
    return {"rate": r, "arm": name, "resolved": len(solved), "scored": len([x for x in rows if x["instance_id"] in valid]),
            "graph_share": graph / calls if calls else 0.0, "tool_calls": calls, "graph_calls": graph, "solved": solved,
            "tasks": tasks}


def next_rate(res, min_gap):
    pts = sorted(res)
    best = max(pts, key=lambda r: (res[r]["resolved"], -r))
    i = pts.index(best)
    gaps = [(best - pts[i - 1], pts[i - 1], best)] if i > 0 else []
    gaps += [(pts[i + 1] - best, best, pts[i + 1])] if i + 1 < len(pts) else []
    gap, lo, hi = max(gaps, default=(0, 0, 0))
    return None if gap <= min_gap else round((lo + hi) / 2)


def report(out, res):
    base = res.get(0)
    lines = ["| probe | forced rate | arm | measured graph-call share | resolved | vs B2: only this / only B2 | McNemar p | wall | why probed |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res):
        x = res[r]
        if base and r != 0:
            b, c = len(set(x["solved"]) - set(base["solved"])), len(set(base["solved"]) - set(x["solved"]))
            cmp_ = f"{b} / {c} | {mcnemar(b, c):.3f}"
        else:
            cmp_ = "(baseline) | "
        lines.append(f"| {x.get('probe', '')} | {r}% | {x['arm']} | {x['graph_share']:.1%} | {x['resolved']}/{x['scored']} | {cmp_} "
                     f"| {x.get('wall_seconds', 0) / 3600:.1f} h | {x.get('why', '')} |")
    best = max(res, key=lambda r: (res[r]["resolved"], -r))
    lines += ["", f"Best so far: {best}% ({res[best]['resolved']}/{res[best]['scored']})."]
    # by problem type: valid tasks only; one row per (category, value), one column per rate
    rates, cat_rows = sorted(res), []
    for c in CATEGORIES:
        for v in sorted({t[c] for r in rates for t in res[r]["tasks"] if t["valid"]}):
            row = {"category": c, "value": v}
            for r in rates:
                g = [t for t in res[r]["tasks"] if t["valid"] and t[c] == v]
                row[f"{r}%"] = f"{sum(t['resolved'] for t in g)}/{len(g)}"
            cat_rows.append(row)
    if cat_rows:
        lines += ["", "## By problem type (resolved / valid tasks in the group)", "",
                  "| category | value | " + " | ".join(f"{r}%" for r in rates) + " |",
                  "|---|---|" + "---|" * len(rates)]
        lines += [f"| {x['category']} | {str(x['value']).replace('|', chr(92) + '|')} | " + " | ".join(x[f"{r}%"] for r in rates) + " |"
                  for x in cat_rows]
        with open(out / "fc_search_by_category.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cat_rows[0]))
            w.writeheader()
            w.writerows(cat_rows)
    (out / "fc_search.md").write_text("\n".join(lines) + "\n")
    tools_md = ["# Tool use vs outcome per probe (valid tasks; association, not causation)", ""]
    valid = {i for i, x in PROFILE.items() if x.get("sub_valid") == "True"} or None
    for r in sorted(res):
        m = out / "runs" / f"{res[r]['arm']}_s1" / f"merged_{res[r]['arm']}_s1"
        tools_md.append(tool_usage.markdown(tool_usage.table(tool_usage.per_task(m, valid)), f"{r}% ({res[r]['arm']})"))
    (out / "fc_search_tools.md").write_text("\n".join(tools_md))
    (out / "fc_search.json").write_text(json.dumps({r: {k: v for k, v in x.items() if k != "tasks"} for r, x in res.items()},
                                                  indent=1))
    with open(out / "fc_search_tasks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(res.values()))["tasks"][0]))
        w.writeheader()
        for r in sorted(res):
            w.writerows(res[r]["tasks"])
    print("\n".join(lines), flush=True)


def selftest():
    assert arm(0) == "B2" and arm(100) == "FC" and arm(37) == "FC037"
    assert abs(mcnemar(0, 1) - 1.0) < 1e-9 and mcnemar(0, 10) < 0.01
    assert next_rate({0: {"resolved": 5}, 100: {"resolved": 3}}, 12) == 50
    assert next_rate({0: {"resolved": 5}, 50: {"resolved": 7}, 100: {"resolved": 3}}, 12) in (25, 75)
    assert next_rate({0: {"resolved": 5}, 6: {"resolved": 9}, 12: {"resolved": 5}, 100: {"resolved": 1}}, 12) is None
    assert next_rate({0: {"resolved": 4}, 100: {"resolved": 4}}, 12) == 50      # tie -> best = lower rate, gap to 100


def main():
    if "--selftest" in sys.argv:
        selftest()
        print("selftest ok")
        return
    argv = sys.argv[1:]
    passthru = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-probes", type=int, default=7)
    ap.add_argument("--min-gap", type=int, default=12, help="stop when the gap next to the best rate is at most this (points)")
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    out = Path(a.out).expanduser()
    res, todo, why = {}, [0, 100], {0: "start: 0% (B2)", 100: "start: 100% (FC)"}
    old = json.load(open(out / "fc_search.json")) if (out / "fc_search.json").exists() else {}
    while todo and len(res) < a.max_probes:
        r = todo.pop(0)
        t0, started = time.time(), datetime.now(timezone.utc).isoformat(timespec="seconds")
        if read(out, r, set()) is None:
            print(f"=== probe {len(res) + 1}: forced rate {r}% (arm {arm(r)}): {why[r]}", flush=True)
            subprocess.run([sys.executable, str(HERE / "dev_runs.py"), "--out", str(out), "--arms", arm(r)] + passthru,
                           check=True)
            prov = {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "wall_seconds": round(time.time() - t0)}
        else:
            prov = {k: old.get(str(r), {}).get(k) for k in ("started", "finished", "wall_seconds")}
        sp = out / "submissions" / f"{arm(r)}_s1" / "prompts" / "system.md"
        if sp.exists():
            (out / "fc_search_prompts").mkdir(exist_ok=True)
            shutil.copy(sp, out / "fc_search_prompts" / f"{arm(r)}.md")
        prof = out / "validity" / "slices" / "profile.csv"
        PROFILE.update({x["instance_id"]: x for x in csv.DictReader(open(prof))})
        valid = {i for i, x in PROFILE.items() if x.get("sub_valid") == "True"}
        res[r] = read(out, r, valid)
        if res[r] is None:
            sys.exit(f"no merged results for {arm(r)} after its run")
        st = out / "settings.json"
        res[r].update(prov, probe=len(res), why=why[r], dev_runs_args=passthru,
                      settings=json.load(open(st)) if st.exists() else None)
        if not prov.get("wall_seconds"):
            res[r]["wall_seconds"] = 0
        report(out, res)
        if not todo:
            n = next_rate(res, a.min_gap)
            todo = [] if n is None or n in res else [n]
            if todo:
                best = max(res, key=lambda q: (res[q]["resolved"], -q))
                why[n] = f"midpoint of the widest gap next to the best rate so far ({best}%, {res[best]['resolved']} resolved)"
    print(f"search done after {len(res)} probes", flush=True)


if __name__ == "__main__":
    main()

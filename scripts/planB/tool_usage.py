#!/usr/bin/env python3
"""Which tools are used, and which go with solving the task? For any merged run folder (task_results.jsonl + traces/).

    python tool_usage.py RUN_DIR [RUN_DIR ...] [--valid PROFILE.csv] [--out OUT.md]

Per tool, over valid tasks (all tasks if no profile): tasks that used it, resolve rate when used vs not used, calls per
resolved and per unresolved task, and its share of all calls. Motivation: tools that are available but not needed can
lower accuracy (When Tools Get in the Way, arXiv 2609.14157), so tool use is tracked against outcome, not just counted.
Association only: harder tasks draw more exploration, so "used and failed" is not "the tool caused the failure".
Also importable: per_task(run_dir) -> {instance_id: {"resolved": bool, "calls": Counter}}; table(per_task_dict) -> rows.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

TOOLS = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch",
         "get_code_neighbors", "search_similar_code", "get_code_subgraph"]


def per_task(run_dir, valid=None):
    run_dir = Path(run_dir)
    out = {}
    for line in open(run_dir / "task_results.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        iid = r["instance_id"]
        if valid is not None and iid not in valid:
            continue
        t = run_dir / "traces" / f"trace_{iid}.json"
        steps = json.load(open(t)).get("steps", []) if t.exists() else []
        out[iid] = {"resolved": bool(r.get("resolved")),
                    "calls": Counter(tc.get("function_name") for s in steps for tc in s.get("tool_calls") or [])}
    return out


def table(pt):
    tools = TOOLS + sorted({k for x in pt.values() for k in x["calls"]} - set(TOOLS) - {None})
    total = sum(sum(x["calls"].values()) for x in pt.values()) or 1
    rows = []
    for tool in tools:
        used = [x for x in pt.values() if x["calls"].get(tool)]
        unused = [x for x in pt.values() if not x["calls"].get(tool)]
        res = [x for x in pt.values() if x["resolved"]]
        unres = [x for x in pt.values() if not x["resolved"]]
        rate = lambda g: f"{sum(x['resolved'] for x in g)}/{len(g)}" + (f" = {sum(x['resolved'] for x in g) / len(g):.0%}" if g else "")
        per = lambda g: f"{sum(x['calls'].get(tool, 0) for x in g) / len(g):.1f}" if g else "-"
        rows.append({"tool": tool, "tasks_using": len(used), "resolved_when_used": rate(used),
                     "resolved_when_not_used": rate(unused), "calls_per_resolved_task": per(res),
                     "calls_per_unresolved_task": per(unres),
                     "share_of_all_calls": f"{sum(x['calls'].get(tool, 0) for x in pt.values()) / total:.1%}"})
    return rows


def markdown(rows, title):
    cols = list(rows[0]) if rows else []
    return "\n".join([f"### {title}", "", "| " + " | ".join(c.replace("_", " ") for c in cols) + " |",
                      "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]) + "\n"


def selftest():
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "traces").mkdir()
    (d / "task_results.jsonl").write_text("".join(json.dumps({"instance_id": f"t{i}", "resolved": i < 2}) + "\n" for i in range(4)))
    for i in range(4):
        names = ["read_file", "edit_file"] + (["get_code_neighbors"] * 2 if i >= 2 else [])
        (d / "traces" / f"trace_t{i}.json").write_text(json.dumps({"steps": [{"tool_calls": [{"function_name": n}]} for n in names]}))
    rows = {r["tool"]: r for r in table(per_task(d))}
    assert rows["get_code_neighbors"]["resolved_when_used"] == "0/2 = 0%"
    assert rows["get_code_neighbors"]["resolved_when_not_used"] == "2/2 = 100%"
    assert rows["read_file"]["tasks_using"] == 4 and rows["get_code_neighbors"]["calls_per_unresolved_task"] == "2.0"
    assert len(table(per_task(d, valid={"t0"}))[0]) == 7
    print("selftest ok")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--valid", help="validity profile.csv: only sub_valid tasks count")
    ap.add_argument("--out")
    a = ap.parse_args()
    valid = {x["instance_id"] for x in csv.DictReader(open(a.valid)) if x.get("sub_valid") == "True"} if a.valid else None
    md = "".join(markdown(table(per_task(r, valid)), Path(r).name) + "\n" for r in a.runs)
    print(md)
    if a.out:
        Path(a.out).write_text(md)


if __name__ == "__main__":
    main()

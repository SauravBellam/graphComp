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
After every probe: fc_search.json and fc_search.md in --out (resolved, measured graph-call share, exact McNemar vs B2).
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRAPH = {"get_code_neighbors", "search_similar_code", "get_code_subgraph"}


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
    calls = graph = 0
    for t in (m / "traces").glob("trace_*.json"):
        for s in json.load(open(t)).get("steps", []):
            for tc in s.get("tool_calls") or []:
                calls += 1
                graph += tc.get("function_name") in GRAPH
    return {"rate": r, "arm": name, "resolved": len(solved), "scored": len([x for x in rows if x["instance_id"] in valid]),
            "graph_share": graph / calls if calls else 0.0, "tool_calls": calls, "solved": solved}


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
    lines = ["| forced rate | arm | measured graph-call share | resolved | vs B2: only this / only B2 | McNemar p |",
             "|---|---|---|---|---|---|"]
    for r in sorted(res):
        x = res[r]
        if base and r != 0:
            b, c = len(set(x["solved"]) - set(base["solved"])), len(set(base["solved"]) - set(x["solved"]))
            cmp_ = f"{b} / {c} | {mcnemar(b, c):.3f}"
        else:
            cmp_ = "(baseline) | "
        lines.append(f"| {r}% | {x['arm']} | {x['graph_share']:.1%} | {x['resolved']}/{x['scored']} | {cmp_} |")
    best = max(res, key=lambda r: (res[r]["resolved"], -r))
    lines += ["", f"Best so far: {best}% ({res[best]['resolved']}/{res[best]['scored']})."]
    (out / "fc_search.md").write_text("\n".join(lines) + "\n")
    (out / "fc_search.json").write_text(json.dumps(res, indent=1))
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
    res, todo = {}, [0, 100]
    while todo and len(res) < a.max_probes:
        r = todo.pop(0)
        if read(out, r, set()) is None:
            print(f"=== probe {len(res) + 1}: forced rate {r}% (arm {arm(r)})", flush=True)
            subprocess.run([sys.executable, str(HERE / "dev_runs.py"), "--out", str(out), "--arms", arm(r)] + passthru,
                           check=True)
        prof = out / "validity" / "slices" / "profile.csv"
        valid = {x["instance_id"] for x in csv.DictReader(open(prof)) if x.get("sub_valid") == "True"}
        res[r] = read(out, r, valid)
        if res[r] is None:
            sys.exit(f"no merged results for {arm(r)} after its run")
        report(out, res)
        if not todo:
            n = next_rate(res, a.min_gap)
            todo = [] if n is None or n in res else [n]
    print(f"search done after {len(res)} probes", flush=True)


if __name__ == "__main__":
    main()

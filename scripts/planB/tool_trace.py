#!/usr/bin/env python3
"""Follow every tool call: what was it for, did it touch or reveal the code the fix needs, and was its output acted on?

    python tool_trace.py RUN_DIR [RUN_DIR ...] --tasks tasks.jsonl [--valid profile.csv] [--out DIR]

Reference = the task's gold patch (tasks.jsonl `patch`): the files it changes and the functions/classes it touches (hunk
headers and changed def/class lines). Function level on purpose: file choice alone barely separates success from failure,
function selection does (TRACEPROBE, arXiv 2607.06184). Per tool call:
  purpose      normalized action: read-src / read-test, edit-src / edit-test-or-script, test, search, inspect, reproduce,
               vcs, graph:<tool>, status, submit (cf. TRACEPROBE's canonical action taxonomy)
  on_gold      its target (file, node, query, command) names a gold file or gold function
  reveals_gold its output names a gold file or function that nothing earlier in the trajectory had named
  followed     one of the next 3 calls targets a .py file or name that first appeared in this call's output
Per trajectory: which call and purpose first reached the gold code, after how many calls, and whether the gold file was
edited. Outputs (in --out, default the current folder): tool_trace.md (per-purpose table for resolved vs unresolved runs,
who first finds the gold code, the shortest resolved routes, purpose bigrams), tool_trace_calls.csv (one row per call),
tool_trace_runs.csv (one row per trajectory). Heuristic text matching: association, not causation.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PY = re.compile(r"[\w./-]+\.py\b")


def gold_of(patch):
    files = list(dict.fromkeys(re.findall(r"^diff --git a/(\S+) b/", patch, re.M) +
                               re.findall(r"^\+\+\+ b/(\S+)", patch, re.M)))      # tasks.jsonl patches have only ---/+++
    funcs = set(re.findall(r"^@@[^@]*@@\s*(?:async\s+)?(?:def|class)\s+(\w+)", patch, re.M))
    funcs |= set(re.findall(r"^[+-]\s*(?:async\s+)?(?:def|class)\s+(\w+)", patch, re.M))
    # library code only: tests and documentation examples (docs_src/: names like `app`, `Item`) are not the fix location;
    # dunders (`__init__` is in 26 gold patches) and names under 4 characters match far too much text to mean anything
    src = [f for f in files if not re.search(r"(^|/)(tests?|testing|docs|docs_src)/|test_[^/]*\.py$|_test\.py$|conftest\.py$", f)]
    return {"files": src or files,
            "funcs": {f for f in funcs if not f.startswith(("test", "__")) and len(f) >= 4} if src else set()}


def purpose(tool, args):
    path = str(args.get("filepath", ""))
    is_test = bool(re.search(r"test|repro|reproduce|scratch|tmp", path, re.I))
    if tool == "read_file":
        return "read-test" if is_test else "read-src"
    if tool in ("edit_file", "write_file"):
        return "edit-test-or-script" if is_test or not path.endswith(".py") else "edit-src"
    if tool == "run_command":
        c = str(args.get("command", ""))
        for name, rx in (("test", r"\bpytest\b|-m\s+pytest|unittest"), ("search", r"\b(grep|rg|find|ls|git\s+grep|ack)\b"),
                         ("vcs", r"\bgit\s+(diff|status|log|show|stash|checkout)"), ("install", r"\bpip\b"),
                         ("reproduce", r"\bpython3?\b\s+(-c\b|[\w./-]+\.py)"), ("inspect", r"\b(cat|sed|head|tail|nl|less|wc)\b")):
            if re.search(rx, c):
                return name
        return "command-other"
    if tool in ("get_code_neighbors", "search_similar_code", "get_code_subgraph"):
        return f"graph:{tool}"
    return {"get_status": "status", "submit_patch": "submit"}.get(tool, tool or "?")


def gold_items(text, gold):
    """The gold files (full path, or basename as a word) and gold functions (as a word) that `text` names."""
    text = str(text)
    files = {f for f in gold["files"] if f in text or re.search(rf"(?<![\w.-]){re.escape(f.split('/')[-1])}\b", text)}
    return files | {fn for fn in gold["funcs"] if re.search(rf"\b{re.escape(fn)}\b", text)}


def names_gold(text, gold):
    return bool(gold_items(text, gold))


def target(args):
    return str(args.get("filepath") or args.get("node") or args.get("query") or args.get("command") or "")


def calls_of(trace):
    out = []
    for s in trace.get("steps", []):
        obs = s.get("observation")
        content = obs.get("content", "") if isinstance(obs, dict) else ""
        for tc in s.get("tool_calls") or []:
            out.append({"tool": tc.get("function_name"), "args": tc.get("arguments") or {}, "output": str(content)})
    return out


def trace_run(trace, gold):
    calls = calls_of(trace)
    seen_text, seen_gold, rows, first = "", set(), [], None
    for i, c in enumerate(calls):
        tgt = target(c["args"])
        p = purpose(c["tool"], c["args"])
        on = names_gold(tgt, gold)
        new_gold = bool(gold_items(c["output"], gold) - seen_gold - gold_items(tgt, gold))
        seen_gold |= gold_items(tgt, gold) | gold_items(c["output"], gold)
        new_names = {m for m in PY.findall(c["output"]) if m not in seen_text}
        followed = any(any(n in target(d["args"]) or n.split("/")[-1] in target(d["args"]) for n in new_names)
                       for d in calls[i + 1:i + 4]) if new_names else False
        if first is None and (on or new_gold):
            first = (i, p)
        rows.append({"i": i, "purpose": p, "tool": c["tool"], "target": tgt[:160], "on_gold": on, "reveals_gold": new_gold,
                     "followed": followed})
        seen_text += " " + tgt + " " + c["output"]
    edited_gold = any(r["purpose"] == "edit-src" and names_gold(r["target"], gold) for r in rows)
    return rows, {"n_calls": len(rows), "first_gold_call": first[0] if first else None,
                  "first_gold_purpose": first[1] if first else None, "edited_gold_file": edited_gold}


def collapse(seq):
    out = []
    for p in seq:
        if out and out[-1][0] == p:
            out[-1][1] += 1
        else:
            out.append([p, 1])
    return " → ".join(p if n == 1 else f"{p}×{n}" for p, n in out)


def analyze(run_dirs, tasks, valid=None):
    calls, runs = [], []
    for rd in map(Path, run_dirs):
        for line in open(rd / "task_results.jsonl"):
            if not line.strip():
                continue
            r = json.loads(line)
            iid = r["instance_id"]
            if (valid is not None and iid not in valid) or iid not in tasks:
                continue
            tp = rd / "traces" / f"trace_{iid}.json"
            if not tp.exists():
                continue
            gold = gold_of(tasks[iid]["patch"])
            rows, summ = trace_run(json.load(open(tp)), gold)
            summ["issue_names_gold_file"] = bool(gold_items(tasks[iid]["problem"], {"files": gold["files"], "funcs": set()}))
            summ["issue_names_gold_func"] = bool(gold_items(tasks[iid]["problem"], {"files": [], "funcs": gold["funcs"]}))
            res = bool(r.get("resolved"))
            for x in rows:
                calls.append({"run": rd.name, "instance_id": iid, "resolved": res, **x})
            runs.append({"run": rd.name, "instance_id": iid, "resolved": res, **summ,
                         "route": collapse([x["purpose"] for x in rows])})
    return calls, runs


def report(calls, runs):
    pct = lambda g, k: f"{sum(x[k] for x in g) / len(g):.0%}" if g else "-"
    L = ["# Tool tracing: what each tool call was for, and whether it reached or revealed the gold code", "",
         f"{len(runs)} trajectories ({sum(r['resolved'] for r in runs)} resolved), {len(calls)} tool calls. "
         "Heuristic text matching against the gold patch; association, not causation.", "",
         "## Per purpose: resolved vs unresolved trajectories", "",
         "| purpose | calls (res / unres) | on gold code (res / unres) | reveals gold code first (res / unres) | output followed within 3 calls (res / unres) |",
         "|---|---|---|---|---|"]
    by = defaultdict(lambda: ([], []))
    for c in calls:
        by[c["purpose"]][0 if c["resolved"] else 1].append(c)
    for p in sorted(by, key=lambda p: -len(by[p][0]) - len(by[p][1])):
        a, b = by[p]
        L.append(f"| {p} | {len(a)} / {len(b)} | {pct(a, 'on_gold')} / {pct(b, 'on_gold')} | {pct(a, 'reveals_gold')} / "
                 f"{pct(b, 'reveals_gold')} | {pct(a, 'followed')} / {pct(b, 'followed')} |")
    L += ["", "## Who first reaches the gold code (first call whose target or new output names it)", "",
          "| first purpose | resolved runs | unresolved runs |", "|---|---|---|"]
    fr = Counter((r["first_gold_purpose"] or "never", r["resolved"]) for r in runs)
    for p in sorted({k[0] for k in fr}, key=lambda p: -(fr[(p, True)] + fr[(p, False)])):
        L.append(f"| {p} | {fr[(p, True)]} | {fr[(p, False)]} |")
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else "-"
    for lab, g in (("resolved", [r for r in runs if r["resolved"]]), ("unresolved", [r for r in runs if not r["resolved"]])):
        L.append(f"\n{lab}: median calls before the gold code is reached {med([r['first_gold_call'] for r in g if r['first_gold_call'] is not None])}; "
                 f"edited a gold file {sum(r['edited_gold_file'] for r in g)}/{len(g)}; never reached it {sum(r['first_gold_call'] is None for r in g)}/{len(g)}")
    for lab, k in (("a gold file", "issue_names_gold_file"), ("a gold function", "issue_names_gold_func")):
        g1, g0 = [r for r in runs if r[k]], [r for r in runs if not r[k]]
        L.append(f"issue text (problem + hints) names {lab}: {len(g1)}/{len(runs)} trajectories; resolved "
                 f"{sum(r['resolved'] for r in g1)}/{len(g1)} when it does vs {sum(r['resolved'] for r in g0)}/{len(g0)} when not")
    L += ["", "## Shortest resolved routes (the best cases)", ""]
    for r in sorted([r for r in runs if r["resolved"]], key=lambda r: r["n_calls"])[:10]:
        L.append(f"- `{r['instance_id']}` ({r['n_calls']} calls): {r['route']}")
    L += ["", "## Purpose bigrams: share of resolved vs unresolved trajectories containing them (top 15 by difference)", "",
          "| bigram | in resolved | in unresolved |", "|---|---|---|"]
    seqs = defaultdict(list)
    for c in calls:
        seqs[(c["run"], c["instance_id"], c["resolved"])].append(c["purpose"])
    big = lambda s: {(a, b) for a, b in zip(s, s[1:])}
    R = [big(s) for k, s in seqs.items() if k[2]]
    U = [big(s) for k, s in seqs.items() if not k[2]]
    allb = set().union(*R, *U) if (R or U) else set()
    share = lambda g, b: sum(b in x for x in g) / len(g) if g else 0
    for b in sorted(allb, key=lambda b: -abs(share(R, b) - share(U, b)))[:15]:
        L.append(f"| {b[0]} → {b[1]} | {share(R, b):.0%} | {share(U, b):.0%} |")
    return "\n".join(L) + "\n"


def write(out, calls, runs):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tool_trace.md").write_text(report(calls, runs))
    for name, rows in (("tool_trace_calls.csv", calls), ("tool_trace_runs.csv", runs)):
        if rows:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


def selftest():
    patch = "diff --git a/pkg/core.py b/pkg/core.py\n@@ -1,3 +1,3 @@ def solve(x):\n-    return 1\n+    return 2\n"
    g = gold_of(patch)
    assert g == {"files": ["pkg/core.py"], "funcs": {"solve"}}, g
    assert gold_of("--- a/x/y.py\n+++ b/x/y.py\n@@ -1 +1 @@ def f(\n-a\n+b\n")["files"] == ["x/y.py"]
    assert purpose("run_command", {"command": "python -m pytest tests -x"}) == "test"
    assert purpose("run_command", {"command": "grep -rn solve pkg"}) == "search"
    assert purpose("run_command", {"command": "python3 reproduce.py"}) == "reproduce"
    assert purpose("edit_file", {"filepath": "pkg/core.py"}) == "edit-src"
    assert purpose("write_file", {"filepath": "reproduce_issue.py"}) == "edit-test-or-script"
    tr = {"steps": [
        {"tool_calls": [{"function_name": "run_command", "arguments": {"command": "grep -rn solve ."}}],
         "observation": {"content": "pkg/core.py:1:def solve(x):"}},
        {"tool_calls": [{"function_name": "read_file", "arguments": {"filepath": "pkg/core.py"}}], "observation": {"content": "..."}},
        {"tool_calls": [{"function_name": "edit_file", "arguments": {"filepath": "pkg/core.py"}}], "observation": {"content": "ok"}}]}
    rows, s = trace_run(tr, g)
    assert rows[0]["reveals_gold"] and rows[0]["followed"] and rows[1]["on_gold"] and not rows[1]["reveals_gold"], rows
    assert s == {"n_calls": 3, "first_gold_call": 0, "first_gold_purpose": "search", "edited_gold_file": True}, s
    assert collapse(["a", "a", "b"]) == "a×2 → b"
    print("selftest ok")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--valid", help="validity profile.csv: only sub_valid tasks")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    tasks = {t["instance_id"]: {"patch": t["patch"], "problem": t.get("problem_statement", "") + "\n" + (t.get("hints_text") or "")}
             for t in map(json.loads, open(a.tasks)) if t.get("patch")}
    valid = {x["instance_id"] for x in csv.DictReader(open(a.valid)) if x.get("sub_valid") == "True"} if a.valid else None
    calls, runs = analyze(a.runs, tasks, valid)
    write(a.out, calls, runs)
    print((Path(a.out) / "tool_trace.md").read_text())


if __name__ == "__main__":
    main()

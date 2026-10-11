#!/usr/bin/env python3
"""A9: three task-level metrics derived from the Plan A audit, and a scorer that uses them.

Inputs: A1 outputs (a1_tasks.csv, a1_gold_symbols.jsonl), A2 analysis outputs (a2_tasks.csv,
a2_gold.jsonl, a2_run_status.csv), the A2 run folders (trace_pre.json / trace_post.json, for per-test
outcomes), and tasks.jsonl. No new runs.

Per task (``task_metrics.csv``):

1. Entry distance ``d_entry``: min directed hops in the provided `calls` graph from the nodes where
   the failing tests enter the library (A2 ``entry`` sources) to a gold function executed by those
   tests. 0 = the gold is itself an entry; None = no executed gold node is reachable. Also the
   max over executed gold nodes (``d_entry_max``, every gold site), and undirected variants.
   Defined wherever the task has a trace and an executed gold node; GraphLocator-style d (issue text
   -> gold) was defined for 47 of 129 tasks.

2. Tool answerability: could a perfect user of get_code_neighbors / search_similar_code /
   get_code_subgraph surface the gold? Components: every gold symbol is a node (``ans_all_gold_nodes``),
   no gold symbol's natural query resolves to a wrong node (``ans_no_misresolution``), at least one
   executed gold node is reachable from the entry set (``ans_reachable``), all of them are
   (``ans_all_reachable``). ``answerable`` = all four; ``answerability_failures`` lists the failing ones.

3. Test discriminativeness: of the tests the harness requires (FAIL_TO_PASS, else the test functions
   added by test_patch), which actually distinguish a fix from no fix?
     required      = expected tests that were collected in the post-fix run
     discriminating = required tests that failed / errored / were not collectable before the fix and
                      pass after it
   ``disc_share`` = |discriminating| / |required|; ``null_patch_resolvable`` = the pre-fix run already
   satisfies the harness criterion (exit 0 and every required test passed), so an empty patch scores.

Scorer (``score_submission``): given per-test outcomes of a submission's post-patch run, returns
  harness_resolved   : no failed/error test and every required test passed (the competition rule)
  strict_resolved    : harness_resolved and >= 1 discriminating test exists and all of them pass
  disc_credit        : share of discriminating tests passing (continuous, None if none exist)
and aggregates them. Two submissions are scored from the audit's own runs as a check: the gold
patch (post trace) and the empty patch (pre trace). The empty patch must score 0 under
strict_resolved; whatever it scores under harness_resolved is the benchmark's free floor.

Outputs: task_metrics.csv, task_tests.jsonl (per required test: pre/post outcome, discriminating),
a9_summary.json, scores_gold_vs_null.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "a2"))
from analyze import _matches, expected_f2p, load_trace  # noqa: E402
from common import load_tasks, read_jsonl, write_csv, write_jsonl  # noqa: E402
from stats import mean_ci, wilson  # noqa: E402


# --- discriminativeness -----------------------------------------------------------------------

def required_tests(task, pre_out: dict, post_out: dict) -> list[dict]:
    """One record per test the harness effectively requires on this task.

    The harness runs pytest on the files test_patch touches and needs exit 0, so every test collected
    there must pass (``named`` = False), and in addition every extracted test name must be present and
    pass (``named`` = True). When test_patch only modifies existing tests the extracted set is empty and
    exit 0 is the whole requirement. ``discriminating`` = failed / errored / not collectable before the
    fix and passing after it: the tests that tell a fix from no fix."""
    reqs, source = expected_f2p(task)
    named_ids: set[str] = set()
    missing_named = []
    for req in reqs:
        hits = [t for t in set(pre_out) | set(post_out) if _matches(t, req)]
        if hits:
            named_ids |= set(hits)
        else:
            missing_named.append(req)
    out = []
    for t in sorted(set(pre_out) | set(post_out)):
        pre = pre_out.get(t, "uncollected")
        post = post_out.get(t, "uncollected")
        out.append({"req": next((r for r in reqs if _matches(t, r)), None) if t in named_ids else None,
                    "test": t, "named": t in named_ids, "pre": pre, "post": post,
                    "discriminating": pre in ("failed", "error", "uncollected") and post == "passed",
                    "source": source if reqs else "all_target_tests", "collected_post": t in post_out})
    for req in missing_named:
        out.append({"req": req, "test": None, "named": True, "pre": "not_collected", "post": "not_collected",
                    "discriminating": False, "source": source, "collected_post": False})
    return out


def score_submission(tests: list[dict], outcomes: dict, exit_zero: bool | None = None) -> dict:
    """Score one submission on one task from its post-patch per-test outcomes.

    ``outcomes``: {pytest nodeid: passed|failed|error|skipped} for the target files. ``exit_zero``:
    pytest exit code == 0 if known; otherwise approximated as no failed/error outcome and >= 1 passed."""
    named = [r for r in tests if r["named"]]
    disc = [r for r in tests if r["discriminating"] and r["test"] is not None]
    if exit_zero is None:
        exit_zero = bool(outcomes) and not any(o in ("failed", "error") for o in outcomes.values()) \
            and any(o == "passed" for o in outcomes.values())
    named_ok = all(r["test"] is not None and outcomes.get(r["test"]) == "passed" for r in named)
    harness = bool(exit_zero and named_ok and any(o == "passed" for o in outcomes.values()))
    disc_pass = sum(1 for r in disc if outcomes.get(r["test"]) == "passed")
    return {"harness_resolved": harness,
            "strict_resolved": bool(harness and disc and disc_pass == len(disc)),
            "disc_credit": (disc_pass / len(disc)) if disc else None,
            "n_named": len(named), "n_discriminating": len(disc), "n_disc_passed": disc_pass}


def aggregate(scores: list[dict]) -> dict:
    n = len(scores)
    return {"n_tasks": n,
            "resolution_rate_harness": wilson(sum(1 for s in scores if s["harness_resolved"]), n),
            "resolution_rate_strict": wilson(sum(1 for s in scores if s["strict_resolved"]), n),
            "mean_disc_credit": mean_ci([s["disc_credit"] for s in scores]),
            "tasks_without_discriminating_tests": sum(1 for s in scores if s["n_discriminating"] == 0)}


# --- main --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--a1", required=True, type=Path)
    ap.add_argument("--a2-analysis", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    a1 = {r["instance_id"]: r for r in csv.DictReader(open(args.a1 / "a1_tasks.csv", newline="", encoding="utf-8"))}
    a2 = {r["instance_id"]: r for r in csv.DictReader(open(args.a2_analysis / "a2_tasks.csv", newline="", encoding="utf-8"))}
    status = {r["instance_id"]: r for r in csv.DictReader(open(args.a2_analysis / "a2_run_status.csv", newline="", encoding="utf-8"))}
    gold_by_task: dict[str, list[dict]] = defaultdict(list)
    for g in read_jsonl(args.a2_analysis / "a2_gold.jsonl"):
        gold_by_task[g["instance_id"]].append(g)

    rows, test_rows = [], []
    gold_scores, null_scores = [], []
    for t in load_tasks(args.data):
        a = a1.get(t.instance_id, {})
        b = a2.get(t.instance_id, {})
        st = status.get(t.instance_id, {})
        row = {"instance_id": t.instance_id, "repo": t.repo, "a2_run_class": st.get("run_class"), "a2_usable": b.get("usable")}

        # 1. entry distance
        execd = [g for g in gold_by_task.get(t.instance_id, []) if g["status"] == "in_graph_executed"]
        hops = [g.get("static_hops_from_entry_directed") for g in execd]
        hops_u = [g.get("static_hops_from_entry_undirected") for g in execd]
        defined = [h for h in hops if h is not None]
        row.update({
            "n_gold_executed": len(execd),
            "n_entry_sources": int(b["n_entry_sources"]) if b.get("n_entry_sources") not in (None, "") else None,
            "d_entry": min(defined) if defined else None,
            "d_entry_max": max(defined) if defined and len(defined) == len(execd) else None,
            "d_entry_undirected": min((h for h in hops_u if h is not None), default=None),
            "n_gold_executed_unreachable": sum(1 for h in hops if h is None),
            "d_entry_defined": bool(execd) and (b.get("n_entry_sources") not in (None, "", "0")),
        })
        # 2. answerability
        n_sym = int(a["n_gold_symbols"]) if a.get("n_gold_symbols") else 0
        n_present = int(a["n_gold_symbols_present"]) if a.get("n_gold_symbols_present") else 0
        n_wrong = int(a["n_gold_query_wrong_node"]) if a.get("n_gold_query_wrong_node") else 0
        comp = {"ans_all_gold_nodes": n_sym > 0 and n_present == n_sym,
                "ans_no_misresolution": n_wrong == 0,
                "ans_reachable": bool(defined),
                "ans_all_reachable": bool(execd) and len(defined) == len(execd)}
        row.update(comp)
        row["answerable"] = all(comp.values())
        row["answerability_failures"] = ";".join(k for k, v in comp.items() if not v)
        row["n_gold_symbols"], row["n_gold_symbols_present"], row["n_gold_query_wrong_node"] = n_sym, n_present, n_wrong
        row["n_gold_async"] = int(a["n_gold_async"]) if a.get("n_gold_async") else 0

        # 3. discriminativeness (needs the traces)
        tr_pre = load_trace(args.runs / t.instance_id / "trace_pre.json")
        tr_post = load_trace(args.runs / t.instance_id / "trace_post.json")
        pre_out = (tr_pre or {}).get("outcomes", {})
        post_out = (tr_post or {}).get("outcomes", {})
        tests = required_tests(t, pre_out, post_out)
        named = [x for x in tests if x["named"] and x["test"] is not None]
        effective = [x for x in tests if x["test"] is not None and x["collected_post"]]
        disc = [x for x in tests if x["discriminating"]]
        pre_exit0 = st.get("pre_exit") == "0"
        row.update({
            "n_expected_names": len({x["req"] for x in tests if x["req"]}),
            "n_named_collected": len(named), "n_named_missing": sum(1 for x in tests if x["named"] and x["test"] is None),
            "n_effective_required": len(effective), "n_discriminating": len(disc),
            "n_named_discriminating": sum(1 for x in named if x["discriminating"]),
            "disc_share_named": round(sum(1 for x in named if x["discriminating"]) / len(named), 4) if named else None,
            "disc_share_effective": round(len(disc) / len(effective), 4) if effective else None,
            "n_named_pass_pre": sum(1 for x in named if x["pre"] == "passed"),
            "n_named_uncollected_pre": sum(1 for x in named if x["pre"] == "uncollected"),
            # the harness criterion is already met before the fix: an empty patch scores
            "null_patch_resolvable": bool(pre_exit0 and any(x["pre"] == "passed" for x in tests)
                                          and all(x["pre"] == "passed" for x in named) and not any(x["named"] and x["test"] is None for x in tests)),
            "pre_exit": st.get("pre_exit"), "post_exit": st.get("post_exit"),
        })
        for x in tests:
            test_rows.append({"instance_id": t.instance_id, **x})
        # the two reference submissions
        if tr_pre is not None and tr_post is not None:
            gs = score_submission(tests, post_out, exit_zero=(st.get("post_exit") == "0"))
            ns = score_submission(tests, pre_out, exit_zero=pre_exit0)
            gold_scores.append({"instance_id": t.instance_id, **gs})
            null_scores.append({"instance_id": t.instance_id, **ns})
            row["gold_patch_harness"], row["gold_patch_strict"], row["gold_patch_disc_credit"] = gs["harness_resolved"], gs["strict_resolved"], gs["disc_credit"]
            row["null_patch_harness"], row["null_patch_strict"] = ns["harness_resolved"], ns["strict_resolved"]
        rows.append(row)

    write_csv(args.out / "task_metrics.csv", rows)
    write_jsonl(args.out / "task_tests.jsonl", test_rows)
    traced = [r for r in rows if r.get("pre_exit") not in (None, "")]
    usable = [r for r in rows if r.get("a2_usable") == "True"]

    def dist(rs, key):
        return dict(Counter(str(r.get(key)) for r in rs))

    summ = {
        "n_tasks": len(rows), "n_with_traces": len(traced), "n_usable": len(usable),
        "entry_distance": {
            "tasks_defined": sum(1 for r in rows if r["d_entry_defined"]),
            "d_entry_hist_all": dist([r for r in rows if r["d_entry_defined"]], "d_entry"),
            "d_entry_hist_usable": dist([r for r in usable if r["d_entry_defined"]], "d_entry"),
            "d_entry_max_hist_usable": dist([r for r in usable if r["d_entry_defined"]], "d_entry_max"),
            "per_repo_usable": {repo: dist([r for r in usable if r["repo"] == repo and r["d_entry_defined"]], "d_entry")
                                for repo in sorted({r["repo"] for r in rows})},
        },
        "answerability": {
            "tasks_answerable": wilson(sum(1 for r in rows if r["answerable"]), len(rows)),
            "tasks_answerable_usable": wilson(sum(1 for r in usable if r["answerable"]), len(usable)),
            "component_failure_counts": {k: sum(1 for r in rows if not r[k]) for k in
                                         ("ans_all_gold_nodes", "ans_no_misresolution", "ans_reachable", "ans_all_reachable")},
            "failure_combinations": dict(Counter(r["answerability_failures"] or "none" for r in rows)),
            "per_repo": {repo: wilson(sum(1 for r in rows if r["repo"] == repo and r["answerable"]), sum(1 for r in rows if r["repo"] == repo))
                         for repo in sorted({r["repo"] for r in rows})},
        },
        "discriminativeness": {
            "named_required_tests": sum(r["n_named_collected"] for r in traced),
            "named_discriminating": sum(r["n_named_discriminating"] for r in traced),
            "named_passing_pre_fix": sum(r["n_named_pass_pre"] for r in traced),
            "named_uncollected_pre_fix": sum(r["n_named_uncollected_pre"] for r in traced),
            "named_missing_post": sum(r["n_named_missing"] for r in traced),
            "effective_required_tests": sum(r["n_effective_required"] for r in traced),
            "discriminating_tests": sum(r["n_discriminating"] for r in traced),
            "tasks_with_no_named_tests": sum(1 for r in traced if r["n_named_collected"] == 0 and r["n_named_missing"] == 0),
            "tasks_named_disc_share_1": sum(1 for r in traced if r["disc_share_named"] == 1.0),
            "tasks_named_disc_share_0": sum(1 for r in traced if r["disc_share_named"] == 0.0),
            "tasks_named_disc_share_between": sum(1 for r in traced if r["disc_share_named"] not in (None, 0.0, 1.0)),
            "tasks_without_any_discriminating_test": [r["instance_id"] for r in traced if r["n_discriminating"] == 0],
            "null_patch_resolvable": [r["instance_id"] for r in rows if r["null_patch_resolvable"]],
            "mean_disc_share_named": mean_ci([r["disc_share_named"] for r in traced]),
            "mean_disc_share_effective": mean_ci([r["disc_share_effective"] for r in traced]),
            "discriminating_per_task_hist": dict(Counter(str(min(r["n_discriminating"], 10)) for r in traced)),
            "per_repo_mean_disc_share_named": {repo: mean_ci([r["disc_share_named"] for r in traced if r["repo"] == repo])
                                               for repo in sorted({r["repo"] for r in rows})},
        },
        "reference_submissions": {"gold_patch": aggregate(gold_scores), "empty_patch": aggregate(null_scores)},
    }
    (args.out / "a9_summary.json").write_text(json.dumps(summ, indent=2))
    (args.out / "scores_gold_vs_null.json").write_text(json.dumps({"gold_patch": gold_scores, "empty_patch": null_scores}, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "entry_distance"}, indent=2))
    print(json.dumps(summ["entry_distance"], indent=2))


if __name__ == "__main__":
    main()

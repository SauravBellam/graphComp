"""B1: merge the subprocess-sandbox gold/empty pass into the task profile and write the two 24-task slices.

    python make_slices.py --profile profile.csv \
        --empty RUN_DIR [RUN_DIR ...] --gold RUN_DIR [RUN_DIR ...] \
        [--import-check import_check_subprocess.jsonl] --out OUT_DIR [--seed 20261012] [--size 24]

RUN_DIR is a run_arm.py results directory (task_results.jsonl). Pass several empty / gold runs to
detect flaky tests: a task is `sub_valid` only if EVERY empty run fails and EVERY gold run passes,
and (when the import check is given) pytest imported the task's own workspace code.

Eligible for the slices (HANDOFF_planB.md section 2): a2_usable AND sub_valid.

Stratified allocation (deterministic given --seed): eligible tasks are sorted by
(repo, cell, gold_has_async, gold_is_entry_point, random tiebreak from the seed), then dealt one by
one to dev / holdout / rest, each time to the group furthest below its proportional target
(size/N, size/N, rest/N). Every stratum, and every prefix of the sort order, is therefore split in
proportion to within one task, and the totals are exactly size / size / N - 2*size.

Writes OUT/profile.csv (input profile + sub_* columns), OUT/dev_slice.txt, OUT/holdout_slice.txt,
OUT/slices_meta.json (seed, rule, counts, balance tables). The slice files are never rewritten by
hand afterwards (Plan B section 2).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

STRATA = ("repo", "cell", "gold_has_async", "gold_is_entry_point")


def read_results(dirs):
    """{instance_id: [resolved per run]} over several run dirs (missing task in a run -> None)."""
    runs = []
    for d in dirs:
        p = Path(d) / "task_results.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
        runs.append({r["instance_id"]: r for r in rows})
    ids = set().union(*[set(r) for r in runs]) if runs else set()
    return {i: [r.get(i) for r in runs] for i in ids}, len(runs)


def fmt(rs):
    return ";".join("NA" if r is None else str(bool(r.get("resolved"))) for r in rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--empty", nargs="+", required=True)
    ap.add_argument("--gold", nargs="+", required=True)
    ap.add_argument("--import-check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--size", type=int, default=24)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = list(csv.DictReader(open(a.profile)))
    E, n_e = read_results(a.empty)
    G, n_g = read_results(a.gold)
    IC = {}
    if a.import_check and Path(a.import_check).exists():
        for l in Path(a.import_check).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                IC[r["instance_id"]] = r

    for row in prof:
        i = row["instance_id"]
        e, g = E.get(i, [None] * n_e), G.get(i, [None] * n_g)
        ic = IC.get(i, {})
        row["sub_import_verdict"] = ic.get("verdict", "" if not IC else "missing")
        row["sub_pytest_file_is_workspace"] = "" if not IC else str(ic.get("verdict") == "workspace tested")
        row["sub_empty_resolved"] = fmt(e)
        row["sub_gold_resolved"] = fmt(g)
        row["sub_gold_exit"] = ";".join("NA" if r is None else str(r.get("test_exit_code")) for r in g)
        # Evaluated in this order (Oct 12): a task whose tests import a host copy is out without running anything;
        # a gold failure settles a task without its empty pass (the empty pass is only run on gold-passing tasks).
        ws_ok = (not IC) or ic.get("verdict") == "workspace tested"
        g_have = [r for r in g if r is not None]
        e_have = [r for r in e if r is not None]
        gold_flaky = len({bool(r.get("resolved")) for r in g_have}) > 1
        empty_flaky = len({bool(r.get("resolved")) for r in e_have}) > 1
        gold_passes = bool(g_have) and len(g_have) == n_g and all(bool(r.get("resolved")) for r in g_have)
        empty_fails = bool(e_have) and len(e_have) == n_e and all(not r.get("resolved") for r in e_have)
        if not ws_ok:
            reason = "released copy tested"
        elif not g_have:
            reason = "incomplete"
        elif gold_flaky:
            reason = "flaky"
        elif not all(bool(r.get("resolved")) for r in g_have):
            reason = "gold fails"
        elif len(g_have) < n_g or len(e_have) < n_e:
            reason = "incomplete"
        elif empty_flaky:
            reason = "flaky"
        elif not empty_fails:
            reason = "empty passes"
        else:
            reason = "ok"
        row["sub_flaky"] = str(gold_flaky or empty_flaky)
        row["sub_valid"] = str(reason == "ok")
        row["sub_reason"] = reason

    fields = list(prof[0].keys())
    with open(out / "profile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(prof)

    elig = [r for r in prof if r["a2_usable"] == "True" and r["sub_valid"] == "True"]
    n = len(elig)
    if n < 2 * a.size:
        raise SystemExit(f"only {n} eligible tasks (a2_usable and sub_valid); need {2 * a.size}. Nothing written for slices.")
    rng = random.Random(a.seed)
    tie = {r["instance_id"]: rng.random() for r in sorted(elig, key=lambda r: r["instance_id"])}
    order = sorted(elig, key=lambda r: tuple(r[k] for k in STRATA) + (tie[r["instance_id"]],))
    target = {"dev": a.size / n, "holdout": a.size / n, "rest": (n - 2 * a.size) / n}
    got = {"dev": [], "holdout": [], "rest": []}
    for k, r in enumerate(order, 1):
        # deficit after placing this task; ties broken dev > holdout > rest, alternating dev/holdout by parity
        pref = ["dev", "holdout", "rest"] if k % 2 else ["holdout", "dev", "rest"]
        grp = max(pref, key=lambda g: (target[g] * k - len(got[g]), -pref.index(g)))
        got[grp].append(r)
    assert len(got["dev"]) == len(got["holdout"]) == a.size, {g: len(v) for g, v in got.items()}

    for g in ("dev", "holdout"):
        (out / f"{g}_slice.txt").write_text("\n".join(sorted(r["instance_id"] for r in got[g])) + "\n")

    def bal(key):
        t = collections.defaultdict(lambda: {"dev": 0, "holdout": 0, "rest": 0})
        for g, rs in got.items():
            for r in rs:
                t[r[key]][g] += 1
        return dict(sorted(t.items()))

    reasons = collections.Counter((r["repo"], r["sub_reason"]) for r in prof)
    meta = {
        "seed": a.seed, "size": a.size, "n_eligible": n,
        "rule": "eligible = a2_usable and sub_valid (every empty run fails, every gold run passes, pytest imports the workspace); "
                "sorted by " + ", ".join(STRATA) + ", seeded tiebreak; dealt to the group furthest below its proportional target",
        "n_empty_runs": n_e, "n_gold_runs": n_g, "import_check_used": bool(IC),
        "sub_reason_by_repo": {f"{k[0]}|{k[1]}": v for k, v in sorted(reasons.items())},
        "n_sub_valid": sum(r["sub_valid"] == "True" for r in prof),
        "n_flaky": sum(r["sub_flaky"] == "True" for r in prof),
        "balance": {k: bal(k) for k in STRATA + ("multi_file",)},
        "dev": sorted(r["instance_id"] for r in got["dev"]),
        "holdout": sorted(r["instance_id"] for r in got["holdout"]),
    }
    (out / "slices_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"eligible {n}; dev {a.size}, holdout {a.size}, rest {n - 2 * a.size}; seed {a.seed}")
    for k in STRATA:
        print(f"\n{k}:")
        for v, c in meta["balance"][k].items():
            print(f"  {v:18s} dev {c['dev']:2d}  holdout {c['holdout']:2d}  rest {c['rest']:2d}")


if __name__ == "__main__":
    main()

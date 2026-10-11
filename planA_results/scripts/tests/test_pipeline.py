"""End-to-end test of the Plan A pipeline on the synthetic fixture (no competition data needed)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "planA"))


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("archive")
    import make_fixture
    make_fixture.main(d)
    return d


@pytest.fixture(scope="session")
def results(fixture_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    cmd = [sys.executable, str(ROOT / "planA" / "run_all.py"), "--data", str(fixture_dir), "--out", str(out),
           "--skip", "a3", "--a1-shards", "2", "--a2-shards", "2", "--jobs", "4", "--a2-timeout", "600"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    print(p.stdout[-4000:], p.stderr[-4000:])
    assert p.returncode == 0, "run_all failed; see logs"
    return out


def read_csv(p: Path):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def test_common_units(fixture_dir):
    from common import extract_identifiers, load_tasks, resolve_node_name, parse_patch, path_to_module_candidates
    tasks = load_tasks(fixture_dir)
    assert [t.instance_id for t in tasks] == ["mini_1", "mini_2", "mini_3"]
    ids = extract_identifiers(tasks[0].problem_statement)
    assert "compute" in ids["backtick"] or "compute" in ids["call"]
    assert "mini.core.compute" in ids["dotted"] or "mini.core.compute" in ids["backtick"]
    nodes = ["mini.core.compute", "mini.core.Engine.run", "tests.test_core.test_compute"]
    assert resolve_node_name("compute", nodes) == ("mini.core.compute", "suffix")
    assert resolve_node_name("RUN", nodes) == ("mini.core.Engine.run", "ci")
    assert resolve_node_name("ngine", nodes) == ("mini.core.Engine.run", "substring")
    assert resolve_node_name("zzz", nodes) == (None, "none")
    fds = parse_patch(tasks[2].patch)
    assert [f.path for f in fds] == ["docs_src/tutorial/tutorial001.py", "src/minisrc/extra.py", "mini/core.py"]
    assert path_to_module_candidates("docs_src/tutorial/tutorial001.py")[1] == ("tutorial.tutorial001", "drop:docs_src")


def test_resolver_matches_harness():
    import random
    from common import Resolver, resolve_node_name
    rng = random.Random(0)
    segs = ["a", "b", "Foo", "foo", "run", "Run", "get", "get_item", "x_y", "tests", "test_run"]
    ids = list(dict.fromkeys(".".join(rng.choice(segs) for _ in range(rng.randint(1, 5))) for _ in range(400)))
    r = Resolver(ids)
    queries = segs + [".".join(rng.choice(segs) for _ in range(rng.randint(1, 3))) for _ in range(300)] + ["FOO", "un", "zz", "et_it"]
    for q in queries:
        assert r.resolve(q) == resolve_node_name(q, ids), q


def test_stats():
    from stats import mcnemar, ratio_ci, wilson
    w = wilson(5, 10)
    assert w["p"] == 0.5 and abs(w["lo"] - 0.2366) < 1e-3 and abs(w["hi"] - 0.7634) < 1e-3
    m = mcnemar([True] * 8 + [False] * 2, [False] * 8 + [True] * 2)
    assert m["a_only"] == 8 and m["b_only"] == 2 and abs(m["p_exact"] - 0.109375) < 1e-6
    r = ratio_ci([1, 2, 3], [2, 4, 6])
    assert r["ratio"] == 0.5 and r["lo"] == 0.5 and r["hi"] == 0.5


def test_a1_outputs(results):
    tasks = {r["instance_id"]: r for r in read_csv(results / "a1" / "a1_tasks.csv")}
    assert set(tasks) == {"mini_1", "mini_2", "mini_3"}
    t1 = tasks["mini_1"]
    assert t1["graph_available"] == "True" and t1["snapshot_available"] == "True"
    assert json.loads(t1["gold_nodes"]) == ["mini.core.compute"]
    assert t1["d_any_undirected"] == "0"          # compute is itself an anchor
    assert t1["v_file_rank"] not in ("", None)
    t2 = tasks["mini_2"]
    assert t2["n_gold_async"] == "1" and t2["n_gold_async_present"] == "0"   # async omission detected
    t3 = tasks["mini_3"]
    assert t3["multi_file"] == "True" and t3["n_gold_src_files"] == "3"
    gold = [json.loads(l) for l in (results / "a1" / "a1_gold_symbols.jsonl").read_text().splitlines()]
    rules = {g["file"]: g["prefix_rule"] for g in gold if g["instance_id"] == "mini_3"}
    assert rules["docs_src/tutorial/tutorial001.py"] == "drop:docs_src"
    assert rules["src/minisrc/extra.py"] == "drop:src"
    new = [g for g in gold if g["instance_id"] == "mini_3" and g["new_symbol"]]
    assert {g["qualname"] for g in new} == {"Engine.double", "extra_new"}
    acomp = next(g for g in gold if g["instance_id"] == "mini_2" and g["qualname"] == "acompute")
    assert acomp["agent_query_outcome"] == "no_match"
    graphs = read_csv(results / "a1" / "a1_graphs.csv")
    assert len(graphs) == 2
    assert json.loads(graphs[0]["edge_types"]) == {"calls": 13}
    assert float(graphs[0]["pct_containment_in_calls"]) > 0
    # merged summary: no double counting across shards that share a commit
    summ = json.loads((results / "a1" / "a1_summary.json").read_text())
    assert summ["merged_from_shards"]
    assert summ["omission_direct_files"]["async_function"] == {**summ["omission_direct_files"]["async_function"], "total": 4, "present": 0}
    assert summ["omission_direct_files"]["function"]["present"] == summ["omission_direct_files"]["function"]["total"]
    # the nested def is found under the flattened id the generator uses
    assert summ["omission_direct_files"]["nested"]["total"] == 2 and summ["omission_direct_files"]["nested"]["present"] == 2
    inner = [r for r in read_csv(results / "a1" / "a1_async_audit.csv") if r["node_id"] == "mini.core.inner"]
    assert inner and all(r["present"] == "True" and r["node_id_locals"] == "mini.core.run.<locals>.inner" for r in inner)
    assert summ["omission_inferred_module_files"]["async_function"]["total"] == 2      # mini/aio.py, only async defs
    # async `total` is absent and its name silently resolves to the sync method Engine.total
    audit = read_csv(results / "a1" / "a1_async_audit.csv")
    tot = [r for r in audit if r["node_id"] == "mini.core.total"]
    assert tot and all(r["q_name_outcome"] == "wrong_node" and r["q_name_resolved"] == "mini.core.Engine.total"
                       and r["q_name_tier"] == "suffix" for r in tot)
    mis = summ["misresolution"]["async_function|absent|q_name"]
    assert mis["outcomes"] == {"no_match": 4, "wrong_node": 2}
    assert summ["task_rates"]["has_gold_async"]["k"] == 1


def test_a2_outputs(results):
    st = json.loads((results / "a2_runs" / "mini_1" / "status.json").read_text())
    assert st["done"], st
    assert st["pre"]["run_class"] == "clean" and st["post"]["exit"] == 0, st
    assert st["test_patch_strategy"] == "exact" and st["gold_patch_strategy"] == "exact"
    assert "pytest" in (results / "a2_runs" / "mini_1" / "env_freeze.txt").read_text().lower()
    tr = json.loads((results / "a2_runs" / "mini_1" / "trace_pre.json").read_text())
    assert tr["outcomes"]["tests/test_core.py::test_compute"] == "failed"
    kinds = dict(zip(tr["functions"], tr["kinds"]))
    assert kinds["mini.util.gen_values"] == "generator" and kinds["mini.core.compute"] == "function"
    # generator resumes are deduplicated: the gen_values edge appears once, from Engine.total
    fn = tr["functions"]
    gen_callers = {fn[c] for c, e, _ in tr["edges_direct"] if fn[e] == "mini.util.gen_values"}
    assert gen_callers == {"mini.core.Engine.total"}

    status = {r["instance_id"]: r for r in read_csv(results / "a2" / "a2_run_status.csv")}
    assert all(r["run_class"] == "clean" and r["usable"] == "True" for r in status.values())
    assert all(r["n_f2p_confirmed"] == r["n_f2p_expected"] == "1" for r in status.values())

    rows = {r["instance_id"]: r for r in read_csv(results / "a2" / "a2_tasks.csv")}
    r1, r2, r3 = rows["mini_1"], rows["mini_2"], rows["mini_3"]
    assert r1["gold_in_graph_executed"] == "1" and r1["gold_any_static_reachable"] == "True"
    # run -> inner is a captured edge now that the nested def maps to its flattened id
    assert int(r1["all_any_absent_nested"]) == 0 and int(r1["all_any_missed_between_nodes"]) >= 1
    assert r1["static_sources_defined"] == "True"
    # entry-point reachability: test_compute calls compute directly, so compute is an entry node (0 hops)
    assert int(r1["n_entry_sources"]) >= 1 and r1["gold_any_static_reachable_entry"] == "True"
    assert r1["gold_min_hops_from_entry_directed"] == "0"
    # the async gold function runs in the failing test but has no node: absent_executed
    assert r2["gold_absent_executed"] == "1" and int(r2["all_any_absent_async"]) >= 1
    # Engine.__init__ executed -> collapsed to the gold class Engine; the new method runs only post-fix
    assert r3["gold_in_graph_executed"] == "1" and r3["gold_new_symbol_executed_post"] == "1"
    assert r3["gold_any_static_reachable"] == "False"

    edges = [json.loads(l) for l in (results / "a2" / "a2_dyn_edges.jsonl").read_text().splitlines()]
    miss = [e for e in edges if e["instance_id"] == "mini_1" and e["cause"] == "missed_between_nodes"]
    assert any(e["src"] == "mini.util.gen_values" and e["dst"] == "mini.util.helper" and e["subcause"] == "sync" for e in miss)
    assert any(e["instance_id"] == "mini_1" and e["cause"] == "captured" and e["src"] == "mini.core.run" and e["dst"] == "mini.core.inner" for e in edges)
    assert any(e["cause"] == "absent_async" and e["dst"] == "mini.core.acompute" for e in edges)

    summ = json.loads((results / "a2" / "a2_summary.json").read_text())
    assert summ["coverage"]["usable"]["k"] == 3
    assert summ["coverage"]["patch_strategies"]["gold"] == {"exact": 3}
    u = summ["usable_tasks"]["all_any"]
    assert u["recall_representable"]["den"] >= u["recall_representable"]["num"] > 0
    assert u["recall_agent_facing"]["den"] > u["recall_representable"]["den"]
    assert set(u["miss_share_by_cause"]) == {"missed_between_nodes", "absent_async", "absent_nested", "absent_sync"}
    assert summ["usable_tasks"]["tasks_gold_absent_but_executed"]["k"] == 1
    # mini_2's failing test calls only the async (absent) acompute, which calls helper: the walk passes
    # through the absent frame, so helper is the entry node (and the gold itself is absent)
    assert summ["usable_tasks"]["tasks_entry_sources_defined"]["k"] == 3 and r2["n_entry_sources"] == "1"
    assert r2["gold_any_static_reachable_entry"] == "False" and r2["gold_min_hops_from_entry_directed"] == ""
    assert summ["usable_tasks"]["f2p_statuses"]["confirmed"] == 3 and summ["usable_tasks"]["f2p_statuses"]["uncollected_pre"] == 0
    assert summ["strict_tasks"]["n_tasks"] == 3
    assert summ["coverage"]["f2p_check"]["tasks_harness_resolved_post"] == 3


def test_a4_a5_a6(results):
    rel = read_csv(results / "a4" / "a4_relations.csv")
    assert {r["relation"] for r in rel} >= {"calls", "same_module", "random", "containment"}
    a5 = {r["instance_id"]: r for r in read_csv(results / "a5" / "a5_tasks.csv")}
    assert a5["mini_1"]["a_harness_hit10_node"] == "True"
    assert a5["mini_1"]["b_bm25_node_rank"] != ""
    s5 = json.loads((results / "a5" / "a5_summary.json").read_text())
    ov = s5["overall_tasks_with_gold_nodes"]
    assert {"k", "n", "lo", "hi"} <= set(ov["a_harness"]["hit10_node"])
    assert "p_exact" in ov["paired_mcnemar"]["a_harness_vs_b_bm25"]["hit10_node"]
    prof = {r["instance_id"]: r for r in read_csv(results / "a6" / "task_profile.csv")}
    assert prof["mini_1"]["cell"].startswith("d0|")
    assert prof["mini_1"]["a2_usable"] == "True" and prof["mini_1"]["a2_recall_representable_any"] != ""
    g2 = json.loads(next((results / "a6" / "graphs_v2").glob("*.json")).read_text())
    types = {e["type"] for e in g2["edges"]}
    assert {"contains", "imports", "calls_static", "calls_dynamic", "tests"} <= types
    ids = {n["id"] for n in g2["nodes"]}
    assert "mini.core.acompute" in ids and "tutorial.tutorial001.read_items" in ids and "minisrc.extra.extra_fn" in ids
    inner = next(n for n in g2["nodes"] if n["id"] == "mini.core.inner")
    assert inner["nested"] and inner["in_provided_graph"] and inner["qualname"] == "run.<locals>.inner"
    assert ("mini.core.run", "mini.core.inner") in {(e["source"], e["target"]) for e in g2["edges"] if e["type"] == "contains"}
    assert not [n for n in g2["nodes"] if n["kind"] == "provided_only"]
    # every edge endpoint is a node (no dangling `tests` edges from test_patch tests)
    assert all(e["source"] in ids and e["target"] in ids for e in g2["edges"])
    report = json.loads((results / "run_report.json").read_text())
    assert report["all_ok"]
    assert report["manifest"]["tasks_jsonl_sha256"] and report["manifest"]["graphs_files"] == 3


def test_classify_pytest():
    sys.path.insert(0, str(ROOT / "planA" / "a2"))
    from run_task import classify_pytest
    imp = classify_pytest(2, "_____ ERROR collecting tests/test_x.py _____\nE   ModuleNotFoundError: No module named 'dirty_equals'\n1 error in 0.2s")
    assert imp["run_class"] == "import_error" and imp["missing_modules"] == ["dirty_equals"]
    assert imp["collection_errors"] == ["tests/test_x.py"]
    assert classify_pytest(5, "no tests ran in 0.01s")["run_class"] == "no_tests"
    assert classify_pytest(1, "1 failed, 3 passed in 0.1s")["run_class"] == "clean"
    assert classify_pytest(None, "")["run_class"] == "timeout"
    assert classify_pytest(4, "ERROR: usage")["run_class"] == "crashed"


def test_flatten_qualname():
    from common import flatten_qualname, provided_node_id
    assert flatten_qualname("HTTPDigestAuth.build_digest_header.<locals>.md5_utf8") == "HTTPDigestAuth.md5_utf8"
    assert flatten_qualname("test_pretty.<locals>.Thing._repr_html_") == "Thing._repr_html_"
    assert flatten_qualname("a.<locals>.b.<locals>.c") == "c"
    assert flatten_qualname("Engine.run") == "Engine.run"
    assert provided_node_id("mini.core", "run.<locals>.inner") == "mini.core.inner"
    assert provided_node_id("mini.core", None) == "mini.core"


def test_check_f2p_statuses():
    sys.path.insert(0, str(ROOT / "planA" / "a2"))
    from analyze import check_f2p
    from common import Task
    tp = "+def test_a():\n+    pass\n+def test_b():\n+    pass\n+    def test_c(self):\n+        pass\n"
    t = Task(instance_id="x", repo="r/r", base_commit="c", test_patch=tp)
    pre = {"tests/t.py::test_a": "failed", "tests/t.py::test_b": "passed"}
    post = {"tests/t.py::test_a": "passed", "tests/t.py::test_b": "passed", "tests/t.py::TestX::test_c": "passed"}
    r = check_f2p(t, pre, post)
    assert r["n_f2p_expected"] == 3 and r["n_f2p_confirmed"] == 1 and r["n_f2p_pre_passed"] == 1
    assert r["n_f2p_uncollected_pre"] == 1 and r["f2p_uncollected_pre"] == ["tests/t.py::TestX::test_c"]
    assert r["harness_all_expected_pass_post"] and not r["f2p_all_confirmed"]
    post2 = dict(post, **{"tests/t.py::test_b": "failed"})
    assert check_f2p(t, pre, post2)["harness_all_expected_pass_post"] is False
    # a collection failure in the pre run with no collected tests at all
    r3 = check_f2p(t, {}, post)
    assert r3["n_f2p_uncollected_pre"] == 3 and r3["n_f2p_confirmed"] == 0 and r3["f2p_all_confirmed_or_uncollected"]


def test_entry_nodes():
    sys.path.insert(0, str(ROOT / "planA" / "a2"))
    from analyze import entry_nodes
    # test_a -> f -> g (g also called by h, which only runs in test_b); test_b -> h -> g; m is import-time
    tr = {"tests": ["tests/t.py::test_a", "tests/t.py::test_b[1]"],
          "functions": ["tests.t.test_a", "pkg.m.f", "tests.t.test_b", "pkg.m.g", "pkg.m.<module>", "pkg.m.h"],
          "per_test_exec": {"tests/t.py::test_a": [0, 1, 3], "tests/t.py::test_b[1]": [2, 5, 3]},
          "edges_direct": [[0, 1, [0]], [1, 3, [0]], [0, 4, [0]], [5, 3, [1]]],
          "edges_vialib": [[2, 5, [1]]]}
    mapped = [{"nid": "tests.t.test_a", "raw": "tests.t.test_a", "status": "node", "module_level": False},
              {"nid": "pkg.m.f", "raw": "pkg.m.f", "status": "node", "module_level": False},
              {"nid": "tests.t.test_b", "raw": "tests.t.test_b", "status": "node", "module_level": False},
              {"nid": "pkg.m.g", "raw": "pkg.m.g", "status": "node", "module_level": False},
              {"nid": "pkg.m", "raw": "pkg.m", "status": "node", "module_level": True},
              {"nid": "pkg.m.h", "raw": "pkg.m.h", "status": "missing", "module_level": False}]
    assert entry_nodes(["tests/t.py::test_a"], tr, mapped) == {"pkg.m.f"}          # g is called by f, not a root
    # h (the true root) is not a node (e.g. an absent async def): the walk passes through it, so g is the entry
    assert entry_nodes(["tests/t.py::test_b[1]"], tr, mapped) == {"pkg.m.g"}
    assert entry_nodes([], tr, mapped) == set() and entry_nodes(["tests/t.py::test_a"], None, mapped) == set()

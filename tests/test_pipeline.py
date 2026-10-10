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


def test_a1_outputs(results):
    tasks = {r["instance_id"]: r for r in read_csv(results / "a1" / "a1_tasks.csv")}
    assert set(tasks) == {"mini_1", "mini_2", "mini_3"}
    t1 = tasks["mini_1"]
    assert t1["graph_available"] == "True" and t1["snapshot_available"] == "True"
    assert "mini.core.compute" in json.loads(t1["gold_nodes"])
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
    graphs = read_csv(results / "a1" / "a1_graphs.csv")
    assert len(graphs) == 2
    g = graphs[0]
    assert json.loads(g["edge_types"]) == {"calls": 12} or json.loads(g["edge_types"]) == {"calls": 13}
    assert float(g["pct_containment_in_calls"]) > 0
    summ = json.loads((results / "a1" / "a1_summary_shard0of2.json").read_text())
    assert "async_audit" in summ


def test_a2_outputs(results):
    st = json.loads((results / "a2_runs" / "mini_1" / "status.json").read_text())
    assert st["done"], st
    assert st["pre"]["exit"] not in (None, 0) and st["post"]["exit"] == 0, st
    tr = json.loads((results / "a2_runs" / "mini_1" / "trace_pre.json").read_text())
    assert tr["outcomes"]["tests/test_core.py::test_compute"] == "failed"
    names = set(tr["functions"])
    assert "mini.core.compute" in names and "mini.core.run.<locals>.inner" in names
    assert "mini.util.gen_values" in names
    rows = {r["instance_id"]: r for r in read_csv(results / "a2" / "a2_tasks.csv")}
    r1 = rows["mini_1"]
    assert r1["n_fail_to_pass"] == "2"      # test_compute and test_run both flip
    assert r1["gold_any_dyn_reachable"] == "True" and r1["gold_any_static_reachable"] == "True"
    assert float(r1["recall_direct"]) > 0 and float(r1["observed_precision"]) > 0
    # generator resumes are deduplicated: the gen_values edge appears once, from Engine.total
    dyn = [(e[0], e[1]) for e in tr["edges_direct"]]
    fn = tr["functions"]
    gen_callers = {fn[c] for c, e in dyn if fn[e] == "mini.util.gen_values"}
    assert gen_callers == {"mini.core.Engine.total"}
    r2 = rows["mini_2"]
    assert r2["gold_any_dyn_reachable"] == "False"  # async gold fn is not a graph node -> not representable
    assert json.loads(r2["map_status"]).get("missing", 0) >= 1
    r3 = rows["mini_3"]
    # Engine.__init__ executed -> collapsed to the gold class Engine; new test function is not a base-graph node
    assert r3["gold_any_dyn_reachable"] == "True" and r3["gold_any_static_reachable"] == "False"
    assert json.loads(r3["f2p_source_levels"]).get("function", 0) == 0


def test_a4_a5_a6(results):
    rel = read_csv(results / "a4" / "a4_relations.csv")
    assert {r["relation"] for r in rel} >= {"calls", "same_module", "random", "containment"}
    a5 = {r["instance_id"]: r for r in read_csv(results / "a5" / "a5_tasks.csv")}
    assert a5["mini_1"]["a_harness_hit10_node"] == "True"
    assert a5["mini_1"]["b_bm25_node_rank"] != ""
    prof = {r["instance_id"]: r for r in read_csv(results / "a6" / "task_profile.csv")}
    assert prof["mini_1"]["cell"].startswith("d0|")
    g2 = json.loads(next((results / "a6" / "graphs_v2").glob("*.json")).read_text())
    types = {e["type"] for e in g2["edges"]}
    assert {"contains", "imports", "calls_static", "calls_dynamic", "tests"} <= types
    ids = {n["id"] for n in g2["nodes"]}
    assert "mini.core.acompute" in ids and "tutorial.tutorial001.read_items" in ids and "minisrc.extra.extra_fn" in ids
    report = json.loads((results / "run_report.json").read_text())
    assert report["all_ok"]

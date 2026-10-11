#!/usr/bin/env python3
"""Check every quantitative claim in paper 1 ("Is the Map Right?") against the committed result files.

Usage (from planA_results/):  python3 papers/verify_paper1_numbers.py
Standard library only. Prints one line per claim: PASS/FAIL, the paper's value, the value recomputed
from the files, and the file it came from. Exit code 1 if any claim fails.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "vm-a2" / "results"
M = ROOT / "vm-main" / "results"
J = lambda p: json.loads(Path(p).read_text())  # noqa: E731
from decimal import ROUND_HALF_UP, Decimal
pct = lambda x: float((Decimal(str(x)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))  # noqa: E731

a1 = J(R / "a1_v2/a1_summary.json")
a7 = J(R / "a7/a7_summary.json")["per_repo"]
a2 = J(R / "a2_v3/a2_summary.json")
u = a2["usable_tasks"]
a8 = J(R / "a8/a8_summary.json")["nontest_edges"]
a3 = J(R / "a3/a3_summary.json")["models"]
a3g = J(R / "a3_gemma2/a3_summary.json")["models"]
a4 = J(M / "a4/a4_summary.json")
a5 = J(R / "a5_v3/a5_summary.json")["overall_all_tasks"]
fig = J(ROOT / "papers/figures/figure_data.json")
gold = [json.loads(l) for l in open(R / "a1_v2/a1_gold_symbols.jsonl", encoding="utf-8")]
tasks1 = list(csv.DictReader(open(R / "a1_v2/a1_tasks.csv", newline="", encoding="utf-8")))
graphs1 = list(csv.DictReader(open(R / "a1_v2/a1_graphs.csv", newline="", encoding="utf-8")))
vstats = list(csv.DictReader(open(R / "a3/a3_vectors_stats.csv", newline="", encoding="utf-8")))
knn = list(csv.DictReader(open(M / "a4/a4_knn.csv", newline="", encoding="utf-8")))
a3pm = list(csv.DictReader(open(R / "a3/a3_per_model.csv", newline="", encoding="utf-8"))) + \
    list(csv.DictReader(open(R / "a3_gemma2/a3_per_model.csv", newline="", encoding="utf-8")))

CLAIMS: list[tuple[str, object, object, str]] = []


def claim(section, text, paper, value, src):
    CLAIMS.append((f"{section} {text}", paper, value, src))


om, oms = a1["omission_direct_files"], a1["omission_direct_source_files"]
mr = a1["misresolution"]
mrr = a1["misresolution_per_repo"]
how = lambda pop, keys: pct(sum(a8[pop]["unique_edges"]["how"].get(k, {}).get("count", 0) for k in keys) / a8[pop]["unique_edges"]["n"])  # noqa: E731
EXPLICIT = ["call_direct", "call_self", "call_super", "call_attr_other", "instantiation", "call_in_nested_scope"]
es = fig["p1_edge_structure"]
hops = u["gold_min_hops_from_entry_hist"]

# ---- abstract / section 3
claim("Abs", "graphs", 127, a1["n_graphs"], "a1_v2/a1_summary.json")
claim("Abs", "tasks", 129, a1["n_tasks"], "a1_v2/a1_summary.json")
claim("Abs", "async defs present / total", "0 of 45,492", f"{om['async_function']['present']} of {om['async_function']['total']:,}", "a1_v2/a1_summary.json")
claim("Abs", "gold symbols that are async (none a node, see 5.1)", "81 of 711", f"{a1['gold_async']} of {a1['gold_symbols']}", "a1_v2/a1_summary.json")
fa = [t for t in tasks1 if t["repo"] == "fastapi/fastapi"]
claim("Abs", "fastapi tasks with an async gold symbol", "31%", f"{round(100 * sum(1 for t in fa if int(t['n_gold_async'] or 0) > 0) / len(fa))}%", "a1_v2/a1_tasks.csv")
claim("Abs", "absent-async bare-name query -> wrong node", "65%", f"{round(100 * mr['async_function|absent|q_name']['wrong_node_rate']['p'])}%", "a1_v2/a1_summary.json")
claim("Abs", "representable recall", "37%", f"{round(100 * u['all_any']['recall_representable']['ratio'])}%", "a2_v3/a2_summary.json")
claim("Abs", "observed precision", "21%", f"{round(100 * u['observed_precision']['ratio'])}%", "a2_v3/a2_summary.json")
claim("Abs", "gold reachable from entry", "69 of 72", f"{u['tasks_gold_all_executed_reachable_entry']['k']} of {u['tasks_gold_all_executed_reachable_entry']['n']}", "a2_v3/a2_summary.json")
claim("S3", "calls edges, all graphs", "452,588", f"{a1['edge_types_union']['calls']:,}", "a1_v2/a1_summary.json")
claim("S3", "edge types", "calls only", "calls only" if set(a1["edge_types_union"]) == {"calls"} else str(set(a1["edge_types_union"])), "a1_v2/a1_summary.json")
claim("S3", "vector dimension", 256, int(vstats[0]["dim"]), "a3/a3_vectors_stats.csv")

# ---- section 4
c = a2["coverage"]["per_repo"]
claim("S4", "usable tasks", 91, a2["coverage"]["usable"]["k"], "a2_v3/a2_summary.json")
claim("S4", "usable fastapi / rich / requests", "45 / 39 / 7",
      f"{c['fastapi/fastapi']['usable']} / {c['Textualize/rich']['usable']} / {c['psf/requests']['usable']}", "a2_v3/a2_summary.json")

# ---- 5.1
for k, lab in (("function", "sync functions"), ("class", "classes"), ("nested", "nested definitions"), ("async_function", "async functions")):
    claim("5.1/Fig1", f"{lab} present", "", f"{om[k]['present']:,} of {om[k]['total']:,} ({pct(om[k]['present'] / om[k]['total'])}%)", "a1_v2/a1_summary.json")
claim("5.1", "non-test source: sync present", "100%", f"{pct(oms['function']['present'] / oms['function']['total']):.0f}%", "a1_v2/a1_summary.json")
claim("5.1", "non-test source: async present", "0 of 28,113", f"{oms['async_function']['present']} of {oms['async_function']['total']:,}", "a1_v2/a1_summary.json")
claim("5.1", "fastapi Config classes listed without model (class-dropped ids)", "1,683",
      f"{a7['fastapi/fastapi']['defs'].get('class|present_class_dropped', 0) + a7['fastapi/fastapi']['defs'].get('nested|present_class_dropped', 0):,}", "a7/a7_summary.json")
claim("5.1", "requests: missing nested defs (methods of classes inside test functions)", 340, a7["psf/requests"]["defs"].get("nested|absent_nested"), "a7/a7_summary.json")
claim("5.1", "missing nested defs that are NOT async, fastapi / rich (rich: 'a single class', once per graph)", "0 / 47",
      f'{a7["fastapi/fastapi"]["defs"].get("nested|absent_nested", 0)} / {a7["Textualize/rich"]["defs"].get("nested|absent_nested", 0)}',
      "a7/a7_summary.json; rich's 47 = one test class, test_inspect.py ...<locals>.Something.Thing, absent in each of 47 graphs")
async_present = [g for g in gold if g["kind"] == "async_function" and g["present_in_graph"]]
claim("5.1", "async gold symbols counted present only through an id collision", "1 (FastAPI.setup.<locals>.openapi)",
      f"{len(async_present)} ({async_present[0]['qualname'] if async_present else '-'})", "a1_v2/a1_gold_symbols.jsonl")
gp = a1["gold_symbol_present_rate"]
claim("5.1", "gold symbols that are nodes", "470 of 711 (66.1%, CI 62.6 to 69.5)", f"{gp['k']} of {gp['n']} ({pct(gp['p'])}%, CI {pct(gp['lo'])} to {pct(gp['hi'])})", "a1_v2/a1_summary.json")
claim("5.1", "tasks with an async gold symbol (fastapi)", "22 (21)", f"{a1['task_rates']['has_gold_async']['k']} ({a1['task_rates_per_repo']['fastapi/fastapi']['has_gold_async']['k']})", "a1_v2/a1_summary.json")
claim("5.1", "usable tasks with an executed gold function that has no node", 3, u["tasks_gold_absent_but_executed"]["k"], "a2_v3/a2_summary.json")

# ---- 5.2
x = mr["async_function|absent|q_name"]
claim("5.2/Fig2", "absent async: wrong node, n", "65.4% of 56,487", f"{pct(x['wrong_node_rate']['p'])}% of {x['n']:,}", "a1_v2/a1_summary.json")
claim("5.2", "absent async: wrong node is a test", "26,851", f"{x['wrong_node_is_test']:,}", "a1_v2/a1_summary.json")
claim("5.2", "absent async: no match", "34.6%", f"{pct(x['outcomes']['no_match'] / x['n'])}%", "a1_v2/a1_summary.json")
claim("5.2", "present sync: wrong node", "41.2%", f"{pct(mr['function|present|q_name']['wrong_node_rate']['p'])}%", "a1_v2/a1_summary.json")
claim("5.2", "present sync wrong node: fastapi / httpx", "46.6% / 62.0%",
      f"{pct(mrr['fastapi/fastapi']['function|present|q_name']['wrong_node_rate']['p'])}% / {pct(mrr['encode/httpx']['function|present|q_name']['wrong_node_rate']['p'])}%", "a1_v2/a1_summary.json")
claim("5.2", "present class: wrong node", "47.9%", f"{pct(mr['class|present|q_name']['wrong_node_rate']['p'])}%", "a1_v2/a1_summary.json")
claim("5.2/Fig2", "query populations (sync, class, async)", "287,841 / 62,552 / 56,487",
      f"{mr['function|present|q_name']['n']:,} / {mr['class|present|q_name']['n']:,} / {x['n']:,}", "a1_v2/a1_summary.json")
claim("5.2", "tasks whose gold query lands on the wrong node", "21 of 129", f"{a1['task_rates']['gold_query_wrong_node']['k']} of {a1['task_rates']['gold_query_wrong_node']['n']}", "a1_v2/a1_summary.json")

# ---- 5.3 / Table 1
def ci(d):
    return f"{pct(d['ratio'])}% ({int(d['num']):,} of {int(d['den']):,}), CI {pct(d['lo'])} to {pct(d['hi'])}"


claim("Tab1", "representable recall", "37.1% (8,032 of 21,653), CI 34.8 to 40.2", ci(u["all_any"]["recall_representable"]), "a2_v3/a2_summary.json")
claim("Tab1", "agent-facing recall", "31.8% (8,032 of 25,267), CI 29.4 to 34.8", ci(u["all_any"]["recall_agent_facing"]), "a2_v3/a2_summary.json")
claim("Tab1", "gold-incident recall", "56.3% (604 of 1,073), CI 39.3 to 68.4", ci(u["gold_any"]["recall_representable"]), "a2_v3/a2_summary.json")
claim("Tab1", "observed precision", "21.4% (8,013 of 37,433), CI 20.0 to 22.8", ci(u["observed_precision"]), "a2_v3/a2_summary.json")
ms = u["all_any"]["miss_share_by_cause"]
claim("5.3", "miss shares: between nodes / sync / nested / async", "79 / 11 / 6 / 4",
      " / ".join(str(round(100 * ms[k]["ratio"])) for k in ("missed_between_nodes", "absent_sync", "absent_nested", "absent_async")), "a2_v3/a2_summary.json")
pr = a2["per_repo_usable"]
claim("5.3/Fig3", "recall fastapi / rich / requests", "65.6 / 30.5 / 34.7",
      " / ".join(str(pct(pr[k]["all_any"]["recall_representable"]["ratio"])) for k in ("fastapi/fastapi", "Textualize/rich", "psf/requests")), "a2_v3/a2_summary.json")
claim("5.3/Fig3", "precision fastapi / requests", "16.5 / 34.3",
      f"{pct(pr['fastapi/fastapi']['observed_precision']['ratio'])} / {pct(pr['psf/requests']['observed_precision']['ratio'])}", "a2_v3/a2_summary.json")
claim("5.3", "non-test nodes executed", "16.7%", f"{pct(u['nontest_node_coverage']['ratio'])}%", "a2_v3/a2_summary.json")

# ---- 5.4 / Figs 4-5
claim("5.4/Fig4", "edge populations captured / unexercised / missed", "7,640 / 35,017 / 12,231",
      " / ".join(f"{a8[p]['unique_edges']['n']:,}" for p in ("captured", "unconfirmed", "missed")), "a8/a8_summary.json (nontest_edges)")
claim("5.4", "captured: explicit / bare / instantiation", "88.6 / 55.3 / 19.0",
      f"{pct(a8['captured']['explicit_callsite_share']['p'])} / {how('captured', ['call_direct'])} / {how('captured', ['instantiation'])}", "a8/a8_summary.json")
claim("5.4", "missed: obj / self / reference / dunder", "23.4 / 15.9 / 28.3 / 17.3",
      " / ".join(str(how("missed", [k])) for k in ("call_attr_other", "call_self", "referenced", "implicit_dunder")), "a8/a8_summary.json")
claim("5.4", "unexercised: no mention / string or comment only", "56.1 / 26.7",
      f"{how('unconfirmed', ['no_reference'])} / {how('unconfirmed', ['text_only'])}", "a8/a8_summary.json")
n_unc, n_mis = es["unconfirmed"]["n"], es["missed"]["n"]
claim("5.4/Fig5", "unexercised: own class / subclass / base class", "22.3 / 13.8 / 13.3",
      " / ".join(str(pct(es["unconfirmed"].get(k, 0) / n_unc)) for k in ("Method to its own class", "Class to a subclass", "Class to its base class")),
      "a8/a8_edges.csv + a6_v3/graphs_v2 (papers/make_figures.py p1_edge_structure)")
claim("5.4/Fig5", "missed: same-class sibling methods", "45.5", pct(es["missed"].get("Method to a sibling method", 0) / n_mis), "same")
cont = [float(g["pct_containment_in_calls"]) for g in graphs1 if g.get("pct_containment_in_calls")]
non_httpx = [float(g["pct_containment_in_calls"]) for g in graphs1 if g.get("pct_containment_in_calls") and "httpx" not in g["repo"]]
httpx_ = [float(g["pct_containment_in_calls"]) for g in graphs1 if g.get("pct_containment_in_calls") and "httpx" in g["repo"]]
claim("5.4", "calls edges per graph pointing to own member: other 126 graphs / single httpx graph", "8 to 21% / 30%",
      f"{min(non_httpx):.0f} to {max(non_httpx):.0f}% / {', '.join(f'{x:.0f}' for x in httpx_)}%",
      "a1_v2/a1_graphs.csv; fastapi 7.9-9.1, rich 15.1-15.6, requests 20.6-20.9, httpx (1 graph) 29.8")

# ---- 5.5 / Fig 6
claim("5.5", "usable tasks with an entry set", "86 of 91", f"{u['tasks_entry_sources_defined']['k']} of {u['tasks_entry_sources_defined']['n']}", "a2_v3/a2_summary.json")
r_ = u["tasks_gold_all_executed_reachable_entry"]
claim("5.5", "all executed gold reachable", "69 of 72 (95.8%, CI 88.5 to 98.6)", f"{r_['k']} of {r_['n']} ({pct(r_['p'])}%, CI {pct(r_['lo'])} to {pct(r_['hi'])})", "a2_v3/a2_summary.json")
claim("5.5", "reachable only against edge direction", 0, u["tasks_gold_reachable_entry_undirected_only"]["k"], "a2_v3/a2_summary.json")
claim("5.5/Fig6", "hops 0..7 / unreachable", "32 9 13 9 2 2 1 1 / 3", " ".join(str(hops.get(str(i), 0)) for i in range(8)) + f" / {hops.get('None', 0)}", "a2_v3/a2_summary.json")
claim("5.5", "gold is an entry / within three hops", "32 / 63", f"{hops.get('0', 0)} / {sum(hops.get(str(i), 0) for i in range(4))}", "a2_v3/a2_summary.json")

# ---- 5.6 / Figs 7-8
best = lambda m, src: (lambda r: (float(r["cos_mean"]), r))(max((r for r in src if r["model"] == m and r["cos_mean"]), key=lambda r: float(r["cos_mean"])))  # noqa: E731
b300 = best("google/embeddinggemma-300m", a3pm)
claim("5.6", "best cosine EmbeddingGemma 300m (variant, prompt)", "0.073 (name, document)", f"{b300[0]:.3f} ({b300[1]['variant']}, {b300[1]['prompt']})", "a3/a3_per_model.csv")
claim("5.6", "best cosine EmbeddingGemma 2 / MiniLM", "0.015 / 0.036",
      f"{best('google/embeddinggemma-2', a3pm)[0]:.3f} / {best('sentence-transformers/all-MiniLM-L6-v2', a3pm)[0]:.3f}", "a3/, a3_gemma2/ a3_per_model.csv")
mx = lambda pre: max(float(r['spearman_pairwise']) for r in a3pm if r['spearman_pairwise'] and r['model'].startswith(pre))  # noqa: E731
claim("5.6", "max pairwise rank correlation: either EmbeddingGemma / MiniLM control", "0.17 / 0.21", f"{mx('google/embeddinggemma'):.2f} / {mx('sentence-transformers'):.2f}",
      "a3 per_model.csv; per model: EmbeddingGemma 300m 0.17, EmbeddingGemma 2 0.14, MiniLM control 0.21")
claim("5.6", "cos to mean vector range", "0.87 to 0.95", f"{min(float(r['cos_to_mean_vector_mean']) for r in vstats):.2f} to {max(float(r['cos_to_mean_vector_mean']) for r in vstats):.2f}", "a3/a3_vectors_stats.csv")
claim("5.6", "PC1 share range", "44% to 75%", f"{min(float(r['pc1_explained']) for r in vstats) * 100:.0f}% to {max(float(r['pc1_explained']) for r in vstats) * 100:.0f}%", "a3/a3_vectors_stats.csv")
claim("5.6", "unit-normalised", False, any(r["unit_norm"] == "True" for r in vstats), "a3/a3_vectors_stats.csv")
claim("5.6", "top-1 neighbour cosine (mean)", "0.999", f"{sum(float(r['top1_cos_mean']) for r in knn) / len(knn):.3f}", "vm-main a4/a4_knn.csv")
claim("Fig7a", "AUROC containment / calls / co-change / same module / name overlap", "0.90 / 0.90 / 0.81 / 0.68 / 0.68",
      " / ".join(f"{a4[k]['auroc_avg']:.2f}" for k in ("containment", "calls", "gold_cochange", "same_module", "name_overlap")), "vm-main a4/a4_summary.json")
kn = fig["p1_embeddings"]["knn"]
claim("Fig7b", "test share of 10-NN, min to max over repos", "45% to 67%", f"{min(v[0] for v in kn.values()) * 100:.0f}% to {max(v[0] for v in kn.values()) * 100:.0f}%", "vm-main a4/a4_knn.csv")
claim("Fig7b", "call-neighbour share of 10-NN, min to max", "2% to 6%", f"{min(v[3] for v in kn.values()) * 100:.0f}% to {max(v[3] for v in kn.values()) * 100:.0f}%", "vm-main a4/a4_knn.csv")
for name, k in (("identifiers", "a0_anchors"), ("harness", "a_harness"), ("harness+1hop", "d_harness_1hop"), ("BM25 node text", "b_bm25"), ("BM25 source", "e_bm25_snapshot")):
    claim("Fig8", f"{name}: node@10 / file@5 / empty", "", f"{a5[k]['hit10_node']['k']} / {a5[k]['hit5_file']['k']} / {a5[k]['empty']['k']}", "a5_v3/a5_summary.json (overall_all_tasks)")
mc = a5["paired_mcnemar"]["a_harness_vs_e_bm25_snapshot"]["hit5_file"]
claim("5.6", "file@5 BM25-source-only / harness-only, McNemar p", "47 / 5, p < 0.0001", f"{mc['b_only']} / {mc['a_only']}, p {'< 0.0001' if mc['p_exact'] < 1e-4 else mc['p_exact']}", "a5_v3/a5_summary.json")
claim("5.6", "cosine step / 1-hop step changes (file@5)", "3 / 2",
      f"{a5['paired_mcnemar']['a0_anchors_vs_a_harness']['hit5_file']['b_only']} / {a5['paired_mcnemar']['a_harness_vs_d_harness_1hop']['hit5_file']['b_only']}", "a5_v3/a5_summary.json")
claim("5.6", "harness mode returns nothing", "54 of 129", f"{a5['a_harness']['empty']['k']} of {a5['a_harness']['empty']['n']}", "a5_v3/a5_summary.json")

# ---- 7
rc = a2["coverage"]["run_class"]
claim("S7", "collection errors / import errors / no-test tasks", "9 / 2 / 4", f"{rc.get('collection_error', 0)} / {rc.get('import_error', 0)} / {rc.get('no_tests', 0)}", "a2_v3/a2_summary.json")
claim("S7", "hand-review sample size", 80, sum(1 for _ in csv.DictReader(open(R / "a8/a8_sample.csv", newline="", encoding="utf-8"))), "a8/a8_sample.csv")


def same(paper, value):
    if paper == "" or paper is None:
        return None          # value listed for reference (shown in a figure)
    return str(paper).replace(",", "") == str(value).replace(",", "")


fails = 0
for text, paper, value, src in CLAIMS:
    s = same(paper, value)
    tag = "INFO" if s is None else ("PASS" if s else "FAIL")
    fails += tag == "FAIL"
    print(f"{tag:4}  {text}\n      paper: {paper}\n      files: {value}   [{src}]")
print(f"\n{sum(1 for c_ in CLAIMS if same(c_[1], c_[2]))} PASS, {fails} FAIL, {sum(1 for c_ in CLAIMS if same(c_[1], c_[2]) is None)} INFO")
sys.exit(1 if fails else 0)

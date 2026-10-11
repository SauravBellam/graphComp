#!/usr/bin/env python3
"""Build every figure of paper 1 and paper 2 from the committed result files.

Usage (from planA_results/):  python papers/make_figures.py
Writes papers/figures/*.pdf and papers/figures/figure_data.json (the numbers each figure plots).
Reads only files under planA_results/ (vm-a2/results/..., vm-main/results/...).
Palette: the dataviz reference palette, light mode, slots 1-5 (validated: CVD dE >= 9.1 adjacent,
normal-vision dE >= 19.6); slots below 3:1 contrast carry direct numeric labels on every mark.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "vm-a2" / "results"
M = ROOT / "vm-main" / "results"
OUT = ROOT / "papers" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
OTHER = "#c3c2b7"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
DATA: dict = {}


def load(p: Path):
    return json.loads(p.read_text())


def hgrid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def stacked_h(ax, labels, parts, colors, names, total_fmt=None, min_label=0.05):
    """100% horizontal stacked bars with a 1.5pt white gap and direct labels."""
    left = [0.0] * len(labels)
    for j, (name, col) in enumerate(zip(names, colors)):
        vals = [p[j] for p in parts]
        ax.barh(labels, vals, left=left, color=col, edgecolor=SURF, linewidth=1.5, height=0.62, label=name)
        for i, v in enumerate(vals):
            if v >= min_label:
                ax.text(left[i] + v / 2, i, f"{v * 100:.0f}%", ha="center", va="center", fontsize=7.5,
                        color=INK if col in (S3, S4, S5, OTHER) else "white")
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.invert_yaxis()


# ---------------------------------------------------------------- paper 1
def p1_presence():
    a1 = load(R / "a1_v2" / "a1_summary.json")["omission_direct_files"]
    rows = [("Sync function", a1["function"]), ("Class", a1["class"]), ("Nested definition", a1["nested"]),
            ("Async function", a1["async_function"])]
    labels = [r[0] for r in rows]
    rate = [r[1]["present"] / r[1]["total"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 1.9))
    bars = ax.barh(labels, rate, color=[S1, S1, S1, S2], height=0.6)
    for b, r, (_, x) in zip(bars, rate, rows):
        ax.text(max(r, 0) + 0.01, b.get_y() + b.get_height() / 2,
                f"{r * 100:.1f}%  ({x['present']:,} of {x['total']:,})", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.invert_yaxis()
    ax.set_xlabel("Definitions present as graph nodes (all files, 127 graphs)")
    hgrid(ax)
    fig.savefig(OUT / "p1_presence.pdf")
    DATA["p1_presence"] = {lab: x for lab, (_, x) in zip(labels, rows)}


def p1_resolver():
    m = load(R / "a1_v2" / "a1_summary.json")["misresolution"]
    cats = [("Sync function, present", m["function|present|q_name"]), ("Class, present", m["class|present|q_name"]),
            ("Async function, absent", m["async_function|absent|q_name"])]
    names = ["Returns the definition", "Wrong node: non-test", "Wrong node: test", "No match"]
    parts, raw = [], {}
    for lab, x in cats:
        n = x["n"]
        wrong = x["outcomes"].get("wrong_node", 0)
        test = x["wrong_node_is_test"]
        vals = [x["outcomes"].get("self", 0), wrong - test, test, x["outcomes"].get("no_match", 0)]
        parts.append([v / n for v in vals])
        raw[lab] = dict(zip(names, vals), n=n)
    fig, ax = plt.subplots(figsize=(5.2, 1.75))
    stacked_h(ax, [c[0] for c in cats], parts, [S1, S2, S4, OTHER], names)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.35, -0.22), handlelength=1.0, columnspacing=1.2)
    ax.set_xlabel("")
    fig.savefig(OUT / "p1_resolver.pdf")
    DATA["p1_resolver"] = raw


def p1_recall_precision():
    a2 = load(R / "a2_v3" / "a2_summary.json")
    groups = [("All usable\n(91 tasks)", a2["usable_tasks"])] + [
        (f"{k.split('/')[1]}\n({v['n_tasks']} tasks)", v) for k, v in a2["per_repo_usable"].items()
        if v.get("n_tasks") and k != "encode/httpx"]
    order = {"All": 0, "fastapi": 1, "rich": 2, "requests": 3}
    groups.sort(key=lambda g: order.get(g[0].split("\n")[0].split(" ")[0], 9))
    rec = [g[1]["all_any"]["recall_representable"] for g in groups]
    pre = [g[1]["observed_precision"] for g in groups]
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    x = range(len(groups))
    w = 0.36
    for off, series, col, name in ((-w / 2 - 0.01, rec, S1, "Representable recall"), (w / 2 + 0.01, pre, S2, "Observed precision")):
        ys = [s["ratio"] for s in series]
        err = [[s["ratio"] - s["lo"] for s in series], [s["hi"] - s["ratio"] for s in series]]
        ax.bar([i + off for i in x], ys, w, color=col, label=name, yerr=err, ecolor=INK2, capsize=2, error_kw={"linewidth": 0.8})
        for i, (y, s_) in enumerate(zip(ys, series)):
            ax.text(i + off, s_["hi"] + 0.025, f"{y * 100:.0f}%", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    hgrid(ax, "y")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "p1_recall_precision.pdf")
    DATA["p1_recall_precision"] = {g[0].replace("\n", " "): {"recall": r, "precision": p} for g, r, p in zip(groups, rec, pre)}


def p1_edge_semantics():
    a8 = load(R / "a8" / "a8_summary.json")["nontest_edges"]
    groups = {"Explicit call": ["call_direct", "call_self", "call_super", "call_attr_other", "instantiation", "call_in_nested_scope"],
              "Reference, no call": ["referenced"], "Implicit dunder": ["implicit_dunder"],
              "String or comment only": ["text_only"], "No mention": ["no_reference"]}
    pops = [("Captured\n(graph has it, ran)", "captured"), ("Unexercised static\n(graph has it, never ran)", "unconfirmed"),
            ("Missed\n(ran, graph lacks it)", "missed")]
    parts, raw = [], {}
    for lab, key in pops:
        how = a8[key]["unique_edges"]["how"]
        n = a8[key]["unique_edges"]["n"]
        vals = [sum(how.get(h, {}).get("count", 0) for h in hs) for hs in groups.values()]
        parts.append([v / n for v in vals])
        raw[key] = dict(zip(groups, vals), n=n)
    fig, ax = plt.subplots(figsize=(5.4, 1.95))
    stacked_h(ax, [p[0] for p in pops], parts, [S1, S3, S5, S4, OTHER], list(groups))
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.38, -0.2), handlelength=1.0, columnspacing=1.0)
    fig.savefig(OUT / "p1_edge_semantics.pdf")
    DATA["p1_edge_semantics"] = raw


def p1_edge_structure():
    """Structural relation of every unexercised non-test static edge (full population, not a sample)."""
    rows = [r for r in csv.DictReader(open(R / "a8" / "a8_edges.csv", newline="", encoding="utf-8"))
            if r["nontest"] == "True"]
    cache: dict = {}

    def graph(gk):
        if gk not in cache:
            d = load(R / "a6_v3" / "graphs_v2" / f"{gk}.json")
            nodes = {n["id"]: n for n in d["nodes"]}
            inh = defaultdict(set)
            for e in d["edges"]:
                if e["type"] == "inherits":
                    inh[e["source"]].add(e["target"])
            cache.clear() if len(cache) > 6 else None
            cache[gk] = (nodes, inh)
        return cache[gk]

    def cls_of(nodes, nid):
        parts = nid.split(".")
        for k in range(len(parts) - 1, 0, -1):
            p = ".".join(parts[:k])
            if p in nodes and nodes[p].get("kind") == "class":
                return p
        return None

    def classify(r):
        nodes, inh = graph(r["graph_key"])
        s, d = r["src"], r["dst"]
        sc, dk, sk = cls_of(nodes, s), nodes.get(d, {}).get("kind"), nodes.get(s, {}).get("kind")
        if dk == "class" and sc == d:
            return "Method to its own class"
        if dk == "class" and sk == "class" and d in inh.get(s, ()):
            return "Class to its base class"
        if dk == "class" and sk == "class" and s in inh.get(d, ()):
            return "Class to a subclass"
        if dk == "class" and sc and d in inh.get(sc, ()):
            return "Method to a base class"
        if sc and d.startswith(sc + "."):
            return "Method to a sibling method"
        if dk == "class":
            return "To another class"
        return "Function to function, other"

    res = {}
    for pop in ("unconfirmed", "missed", "captured"):
        rs = sorted((r for r in rows if r["population"] == pop), key=lambda r: r["graph_key"])
        res[pop] = Counter(classify(r) for r in rs)
        res[pop]["n"] = len(rs)
    order = ["Method to its own class", "Class to a subclass", "Class to its base class", "Method to a base class",
             "To another class", "Method to a sibling method", "Function to function, other"]
    pops = [("Captured", "captured"), ("Unexercised static", "unconfirmed"), ("Missed", "missed")]
    cols = [S2, S4, S5, S3, S1, "#6250d6", OTHER]
    parts = [[res[k][o] / res[k]["n"] for o in order] for _, k in pops]
    fig, ax = plt.subplots(figsize=(5.4, 1.9))
    stacked_h(ax, [p[0] for p in pops], parts, cols, order, min_label=0.06)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.4, -0.2), handlelength=1.0, columnspacing=0.9, fontsize=7.5)
    fig.savefig(OUT / "p1_edge_structure.pdf")
    DATA["p1_edge_structure"] = {k: dict(v) for k, v in res.items()}


def p1_hops():
    h = load(R / "a2_v3" / "a2_summary.json")["usable_tasks"]["gold_min_hops_from_entry_hist"]
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "none"]
    vals = [h.get(k if k != "none" else "None", 0) for k in labels]
    fig, ax = plt.subplots(figsize=(5.2, 1.9))
    bars = ax.bar(labels, vals, color=[S1] * 8 + [S2], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, str(v), ha="center", fontsize=7.5, color=INK)
    ax.set_xlabel("Min. directed calls hops, test entry point to gold function (none = unreachable)")
    ax.set_ylabel("Tasks")
    ax.set_ylim(0, max(vals) * 1.18)
    hgrid(ax, "y")
    fig.savefig(OUT / "p1_hops.pdf")
    DATA["p1_hops"] = dict(zip([l if l != "none" else "unreachable" for l in labels], vals))


def p1_embeddings():
    a4 = load(M / "a4" / "a4_summary.json")
    rels = [("Containment", "containment"), ("Call-adjacent", "calls"), ("Gold co-change", "gold_cochange"),
            ("Same module", "same_module"), ("Name overlap", "name_overlap")]
    knn = list(csv.DictReader(open(M / "a4" / "a4_knn.csv", newline="", encoding="utf-8")))
    by = defaultdict(list)
    for r in knn:
        by[r["repo"]].append(r)
    repos = [("fastapi", "fastapi/fastapi"), ("requests", "psf/requests"), ("rich", "Textualize/rich")]
    comp_names = ["Test nodes", "Same module", "Name overlap", "Call neighbours"]
    comp_keys = ["top10_share_test_node", "top10_share_same_module", "top10_share_name_overlap", "top10_share_call_neighbor"]
    comp = {lab: [sum(float(r[k]) for r in by[full]) / len(by[full]) for k in comp_keys] for lab, full in repos}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw={"width_ratios": [1.0, 1.1]})
    vals = [a4[k]["auroc_avg"] for _, k in rels]
    bars = a.barh([r[0] for r in rels], vals, color=S1, height=0.6)
    for bb, v in zip(bars, vals):
        a.text(v + 0.01, bb.get_y() + bb.get_height() / 2, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
    a.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    a.text(0.51, -0.62, "chance", fontsize=7, color=INK2)
    a.set_xlim(0.4, 1.0)
    a.set_ylim(4.5, -0.8)
    a.set_xlabel("AUROC, pair type vs random pairs")
    a.set_title("(a) What cosine similarity tracks", loc="left")
    hgrid(a)
    x = range(len(repos))
    w = 0.19
    for j, (name, col) in enumerate(zip(comp_names, [S2, S1, S3, S4])):
        ys = [comp[lab][j] for lab, _ in repos]
        b.bar([i + (j - 1.5) * (w + 0.015) for i in x], ys, w, color=col, label=name)
        for i, y in enumerate(ys):
            b.text(i + (j - 1.5) * (w + 0.015), y + 0.015, f"{y * 100:.0f}", ha="center", fontsize=6.5, color=INK)
    b.set_xticks(list(x))
    b.set_xticklabels([r[0] for r in repos])
    b.set_ylim(0, 0.8)
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    b.set_title("(b) Who the 10 nearest neighbours are", loc="left")
    b.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, handlelength=0.9, columnspacing=0.8)
    hgrid(b, "y")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "p1_embeddings.pdf")
    DATA["p1_embeddings"] = {"auroc": dict(zip([r[0] for r in rels], vals)), "knn": comp}


def p1_retrieval():
    a5 = load(R / "a5_v3" / "a5_summary.json")["overall_all_tasks"]
    modes = [("Issue identifiers\nresolved", "a0_anchors"), ("Harness:\n+ cosine", "a_harness"),
             ("Harness\n+ 1-hop", "d_harness_1hop"), ("BM25 over\nnode text", "b_bm25"), ("BM25 over\nsource files", "e_bm25_snapshot")]
    h10 = [a5[k]["hit10_node"]["k"] for _, k in modes]
    h5 = [a5[k]["hit5_file"]["k"] for _, k in modes]
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    x = range(len(modes))
    w = 0.36
    for off, ys, col, name in ((-w / 2 - 0.01, h10, S1, "Gold function in top 10"), (w / 2 + 0.01, h5, S2, "Gold file in top 5")):
        ax.bar([i + off for i in x], ys, w, color=col, label=name)
        for i, y in enumerate(ys):
            ax.text(i + off, y + 1.5, str(y), ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m[0] for m in modes], fontsize=7.5)
    ax.set_ylabel("Tasks (of 129)")
    ax.set_ylim(0, 100)
    hgrid(ax, "y")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "p1_retrieval.pdf")
    DATA["p1_retrieval"] = {m[0].replace("\n", " "): {"hit10_node": a, "hit5_file": b} for m, a, b in zip(modes, h10, h5)}


# ---------------------------------------------------------------- paper 2
def p2_named_tests():
    tests = [json.loads(l) for l in open(R / "a9" / "task_tests.jsonl", encoding="utf-8")]
    named = [t for t in tests if t["named"]]
    repo_of = {r["instance_id"]: r["repo"] for r in csv.DictReader(open(R / "a9" / "task_metrics.csv", newline="", encoding="utf-8"))}
    cats = ["Fails before, passes after", "Not collectable before, passes after", "Already passes before the fix",
            "Other (fails after, or never collected)"]

    def cat(t):
        if t["test"] is None:
            return cats[3]
        if t["post"] == "passed" and t["pre"] in ("failed", "error"):
            return cats[0]
        if t["post"] == "passed" and t["pre"] == "uncollected":
            return cats[1]
        if t["pre"] == "passed":
            return cats[2]
        return cats[3]

    by = defaultdict(Counter)
    for t in named:
        by["All repositories"][cat(t)] += 1
        by[repo_of[t["instance_id"]].split("/")[1]][cat(t)] += 1
    labels = ["All repositories", "fastapi", "rich", "requests"]
    parts = [[by[l][c] / sum(by[l].values()) for c in cats] for l in labels]
    ylabels = [f"{l}  (n={sum(by[l].values())})" for l in labels]
    fig, ax = plt.subplots(figsize=(5.6, 2.0))
    stacked_h(ax, ylabels, parts, [S1, S3, S2, OTHER], cats)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.38, -0.18), handlelength=1.0, columnspacing=1.0)
    fig.savefig(OUT / "p2_named_tests.pdf")
    DATA["p2_named_tests"] = {l: dict(by[l]) for l in labels}


def p2_disc_per_task():
    rows = [r for r in csv.DictReader(open(R / "a9" / "task_metrics.csv", newline="", encoding="utf-8")) if r["pre_exit"] != ""]
    bins = ["0", "1", "2", "3", "4", "5-9", "10+"]

    def b(n):
        return "0" if n == 0 else str(n) if n <= 4 else "5-9" if n < 10 else "10+"
    c = Counter(b(int(r["n_discriminating"])) for r in rows)
    vals = [c[k] for k in bins]
    fig, ax = plt.subplots(figsize=(5.0, 1.9))
    bars = ax.bar(bins, vals, color=[S2] + [S1] * 6, width=0.62)
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.8, str(v), ha="center", fontsize=7.5, color=INK)
    ax.set_xlabel("Discriminating tests per task (all tests in the target files)")
    ax.set_ylabel("Tasks")
    ax.set_ylim(0, max(vals) * 1.18)
    hgrid(ax, "y")
    fig.savefig(OUT / "p2_disc_per_task.pdf")
    DATA["p2_disc_per_task"] = dict(zip(bins, vals))


def p2_reference_scores():
    s = load(R / "a9" / "a9_summary.json")["reference_submissions"]
    subs = [("Gold patch", s["gold_patch"]), ("Empty patch", s["empty_patch"])]
    metrics = [("Harness rule", lambda a: a["resolution_rate_harness"]["p"], lambda a: f"{a['resolution_rate_harness']['k']}/{a['n_tasks']}"),
               ("Strict rule", lambda a: a["resolution_rate_strict"]["p"], lambda a: f"{a['resolution_rate_strict']['k']}/{a['n_tasks']}"),
               ("Mean credit", lambda a: a["mean_disc_credit"]["mean"], lambda a: f"{a['mean_disc_credit']['mean']:.2f}")]
    fig, ax = plt.subplots(figsize=(4.8, 2.0))
    x = range(len(subs))
    w = 0.25
    for j, ((name, f, lab), col) in enumerate(zip(metrics, [S1, S2, S3])):
        ys = [f(a) for _, a in subs]
        ax.bar([i + (j - 1) * (w + 0.02) for i in x], ys, w, color=col, label=name)
        for i, ((_, a), y) in enumerate(zip(subs, ys)):
            ax.text(i + (j - 1) * (w + 0.02), y + 0.03, lab(a), ha="center", fontsize=7, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s_[0] for s_ in subs])
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    hgrid(ax, "y")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.savefig(OUT / "p2_reference_scores.pdf")
    DATA["p2_reference_scores"] = {n: a for n, a in subs}


def p2_strata():
    rows = list(csv.DictReader(open(R / "a9" / "task_metrics.csv", newline="", encoding="utf-8")))
    us = [r for r in rows if r["a2_usable"] == "True"]

    def stratum(r):
        if r["d_entry_defined"] != "True":
            return "no executed gold node"
        d = r["d_entry"]
        if d == "":
            return "unreachable"
        d = int(float(d))
        return "0 hops" if d == 0 else "1-2 hops" if d <= 2 else "3+ hops"
    order = ["0 hops", "1-2 hops", "3+ hops", "unreachable", "no executed gold node"]
    ans = Counter((stratum(r), r["answerable"] == "True") for r in us)
    fig, ax = plt.subplots(figsize=(5.2, 2.0))
    a_vals = [ans[(o, True)] for o in order]
    n_vals = [ans[(o, False)] for o in order]
    ax.barh(order, a_vals, color=S1, height=0.6, edgecolor=SURF, linewidth=1.5, label="Answerable by the provided tools")
    ax.barh(order, n_vals, left=a_vals, color=OTHER, height=0.6, edgecolor=SURF, linewidth=1.5, label="Not answerable")
    for i, (a, n) in enumerate(zip(a_vals, n_vals)):
        if a:
            ax.text(a / 2, i, str(a), ha="center", va="center", fontsize=7.5, color="white")
        if n:
            ax.text(a + n / 2, i, str(n), ha="center", va="center", fontsize=7.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 48)
    ax.set_xlabel("Usable tasks (91), by entry distance")
    ax.legend(loc="center right", fontsize=7.5)
    hgrid(ax)
    fig.savefig(OUT / "p2_strata.pdf")
    DATA["p2_strata"] = {o: {"answerable": a, "not": n} for o, a, n in zip(order, a_vals, n_vals)}


if __name__ == "__main__":
    for f in (p1_presence, p1_resolver, p1_recall_precision, p1_edge_semantics, p1_edge_structure, p1_hops,
              p1_embeddings, p1_retrieval, p2_named_tests, p2_disc_per_task, p2_reference_scores, p2_strata):
        f()
        plt.close("all")
        print("ok", f.__name__, flush=True)
    (OUT / "figure_data.json").write_text(json.dumps(DATA, indent=2, default=str))

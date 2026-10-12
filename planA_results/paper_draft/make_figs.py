import csv, json, statistics as s
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

U = "/mnt/user-data/uploads/planA_results"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "violet": "#6250d6", "red": "#e34948", "gray": "#9a9892"}
REPO_C = {"fastapi/fastapi": C["blue"], "Textualize/rich": C["orange"], "psf/requests": C["aqua"], "encode/httpx": C["violet"]}
SHORT = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})

def ff(rows, k):
    return [float(r[k]) for r in rows if r[k] not in ("", "None")]

# ---------- Fig 1: node census by kind (A1, source files, all repos) ----------
a1 = json.load(open(f"{U}/vm-a2/results/a1_v2/a1_summary.json"))
kinds = ["function", "class", "nested", "async_function"]
labels = ["sync function", "class", "nested def", "async function"]
src = a1["omission_direct_source_files"]; tst = a1["omission_direct_test_files"]
fig, ax = plt.subplots(figsize=(5.2, 2.6))
x = np.arange(len(kinds)); w = 0.36
ps = [100 * src[k]["present_rate"]["p"] for k in kinds]
pt = [100 * tst[k]["present_rate"]["p"] for k in kinds]
b1 = ax.bar(x - w / 2, ps, w, color=C["blue"], label="source files")
b2 = ax.bar(x + w / 2, pt, w, color=C["orange"], label="test files")
for b, v, k in zip(b1, ps, kinds):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%\n(n={src[k]['total']:,})", ha="center", va="bottom", fontsize=6.5, color="#52514e")
for b, v, k in zip(b2, pt, kinds):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%\n(n={tst[k]['total']:,})", ha="center", va="bottom", fontsize=6.5, color="#52514e")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 118); ax.set_ylabel("definitions present in graph (%)")
ax.legend(loc="upper right", fontsize=7.5); ax.set_yticks([0, 25, 50, 75, 100])
fig.tight_layout(); fig.savefig("figs/fig1_census.pdf"); plt.close(fig)

# ---------- Fig 2: per-graph structure (isolated %, test-node %) ----------
g = list(csv.DictReader(open(f"{U}/vm-a2/results/a1_v2/a1_graphs.csv")))
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
for repo in ["fastapi/fastapi", "Textualize/rich", "psf/requests", "encode/httpx"]:
    rows = [r for r in g if r["repo"] == repo]
    axs[0].scatter(ff(rows, "n_nodes"), ff(rows, "pct_isolated"), s=14, color=REPO_C[repo], label=f"{SHORT[repo]} ({len(rows)})", alpha=0.85)
    axs[1].scatter(ff(rows, "n_nodes"), ff(rows, "pct_test_nodes"), s=14, color=REPO_C[repo], alpha=0.85)
axs[0].set_xlabel("nodes per graph"); axs[0].set_ylabel("isolated nodes (%)"); axs[0].legend(fontsize=7)
axs[1].set_xlabel("nodes per graph"); axs[1].set_ylabel("test-file nodes (%)")
for a in axs: a.set_ylim(0, 100)
fig.tight_layout(); fig.savefig("figs/fig2_graph_structure.pdf"); plt.close(fig)

# ---------- Fig 3: A2 recall / precision per task by repo ----------
a2 = list(csv.DictReader(open(f"{U}/vm-a2/results/a2_v3/a2_tasks.csv")))
us = [r for r in a2 if r["usable"] == "True"]
def recall(r):
    cap = float(r["nontest_direct_captured"]); miss = float(r["nontest_direct_missed_between_nodes"])
    return cap / (cap + miss) if cap + miss > 0 else None
def prec(r):
    o = float(r["n_static_observed"]); c = float(r["n_static_confirmed"])
    return c / o if o > 0 else None
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
order = ["fastapi/fastapi", "Textualize/rich", "psf/requests"]
for i, (fn, name) in enumerate([(recall, "representable recall (non-test, direct)"), (prec, "observed precision")]):
    data = [[v for v in (fn(r) for r in us if r["repo"] == repo) if v is not None] for repo in order]
    bp = axs[i].boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color="#0b0b0b"))
    for patch, repo in zip(bp["boxes"], order):
        patch.set_facecolor(REPO_C[repo]); patch.set_alpha(0.55); patch.set_edgecolor("#52514e")
    for j, (d, repo) in enumerate(zip(data, order)):
        xs = np.random.default_rng(j).normal(j + 1, 0.06, len(d))
        axs[i].scatter(xs, d, s=7, color=REPO_C[repo], alpha=0.9, zorder=3)
        axs[i].text(j + 1, -0.08, f"n={len(d)}", ha="center", fontsize=7, color="#52514e")
    axs[i].set_xticks([1, 2, 3]); axs[i].set_xticklabels([SHORT[r] for r in order]); axs[i].set_ylim(-0.12, 1.02)
    axs[i].set_ylabel(name, fontsize=8)
fig.tight_layout(); fig.savefig("figs/fig3_recall_precision.pdf"); plt.close(fig)

# ---------- Fig 4: A8 edge audit: how the callee appears in caller source ----------
a8 = json.load(open(f"{U}/vm-a2/results/a8/a8_summary.json"))["nontest_edges"]
hows = ["call_direct", "instantiation", "call_self", "call_attr_other", "call_in_nested_scope", "referenced", "implicit_dunder", "text_only", "no_reference", "call_super"]
hlab = ["direct call", "instantiation", "self./cls. call", "obj.attr call", "call in nested scope", "referenced, not called", "implicit dunder", "string/comment only", "no reference", "super() call"]
hcol = [C["blue"], C["blue"], C["blue"], C["blue"], C["blue"], C["yellow"], C["yellow"], C["red"], C["red"], C["blue"]]
pops = [("captured", "captured (static & dynamic)"), ("missed", "missed (dynamic only)"), ("unconfirmed", "unconfirmed (static only)")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for ax, (p, title) in zip(axs, pops):
    how = a8[p]["unique_edges"]["how"]; n = a8[p]["unique_edges"]["n"]
    vals = [100 * how.get(h, {}).get("share", 0) for h in hows]
    ax.barh(range(len(hows)), vals, color=hcol, height=0.7)
    for i, v in enumerate(vals):
        if v >= 1: ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=6.5, color="#52514e")
    ax.set_yticks(range(len(hows))); ax.set_yticklabels(hlab, fontsize=7); ax.invert_yaxis()
    ax.set_title(f"{title}\nn = {n:,}", fontsize=8); ax.set_xlim(0, 100); ax.set_xlabel("share of edges (%)", fontsize=7.5)
fig.tight_layout(); fig.savefig("figs/fig4_edge_audit.pdf"); plt.close(fig)

# ---------- Fig 5: A5 retrieval modes ----------
a5 = json.load(open(f"{U}/vm-a2/results/a5_v3/a5_summary.json"))["overall_all_tasks"]
modes = ["a0_anchors", "a_harness", "d_harness_1hop", "b_bm25", "e_bm25_snapshot"]
mlab = ["anchors only\n(a0)", "harness\n(a)", "harness+1-hop\n(d)", "BM25 graph text\n(b)", "BM25 snapshot\nsource (e)"]
fig, ax = plt.subplots(figsize=(5.6, 2.7))
x = np.arange(len(modes)); w = 0.38
for off, metric, col, lab in [(-w / 2, "hit10_node", C["blue"], "gold node in top 10"), (w / 2, "hit5_file", C["orange"], "gold file in top 5")]:
    p = [100 * a5[m][metric]["p"] for m in modes]
    lo = [100 * (a5[m][metric]["p"] - a5[m][metric]["lo"]) for m in modes]
    hi = [100 * (a5[m][metric]["hi"] - a5[m][metric]["p"]) for m in modes]
    ax.bar(x + off, p, w, color=col, label=lab, yerr=[lo, hi], error_kw=dict(ecolor="#52514e", lw=0.8, capsize=2))
    for xi, v in zip(x + off, p): ax.text(xi, v + 7, f"{v:.0f}", ha="center", fontsize=7, color="#52514e")
ax.set_xticks(x); ax.set_xticklabels(mlab, fontsize=7.5); ax.set_ylabel("tasks (%) , n = 129"); ax.set_ylim(0, 90)
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig("figs/fig5_retrieval.pdf"); plt.close(fig)

# ---------- Fig 6: A3b GNN fit gain per repo ----------
gnn = list(csv.DictReader(open(f"{U}/vm-gpu/results/a3b_gpu/a3b_gnn.csv")))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
repos = ["Textualize/rich", "fastapi/fastapi", "psf/requests", "encode/httpx"]
feats = ["structural", "text", "both"]; fc = [C["gray"], C["blue"], C["violet"]]
x = np.arange(len(repos)); w = 0.26
for j, (f, col) in enumerate(zip(feats, fc)):
    gains = []
    for repo in repos:
        rows = [r for r in gnn if r["repo"] == repo and r["objective"] == "fit" and r["features"] == f]
        gg = [(float(r["heldout_cos_mean"]) - float(r["heldout_globalmean_baseline_cos"])) / (1 - float(r["heldout_globalmean_baseline_cos"])) for r in rows]
        gains.append(s.mean(gg))
    axs[0].bar(x + (j - 1) * w, gains, w, color=col, label=f)
    for xi, v in zip(x + (j - 1) * w, gains): axs[0].text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6, color="#52514e")
axs[0].set_xticks(x); axs[0].set_xticklabels([SHORT[r] for r in repos]); axs[0].set_ylabel("fit: share of gap closed\n(held-out, vs mean-vector baseline)", fontsize=7.5); axs[0].legend(fontsize=7, loc="lower right", ncol=3, columnspacing=0.8); axs[0].set_ylim(0, 1.12)
feats2 = ["random_control", "structural", "text", "both"]; fc2 = [C["gray"], "#52514e", C["blue"], C["violet"]]
w = 0.2
for j, (f, col) in enumerate(zip(feats2, fc2)):
    vals = []
    for repo in repos:
        rows = [r for r in gnn if r["repo"] == repo and r["objective"] == "linkpred" and r["features"] == f]
        vals.append(s.mean(float(r["knn10_overlap"]) for r in rows))
    axs[1].bar(x + (j - 1.5) * w, vals, w, color=col, label=f.replace("_", " "))
axs[1].set_xticks(x); axs[1].set_xticklabels([SHORT[r] for r in repos]); axs[1].set_ylabel("linkpred: k-NN@10 overlap\nwith provided vectors (held-out)", fontsize=7.5); axs[1].legend(fontsize=7); axs[1].set_ylim(0, 0.5)
fig.tight_layout(); fig.savefig("figs/fig6_gnn.pdf"); plt.close(fig)

# ---------- Fig 7: A4 what cosine tracks (AUROC) + kNN composition ----------
rel = list(csv.DictReader(open(f"{U}/vm-main/results/a4/a4_relations.csv")))
knn = list(csv.DictReader(open(f"{U}/vm-main/results/a4/a4_knn.csv")))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
rels = ["calls", "containment", "gold_cochange", "same_module", "name_overlap"]
rlab = ["call-adjacent", "containment", "gold co-change", "same module", "name overlap"]
data = [[float(r["auroc_vs_random"]) for r in rel if r["relation"] == k and r["auroc_vs_random"] not in ("", "None")] for k in rels]
bp = axs[0].boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="#0b0b0b"))
for patch in bp["boxes"]: patch.set_facecolor(C["blue"]); patch.set_alpha(0.5); patch.set_edgecolor("#52514e")
for i, d in enumerate(data): axs[0].text(i + 1, 0.47, f"n={len(d)}", ha="center", fontsize=6.5, color="#52514e")
axs[0].axhline(0.5, color="#52514e", lw=0.8, ls="--"); axs[0].set_xticks(range(1, 6)); axs[0].set_xticklabels(rlab, fontsize=7, rotation=20, ha="right")
axs[0].set_ylabel("AUROC: related pair vs random pair"); axs[0].set_ylim(0.45, 1.0)
cols = ["top10_share_test_node", "top10_share_name_overlap", "top10_share_same_module", "top10_share_call_neighbor"]
clab = ["test-file node", "name overlap", "same module", "call neighbour"]
rr = ["fastapi/fastapi", "Textualize/rich", "psf/requests", "encode/httpx"]
x = np.arange(len(rr)); w = 0.2
for j, (c, lab, col) in enumerate(zip(cols, clab, [C["red"], C["yellow"], C["aqua"], C["blue"]])):
    vals = [100 * s.mean(float(r[c]) for r in knn if r["repo"] == repo) for repo in rr]
    axs[1].bar(x + (j - 1.5) * w, vals, w, color=col, label=lab)
axs[1].set_xticks(x); axs[1].set_xticklabels([SHORT[r] for r in rr]); axs[1].set_ylabel("share of 10 nearest neighbours (%)"); axs[1].legend(fontsize=6.5, ncol=2); axs[1].set_ylim(0, 100)
fig.tight_layout(); fig.savefig("figs/fig7_similarity.pdf"); plt.close(fig)

# ---------- Fig 8: gold hops from entry + d x v cells ----------
a2s = json.load(open(f"{U}/vm-a2/results/a2_v3/a2_summary.json"))["usable_tasks"]
h = a2s["gold_min_hops_from_entry_hist"]
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
ks = ["0", "1", "2", "3", "4", "5", "6", "7", "None"]
axs[0].bar(range(len(ks)), [h.get(k, 0) for k in ks], color=[C["blue"]] * 8 + [C["red"]])
for i, k in enumerate(ks): axs[0].text(i, h.get(k, 0) + 0.5, str(h.get(k, 0)), ha="center", fontsize=7, color="#52514e")
axs[0].set_xticks(range(len(ks))); axs[0].set_xticklabels(ks[:-1] + ["unreach."]); axs[0].set_xlabel("directed hops, test entry node → executed gold function"); axs[0].set_ylabel("tasks (n = 72)")
cells = json.load(open(f"{U}/vm-a2/results/a6_v3/a6_summary.json"))["task_profile_cells"]
drow = ["d0", "d1", "d2+", "d_undef"]; vcol = ["v_high", "v_low", "v_undef"]
M = np.array([[cells.get(f"{d}|{v}", 0) for v in vcol] for d in drow])
im = axs[1].imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.1)
for i in range(len(drow)):
    for j in range(len(vcol)):
        axs[1].text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8, color="#0b0b0b" if M[i, j] < 30 else "white")
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["v high\n(file rank ≤5)", "v low", "v undef."], fontsize=7)
axs[1].set_yticks(range(4)); axs[1].set_yticklabels(["d = 0", "d = 1", "d ≥ 2", "d undefined"], fontsize=7)
axs[1].set_title("task profile cells (n = 129)", fontsize=8); axs[1].grid(False)
fig.tight_layout(); fig.savefig("figs/fig8_reach_cells.pdf"); plt.close(fig)

# ---------- Fig 9: A1 misresolution by name query ----------
mis = a1["misresolution"]
fig, ax = plt.subplots(figsize=(5.4, 2.7))
cats = [("function|present|q_name", "sync function, present\n(bare name)"), ("class|present|q_name", "class, present\n(bare name)"),
        ("async_function|absent|q_name", "async function, absent\n(bare name)"), ("function|present|q_qual", "sync function, present\n(qualified name)"),
        ("async_function|absent|q_qual", "async function, absent\n(qualified name)")]
y = np.arange(len(cats))
selfs, wrong, nomatch = [], [], []
for k, _ in cats:
    o = mis[k]["outcomes"]; n = mis[k]["n"]
    selfs.append(100 * o.get("self", 0) / n); wrong.append(100 * o.get("wrong_node", 0) / n); nomatch.append(100 * o.get("no_match", 0) / n)
ax.barh(y, selfs, color=C["aqua"], label="resolves to itself", height=0.65)
ax.barh(y, wrong, left=selfs, color=C["red"], label="wrong node", height=0.65)
ax.barh(y, nomatch, left=np.array(selfs) + np.array(wrong), color=C["gray"], label="no match", height=0.65)
for i, (k, _) in enumerate(cats):
    ax.text(101, i, f"n={mis[k]['n']:,}", va="center", fontsize=6.5, color="#52514e")
ax.set_yticks(y); ax.set_yticklabels([c[1] for c in cats], fontsize=7); ax.invert_yaxis(); ax.set_xlim(0, 125); ax.set_xlabel("queries (%)")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=3); ax.set_xticks([0, 25, 50, 75, 100])
fig.tight_layout(); fig.savefig("figs/fig9_resolver.pdf"); plt.close(fig)
print("done")

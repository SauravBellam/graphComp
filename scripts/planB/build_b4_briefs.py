"""B4 (push): build the structural brief for each task, on CPU, from the competition archive.

    python build_b4_briefs.py --data archive --tasks-file official_slices/dev_slice.txt --out OUT_DIR

Plan B table, B4: "B2 plus a short structural brief (1-hop callers and callees of resolved issue symbols, test callers
filtered, grouped by file) prepended to the problem statement". Only information the agent could get itself is used:
the issue text, the provided graph (the same file the graph tools read) and the snapshot's file list. The gold patch is
read ONLY for diagnostics.json (does the brief name a gold file?), never for the brief.

Rules (fixed before any B4 run):
- identifiers: Plan A's deterministic extractor (common.extract_identifiers) on the problem statement;
- anchors: identifiers the harness's own resolver maps at the exact or suffix tier (Plan A: the ci / substring tiers
  misresolve), minus test and docs_src example nodes; at most MAX_ANCHORS, in extraction order;
- per anchor: callers (in-edges) and callees (out-edges) of the provided graph, minus test / docs_src nodes and minus
  containment edges (class -> own member, Plan A: class structure labelled `calls`), at most
  MAX_NEIGHBORS each; each node is shown with its file (module id -> snapshot path, learned like Plan A's ModuleIndex);
- grouped by the anchor's file; cut at MAX_CHARS; tasks with no anchor get no brief (the run is then identical to B2).

Writes OUT/<instance_id>.md (non-empty briefs only; also what scorecard.py --brief-dir reads), OUT/briefs.json
({instance_id: {"brief", "problem_sha256"}}, embedded in the Kaggle notebook) and OUT/diagnostics.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "planA_results/scripts/planA"))
from common import (Resolver, extract_identifiers, flat_identifiers, find_snapshot, get_graph, is_test_path,  # noqa: E402
                    load_tasks, path_to_module_candidates)

MAX_ANCHORS, MAX_NEIGHBORS, MAX_CHARS = 6, 8, 3000
HEADER = ("## Code-graph brief\n"
          "Generated from the repository's code graph before you start: the code elements named in the issue, the functions "
          "that call them and the functions they call (tests and documentation examples left out). The graph has no async "
          "functions and no nested functions, so treat this as a starting point, not a complete list.\n")


def module_files(snapshot: Path, graph) -> dict[str, str]:
    """{module id present in the graph: repo-relative .py path}; the first path mapping to a module wins."""
    prefixes = {".".join(n.split(".")[:k]) for n in graph.nodes for k in range(1, len(n.split(".")) + 1)}
    out = {}
    with tarfile.open(snapshot) as tf:
        names = sorted(m[2:] if m.startswith("./") else m for m in tf.getnames() if m.endswith(".py"))
    for p in names:
        for mod, _ in path_to_module_candidates(p):
            if mod in prefixes:
                out.setdefault(mod, p)
                break
    return out


def node_file(nid: str, mods: dict[str, str]) -> str | None:
    parts = nid.split(".")
    for k in range(len(parts) - 1, 0, -1):
        f = mods.get(".".join(parts[:k]))
        if f:
            return f
    return None


def skip(nid: str, mods) -> bool:
    f = node_file(nid, mods)
    return f is None or is_test_path(f) or f.startswith("docs_src/") or f.startswith("docs/")


def brief_for(task, graph, mods) -> tuple[str, dict]:
    res = Resolver(graph.nodes)
    anchors = []
    for tok in flat_identifiers(extract_identifiers(task.problem_statement)):
        nid, tier = res.resolve(tok)
        if nid and tier in ("exact", "suffix") and nid not in anchors and not skip(nid, mods):
            anchors.append(nid)
    anchors = anchors[:MAX_ANCHORS]
    if not anchors:
        return "", {"anchors": [], "files": []}
    fmt = lambda n: f"`{n}` ({node_file(n, mods)})"  # noqa: E731
    by_file: dict[str, list[str]] = {}
    files = set()
    for a in anchors:
        # containment edges (a class "calls" its own members) are left out: Plan A found them mislabelled structure
        callers = [s for s, _ in graph.in_edges.get(a, ()) if s != a and not a.startswith(s + ".") and not skip(s, mods)]
        callees = [d for d, _ in graph.out_edges.get(a, ()) if d != a and not d.startswith(a + ".") and not skip(d, mods)]
        callers, callees = list(dict.fromkeys(callers))[:MAX_NEIGHBORS], list(dict.fromkeys(callees))[:MAX_NEIGHBORS]
        files |= {node_file(n, mods) for n in [a] + callers + callees}
        lines = [f"- `{a}`",
                 "  - called by: " + (", ".join(map(fmt, callers)) or "nothing in the graph"),
                 "  - calls: " + (", ".join(map(fmt, callees)) or "nothing in the graph")]
        by_file.setdefault(node_file(a, mods), []).extend(lines)
    body = HEADER
    for f, lines in by_file.items():
        block = f"\n### {f}\n" + "\n".join(lines) + "\n"
        if len(body) + len(block) > MAX_CHARS:
            body += "\n(brief cut at its length limit)\n"
            break
        body += block
    return body, {"anchors": anchors, "files": sorted(files)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="competition archive (tasks.jsonl, graphs/, snapshots/)")
    ap.add_argument("--tasks-file", help="one instance id per line (default: all tasks)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.tasks_file) if l.strip()] if a.tasks_file else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    briefs, diag = {}, {}
    for t in load_tasks(Path(a.data), ids):
        g, snap = get_graph(Path(a.data), t), find_snapshot(Path(a.data), t)
        text, info = brief_for(t, g, module_files(snap, g)) if g and snap else ("", {"anchors": [], "files": []})
        gold = [f for f in t.gold_files if not is_test_path(f)]
        diag[t.instance_id] = {**info, "chars": len(text), "gold_files": gold,
                               "gold_file_in_brief": bool(set(gold) & set(info["files"]))}
        if text:
            briefs[t.instance_id] = {"brief": text, "problem_sha256": hashlib.sha256(t.problem_statement.encode()).hexdigest()}
            (out / f"{t.instance_id}.md").write_text(text)
        print(f"{t.instance_id}: {len(info['anchors'])} anchors, {len(text)} chars, gold file in brief: "
              f"{diag[t.instance_id]['gold_file_in_brief']}")
    (out / "briefs.json").write_text(json.dumps(briefs, indent=1))
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=1))
    n = len(diag)
    print(f"\n{len(briefs)}/{n} tasks get a brief; gold file named in {sum(d['gold_file_in_brief'] for d in diag.values())}/{n}; "
          f"median chars {sorted(d['chars'] for d in diag.values())[n // 2]}")


if __name__ == "__main__":
    main()

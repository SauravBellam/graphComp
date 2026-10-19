"""Plan B scorecard: score graph-tool use in swegemma 0.2.11 traces against the Plan A oracle.

Usage
-----
    python scorecard.py --results RUN_DIR [RUN_DIR ...] --oracle-pack oracle_pack.json.gz --out OUT
        [--arm B2] [--no-tools-results RUN_DIR ...] [--brief-dir DIR] [--graphs-dir DIR] [--dump-format]

RUN_DIR is a swegemma --results-dir (traces/trace_<id>.json, task_results.jsonl, logs/<id>.log).
Writes OUT/scorecard_calls.jsonl (one row per graph call), OUT/scorecard_tasks.csv (one row per
run x task with every component and every version's primary), OUT/scorecard_summary.json (pooled
rates with Wilson intervals, per repo and per cell), OUT/parse_report.json.

Trace format (read from adk_eval_core 0.1.0 tracing/atif_converter.py, ATIF-v1.7):
  steps[] with source in {user, agent, system}. An agent step holds tool_calls[] (function_name,
  arguments, extra.usage) and at most ONE observation (the first tool response). Further responses
  of the same step become separate system steps carrying observation (+extra.tool_name). Observations
  carry no tool_call_id, so calls and results are matched FIFO by tool name, then by order.
  Compaction = system step with extra.event_type == "compaction". Harness nudges = system step with
  extra.event_type == "continuation_nudge" (message = the nudge text, which names the cause).
  Raw tool returns: observation.content is the json.dumps of the full parsed tool response.

Component definitions and versions: HANDOFF_planB.md section 4.2/4.3; the concrete choices made
where the handoff leaves room are in VERSIONS_DOC below and are written to OUT/scorecard_versions.md.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path

SCORECARD_VERSION = "0.3.2"

GRAPH_TOOLS = ("get_code_neighbors", "search_similar_code", "get_code_subgraph")
HARNESS_TOOLS = {"run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch",
                 *GRAPH_TOOLS}
# Gemma 4 tool-call markup that leaked into assistant text = the gemma4 parser did not extract a call.
RAW_TOOLCALL_RE = re.compile(r"<\|?tool_call\|?>|<tool_call>|\bcall:[A-Za-z_]\w*\s*\{")
# Wording in logs/<id>.log or the vLLM log for a parser failure (adjust after B0 session A).
PARSER_ERROR_RE = re.compile(r"(error in extracting tool call|failed to parse tool call|tool[_ ]call.*(pars|decod)\w* (error|fail))", re.I)
MALFORMED_ERR_TYPES = {"ValidationError", "TypeError", "ValueError", "InvalidArguments", "ArgumentError",
                       "MissingArgument", "ToolNotFound", "UnknownTool"}
SHELL_READ_RE = re.compile(r"\b(cat|sed|head|tail|nl|less|more|awk|grep|rg|bat)\b")
PATH_TOKEN_RE = re.compile(r"[\w./-]+\.py\b")

PARAMS_DEFAULT = {"R": "R2", "window": 2, "label": "v1", "hub_pct": 95, "hub_share": 0.5}

VERSIONS_DOC = """# Scorecard versions (scorecard.py {ver})

Pre-registered list from HANDOFF_planB.md 4.3. Add versions; never replace one silently.
Components are task-level rates (NaN when their denominator is 0). `yield_R*` = share of graph calls
whose returned nodes intersect R; `novel_yield_R*` counts only returned R-nodes not returned by an
earlier graph call; `follow_w*` = share of graph calls with returned nodes followed within W tool calls
by a read_file on a returned node's file; `necessity_v*` = 1 if (made a graph call) == (need label);
`waste_p*_s*` = share of graph calls that returned only already-returned nodes, or whose resolved
query node is a hub (provided-graph in-degree above the repo-graph percentile) with test share of
returned nodes at or above s; `halluc_rate` = (unknown tool names + malformed arguments + unresolvable
graph arguments) / tool calls, parser failures excluded; `async_hazard` = share of graph calls naming
a gold/R2 node that is absent from the provided graph.

| Version | Primary | Date | Parameters | Why |
| --- | --- | --- | --- | --- |
| v0 | n_graph_calls | 2026-10-12 | none | strawman |
| v1 | mean of available(yield_R2, follow_w2, necessity_v1, 1 - waste_p95_s50) | 2026-10-12 | R2, W2, label v1, hub 95/50 | Oct 11 proposal design. With no graph call the primary is necessity_v1 alone |
| v2 | v1 with necessity_v2 | 2026-10-12 | label v2 (needs --no-tools-results) | counterfactual need |
| v3 | v1 with yield_R1 | 2026-10-12 | R = gold only | strict yield |
| v4 | novel_yield_R2 on need_v1 = 1 tasks (0 if no call), NaN on need_v1 = 0; 0 if halluc_rate > 0.25 | 2026-10-12 | R2, label v1, veto 0.25 | selection-rule form; novel yield because repeated good calls inflated raw yield in the 0.2.0 fixture |
| v5 | follow_w2 | 2026-10-12 | W2 | delivery only |
| v6 | logistic(components) fitted with --fit-v6 on B2 seed 1 of the 81 non-slice tasks | 2026-10-12 | coefficients file | learned composite, expected to overfit |
| v7 | tqs: port of TQS (Li et al. 2026, arXiv 2610.13152), graph-agnostic trajectory-structure score | 2026-10-12 | TQS's published weights and constants; deviations below | external baseline: a graph-specific version must beat a validated generic trajectory score, not only v0 |

v7 port (TQS section 3.3, weights OEC .30, IAC .25, PSN/TTE/PED/DPI .10, SCP .05, renormalised over defined
components; NaN when fewer than 3 tool calls or fewer than 4 of 7 components defined). Deviations from the
paper, fixed here: steps are tool calls (thinking-only steps are not actions); action alphabet
(read_file, write_file, execute_bash, search, submit, other, think) with read_file -> read_file,
edit_file/write_file -> write_file, run_command -> search when the command starts with grep/rg/find/ls/ag/ack
else execute_bash, the three graph tools -> search, submit_patch -> submit, get_status -> other; F_t = file
paths in the call's arguments (read/edit/write filepath, *.py tokens in run_command); error step = result
status "error", or a non-zero exit code; IAC lexicon (ours; the paper does not publish its lexicon) in
TQS_LEXICON, applied to the last 400 characters of the thought text preceding the call, the category whose
keyword occurs last wins; DPI: early = fewer than 5 tool calls, no_edit = no write_file without error,
lp = share of run_command calls repeating an earlier identical command, er = share of error steps repeating
an earlier error step's (action, files, first 100 characters of the error).

Length-confound rule (pre-registered 2026-10-12, from TQS section 3.4 and appendix G): every version's
|Spearman rho| with n_tool_calls is reported (`length_check.json`); a version with |rho| > 0.25 is flagged
`length_confounded` and its test-1 AUROC must also be reported inside the interquartile band of n_tool_calls.
A version that passes test 1 only outside that band does not advance.

Changelog: 0.3.2 (2026-10-12): v7 (TQS port); length-confound check; per-task `end_reason`
(context_exceeded / time_limit / tool_budget / error / none) from task_results.jsonl, so ContextWindowExceededError
endings under max_output_tokens 16384 are counted separately (decision of Oct 12). No earlier definition changed.
0.3.1 (2026-10-12, B0 session A traces): per-call token usage read from the agent step's `metrics`
when `tool_call.extra.usage` is absent (it was absent for all 128 calls on Kaggle). No component definition changed.

Choices made here where the handoff left room (change = new version): the v1 combination rule (equal
weights over available components); the v4 veto threshold 0.25; resolution precision counts a call
precise when every named argument resolves exactly, by a unique delimiter-suffix match, or by a unique
case-insensitive match (a suffix tier with more than one candidate is a same-name collision);
search_similar_code arguments are replayed through the same resolver (an approximation of the
embedding-key lookup).
"""


# --------------------------------------------------------------------------------------------
# small stats helpers
# --------------------------------------------------------------------------------------------
def wilson(k, n, z=1.959964):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def nanmean(xs):
    v = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v) if v else float("nan")


def rate(k, n):
    return k / n if n else float("nan")


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


# --------------------------------------------------------------------------------------------
# resolver: line-for-line port of swegemma 0.2.11 graph_utils.resolve_node_name, plus the tier
# --------------------------------------------------------------------------------------------
def resolve_with_tier(query, node_list, node_set=None):
    """Return (resolved_node or None, tier, n_candidates_in_tier)."""
    q = str(query or "").strip()
    if not q or not node_list:
        return None, "none", 0
    node_set = node_set if node_set is not None else set(node_list)
    bare = q.split(";", 1)[1] if ";" in q else q
    if q in node_set:
        return q, "exact", 1
    if bare in node_set:
        return bare, "exact", 1
    key = lambda x: (len(x.split(".")), len(x), x)
    cands = [n for n in node_list if n.endswith(f".{bare}") or n.endswith(f";{bare}") or n.endswith(f"/{bare}")]
    if cands:
        return min(cands, key=key), "suffix", len(cands)
    lq, lf = bare.lower(), q.lower()
    ci = [n for n in node_list if n.lower() == lf or n.lower() == lq]
    if ci:
        return min(ci, key=key), "ci_exact", len(ci)
    cis = [n for n in node_list if n.lower().endswith(f".{lq}") or n.lower().endswith(f";{lq}") or n.lower().endswith(f"/{lq}")]
    if cis:
        return min(cis, key=key), "ci_suffix", len(cis)
    if not any(c.isspace() for c in bare):
        sub = [n for n in node_list if bare in n or lq in n.lower()]
        if sub:
            return min(sub, key=key), "substring", len(sub)
    return None, "none", 0


def precise_tier(tier, n):
    return tier == "exact" or (tier in ("suffix", "ci_exact", "ci_suffix") and n == 1)


def parity_check(graph_json_path, args, out=None):
    """Compare this port with swegemma's resolve_node_name on the given arguments (needs swegemma)."""
    from swegemma.graph.graph_utils import load_graph_from_json, resolve_node_name
    g = load_graph_from_json(str(graph_json_path))
    nodes = [str(n) for n in g.nodes()]
    ns = set(nodes)
    bad = []
    for a in args:
        mine = resolve_with_tier(a, nodes, ns)[0]
        theirs = resolve_node_name(a, g)
        if mine != theirs:
            bad.append({"arg": a, "port": mine, "swegemma": theirs})
    res = {"n": len(args), "mismatches": bad}
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------------------------
class Oracle:
    def __init__(self, pack_path, profile_csv=None):
        op = gzip.open if str(pack_path).endswith(".gz") else open
        with op(pack_path, "rt") as f:
            self.pack = json.load(f)
        self.tasks = self.pack["tasks"]
        self.graphs = self.pack["graphs"]
        self._nodelists = {}
        self.profile = {}
        if profile_csv and Path(profile_csv).exists():
            self.profile = {r["instance_id"]: r for r in csv.DictReader(open(profile_csv))}

    def task(self, iid):
        return self.tasks.get(iid)

    def graph(self, key):
        return self.graphs.get(key)

    def node_list(self, key):
        if key not in self._nodelists:
            g = self.graphs.get(key) or {"nodes": {}}
            nl = list(g["nodes"].keys())
            self._nodelists[key] = (nl, set(nl))
        return self._nodelists[key]

    def node_file(self, key, node):
        g = self.graphs.get(key)
        if not g:
            return None
        v = g["nodes"].get(node)
        if v is None:
            return None
        f = g["files"][v[0]]
        return f or None

    def node_is_test(self, key, node):
        g = self.graphs.get(key)
        v = g["nodes"].get(node) if g else None
        if v is None:
            return _looks_test_node(node)
        return bool(v[1])

    def indeg(self, key, node):
        g = self.graphs.get(key)
        v = g["nodes"].get(node) if g else None
        return v[2] if v else 0

    def hub_threshold(self, key, pct):
        g = self.graphs.get(key) or {}
        return g.get(f"indeg_p{pct}", 10 ** 9)


def _looks_test_node(n):
    parts = n.split(".")
    return any(p in ("tests", "test", "testing") or p.startswith("test_") for p in parts)


def norm_path(p):
    p = str(p or "").strip().strip("'\"")
    for pre in ("/workspace/", "workspace/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p.lstrip("/")


def path_match(agent_path, true_path):
    a, t = norm_path(agent_path), norm_path(true_path)
    if not a or not t:
        return False
    return a == t or t.endswith("/" + a) or a.endswith("/" + t)


def strip_repo(n):
    s = str(n)
    return s.split(";", 1)[1] if ";" in s else s


def need_v1(cell):
    if not cell or "|" not in cell:
        return None
    d, v = cell.split("|", 1)
    if d in ("d1", "d2+") or v == "v_low":
        return 1
    if v == "v_undef" and d == "d_undef":
        return None
    return 0


# --------------------------------------------------------------------------------------------
# trace parsing
# --------------------------------------------------------------------------------------------
def _parse_content(c):
    if isinstance(c, dict):
        return c
    if isinstance(c, list):
        txt = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    else:
        txt = str(c or "")
    try:
        v = json.loads(txt)
        return v if isinstance(v, dict) else {"result": v}
    except Exception:
        return {"raw": txt}


def _content_len(c):
    if isinstance(c, str):
        return len(c)
    return len(json.dumps(c, default=str))


def parse_trace(path):
    """Return dict with calls (ordered), events, diagnostics."""
    d = json.load(open(path))
    steps = d.get("steps", [])
    calls, pending = [], []
    diag = collections.Counter()
    events = []  # (kind, call_index_at_time, detail)
    turn = 0
    thought_buf = []  # agent text (thinking / text) since the last tool call, for TQS intent-action consistency
    for s in steps:
        src = s.get("source")
        ex = s.get("extra") or {}
        et = ex.get("event_type")
        if src == "agent" and s.get("message") and et in ("thinking", "text", None):
            thought_buf.append(str(s.get("message")))
        diag[f"step:{src}:{et or ('tool_calls' if s.get('tool_calls') else 'obs' if s.get('observation') else 'other')}"] += 1
        if src == "agent" and (s.get("tool_calls") or et in ("thinking", "text", "final")):
            turn += 1 if (s.get("tool_calls") or et in ("text", "final")) else 0
        if et == "compaction":
            events.append({"kind": "compaction", "at_call": len(calls), "chars": len(str(s.get("message") or ""))})
        elif et == "continuation_nudge":
            msg = str(s.get("message") or "")
            cause = ("truncated_tool_call" if "was cut off" in msg else
                     "length" if "token limit while thinking" in msg else "no_call")
            events.append({"kind": "nudge", "cause": cause, "at_call": len(calls)})
        elif src == "agent" and et in ("text", "final") and RAW_TOOLCALL_RE.search(str(s.get("message") or "")):
            events.append({"kind": "parser_failure_text", "at_call": len(calls),
                           "snippet": str(s.get("message"))[:200]})
        step_metrics = s.get("metrics") if s.get("tool_calls") else None
        for j, tc in enumerate(s.get("tool_calls") or []):
            tx = tc.get("extra") or {}
            # 0.3.1: on Kaggle (swegemma 0.2.11) usage arrives as separate 'usage' events that the ATIF converter
            # merges into the step's metrics, not into tool_call.extra.usage; attach it to the step's first call.
            if not tx.get("usage") and j == 0 and step_metrics:
                tx = {**tx, "usage": step_metrics}
            c = {"idx": len(calls), "step_id": s.get("step_id"), "turn": turn,
                 "name": tc.get("function_name") or "", "args": tc.get("arguments") or {},
                 "author": tx.get("author") or ex.get("author") or "",
                 "usage": tx.get("usage") or None, "elapsed_s": tx.get("elapsed_s"),
                 "thought": "\n".join(thought_buf) if j == 0 else "",
                 "result": None, "result_raw_len": None, "result_text": "", "matched_by": None}
            if j == len(s.get("tool_calls") or []) - 1:
                thought_buf = []
            calls.append(c)
            pending.append(c)
            if tx.get("usage"):
                diag["calls_with_usage"] += 1
        obs = s.get("observation")
        if obs is not None:
            oname = (obs.get("extra") or {}).get("tool_name") or ""
            target = None
            for i, c in enumerate(pending):
                if oname and c["name"] == oname:
                    target = pending.pop(i)
                    target["matched_by"] = "name"
                    break
            if target is None and pending:
                target = pending.pop(0)
                target["matched_by"] = "order"
            if target is None:
                diag["results_unmatched"] += 1
                continue
            target["result"] = _parse_content(obs.get("content"))
            target["result_raw_len"] = _content_len(obs.get("content"))
            oc = obs.get("content")
            target["result_text"] = oc if isinstance(oc, str) else json.dumps(oc, default=str)
    for c in pending:
        c["matched_by"] = "no_result"
    diag["calls"] = len(calls)
    diag["matched_by_name"] = sum(c["matched_by"] == "name" for c in calls)
    diag["matched_by_order"] = sum(c["matched_by"] == "order" for c in calls)
    diag["calls_without_result"] = sum(c["matched_by"] == "no_result" for c in calls)
    diag["compactions"] = sum(e["kind"] == "compaction" for e in events)
    diag["nudges"] = sum(e["kind"] == "nudge" for e in events)
    fm = d.get("final_metrics") or {}
    return {"calls": calls, "events": events, "diag": dict(diag), "final_metrics": fm,
            "schema": d.get("schema_version"), "n_steps": len(steps), "step_keys": sorted({k for s in steps for k in s})}


def returned_nodes(call):
    r = call.get("result") or {}
    name = call["name"]
    out = []
    if name == "get_code_neighbors":
        out = r.get("neighbors") or []
    elif name == "search_similar_code":
        out = [x.get("node_name") if isinstance(x, dict) else x for x in (r.get("results") or [])]
    elif name == "get_code_subgraph":
        out = list(r.get("nodes") or [])
        for e in r.get("edges") or []:
            if isinstance(e, dict):
                out += [e.get("from"), e.get("to")]
    seen, res = set(), []
    for n in out:
        if not n:
            continue
        n = strip_repo(n)
        if n not in seen:
            seen.add(n)
            res.append(n)
    return res


def named_args(call):
    a = call.get("args") or {}
    name = call["name"]
    if name == "get_code_neighbors":
        v = a.get("node")
        return [v] if v not in (None, "") else []
    if name == "search_similar_code":
        v = a.get("query")
        return [v] if v not in (None, "") else []
    if name == "get_code_subgraph":
        v = a.get("nodes")
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                v = [x.strip() for x in v.strip("[]").split(",") if x.strip()]
        return [str(x) for x in (v or [])]
    return []


def call_files_read(call):
    """Files this call reads: (path, via) list."""
    a = call.get("args") or {}
    if call["name"] == "read_file":
        return [(norm_path(a.get("filepath") or a.get("path") or ""), "read_file")]
    if call["name"] == "run_command":
        cmd = str(a.get("command") or "")
        if SHELL_READ_RE.search(cmd):
            return [(norm_path(p), "shell") for p in PATH_TOKEN_RE.findall(cmd)]
    return []


def call_error(call):
    r = call.get("result") or {}
    if r.get("status") == "error":
        return r.get("error_type") or "error", str(r.get("error_message") or "")[:200]
    if "error" in r and isinstance(r.get("error"), str):
        return "error", r["error"][:200]
    return None, None


# --------------------------------------------------------------------------------------------
# scoring one trace
# --------------------------------------------------------------------------------------------
def score_trace(iid, parsed, oracle, run_meta, brief_files=None, extra_known_tools=()):
    t = oracle.task(iid) or {}
    key = t.get("graph_key")
    nodes, nodeset = oracle.node_list(key) if key else ([], set())
    calls = parsed["calls"]
    R = {k: set(t.get(k, [])) for k in ("R1", "R2", "R3")}
    gold_files = list(t.get("gold_files", []))
    missing = set(t.get("R2_missing_from_provided", []))
    known = HARNESS_TOOLS | set(extra_known_tools)

    # reads in order
    reads = []  # (call_idx, path, via)
    for c in calls:
        for p, via in call_files_read(c):
            if p:
                reads.append((c["idx"], p, via))

    call_rows = []
    seen_nodes = set()
    seen_R = {k: set() for k in R}
    n_unknown = n_malformed = n_unresolvable = 0
    for c in calls:
        etype, emsg = call_error(c)
        if c["name"] not in known:
            n_unknown += 1
        elif etype and (etype in MALFORMED_ERR_TYPES or "argument" in (emsg or "").lower()):
            n_malformed += 1
        if c["name"] not in GRAPH_TOOLS:
            continue
        args = named_args(c)
        res = [resolve_with_tier(a, nodes, nodeset) for a in args]
        precise = bool(args) and all(precise_tier(tier, n) for _, tier, n in res)
        unresolved = [a for a, (r, _, _) in zip(args, res) if r is None]
        if unresolved:
            n_unresolvable += 1
        ret = returned_nodes(c)
        ret_set = set(ret)
        ret_nontest = [n for n in ret if not oracle.node_is_test(key, n)]
        row = {
            "run": run_meta["run"], "arm": run_meta["arm"], "seed": run_meta["seed"], "instance_id": iid,
            "call_idx": c["idx"], "turn": c["turn"], "tool": c["name"], "args": c["args"], "author": c["author"],
            "status": (c.get("result") or {}).get("status"), "error_type": etype,
            "matched_by": c["matched_by"], "result_raw_len": c["result_raw_len"],
            "resolved": [r for r, _, _ in res], "resolve_tier": [tier for _, tier, _ in res],
            "resolve_n_candidates": [n for _, _, n in res], "precise": precise,
            "n_returned": len(ret), "n_returned_test": len(ret) - len(ret_nontest),
        }
        for k in R:
            hit = ret_set & R[k]
            row[f"yield_{k}"] = bool(hit)
            row[f"novel_yield_{k}"] = bool(hit - seen_R[k])
            seen_R[k] |= hit
        row["returned_all_seen"] = bool(ret) and ret_set <= seen_nodes
        seen_nodes |= ret_set
        # hub
        qnode = next((r for r, _, _ in res if r), None)
        row["query_indeg"] = oracle.indeg(key, qnode) if qnode else 0
        row["test_share"] = (len(ret) - len(ret_nontest)) / len(ret) if ret else 0.0
        for pct in (90, 95):
            thr = oracle.hub_threshold(key, pct)
            for sh in (0.5, 0.75):
                hub = bool(qnode) and row["query_indeg"] > thr and bool(ret) and row["test_share"] >= sh
                row[f"waste_p{pct}_s{int(sh*100)}"] = row["returned_all_seen"] or hub
        # link-following
        ret_files = {oracle.node_file(key, n) for n in ret_nontest}
        ret_files.discard(None)
        if brief_files:
            ret_files |= set(brief_files)
        row["returned_files"] = sorted(ret_files)
        for w in (1, 2, 3):
            nxt = [r for r in reads if c["idx"] < r[0] <= c["idx"] + w and r[2] == "read_file"]
            row[f"follow_w{w}"] = any(path_match(p, f) for _, p, _ in nxt for f in ret_files) if ret_files else None
        # async hazard: argument names a gold/R2 node absent from the provided graph
        haz = []
        for a in args:
            b = strip_repo(a)
            for m in missing:
                if m == b or m.endswith("." + b):
                    haz.append(m)
        row["async_hazard"] = bool(haz)
        row["hazard_nodes"] = sorted(set(haz))
        call_rows.append(row)

    G = len(call_rows)
    with_ret = [r for r in call_rows if r["n_returned"]]
    task = {
        "run": run_meta["run"], "arm": run_meta["arm"], "seed": run_meta["seed"], "instance_id": iid,
        "repo": t.get("repo") or iid.split("_")[0], "cell": t.get("cell"), "graph_key": key,
        "in_oracle": bool(t), "n_tool_calls": len(calls), "n_graph_calls": G,
        "n_graph_calls_with_return": len(with_ret),
        "resolution_precision": rate(sum(r["precise"] for r in call_rows), sum(bool(named_args({"name": r["tool"], "args": r["args"]})) for r in call_rows)),
    }
    for k in R:
        task[f"yield_{k}"] = rate(sum(r[f"yield_{k}"] for r in call_rows), G)
        task[f"novel_yield_{k}"] = rate(sum(r[f"novel_yield_{k}"] for r in call_rows), G)
    for w in (1, 2, 3):
        fr = [r for r in call_rows if r[f"follow_w{w}"] is not None]
        task[f"follow_w{w}"] = rate(sum(bool(r[f"follow_w{w}"]) for r in fr), len(fr))
    for pct in (90, 95):
        for sh in (50, 75):
            task[f"waste_p{pct}_s{sh}"] = rate(sum(r[f"waste_p{pct}_s{sh}"] for r in call_rows), G)
    task["n_unknown_tool"] = n_unknown
    task["n_malformed_args"] = n_malformed
    task["n_unresolvable_graph_args"] = n_unresolvable
    task["halluc_rate"] = rate(n_unknown + n_malformed + n_unresolvable, len(calls))
    task["n_async_hazard"] = sum(r["async_hazard"] for r in call_rows)
    task["async_hazard"] = rate(task["n_async_hazard"], G)
    task["n_parser_failure_text"] = sum(e["kind"] == "parser_failure_text" for e in parsed["events"])
    task["n_nudge_truncated_tool_call"] = sum(e.get("cause") == "truncated_tool_call" for e in parsed["events"])
    task["n_nudge_length"] = sum(e.get("cause") == "length" for e in parsed["events"])
    task["n_nudge_no_call"] = sum(e.get("cause") == "no_call" for e in parsed["events"])
    task["n_compactions"] = sum(e["kind"] == "compaction" for e in parsed["events"])
    task["first_compaction_at_call"] = next((e["at_call"] for e in parsed["events"] if e["kind"] == "compaction"), None)
    # necessity
    nv1 = need_v1(t.get("cell"))
    task["need_v1"] = nv1
    task["made_graph_call"] = int(G > 0)
    task["necessity_v1"] = float(task["made_graph_call"] == nv1) if nv1 is not None else float("nan")
    task["need_v2"] = None
    task["necessity_v2"] = float("nan")
    # localization outcomes
    gold_read_at, read_gold = None, set()
    for ci, p, via in reads:
        for gf in gold_files:
            if path_match(p, gf):
                read_gold.add(gf)
                if gold_read_at is None:
                    gold_read_at = ci + 1
    task["n_gold_files"] = len(gold_files)
    task["steps_to_first_gold_read"] = gold_read_at
    task["gold_file_coverage"] = rate(len(read_gold), len(gold_files))
    task["second_file_discovery"] = (float(len(read_gold) >= 2) if len(gold_files) >= 2 else float("nan"))
    task["gold_file_read"] = int(bool(read_gold))
    # tokens
    fm = parsed.get("final_metrics") or {}
    task["total_prompt_tokens"] = fm.get("total_prompt_tokens")
    task["total_completion_tokens"] = fm.get("total_completion_tokens")
    task["resolved"] = run_meta.get("resolved", {}).get(iid)
    task["end_reason"] = end_reason(run_meta.get("records", {}).get(iid), len(calls))
    task["ended_context_exceeded"] = int(task["end_reason"] == "context_exceeded")
    task.update(tqs_score(calls))
    return task, call_rows


CONTEXT_RE = re.compile(r"ContextWindowExceeded|context.?window|maximum context|context length", re.I)
TIME_RE = re.compile(r"time.?limit|max_time|timed? ?out|TimeoutError", re.I)
BUDGET_RE = re.compile(r"max_tool_calls|tool.?call (budget|limit)|max_turns|turn limit", re.I)


def end_reason(rec, n_calls, max_tool_calls=50):
    """How the agent run ended, from the task_results.jsonl record (error-like string fields only)."""
    if not rec:
        return "unknown"
    txt = " ".join(str(v) for k, v in rec.items()
                   if isinstance(v, str) and re.search(r"error|reason|message|status|exception|stop", k, re.I))
    if CONTEXT_RE.search(txt):
        return "context_exceeded"
    if TIME_RE.search(txt):
        return "time_limit"
    if BUDGET_RE.search(txt) or n_calls >= max_tool_calls:
        return "tool_budget"
    if txt.strip() and re.search(r"error|exception|fail", txt, re.I):
        return "error"
    return "none"


# --------------------------------------------------------------------------------------------
# v7: TQS port (Li et al. 2026, arXiv 2610.13152, section 3.3). Deviations: see VERSIONS_DOC.
# --------------------------------------------------------------------------------------------
TQS_WEIGHTS = {"oec": 0.30, "iac": 0.25, "psn": 0.10, "tte": 0.10, "ped": 0.10, "dpi": 0.10, "scp": 0.05}
TQS_ALPHABET = ("think", "read_file", "write_file", "execute_bash", "search", "submit", "other")
TQS_LEXICON = {  # coarse intent -> regex (ours; the paper's lexicon is not published)
    "reading": r"\b(read|look at|inspect|open|view|examine|check (the )?(file|code|implementation|source)|understand)\b",
    "editing": r"\b(edit|modify|change|fix|patch|update|replace|rewrite|implement|add (a |the )?(check|line|parameter|argument|condition))\b",
    "execution": r"\b(run|test|execute|reproduce|pytest|verify|try running|install)\b",
    "search": r"\b(search|find|grep|locate|look for|where is|neighbou?rs|callers|callees|subgraph)\b",
}
TQS_ACTION_INTENT = {"read_file": "reading", "write_file": "editing", "execute_bash": "execution", "search": "search"}
SEARCH_CMD_RE = re.compile(r"^\s*(cd [^;&]+[;&]+\s*)?(grep|rg|find|ls|ag|ack|tree)\b")


def tqs_action(c):
    n = c["name"]
    if n == "read_file":
        return "read_file"
    if n in ("edit_file", "write_file"):
        return "write_file"
    if n == "run_command":
        return "search" if SEARCH_CMD_RE.search(str((c.get("args") or {}).get("command") or "")) else "execute_bash"
    if n in GRAPH_TOOLS:
        return "search"
    if n == "submit_patch":
        return "submit"
    return "other"


def tqs_files(c):
    a = c.get("args") or {}
    if c["name"] in ("read_file", "edit_file", "write_file"):
        p = norm_path(a.get("filepath") or a.get("path") or "")
        return {p} if p else set()
    if c["name"] == "run_command":
        return {norm_path(p) for p in PATH_TOKEN_RE.findall(str(a.get("command") or ""))}
    return set()


def tqs_error(c):
    r = c.get("result") or {}
    if r.get("status") == "error":
        return True
    for k in ("exit_code", "returncode", "return_code"):
        v = r.get(k)
        if isinstance(v, (int, float)) and v != 0:
            return True
    return False


def _char_entropy(s):
    if not s:
        return 0.0
    cnt = collections.Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in cnt.values())


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def kendall_tau_b(x, y):
    n = len(x)
    nc = nd = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                nc += 1
            else:
                nd += 1
    den = math.sqrt((nc + nd + tx) * (nc + nd + ty))
    return (nc - nd) / den if den else 0.0


def tqs_score(calls):
    """Return dict of TQS components (NaN = undefined) and the composite `tqs`."""
    n = len(calls)
    nan = float("nan")
    out = {f"tqs_{k}": nan for k in TQS_WEIGHTS}
    out["tqs"] = nan
    if n < 3:
        return out
    acts = [tqs_action(c) for c in calls]
    files = [tqs_files(c) for c in calls]
    errs = [tqs_error(c) for c in calls]
    # A1 OEC
    H = [_char_entropy(c.get("result_text") or "") for c in calls]
    k0 = min(5, n)
    h_open = sum(H[:k0]) / k0
    if h_open > 0:
        sm = [sum(H[max(0, t - 2):t + 1]) / len(H[max(0, t - 2):t + 1]) for t in range(n)]
        out["tqs_oec"] = min(1.0, max(0.0, 1 - min(x / h_open for x in sm)))
    # A2 PSN: distinct files in a 5-step window, Kendall tau vs step index over the second half
    scope = [len(set().union(*files[max(0, t - 4):t + 1])) for t in range(n)]
    half = list(range(n // 2, n))
    if len(half) >= 3:
        tau = kendall_tau_b([float(t) for t in half], [float(scope[t]) for t in half])
        out["tqs_psn"] = min(1.0, max(0.0, (1 - tau) / 2))
    # B1 IAC
    hits = tot = 0
    for c, a in zip(calls, acts):
        if a not in TQS_ACTION_INTENT:
            continue
        txt = (c.get("thought") or "")[-400:].lower()
        best, pos = None, -1
        for intent, rx in TQS_LEXICON.items():
            for m in re.finditer(rx, txt):
                if m.start() > pos:
                    best, pos = intent, m.start()
        if best is None:
            continue
        tot += 1
        hits += best == TQS_ACTION_INTENT[a]
    if tot:
        out["tqs_iac"] = hits / tot
    # C1 TTE
    pairs = collections.Counter(zip(acts, acts[1:]))
    m = sum(pairs.values())
    if m:
        out["tqs_tte"] = -sum(v / m * math.log2(v / m) for v in pairs.values()) / math.log2(len(TQS_ALPHABET) ** 2)
    # C2 SCP
    first_edit = next((t for t in range(n) if acts[t] == "write_file" and not errs[t]), None)
    if first_edit is None:
        out["tqs_scp"] = 0.0
    else:
        u = first_edit / n
        d = max(0.0, 0.2 - u, u - 0.5)
        out["tqs_scp"] = math.exp(-d * d / (2 * 0.2 ** 2))
    # C3 PED
    E = [t for t in range(n - 1) if errs[t]]
    if E:
        ds = []
        for t in E:
            a, b = files[t], files[t + 1]
            jd = (1 - len(a & b) / len(a | b)) if (a | b) else 0.0
            ds.append(0.5 * ((acts[t + 1] != acts[t]) + jd))
        out["tqs_ped"] = sum(ds) / len(ds)
    # D1 DPI
    cmds = [str((c.get("args") or {}).get("command") or "") for c in calls if c["name"] == "run_command"]
    lp = (sum(1 for i, x in enumerate(cmds) if x in cmds[:i]) / len(cmds)) if cmds else 0.0
    ekeys = [(acts[t], tuple(sorted(files[t])), (calls[t].get("result_text") or "")[:100]) for t in range(n) if errs[t]]
    er = (sum(1 for i, x in enumerate(ekeys) if x in ekeys[:i]) / len(ekeys)) if ekeys else 0.0
    p = 0.4 * (n < 5) + 0.4 * (first_edit is None) + min(0.3, 3 * lp) + min(0.3, 0.6 * er)
    out["tqs_dpi"] = 1 - min(1.0, p)
    defined = {k: out[f"tqs_{k}"] for k in TQS_WEIGHTS if not isnan(out[f"tqs_{k}"])}
    if len(defined) >= 4:
        out["tqs"] = sum(TQS_WEIGHTS[k] * v for k, v in defined.items()) / sum(TQS_WEIGHTS[k] for k in defined)
    return out


# --------------------------------------------------------------------------------------------
# length confound and AUROC (pre-registered rule: VERSIONS_DOC)
# --------------------------------------------------------------------------------------------
def spearman(x, y):
    pts = [(a, b) for a, b in zip(x, y) if not isnan(a) and not isnan(b)]
    if len(pts) < 3:
        return float("nan")
    rx, ry = _ranks([a for a, _ in pts]), _ranks([b for _, b in pts])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy) if sx and sy else float("nan")


def auroc(scores, labels):
    pts = [(s, l) for s, l in zip(scores, labels) if not isnan(s) and l is not None]
    pos = [s for s, l in pts if l]
    neg = [s for s, l in pts if not l]
    if not pos or not neg:
        return float("nan")
    r = _ranks([s for s, _ in pts])
    rp = sum(ri for ri, (_, l) in zip(r, pts) if l)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


VERSION_KEYS = ("v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7")


def length_check(task_rows):
    L = [float(t["n_tool_calls"]) for t in task_rows]
    cov = [t["gold_file_coverage"] for t in task_rows if not isnan(t.get("gold_file_coverage"))]
    med = sorted(cov)[len(cov) // 2] if cov else None
    res_lab = [t["resolved"] if t.get("resolved") is not None else None for t in task_rows]
    cov_lab = [None if (med is None or isnan(t.get("gold_file_coverage"))) else t["gold_file_coverage"] > med for t in task_rows]
    sl = sorted(L)
    q1, q3 = (sl[len(sl) // 4], sl[(3 * len(sl)) // 4]) if sl else (None, None)
    band = [q1 is not None and q1 <= x <= q3 for x in L]
    out = {"n_tasks": len(task_rows), "length_field": "n_tool_calls", "iqr_band": [q1, q3],
           "n_in_band": sum(band), "gold_coverage_median": med, "versions": {}}
    for v in VERSION_KEYS:
        s = [float(t[v]) if not isnan(t.get(v)) else float("nan") for t in task_rows]
        rho = spearman(s, L)
        rnd = lambda x: None if isnan(x) else round(x, 4)
        out["versions"][v] = {
            "spearman_rho_length": rnd(rho),
            "length_confounded": None if isnan(rho) else abs(rho) > 0.25,
            "auroc_resolved": rnd(auroc(s, res_lab)),
            "auroc_resolved_in_band": rnd(auroc([x for x, b in zip(s, band) if b], [l for l, b in zip(res_lab, band) if b])),
            "auroc_gold_cov_hi": rnd(auroc(s, cov_lab)),
            "auroc_gold_cov_hi_in_band": rnd(auroc([x for x, b in zip(s, band) if b], [l for l, b in zip(cov_lab, band) if b])),
        }
    return out


# --------------------------------------------------------------------------------------------
# versions
# --------------------------------------------------------------------------------------------
def _neg(x):
    return float("nan") if isnan(x) else 1.0 - x


def v1_like(t, yield_key="yield_R2", nec_key="necessity_v1"):
    if t["n_graph_calls"] == 0:
        return t[nec_key]
    return nanmean([t[yield_key], t["follow_w2"], t[nec_key], _neg(t["waste_p95_s50"])])


def v4(t, veto=0.25):
    if t["need_v1"] != 1:
        return float("nan")
    if not isnan(t["halluc_rate"]) and t["halluc_rate"] > veto:
        return 0.0
    return 0.0 if t["n_graph_calls"] == 0 else t["novel_yield_R2"]


V6_FEATURES = ["resolution_precision", "yield_R2", "follow_w2", "necessity_v1", "waste_p95_s50",
               "halluc_rate", "async_hazard"]


def v6(t, coef):
    if not coef:
        return float("nan")
    z = coef["intercept"]
    for f in V6_FEATURES:
        x = t.get(f)
        x = coef["fill"][f] if isnan(x) else x
        z += coef["w"][f] * (x - coef["mean"][f]) / (coef["sd"][f] or 1.0)
    return 1 / (1 + math.exp(-z))


def fit_v6(task_rows, target="gold_file_coverage", l2=1.0, iters=3000, lr=0.1):
    rows = [t for t in task_rows if not isnan(t.get(target))]
    if len(rows) < 20:
        return None
    med = sorted(t[target] for t in rows)[len(rows) // 2]
    y = [1.0 if t[target] > med else 0.0 for t in rows]
    fill, mean, sd = {}, {}, {}
    for f in V6_FEATURES:
        vals = [t[f] for t in rows if not isnan(t.get(f))]
        fill[f] = sum(vals) / len(vals) if vals else 0.0
        xs = [fill[f] if isnan(t.get(f)) else t[f] for t in rows]
        mean[f] = sum(xs) / len(xs)
        sd[f] = math.sqrt(sum((x - mean[f]) ** 2 for x in xs) / len(xs)) or 1.0
    X = [[((fill[f] if isnan(t.get(f)) else t[f]) - mean[f]) / sd[f] for f in V6_FEATURES] for t in rows]
    w, b = [0.0] * len(V6_FEATURES), 0.0
    n = len(rows)
    for _ in range(iters):
        gw, gb = [l2 * wi / n for wi in w], 0.0
        for xi, yi in zip(X, y):
            p = 1 / (1 + math.exp(-(b + sum(a * c for a, c in zip(w, xi)))))
            e = p - yi
            gb += e / n
            for j in range(len(w)):
                gw[j] += e * xi[j] / n
        w = [wi - lr * g for wi, g in zip(w, gw)]
        b -= lr * gb
    return {"target": target, "median": med, "n": n, "intercept": b, "w": dict(zip(V6_FEATURES, w)),
            "fill": fill, "mean": mean, "sd": sd}


def add_versions(t, coef=None):
    t["v0"] = t["n_graph_calls"]
    t["v1"] = v1_like(t)
    t["v2"] = v1_like(t, nec_key="necessity_v2") if not isnan(t["necessity_v2"]) else float("nan")
    t["v3"] = v1_like(t, yield_key="yield_R1")
    t["v4"] = v4(t)
    t["v5"] = t["follow_w2"]
    t["v6"] = v6(t, coef)
    t["v7"] = t.get("tqs", float("nan"))
    return t


# --------------------------------------------------------------------------------------------
# run directories
# --------------------------------------------------------------------------------------------
def run_meta_for(run_dir, arm=None):
    run_dir = Path(run_dir)
    name = run_dir.name
    m = re.search(r"_?s(?:eed)?(\d+)$", name)
    seed = int(m.group(1)) if m else None
    resolved, records = {}, {}
    tr = run_dir / "task_results.jsonl"
    if tr.exists():
        for line in open(tr):
            try:
                r = json.loads(line)
            except Exception:
                continue
            resolved[r.get("instance_id")] = bool(r.get("resolved"))
            records[r.get("instance_id")] = r
    return {"run": str(run_dir), "arm": arm or re.sub(r"_?s(?:eed)?\d+$", "", name), "seed": seed,
            "resolved": resolved, "records": records}


def iter_traces(run_dir):
    for p in sorted(Path(run_dir, "traces").glob("trace_*.json")):
        yield p.stem[len("trace_"):], p


def log_parser_hits(run_dir, iid):
    p = Path(run_dir, "logs", f"{iid}.log")
    if not p.exists():
        return None
    txt = p.read_text(errors="replace")
    return len(PARSER_ERROR_RE.findall(txt))


def gold_found_by_task(task_rows):
    by = collections.defaultdict(list)
    for t in task_rows:
        by[t["instance_id"]].append(t["gold_file_read"])
    return {k: (sum(v) / len(v)) >= 0.5 for k, v in by.items()}


# --------------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------------
COMPONENT_EVENTS = {
    # component: (numerator field on call rows or task rows, denominator)
    "resolution_precision": ("calls", lambda r: r["precise"], lambda r: True),
    "yield_R1": ("calls", lambda r: r["yield_R1"], lambda r: True),
    "yield_R2": ("calls", lambda r: r["yield_R2"], lambda r: True),
    "yield_R3": ("calls", lambda r: r["yield_R3"], lambda r: True),
    "novel_yield_R2": ("calls", lambda r: r["novel_yield_R2"], lambda r: True),
    "follow_w1": ("calls", lambda r: bool(r["follow_w1"]), lambda r: r["follow_w1"] is not None),
    "follow_w2": ("calls", lambda r: bool(r["follow_w2"]), lambda r: r["follow_w2"] is not None),
    "follow_w3": ("calls", lambda r: bool(r["follow_w3"]), lambda r: r["follow_w3"] is not None),
    "waste_p95_s50": ("calls", lambda r: r["waste_p95_s50"], lambda r: True),
    "waste_p90_s75": ("calls", lambda r: r["waste_p90_s75"], lambda r: True),
    "async_hazard": ("calls", lambda r: r["async_hazard"], lambda r: True),
    "necessity_v1": ("tasks", lambda t: t["necessity_v1"] == 1.0, lambda t: not isnan(t["necessity_v1"])),
    "necessity_v2": ("tasks", lambda t: t["necessity_v2"] == 1.0, lambda t: not isnan(t["necessity_v2"])),
    "made_graph_call": ("tasks", lambda t: t["made_graph_call"] == 1, lambda t: True),
    "gold_file_read": ("tasks", lambda t: t["gold_file_read"] == 1, lambda t: t["n_gold_files"] > 0),
    "resolved": ("tasks", lambda t: t["resolved"] is True, lambda t: t["resolved"] is not None),
}


def summarize(task_rows, call_rows):
    def block(trows, crows):
        out = {"n_tasks": len(trows), "n_graph_calls": len(crows),
               "n_tool_calls": sum(t["n_tool_calls"] for t in trows)}
        for comp, (lvl, num, den) in COMPONENT_EVENTS.items():
            rows = crows if lvl == "calls" else trows
            d = [r for r in rows if den(r)]
            k = sum(bool(num(r)) for r in d)
            out[comp] = {"k": k, "n": len(d), "rate": round(k / len(d), 4) if d else None, "wilson95": wilson(k, len(d))}
        hal = sum(t["n_unknown_tool"] + t["n_malformed_args"] + t["n_unresolvable_graph_args"] for t in trows)
        out["hallucination_channel"] = {"k": hal, "n": out["n_tool_calls"], "rate": round(hal / out["n_tool_calls"], 4) if out["n_tool_calls"] else None,
                                        "wilson95": wilson(hal, out["n_tool_calls"])}
        out["parser_failures_text"] = sum(t["n_parser_failure_text"] for t in trows)
        out["end_reason"] = dict(collections.Counter(t["end_reason"] for t in trows))
        ce = sum(t["ended_context_exceeded"] for t in trows)
        out["ended_context_exceeded"] = {"k": ce, "n": len(trows), "wilson95": wilson(ce, len(trows))}
        for v in VERSION_KEYS:
            out[f"mean_{v}"] = (lambda m: None if isnan(m) else round(m, 4))(nanmean([t[v] for t in trows]))
        return out

    s = {"scorecard_version": SCORECARD_VERSION, "pooled": block(task_rows, call_rows), "by_arm": {}, "by_repo": {}, "by_cell": {}}
    for field, dest in (("arm", "by_arm"), ("repo", "by_repo"), ("cell", "by_cell")):
        keys = sorted({str(t[field]) for t in task_rows})
        for k in keys:
            tr = [t for t in task_rows if str(t[field]) == k]
            ids = {(t["run"], t["instance_id"]) for t in tr}
            cr = [c for c in call_rows if (c["run"], c["instance_id"]) in ids]
            s[dest][k] = block(tr, cr)
    return s


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def dump_format(run_dirs, out=sys.stdout):
    for rd in run_dirs:
        for iid, p in iter_traces(rd):
            parsed = parse_trace(p)
            print(f"== {p}  schema={parsed['schema']} steps={parsed['n_steps']}", file=out)
            print("   step keys:", parsed["step_keys"], file=out)
            print("   diag:", json.dumps(parsed["diag"], sort_keys=True), file=out)
            print("   events:", collections.Counter(e["kind"] + (":" + e["cause"] if "cause" in e else "") for e in parsed["events"]), file=out)
            print("   tool names:", collections.Counter(c["name"] for c in parsed["calls"]), file=out)
            for c in parsed["calls"][:8]:
                r = c["result"] or {}
                print(f"   #{c['idx']:>2} turn={c['turn']} {c['name']}({json.dumps(c['args'])[:120]}) -> "
                      f"status={r.get('status')} raw_len={c['result_raw_len']} matched={c['matched_by']} usage={c['usage']}", file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--oracle-pack", required=True)
    ap.add_argument("--profile")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm")
    ap.add_argument("--no-tools-results", nargs="*", default=[])
    ap.add_argument("--brief-dir")
    ap.add_argument("--extra-tools", nargs="*", default=[])
    ap.add_argument("--fit-v6", help="task csv to fit v6 on (B2 seed 1, non-slice tasks)")
    ap.add_argument("--v6-coef", help="json with v6 coefficients")
    ap.add_argument("--dump-format", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.dump_format:
        with open(out / "dump_format.txt", "w") as f:
            dump_format(a.results, out=f)
        print((out / "dump_format.txt").read_text())

    oracle = Oracle(a.oracle_pack, a.profile)
    coef = json.load(open(a.v6_coef)) if a.v6_coef else None
    task_rows, call_rows, parse_report = [], [], {}
    for rd in a.results:
        meta = run_meta_for(rd, a.arm)
        for iid, p in iter_traces(rd):
            parsed = parse_trace(p)
            brief = None
            if a.brief_dir and Path(a.brief_dir, f"{iid}.md").exists():
                brief = sorted(set(PATH_TOKEN_RE.findall(Path(a.brief_dir, f"{iid}.md").read_text())))
            t, cr = score_trace(iid, parsed, oracle, meta, brief, a.extra_tools)
            t["log_parser_error_hits"] = log_parser_hits(rd, iid)
            task_rows.append(t)
            call_rows.extend(cr)
            parse_report[f"{meta['run']}::{iid}"] = parsed["diag"]

    if a.no_tools_results:
        nt_rows = []
        for rd in a.no_tools_results:
            meta = run_meta_for(rd, "no_tools")
            for iid, p in iter_traces(rd):
                t, _ = score_trace(iid, parse_trace(p), oracle, meta)
                nt_rows.append(t)
        found_tools, found_nt = gold_found_by_task(task_rows), gold_found_by_task(nt_rows)
        for t in task_rows:
            if t["instance_id"] in found_nt:
                need = int(found_tools.get(t["instance_id"], False) and not found_nt[t["instance_id"]])
                t["need_v2"] = need
                t["necessity_v2"] = float(t["made_graph_call"] == need)

    if a.fit_v6:
        rows = list(csv.DictReader(open(a.fit_v6)))
        for r in rows:
            for k, v in r.items():
                try:
                    r[k] = float(v) if v not in ("", "None") else float("nan")
                except ValueError:
                    pass
        coef = fit_v6(rows)
        (out / "v6_coef.json").write_text(json.dumps(coef, indent=1))

    for t in task_rows:
        add_versions(t, coef)

    with open(out / "scorecard_calls.jsonl", "w") as f:
        for r in call_rows:
            f.write(json.dumps(r, default=str) + "\n")
    if task_rows:
        cols = list(task_rows[0].keys())
        with open(out / "scorecard_tasks.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for t in task_rows:
                w.writerow({k: ("" if v is None or (isinstance(v, float) and math.isnan(v)) else v) for k, v in t.items()})
    summ = summarize(task_rows, call_rows)
    lc = length_check(task_rows)
    summ["length_check"] = lc
    (out / "length_check.json").write_text(json.dumps(lc, indent=1))
    summ["inputs"] = {"results": a.results, "oracle_pack": a.oracle_pack, "oracle_pack_version": oracle.pack.get("version"),
                      "no_tools_results": a.no_tools_results, "params_default": PARAMS_DEFAULT}
    (out / "scorecard_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    (out / "parse_report.json").write_text(json.dumps(parse_report, indent=1))
    (out / "scorecard_versions.md").write_text(VERSIONS_DOC.format(ver=SCORECARD_VERSION))
    print(f"scorecard {SCORECARD_VERSION}: {len(task_rows)} task rows, {len(call_rows)} graph-call rows -> {out}")
    return task_rows, call_rows, summ


if __name__ == "__main__":
    main()

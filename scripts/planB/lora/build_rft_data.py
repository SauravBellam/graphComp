#!/usr/bin/env python3
"""Build LoRA training data (RFT / distillation) from swegemma ATIF traces (Plan B/C, Oct 14).

    python build_rft_data.py --runs RUN_DIR [RUN_DIR ...] --profile PROFILE.csv --out OUT_DIR
                             [--max-per-task 2] [--require-graph-call] [--max-est-tokens 16000] [--max-repeat-calls N]

RUN_DIR = a merged run folder (task_results.jsonl + traces/). PROFILE = the validity profile of the environment the runs
came from (sub_valid). A trajectory is kept only if: resolved; sub_valid; NOT in the official dev or holdout slice
(scripts/planB/official_slices; checked again at the end, the script refuses to write otherwise); no infra error, no
context-exceeded / error ending; every agent step has at most one tool call with its result. Up to --max-per-task
trajectories per task (fewest tool calls first), so easy tasks solved by many seeds do not dominate.

Output (OUT_DIR): train.jsonl, one trajectory per line:
  {"instance_id", "repo", "source", "n_turns", "est_tokens", "n_graph_calls",
   "messages": [system, user, (assistant{reasoning_content, content, tool_calls}, tool{tool_call_id, name, content})..., final assistant]}
The agent step's text is stored as reasoning_content (ATIF event_type "thinking"); the final step's text as content. The trainer
renders these with the model's own chat template and puts the loss on assistant turns only. Also: manifest.json (selection,
exclusions with reasons, per-repo counts, sources) and excluded.csv.
"""
import argparse, csv, hashlib, json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLICES = HERE.parent / "official_slices"
GRAPH_TOOLS = {"get_code_neighbors", "search_similar_code", "get_code_subgraph"}


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def longest_repeat(msgs):
    """Longest run of consecutive identical tool calls (same name and arguments): the agent spinning."""
    calls = [json.dumps(m["tool_calls"][0]["function"], sort_keys=True) for m in msgs if m.get("tool_calls")]
    best = run = 0
    for i, c in enumerate(calls):
        run = run + 1 if i and c == calls[i - 1] else 1
        best = max(best, run)
    return best


def trace_to_messages(trace):
    steps = trace["steps"]
    msgs, n_graph, problems = [], 0, []
    sys_ = [s for s in steps if s.get("source") == "system"]
    usr = [s for s in steps if s.get("source") == "user"]
    if not sys_ or not usr:
        return None, 0, ["no system or user step"]
    msgs.append({"role": "system", "content": sys_[0]["message"]})
    msgs.append({"role": "user", "content": usr[0]["message"]})
    agent = [s for s in steps if s.get("source") == "agent"]
    k, pending = 0, None
    for s in agent:
        tcs = s.get("tool_calls") or []
        ev = (s.get("extra") or {}).get("event_type")
        if not tcs:
            if ev == "final" or s is agent[-1]:
                # the text-only "thinking" step just before "final" is the reasoning of the final answer
                m = {"role": "assistant", "content": s.get("message", ""),
                     "_prompt_tokens": (pending or s).get("metrics", {}).get("prompt_tokens") if (pending or s).get("metrics") else None}
                if pending is not None and pending.get("message") and pending.get("message") != s.get("message"):
                    m["reasoning_content"] = pending["message"]
                msgs.append(m)
            else:
                pending = s
            continue
        pending = None
        if len(tcs) > 1:
            problems.append(f"step {s.get('step_id')}: {len(tcs)} tool calls (only the first result is stored)")
        obs = s.get("observation")
        if not isinstance(obs, dict) or "content" not in obs:
            problems.append(f"step {s.get('step_id')}: no tool result")
        calls = []
        for tc in tcs:
            k += 1
            cid = tc.get("tool_call_id") or f"call_{k}"
            name = tc.get("function_name")
            n_graph += name in GRAPH_TOOLS
            calls.append({"id": cid, "type": "function",
                          # a mapping, not a JSON string: the Gemma 4 chat template raises on string arguments
                          "function": {"name": name, "arguments": tc.get("arguments") or {}}})
        msgs.append({"role": "assistant", "content": "", "reasoning_content": s.get("message", ""), "tool_calls": calls,
                     # recorded prompt size of this LLM call: train_lora.py --check compares its renderings against it
                     "_prompt_tokens": (s.get("metrics") or {}).get("prompt_tokens")})
        msgs.append({"role": "tool", "tool_call_id": calls[0]["id"], "name": calls[0]["function"]["name"],
                     "content": (obs or {}).get("content", "")})
    if msgs[-1]["role"] != "assistant" or msgs[-1].get("tool_calls"):
        problems.append("does not end with a final assistant message")
    return msgs, n_graph, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-task", type=int, default=2)
    ap.add_argument("--require-graph-call", action="store_true", help="adoption data: keep only trajectories with a graph call")
    ap.add_argument("--max-repeat-calls", type=int, help="drop trajectories with more than N consecutive identical tool calls "
                    "(spinning; off by default)")
    ap.add_argument("--max-est-tokens", type=int, default=16384, help="drop trajectories whose longest context is above this (16384 = the effective input budget at max_output_tokens 16384)")
    a = ap.parse_args()

    dev = {l.strip() for l in open(SLICES / "dev_slice.txt") if l.strip()}
    hold = {l.strip() for l in open(SLICES / "holdout_slice.txt") if l.strip()}
    prof = {r["instance_id"]: r for r in csv.DictReader(open(a.profile))}
    cands, excluded = {}, []
    for rd in map(Path, a.runs):
        for r in read_jsonl(rd / "task_results.jsonl"):
            iid = r["instance_id"]
            why = None
            if iid in dev or iid in hold:
                why = "in an official slice (dev/holdout): never trained on"
            elif not r.get("resolved"):
                why = "not resolved"
            elif prof.get(iid, {}).get("sub_valid") != "True":
                why = f"not valid here ({prof.get(iid, {}).get('sub_reason')})"
            elif r.get("_infra_error") or r.get("error"):
                why = f"error: {str(r.get('error'))[:60]}"
            tp = rd / "traces" / f"trace_{iid}.json"
            if why is None and not tp.exists():
                why = "trace missing"
            if why:
                excluded.append((str(rd), iid, why))
                continue
            trace = json.load(open(tp))
            msgs, n_graph, problems = trace_to_messages(trace)
            if msgs is None or problems:
                excluded.append((str(rd), iid, "; ".join(problems)[:200]))
                continue
            pt = [((s.get("metrics") or {}).get("prompt_tokens") or 0) for s in trace["steps"] if s.get("source") == "agent"]
            est = max(pt) if pt else 0
            if est > a.max_est_tokens:
                excluded.append((str(rd), iid, f"longest context {est} > {a.max_est_tokens}"))
                continue
            if a.max_repeat_calls and (rep_ := longest_repeat(msgs)) > a.max_repeat_calls:
                excluded.append((str(rd), iid, f"spinning: {rep_} identical tool calls in a row > {a.max_repeat_calls}"))
                continue
            if a.require_graph_call and n_graph == 0:
                excluded.append((str(rd), iid, "no graph call (--require-graph-call)"))
                continue
            n_turns = sum(1 for m in msgs if m["role"] == "assistant")
            cands.setdefault(iid, []).append({"instance_id": iid, "repo": iid.split("_")[0], "source": str(rd),
                                              "n_turns": n_turns, "est_tokens": est, "n_graph_calls": n_graph,
                                              "tool_calls": int(r.get("tool_calls") or 0), "messages": msgs})
    rows = []
    for iid, lst in sorted(cands.items()):
        lst.sort(key=lambda x: (x["tool_calls"], x["source"]))
        for x in lst[a.max_per_task:]:
            excluded.append((x["source"], iid, f"over --max-per-task {a.max_per_task}"))
        rows += lst[:a.max_per_task]
    leak = sorted({x["instance_id"] for x in rows} & (dev | hold))
    if leak:
        sys.exit(f"REFUSING to write: dev/holdout tasks in the training set: {leak}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for x in rows:
            x = {k: v for k, v in x.items() if k != "tool_calls"}
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(out / "excluded.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "instance_id", "reason"])
        w.writerows(excluded)
    reasons = Counter(r for _, _, r in excluded)
    man = {"n_trajectories": len(rows), "n_tasks": len({x["instance_id"] for x in rows}),
           "by_repo": dict(Counter(x["repo"] for x in rows)),
           "assistant_turns": sum(x["n_turns"] for x in rows), "graph_calls": sum(x["n_graph_calls"] for x in rows),
           "est_tokens_max_context": {"median": sorted(x["est_tokens"] for x in rows)[len(rows) // 2] if rows else None,
                                      "max": max((x["est_tokens"] for x in rows), default=None)},
           "excluded_reasons": dict(reasons.most_common()), "runs": a.runs, "profile": a.profile,
           "rules": {"max_per_task": a.max_per_task, "require_graph_call": a.require_graph_call, "max_est_tokens": a.max_est_tokens,
                     "max_repeat_calls": a.max_repeat_calls,
                     "never": "official dev and holdout slices"},
           "train_sha256": hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()[:16]}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("n_trajectories", "n_tasks", "by_repo", "assistant_turns", "graph_calls",
                                          "est_tokens_max_context", "excluded_reasons")}, indent=1))


if __name__ == "__main__":
    main()

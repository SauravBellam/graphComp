"""Synthetic swegemma trace built with the REAL adk_eval_core 0.1.0 ATIF converter (no google.adk needed).

Writes a results dir: <out>/traces/trace_<id>.json, task_results.jsonl, logs/<id>.log.
Scenarios on rich_3518 (gold file rich/table.py, R1 = rich.table.Table.add_row):
  good:   neighbors(Table.add_row) -> returns R nodes -> read_file rich/table.py (follow) ; repeated call (waste: all seen)
  spam:   graph call every turn on the same node (anti-Goodhart shape)
  fastapi_12942: call naming a gold async node absent from the provided graph (async hazard) + unknown tool + parser-failure text
"""
import importlib.util, json, sys, types, os
from pathlib import Path

site = Path(next(p for p in sys.path if p.endswith("dist-packages") and Path(p, "adk_eval_core").exists()))
pkg = types.ModuleType("adk_eval_core"); pkg.__path__ = [str(site / "adk_eval_core")]
tr = types.ModuleType("adk_eval_core.tracing"); tr.__path__ = [str(site / "adk_eval_core/tracing")]
sys.modules["adk_eval_core"] = pkg; sys.modules["adk_eval_core.tracing"] = tr
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
atif = load("adk_eval_core.tracing.atif", site / "adk_eval_core/tracing/atif.py")
conv = load("adk_eval_core.tracing.atif_converter", site / "adk_eval_core/tracing/atif_converter.py")


class E:  # TraceEntry stand-in with the same fields
    def __init__(self, event_type, **kw):
        self.event_type = event_type; self.elapsed = kw.pop("elapsed", 0.0); self.timestamp = 0.0
        self.author = kw.pop("author", "swe_b2_agent"); self.content = kw.pop("content", "")
        self.tool_name = kw.pop("tool_name", ""); self.tool_args = kw.pop("tool_args", None)
        self.tool_result = kw.pop("tool_result", ""); self.usage = kw.pop("usage", None)
        self.model_version = ""; self.metadata = kw.pop("metadata", None)


class T:
    def __init__(self, entries): self.entries = entries
    def summarize(self): return {"total_prompt_tokens": 1000, "total_completion_tokens": 200, "total_tokens": 1200}


def call(name, args, result, usage=True):
    u = {"prompt_tokens": 900, "completion_tokens": 50, "total_tokens": 950} if usage else None
    return [E("thinking", content="hmm"), E("tool_call", tool_name=name, tool_args=args, usage=u),
            E("tool_response", tool_name=name, tool_result=json.dumps(result), author="swe_b2_agent")]


def write(out, iid, entries, resolved):
    t = conv.convert_trace_to_atif(T(entries), agent_name="adk-eval-core")
    (out / "traces").mkdir(parents=True, exist_ok=True); (out / "logs").mkdir(exist_ok=True)
    (out / "traces" / f"trace_{iid}.json").write_text(json.dumps(t.to_dict(), indent=1))
    (out / "logs" / f"{iid}.log").write_text("turn 1\n")
    with open(out / "task_results.jsonl", "a") as f:
        f.write(json.dumps({"instance_id": iid, "resolved": resolved}) + "\n")


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "task_results.jsonl").write_text("")
    pre = [E("system_instruction", content="sys", author="harness"), E("task_prompt", content="task", author="harness")]
    nb = {"status": "ok", "node": "Table.add_row", "neighbors": ["rich_x;rich.table.Table.add_row", "rich_x;rich.table.Table._calculate_column_widths", "rich_x;tests.test_table.test_add_row"], "count": 3}
    good = pre + call("get_code_neighbors", {"node": "Table.add_row"}, nb) \
        + call("read_file", {"filepath": "rich/table.py", "start_line": 1, "end_line": 150}, {"status": "ok", "content": "..."}) \
        + call("get_code_neighbors", {"node": "rich.table.Table.add_row"}, nb) \
        + [E("compaction", content="summary", author="compactor")] \
        + call("edit_file", {"filepath": "rich/table.py", "old_string": "a", "new_string": "b"}, {"status": "ok"}) \
        + call("submit_patch", {}, {"status": "ok", "patch_size": 10, "files_changed": 1}) \
        + [E("final", content="done")]
    write(out, "rich_3518", good, True)
    haz = pre + call("get_code_neighbors", {"node": "solve_dependencies", "edge_type": "CALLS"},
                     {"status": "ok", "node": "field_annotation_is_complex", "neighbors": [], "count": 0}) \
        + call("grep_repo", {"pattern": "x"}, {"status": "error", "error_type": "ToolNotFound", "error_message": "unknown tool grep_repo"}) \
        + [E("text", content="<|tool_call>call:read_file{filepath:<|\"|>fastapi/_compat.py<|\"|>}")] \
        + [E("continuation_nudge", content="Your previous response reached the token limit before the tool call finished closing (<|tool_call|> was cut off).", author="harness")] \
        + call("read_file", {"filepath": "/workspace/fastapi/_compat.py"}, {"status": "ok", "content": "..."})
    write(out, "fastapi_12942", haz, False)
    # parallel calls in one step: two tool_calls then two responses
    par = pre + [E("tool_call", tool_name="read_file", tool_args={"filepath": "rich/panel.py"}),
                 E("tool_call", tool_name="get_code_subgraph", tool_args={"nodes": ["Panel.__rich_console__", "render"]}),
                 E("tool_response", tool_name="read_file", tool_result=json.dumps({"status": "ok"})),
                 E("tool_response", tool_name="get_code_subgraph", tool_result=json.dumps({"status": "ok", "nodes": ["rich.panel.Panel.__rich_console__"], "edges": []}))]
    write(out, "rich_3718", par, False)
    print("fixture at", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixture_run_s1")

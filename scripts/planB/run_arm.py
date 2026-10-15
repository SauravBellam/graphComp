"""Run one Plan B arm (or the gold / empty-patch passes) through swegemma 0.2.11's own Evaluator.

    python run_arm.py --mode agent --submission SUBDIR --results OUT --task-ids a b c ...
    python run_arm.py --mode empty --results OUT --task-ids ...      # Phase 2 with an empty patch
    python run_arm.py --mode gold  --results OUT --task-ids ...      # Phase 2 with the reference patch

Why not `swegemma eval`: the CLI's EvalConfig leaves events_compaction_config and
context_cache_config at None, while the competition's scripts/inference.py sets them; and the CLI
registry skips adk_submission's litellm reasoning patch. Plan B arms must match the scoring path, so
this runner sets both (values from the harness README section 7.2) and builds the model registry
with VllmServer.create_model_registry, like inference.py. Pass --cli-parity to get the CLI's
behaviour instead. Every choice is written to OUT/run_config.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import time
from pathlib import Path

MODEL_ALIAS = "gemma-4-31b-it-qat-w4a16-ct"


def build_models(a):
    if a.mode != "agent":
        from swegemma.cli import _build_cli_model_registry
        return _build_cli_model_registry(submission_dir=a.submission)
    if a.cli_parity:
        os.environ.setdefault("LOCAL_INFERENCE_URL", f"http://127.0.0.1:{a.port}/v1")
        from swegemma.cli import _build_cli_model_registry
        return _build_cli_model_registry(submission_dir=a.submission)
    # As the sample notebook: litellm.drop_params, openai/ prefix, aliases = declared model + target name.
    import litellm
    litellm.drop_params = True
    from adk_submission import VllmConfig, VllmServer
    try:  # newer swegemma (Kaggle wheelhouse); 0.2.11 has no swegemma.models.discovery
        from swegemma.models.discovery import validate_single_declared_model
        declared = validate_single_declared_model(Path(a.submission))
    except ImportError:
        from adk_submission.discovery import discover_declared_models
        declared = sorted(discover_declared_models(a.submission))[0]
    # Oct 14: register the submission's own adapters (submission/adapters/<name>, agent.yaml "adapter: <name>") so the agent
    # can route to a LoRA served by vLLM under that name. No adapters folder -> None -> exactly the previous behaviour.
    try:
        from adk_submission import discover_adapters
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
        manifest = discover_adapters(str(a.submission), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS) \
            if (Path(a.submission) / "adapters").is_dir() else None
    except Exception:
        manifest = None
    srv = VllmServer(VllmConfig(model=a.model_path, port=a.port), adapter_manifest=manifest)
    return srv.create_model_registry(aliases=list(dict.fromkeys([declared, a.model_alias])),
                                     model_prefix="openai/", api_key="EMPTY")


def src_layout_shim():
    """requests and httpx keep their package in src/. Docker scoring exposes it through the editable install, but the
    subprocess sandbox venv has no pip (ensurepip unavailable), so their tests imported the host's installed copy
    (import_check.py, Oct 12). Put /workspace/src first on PYTHONPATH for every sandbox command: the agent's tools and
    the harness's own verify_task. No effect on repos without src/ (fastapi, rich)."""
    from swegemma.sandbox import SubprocessManager
    orig = SubprocessManager.exec

    def exec_(self, sandbox_id, command, *, timeout=None):
        return orig(self, sandbox_id, 'export PYTHONPATH="/workspace/src${PYTHONPATH:+:$PYTHONPATH}"; ' + command,
                    timeout=timeout)
    SubprocessManager.exec = exec_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["agent", "empty", "gold"], required=True)
    ap.add_argument("--tasks", required=True, help="tasks.jsonl")
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"])
    ap.add_argument("--task-ids", nargs="+")
    ap.add_argument("--max-tool-calls", type=int, default=50)
    ap.add_argument("--max-time-minutes", type=float, default=30.0)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--timeout-seconds", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--shard-index", type=int)
    ap.add_argument("--num-shards", type=int)
    ap.add_argument("--model-path", default="")
    ap.add_argument("--model-alias", default=MODEL_ALIAS)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cli-parity", action="store_true", help="no compaction/cache, CLI registry")
    ap.add_argument("--compaction-interval", type=int, default=15)
    ap.add_argument("--briefs", help="B4 push: JSON {instance_id: {brief, problem_sha256}} prepended to the problem statement")
    a = ap.parse_args()

    from swegemma.evaluate import EvalConfig, Evaluator
    if a.sandbox == "subprocess":
        src_layout_shim()

    cache_cfg = comp_cfg = None
    if not a.cli_parity:
        from google.adk.agents.context_cache_config import ContextCacheConfig
        try:
            from google.adk.apps._configs import EventsCompactionConfig
        except ImportError:
            from google.adk.apps.app import EventsCompactionConfig
        # Values from the competition sample notebook (compaction_interval=15; the README's section 7.2 says 5).
        comp_cfg = EventsCompactionConfig(compaction_interval=a.compaction_interval, overlap_size=2,
                                          token_threshold=14336, event_retention_size=5)
        cache_cfg = ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10)

    results = Path(a.results)
    results.mkdir(parents=True, exist_ok=True)
    # As the sample notebook: explicit limits, generation constraints, adapters, graph/embedding/wheel dirs.
    from swegemma.config import build_submission_limits
    limits, gen_constraints = build_submission_limits()
    data_dir = Path(a.tasks).parent
    try:
        from adk_submission import discover_adapters
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
        adapters = discover_adapters(str(a.submission), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    except Exception:
        adapters = None
    extra = dict(limits=limits, generation_constraints=gen_constraints, adapter_manifest=adapters,
                 graph_dir=str(data_dir / "graphs"), embeddings_dir=str(data_dir / "embeddings"))
    if (data_dir / "wheels").is_dir():
        extra["wheels_dir"] = data_dir / "wheels"
    cfg = EvalConfig(
        tasks_path=Path(a.tasks), snapshots_dir=Path(a.snapshots), results_dir=results,
        submission_dir=Path(a.submission), models=build_models(a), sandbox=a.sandbox,
        timeout_seconds=a.timeout_seconds, max_tool_calls=a.max_tool_calls, max_turns=a.max_turns,
        max_time_minutes=a.max_time_minutes, task_ids=a.task_ids, skip_agent_patch=(a.mode == "empty"),
        concurrency=a.concurrency, shard_index=a.shard_index, num_shards=a.num_shards,
        display_mode="quiet", context_cache_config=cache_cfg, events_compaction_config=comp_cfg, **extra,
    )

    if a.mode == "gold":
        from adk_eval_core.tracing import SessionTrace

        class GoldEvaluator(Evaluator):
            async def _run_agent_sandbox(self, task, snapshot_path, task_index, total_tasks, **kw):
                task = self._hydrate_task_from_secret(task)
                return task.patch, None, SessionTrace()

        ev = GoldEvaluator(cfg)
    elif a.briefs:
        # B4 (push): the brief is prepended to the problem statement, so it is part of the first user message
        # (swegemma builds that message from task.problem_statement). Tasks without a brief run exactly as B2.
        import dataclasses
        import hashlib
        briefs = json.loads(Path(a.briefs).read_text())

        class PushEvaluator(Evaluator):
            def _hydrate_task_from_secret(self, task):
                task = super()._hydrate_task_from_secret(task)
                b = briefs.get(task.instance_id)
                if not b or task.problem_statement.startswith(b["brief"]):
                    return task
                if hashlib.sha256(task.problem_statement.encode()).hexdigest() != b["problem_sha256"]:
                    raise SystemExit(f"{task.instance_id}: problem statement differs from the one the brief was built from")
                return dataclasses.replace(task, problem_statement=b["brief"] + "\n## Issue\n" + task.problem_statement)

        ev = PushEvaluator(cfg)
    else:
        ev = Evaluator(cfg)

    import importlib.metadata as md
    vers = {}
    for p in ("swegemma", "adk-submission", "adk-eval-core", "google-adk", "litellm", "vllm", "torch"):
        try:
            vers[p] = md.version(p)
        except Exception:
            vers[p] = None
    run_cfg = {"argv": sys.argv, "mode": a.mode, "sandbox": a.sandbox, "task_ids": a.task_ids,
               "budget": {"max_tool_calls": a.max_tool_calls, "max_time_minutes": a.max_time_minutes,
                          "max_turns": a.max_turns, "timeout_seconds": a.timeout_seconds},
               "events_compaction": None if comp_cfg is None else comp_cfg.model_dump(exclude={"summarizer"}),
               "context_cache": None if cache_cfg is None else cache_cfg.model_dump(),
               "registry": "cli" if (a.cli_parity or a.mode != "agent") else "VllmServer.create_model_registry(openai/, drop_params) as sample notebook",
               "model_path": a.model_path, "wheels_dir": str(extra.get("wheels_dir")), "model_alias": a.model_alias, "python": platform.python_version(),
               "versions": vers, "src_layout_shim": a.sandbox == "subprocess", "graph_dir": cfg.graph_dir, "embeddings_dir": cfg.embeddings_dir,
               "started_unix": time.time(),
               "briefs": a.briefs and {"file": a.briefs, "n_tasks_with_brief": len(json.loads(Path(a.briefs).read_text()))}}
    (results / "run_config.json").write_text(json.dumps(run_cfg, indent=1, default=str))
    t0 = time.time()
    res = asyncio.run(ev.run())
    run_cfg["wall_seconds"] = round(time.time() - t0, 1)
    (results / "run_config.json").write_text(json.dumps(run_cfg, indent=1, default=str))
    print(f"[{a.mode}] {res.resolved}/{res.total} resolved in {run_cfg['wall_seconds']} s -> {results}")


if __name__ == "__main__":
    main()

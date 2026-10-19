"""Add --briefs (B4 push) to a run_arm.py source text. Used for the repo copy and the notebook copy."""
def patch(src: str) -> str:
    reps = [
        ('    ap.add_argument("--compaction-interval", type=int, default=15)\n',
         '    ap.add_argument("--compaction-interval", type=int, default=15)\n'
         '    ap.add_argument("--briefs", help="B4 push: JSON {instance_id: {brief, problem_sha256}} prepended to the problem statement")\n'),
        ('    else:\n        ev = Evaluator(cfg)\n',
         '    elif a.briefs:\n'
         '        # B4 (push): the brief is prepended to the problem statement, so it is part of the first user message\n'
         '        # (swegemma builds that message from task.problem_statement). Tasks without a brief run exactly as B2.\n'
         '        import dataclasses\n'
         '        import hashlib\n'
         '        briefs = json.loads(Path(a.briefs).read_text())\n'
         '\n'
         '        class PushEvaluator(Evaluator):\n'
         '            def _hydrate_task_from_secret(self, task):\n'
         '                task = super()._hydrate_task_from_secret(task)\n'
         '                b = briefs.get(task.instance_id)\n'
         '                if not b or task.problem_statement.startswith(b["brief"]):\n'
         '                    return task\n'
         '                if hashlib.sha256(task.problem_statement.encode()).hexdigest() != b["problem_sha256"]:\n'
         '                    raise SystemExit(f"{task.instance_id}: problem statement differs from the one the brief was built from")\n'
         '                return dataclasses.replace(task, problem_statement=b["brief"] + "\\n## Issue\\n" + task.problem_statement)\n'
         '\n'
         '        ev = PushEvaluator(cfg)\n'
         '    else:\n        ev = Evaluator(cfg)\n'),
        ('               "started_unix": time.time()}\n',
         '               "started_unix": time.time(),\n'
         '               "briefs": a.briefs and {"file": a.briefs, "n_tasks_with_brief": len(json.loads(Path(a.briefs).read_text()))}}\n'),
    ]
    for old, new in reps:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src

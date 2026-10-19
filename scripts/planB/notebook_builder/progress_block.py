import json as _json, subprocess as _sp, urllib.request as _ur

def gpu_status():
    try:
        out = _sp.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits",
                      shell=True, capture_output=True, text=True, timeout=10).stdout.split("\n")
        rows = [l.split(",") for l in out if l.strip()]
        return "GPU util " + "/".join(r[0].strip() for r in rows) + "% mem " + "/".join(r[1].strip() for r in rows) + " MiB" if rows else "GPU n/a"
    except Exception:
        return "GPU n/a"

def vllm_status():
    try:
        txt = _ur.urlopen("http://127.0.0.1:8000/metrics", timeout=3).read().decode(errors="replace")
        val = lambda k: sum(float(l.rsplit(" ", 1)[1]) for l in txt.splitlines() if l.startswith(k) and not l.startswith("#"))
        return f"vLLM running {val('vllm:num_requests_running'):.0f} waiting {val('vllm:num_requests_waiting'):.0f}"
    except Exception:
        return "vLLM not answering"

class Progress:
    # While a long command runs: print one line per finished task as soon as its record lands in task_results.jsonl
    # (outcome, tool calls, graph calls, how it ended, seconds), and a heartbeat every `every` seconds with counts,
    # elapsed time, GPU utilisation and the vLLM queue.
    GRAPH = ("get_code_neighbors", "search_similar_code", "get_code_subgraph")
    def __init__(self, path, total, label, every=120, poll=15):
        self.path, self.total, self.label, self.every, self.poll = Path(path), total, label, every, poll
        self._stop = threading.Event()
        self._lines = 0
        self.n = self.resolved = 0
    def _new(self):
        try:
            lines = [l for l in open(self.path) if l.strip()]
        except FileNotFoundError:
            return []
        new, self._lines = lines[self._lines:], len(lines)
        out = []
        for l in new:
            try:
                out.append(_json.loads(l))
            except Exception:
                pass
        return out
    def _task_line(self, r):
        self.n += 1
        self.resolved += bool(r.get("resolved"))
        iid = r.get("instance_id")
        extra = ""
        tp = self.path.parent / "traces" / f"trace_{iid}.json"
        if tp.exists():
            try:
                d = _json.load(open(tp))
                calls = [tc for s in d.get("steps", []) for tc in (s.get("tool_calls") or [])]
                extra = f" tool_calls={len(calls)} graph_calls={sum(tc.get('function_name') in self.GRAPH for tc in calls)}"
            except Exception:
                pass
        err = " ".join(str(v) for k, v in r.items() if isinstance(v, str) and re.search(r"error|reason", k, re.I))[:100]
        dur = r.get("duration_seconds")
        plog(f"    [{self.label} {self.n}/{self.total}] {iid} resolved={r.get('resolved')}{extra}"
             f"{' ' + str(round(dur)) + 's' if dur else ''}{' | ' + err if err else ''}")
    def _heartbeat(self):
        plog(f"    ... {self.label}: {self.n}/{self.total} done, {self.resolved} resolved, "
             f"{_el(time.time() - self._t0)} in this step | {gpu_status()} | {vllm_status()}")
    def _run(self):
        last = time.time()
        while not self._stop.wait(self.poll):
            for r in self._new():
                self._task_line(r)
            if time.time() - last >= self.every:
                last = time.time()
                self._heartbeat()
    def __enter__(self):
        self._t0 = time.time()
        plog(f"    {self.label}: started ({self.total} tasks) | {gpu_status()}")
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self
    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)
        for r in self._new():
            self._task_line(r)
        plog(f"    {self.label}: finished, {self.n}/{self.total} task results, {self.resolved} resolved, "
             f"{_el(time.time() - self._t0)}")


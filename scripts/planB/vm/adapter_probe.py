# Does the served LoRA change the model's output? Greedy-decode one prompt as base and as adapter on the running
# dev_runs vLLM server (port 8000), compare tokens/logprobs, and scan vLLM logs for ignored LoRA modules.
import glob, json, sys, urllib.request
URL, ADAPTER = "http://127.0.0.1:8000/v1", (sys.argv[1] if len(sys.argv) > 1 else "rft_lora")
def call(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))
ids = [m["id"] for m in call("/models")["data"]]
base = next(i for i in ids if i != ADAPTER)
assert ADAPTER in ids, f"adapter {ADAPTER!r} not served; served ids: {ids}"
msgs = [{"role": "user", "content": "A pytest test fails with ImportError in fastapi/routing.py. "
         "Which single tool call do you make first? Answer in one sentence."}]
out = {}
for name in (base, ADAPTER):
    r = call("/chat/completions", {"model": name, "messages": msgs, "temperature": 0, "max_tokens": 48, "logprobs": True,
                                   "top_logprobs": 1, "chat_template_kwargs": {"enable_thinking": False}})
    c = r["choices"][0]
    out[name] = ([t["token"] for t in c["logprobs"]["content"]], [t["logprob"] for t in c["logprobs"]["content"]])
    print(f"{name}: {c['message']['content']!r}")
(tb, lb), (ta, la) = out[base], out[ADAPTER]
n = min(len(tb), len(ta)); same = sum(x == y for x, y in zip(tb, ta))
diverge = next((i for i in range(n) if tb[i] != ta[i]), None)
dlp = max((abs(x - y) for x, y in zip(lb[:diverge or n], la[:diverge or n])), default=0.0)
print(json.dumps({"tokens_compared": n, "identical_tokens": same, "first_divergence": diverge,
                  "max_abs_logprob_diff_before_divergence": round(dlp, 4)}))
logs = glob.glob("/home/*/gc/runs_dev_e4b_rft_lora/provenance/vllm_*.log") + glob.glob("/tmp/*vllm*.log")
hits = [l.strip()[:200] for f in logs for l in open(f, errors="replace") if "will be ignored" in l or "LoRA" in l and "rror" in l]
print(f"vLLM logs scanned: {len(logs)}; LoRA ignore/error lines: {len(hits)}"); print(*hits[:10], sep="\n")
print("VERDICT:", "ADAPTER IS A NO-OP" if diverge is None and dlp < 1e-3 else "adapter changes the output")

# Dev-slice runs on a single L4 (2026-10-13T12:28:11Z)

- Serving: {'util': 0.9, 'max_model_len': 32768, 'kv': ['GPU KV cache size: 98,512 tokens', 'Maximum concurrency for 32,768 tokens per request: 15.06x'], 'launch': '/home/sauravbellam/gc/venv/bin/python -m vllm.entrypoints.openai.api_server --model /home/sauravbellam/gc/data/model_e4b --host 127.0.0.1 --port 8000 --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --reasoning-config \'{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}\' --default-chat-template-kwargs \'{"enable_thinking": true}\' --no-scheduler-reserve-full-isl', 'tp': 1}; concurrency 4 (fixed for every arm)
- Dev slice: 24 tasks, 23 resolvable on this VM (validity measured here)
- Comparisons are paired on this VM only (B2 here is the baseline; Kaggle runs are a different machine type).

| Arm | What | Tasks run | Resolved (valid tasks) | Made a graph call | Graph / tool calls | End reasons | Median s | Arm-only vs B2-only resolved, exact McNemar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B2 | baseline: all nine tools, sample prompt | 24/24 | 3/23 = 13% [5%, 32%] | 11/24 = 46% [28%, 65%] | 22/302 | {'context_exceeded': 5, 'none': 19} | 184 | (baseline) |
| FC | forced graph call every turn | 24/24 | 2/23 = 9% [2%, 27%] | 24/24 = 100% [86%, 100%] | 76/380 | {'none': 23, 'context_exceeded': 1} | 187 | 0 vs 1, p = 1.000 |
| QN | qualified names in graph-tool arguments | 24/24 | 2/23 = 9% [2%, 27%] | 13/24 = 54% [35%, 72%] | 42/318 | {'none': 22, 'context_exceeded': 2} | 130 | 0 vs 1, p = 1.000 |
| T0 | thinking budget 0 | 24/24 | 1/23 = 4% [1%, 21%] | 14/24 = 58% [39%, 76%] | 34/338 | {'none': 23, 'context_exceeded': 1} | 179 | 0 vs 2, p = 0.500 |
| O8K | max_output_tokens 8192 | 24/24 | 2/23 = 9% [2%, 27%] | 13/24 = 54% [35%, 72%] | 25/369 | {'none': 24} | 231 | 0 vs 1, p = 1.000 |

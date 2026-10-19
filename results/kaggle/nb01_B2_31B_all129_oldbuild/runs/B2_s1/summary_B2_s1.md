# B2_s1 on Kaggle (2026-10-12T20:19:35Z), scorecard 0.3.2

- Tasks with results: 129/129 (24/24 dev); complete: True
- Serving: {'util': 0.8, 'max_model_len': 32768, 'tp': 4, 'kv': ['GPU KV cache size: 27,712 tokens', 'Maximum concurrency for 32,768 tokens per request: 5.23x']}; concurrency 4
- This session stopped because: all remaining tasks run

## all tasks
- Made any graph call (adoption): 14/129 = 10.9% [6.6%, 17.4%]
- Graph calls / tool calls: 15/4207
- Resolved, sub_valid tasks only: 38/103 = 36.9% [28.2%, 46.5%]
- Gold file read: needs the oracle pack (laptop)
- End reasons: {'none': 73, 'tool_budget': 35, 'context_exceeded': 20, 'time_limit': 1}
- By repo (adoption | resolved on sub_valid):
  - fastapi: 11/67 = 16.4% [9.4%, 27.1%] | 22/60 = 36.7% [25.6%, 49.3%]
  - httpx: 0/1 = 0.0% [0.0%, 79.3%] | 0/0
  - requests: 2/13 = 15.4% [4.3%, 42.2%] | 0/0
  - rich: 1/48 = 2.1% [0.4%, 10.9%] | 16/43 = 37.2% [24.4%, 52.1%]

## dev slice
- Made any graph call (adoption): 2/24 = 8.3% [2.3%, 25.9%]
- Graph calls / tool calls: 2/745
- Resolved, sub_valid tasks only: 11/23 = 47.8% [29.2%, 67.0%]
- Gold file read: needs the oracle pack (laptop)
- End reasons: {'none': 18, 'tool_budget': 5, 'context_exceeded': 1}
- By repo (adoption | resolved on sub_valid):
  - fastapi: 2/14 = 14.3% [4.0%, 39.9%] | 7/13 = 53.8% [29.1%, 76.8%]
  - rich: 0/10 = 0.0% [0.0%, 27.8%] | 4/10 = 40.0% [16.8%, 68.7%]

## Timing
- Per-task wall seconds: median 434, mean 483, max 1806 (n 129)
- Agent time over all sessions: 5.52 h for 129 tasks = 23.4 tasks/h at concurrency 4; full 129 at this rate: 5.5 h



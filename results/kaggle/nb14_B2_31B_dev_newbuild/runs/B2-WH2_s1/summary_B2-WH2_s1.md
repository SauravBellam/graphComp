# B2-WH2_s1 on Kaggle (2026-10-13T14:02:23Z), scorecard 0.3.2

- Tasks with results: 24/24 (24/24 dev); complete: True
- Stack build: 0c81a00daf (REBUILT wheelhouse, differs from the validity session: pair only with B2 on this build)
- Serving: {'util': 0.8, 'max_model_len': 32768, 'tp': 4, 'kv': ['GPU KV cache size: 27,712 tokens', 'Maximum concurrency for 32,768 tokens per request: 5.23x']}; concurrency 4
- This session stopped because: all remaining tasks run

## all tasks
- Made any graph call (adoption): 3/24 = 12.5% [4.3%, 31.0%]
- Graph calls / tool calls: 3/818
- Resolved, sub_valid tasks only: 10/23 = 43.5% [25.6%, 63.2%]
- Gold file read: 22/24 = 91.7% [74.2%, 97.7%]
- End reasons: {'none': 18, 'tool_budget': 4, 'context_exceeded': 2}
- By repo (adoption | resolved on sub_valid):
  - fastapi: 3/14 = 21.4% [7.6%, 47.6%] | 5/13 = 38.5% [17.7%, 64.5%]
  - rich: 0/10 = 0.0% [0.0%, 27.8%] | 5/10 = 50.0% [23.7%, 76.3%]

## dev slice
- Made any graph call (adoption): 3/24 = 12.5% [4.3%, 31.0%]
- Graph calls / tool calls: 3/818
- Resolved, sub_valid tasks only: 10/23 = 43.5% [25.6%, 63.2%]
- Gold file read: 22/24 = 91.7% [74.2%, 97.7%]
- End reasons: {'none': 18, 'tool_budget': 4, 'context_exceeded': 2}
- By repo (adoption | resolved on sub_valid):
  - fastapi: 3/14 = 21.4% [7.6%, 47.6%] | 5/13 = 38.5% [17.7%, 64.5%]
  - rich: 0/10 = 0.0% [0.0%, 27.8%] | 5/10 = 50.0% [23.7%, 76.3%]

## Timing
- Per-task wall seconds: median 530, mean 596, max 1543 (n 24)
- Agent time over all sessions: 1.30 h for 24 tasks = 18.4 tasks/h at concurrency 4; full 129 at this rate: 7.0 h

## Length check (Spearman rho with n_tool_calls; AUROC resolved overall / in IQR band)
- v0: rho 0.2733, confounded True, AUROC 0.4786 / 0.5119
- v1: rho 0.3603, confounded True, AUROC 0.3714 / 0.5952
- v2: rho None, confounded None, AUROC None / None
- v3: rho 0.3603, confounded True, AUROC 0.3714 / 0.5952
- v4: rho None, confounded None, AUROC 0.5 / 0.5
- v5: rho None, confounded None, AUROC 0.0 / 0.0
- v6: rho None, confounded None, AUROC None / None
- v7: rho -0.3356, confounded True, AUROC 0.55 / 0.5238

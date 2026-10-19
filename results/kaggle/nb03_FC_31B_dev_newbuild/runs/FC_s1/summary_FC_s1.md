# FC_s1 on Kaggle (2026-10-14T07:41:19Z), scorecard 0.3.2

- Tasks with results: 24/24 (24/24 dev); complete: True
- Stack build: 0c81a00daf (REBUILT wheelhouse, differs from the validity session: pair only with B2 on this build)
- Serving: {'util': 0.8, 'max_model_len': 32768, 'tp': 4, 'kv': ['GPU KV cache size: 27,712 tokens', 'Maximum concurrency for 32,768 tokens per request: 5.23x']}; concurrency 4
- This session stopped because: all remaining tasks run

## all tasks
- Made any graph call (adoption): 24/24 = 100.0% [86.2%, 100.0%]
- Graph calls / tool calls: 57/801
- Resolved, sub_valid tasks only: 8/23 = 34.8% [18.8%, 55.1%]
- Gold file read: 20/24 = 83.3% [64.1%, 93.3%]
- End reasons: {'none': 12, 'context_exceeded': 6, 'tool_budget': 6}
- By repo (adoption | resolved on sub_valid):
  - fastapi: 14/14 = 100.0% [78.5%, 100.0%] | 5/13 = 38.5% [17.7%, 64.5%]
  - rich: 10/10 = 100.0% [72.2%, 100.0%] | 3/10 = 30.0% [10.8%, 60.3%]

## dev slice
- Made any graph call (adoption): 24/24 = 100.0% [86.2%, 100.0%]
- Graph calls / tool calls: 57/801
- Resolved, sub_valid tasks only: 8/23 = 34.8% [18.8%, 55.1%]
- Gold file read: 20/24 = 83.3% [64.1%, 93.3%]
- End reasons: {'none': 12, 'context_exceeded': 6, 'tool_budget': 6}
- By repo (adoption | resolved on sub_valid):
  - fastapi: 14/14 = 100.0% [78.5%, 100.0%] | 5/13 = 38.5% [17.7%, 64.5%]
  - rich: 10/10 = 100.0% [72.2%, 100.0%] | 3/10 = 30.0% [10.8%, 60.3%]

## Timing
- Per-task wall seconds: median 529, mean 515, max 1322 (n 24)
- Agent time over all sessions: 1.02 h for 24 tasks = 23.6 tasks/h at concurrency 4; full 129 at this rate: 5.5 h

## Length check (Spearman rho with n_tool_calls; AUROC resolved overall / in IQR band)
- v0: rho 0.4324, confounded True, AUROC 0.3867 / 0.4667
- v1: rho -0.2107, confounded False, AUROC 0.5078 / 0.55
- v2: rho None, confounded None, AUROC None / None
- v3: rho -0.2457, confounded False, AUROC 0.5 / 0.6
- v4: rho -0.0894, confounded False, AUROC 0.4286 / 0.4048
- v5: rho -0.0821, confounded False, AUROC 0.4286 / 0.5093
- v6: rho None, confounded None, AUROC None / None
- v7: rho 0.2214, confounded False, AUROC 0.5625 / 0.4833

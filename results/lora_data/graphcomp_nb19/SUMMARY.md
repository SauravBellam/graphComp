# Notebook 19 summary (2026-10-15T12:16:21Z)

- Teacher runs complete: False; training trajectories: 44
- LoRAs trained: 31B False, STUDENT_12B False
- Holdout: 27 scored tasks (official 24 + ['fastapi_15661', 'rich_3471', 'rich_3718']); coverage-only ['requests_6589', 'httpx_3672'] (not scorable here)
- Results (Wilson 95%):
  - BASE31: 14/27 [34.0%, 69.3%] (official 24: 13); finished 27; adapter ignored False; coverage {'requests_6589': {'duration_seconds': 391.34, 'tool_calls': 24, 'agent_patch_size': 918, 'error': None}, 'httpx_3672': {'duration_seconds': 425.52, 'tool_calls': 30, 'agent_patch_size': 2761, 'error': None}}

**SUBMISSION: BASE31**
- deployment budget {'max_tool_calls': 30, 'max_time_minutes': 5.0, 'max_turns': 80, 'timeout_seconds': 240}; worst case if all 120 tasks hit the cap: 11.2 h of 12
- /kaggle/working/submission (0.0 MB) and /kaggle/working/submission.zip
- Noise bar: two runs of one condition disagree on about 3 of 23 tasks; a smaller gap is not evidence.

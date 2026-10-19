# graphComp trajectories v1 (2026-10-15)

Built by `scripts/planB/lora/make_traj_dataset.py` (branch `orx/notebook-20-45-h-iterative-rft-campaign-round-1`) from notebook 01's seed-1 run (31B, all 129 tasks) and notebook 19's output (31B teacher seeds 11, 12, 13 and the BASE31 holdout run).

## Contents
| Path | What |
|---|---|
| `attempts.jsonl` | **every attempt (280)**: source, seed, task, split (pool / dev / holdout / extra_holdout / coverage), valid, resolved, seconds, tool calls, graph calls, error, trace path |
| `traces/<source>/` | every trace, resolved or not, as recorded by the harness |
| `train/verified_all.jsonl` | **45** resolved trajectories on valid pool tasks, fastest first, chat-message format for `train_lora.py` |
| `train/verified_accurate.jsonl` | **42**: the same, keeping only accurate tool use (every source edit on gold code, a reproduction or test after the last edit) |
| `train/nb19_train.jsonl` | **44**: exactly what notebook 19 trained on (fewest tool calls first, max 3 per task) |
| `tool_trace/` | tool tracing over every attempt on valid tasks (purpose, gold-code contact, shortest resolved routes, bigrams) |

Each `*_manifest.json` / `*_excluded.csv` gives the selection rules and why every other attempt was left out.

## Numbers
- 280 attempts: nb01 seed 1 129, nb19 s11 53, s12 53, s13 16, BASE31 holdout 29.
- 56 resolved attempts on valid pool tasks, covering **23 distinct tasks**.
- BASE31 on the 27-task holdout: **14/27** (13/24 official).

## Never trained on
The dev slice (24), the holdout slice (24), the extra holdout tasks (`fastapi_15661`, `rich_3471`, `rich_3718`; `never_train.txt`) and the coverage-only tasks. Holdout attempts are kept in `attempts.jsonl` / `traces/` for analysis only.

## Resolved attempts not in `verified_all` (counts include attempts outside the 56, e.g. extra-holdout and invalid tasks)
- 7 have no final assistant message: they ended on the tool or time cap, and their working tree was graded.
- 4 have a longest context above 16,384 tokens.
- 3 are extra-holdout tasks.
- 3 are on tasks that are not valid in this sandbox.

See `train/verified_all_excluded.csv` for the full list.

## Not yet included
- The 10 requests/httpx tasks made valid on the VM (no 31B attempts on them yet).
- External and synthetic tasks.

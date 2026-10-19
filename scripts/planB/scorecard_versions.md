# Scorecard versions (scorecard.py 0.3.2)

Pre-registered list from HANDOFF_planB.md 4.3. Add versions; never replace one silently.
Components are task-level rates (NaN when their denominator is 0). `yield_R*` = share of graph calls
whose returned nodes intersect R; `novel_yield_R*` counts only returned R-nodes not returned by an
earlier graph call; `follow_w*` = share of graph calls with returned nodes followed within W tool calls
by a read_file on a returned node's file; `necessity_v*` = 1 if (made a graph call) == (need label);
`waste_p*_s*` = share of graph calls that returned only already-returned nodes, or whose resolved
query node is a hub (provided-graph in-degree above the repo-graph percentile) with test share of
returned nodes at or above s; `halluc_rate` = (unknown tool names + malformed arguments + unresolvable
graph arguments) / tool calls, parser failures excluded; `async_hazard` = share of graph calls naming
a gold/R2 node that is absent from the provided graph.

| Version | Primary | Date | Parameters | Why |
| --- | --- | --- | --- | --- |
| v0 | n_graph_calls | 2026-10-12 | none | strawman |
| v1 | mean of available(yield_R2, follow_w2, necessity_v1, 1 - waste_p95_s50) | 2026-10-12 | R2, W2, label v1, hub 95/50 | Oct 11 proposal design. With no graph call the primary is necessity_v1 alone |
| v2 | v1 with necessity_v2 | 2026-10-12 | label v2 (needs --no-tools-results) | counterfactual need |
| v3 | v1 with yield_R1 | 2026-10-12 | R = gold only | strict yield |
| v4 | novel_yield_R2 on need_v1 = 1 tasks (0 if no call), NaN on need_v1 = 0; 0 if halluc_rate > 0.25 | 2026-10-12 | R2, label v1, veto 0.25 | selection-rule form; novel yield because repeated good calls inflated raw yield in the 0.2.0 fixture |
| v5 | follow_w2 | 2026-10-12 | W2 | delivery only |
| v6 | logistic(components) fitted with --fit-v6 on B2 seed 1 of the 81 non-slice tasks | 2026-10-12 | coefficients file | learned composite, expected to overfit |
| v7 | tqs: port of TQS (Li et al. 2026, arXiv 2610.13152), graph-agnostic trajectory-structure score | 2026-10-12 | TQS's published weights and constants; deviations below | external baseline: a graph-specific version must beat a validated generic trajectory score, not only v0 |

v7 port (TQS section 3.3, weights OEC .30, IAC .25, PSN/TTE/PED/DPI .10, SCP .05, renormalised over defined
components; NaN when fewer than 3 tool calls or fewer than 4 of 7 components defined). Deviations from the
paper, fixed here: steps are tool calls (thinking-only steps are not actions); action alphabet
(read_file, write_file, execute_bash, search, submit, other, think) with read_file -> read_file,
edit_file/write_file -> write_file, run_command -> search when the command starts with grep/rg/find/ls/ag/ack
else execute_bash, the three graph tools -> search, submit_patch -> submit, get_status -> other; F_t = file
paths in the call's arguments (read/edit/write filepath, *.py tokens in run_command); error step = result
status "error", or a non-zero exit code; IAC lexicon (ours; the paper does not publish its lexicon) in
TQS_LEXICON, applied to the last 400 characters of the thought text preceding the call, the category whose
keyword occurs last wins; DPI: early = fewer than 5 tool calls, no_edit = no write_file without error,
lp = share of run_command calls repeating an earlier identical command, er = share of error steps repeating
an earlier error step's (action, files, first 100 characters of the error).

Length-confound rule (pre-registered 2026-10-12, from TQS section 3.4 and appendix G): every version's
|Spearman rho| with n_tool_calls is reported (`length_check.json`); a version with |rho| > 0.25 is flagged
`length_confounded` and its test-1 AUROC must also be reported inside the interquartile band of n_tool_calls.
A version that passes test 1 only outside that band does not advance.

Changelog: 0.3.2 (2026-10-12): v7 (TQS port); length-confound check; per-task `end_reason`
(context_exceeded / time_limit / tool_budget / error / none) from task_results.jsonl, so ContextWindowExceededError
endings under max_output_tokens 16384 are counted separately (decision of Oct 12). No earlier definition changed.
0.3.1 (2026-10-12, B0 session A traces): per-call token usage read from the agent step's `metrics`
when `tool_call.extra.usage` is absent (it was absent for all 128 calls on Kaggle). No component definition changed.

Choices made here where the handoff left room (change = new version): the v1 combination rule (equal
weights over available components); the v4 veto threshold 0.25; resolution precision counts a call
precise when every named argument resolves exactly, by a unique delimiter-suffix match, or by a unique
case-insensitive match (a suffix tier with more than one candidate is a same-name collision);
search_similar_code arguments are replayed through the same resolver (an approximation of the
embedding-key lookup).

# B2_s1 (Kaggle, 4 x L4) evaluation: Oct 12, 2026

Source: `runs/B2_s1/` (129/129 tasks, complete), `validity/`, and `scorecard/B2_s1/` (scorecard 0.3.2,
run on the laptop with `kaggle_upload/graphcomp-planb/oracle_pack.json.gz`, since the pack was not attached on Kaggle).
Serving 0.80 / 32,768 / tp 4, LoRAs mounted, concurrency 4. 5.5 h of agent time (median 434 s per task).
All intervals Wilson 95%. Denominators = this run's own validity (`sub_valid`, 103/129).

## 1. Validity (GPU image, test deps installed)
- Gold pass 105/115 measured; empty pass 2/105. sub_valid 103: fastapi 60/67, rich 43/48, requests 0/13, httpx 0/1.
- **The test-dependency fix worked: fastapi valid 29 -> 60** (CPU B1 had 29). Remaining fastapi exclusions: gold fails 5
  (incl. 15763 / 15785, as on CPU), empty passes 2 (14077, 14851).
- Official dev slice used (identical to `scripts/planB/official_slices/dev_slice.txt`). **One dev task is invalid here:
  `fastapi_14356` (gold fails)**, so dev denominators are 23. Holdout: all 24 valid.
- rich_4070 and rich_3782 are valid on the GPU image (4070 failed gold on CPU; 3782 was the flaky one): validity is
  environment-specific, as the protocol assumed.

## 2. Resolution (sub_valid only)
| Group | Resolved |
| --- | --- |
| all | 38/103 = 36.9% [28.2, 46.5] |
| fastapi | 22/60 = 36.7% [25.6, 49.3] |
| rich | 16/43 = 37.2% [24.4, 52.1] |
| dev | 11/23 = 47.8% [29.2, 67.0] |
| holdout | 10/24 = 41.7% [24.5, 61.2] |
| rest | 17/56 = 30.4% [19.9, 43.3] |

Three more tasks "resolve" but are invalid here (fastapi_14077, 14851 empty-pass; requests_6644 host copy): 41/129 raw.

Task features (sub_valid): single-file 35/73 = 47.9% vs multi-file 3/30 = 10.0%; no async in gold 35/85 = 41.2% vs async
3/18 = 16.7%; gold not entry point 29/69 = 42.0% vs entry point 9/34 = 26.5%. Cells: d0/d1 13/18 resolved; d2+ 4/16;
d_undef 21/69.

## 3. How tasks end: the context wall is the biggest single loss
| end_reason (sub_valid) | Resolved |
| --- | --- |
| none | 33/60 = 55.0% [42.5, 66.9] |
| tool_budget (50 calls) | 5/26 = 19.2% [8.5, 37.9] |
| **context_exceeded** | **0/16 = 0.0% [0.0, 19.4]** |
| time_limit | 0/1 |

- 20/129 tasks (15.5% [10.3, 22.7]) died with `ContextWindowExceededError`; **none resolved**. Half died within 12 tool calls.
- ~~Compaction never fired: 0 events in 129 tasks.~~ **CORRECTED Oct 14:** compaction does fire. The ATIF trace does not
  record compaction events, so the scorecard's `n_compactions` is 0 by construction. The recorded per-call `prompt_tokens`
  show it: in 13 of the 17 resolved non-slice trajectories the prompt falls sharply once it nears about 14-16k tokens
  (e.g. 14,392 -> 5,399), and our re-rendering of the trace matches the recorded size to within about 27 tokens until that
  point (train_lora.py --check). So tasks die at the context wall when the prompt crosses 16,384 before the next compaction
  check (interval 15 calls); the earlier claim below that compaction "never fired" is withdrawn.
- (original note, superseded) Last recorded prompts were 12.7k-16.3k tokens, against the
  16,384-token effective input (32,768 - 16,384 reserved output). The compaction check (interval 15, threshold 14,336)
  comes too late for tasks that hit the wall in under 15 calls, and did not rescue the ones that went longer.
- Cause: unbounded tool returns. Typical killers are whole-file `read_file` (tests/test_syntax.py 11.4k chars,
  tests/test_markdown.py 12.6k), 150-line windows of large modules (~6-7k chars each), and pytest / `grep -r` / `git log`
  outputs of 5-11k chars. Mean return: read_file 3.3k chars, run_command 0.6k.
- This confirms session A finding 2 at scale. It is a harness/configuration condition common to every arm, so it does
  not bias paired comparisons, but it caps every arm's resolution and adds noise. O8K (max_output_tokens 8192, doubling
  input room) is the pre-registered test of it; it should be prioritised.
- Tool budget: 35 tasks ended at or near 50 calls (30 at exactly 50; 14 tasks submitted an empty patch overall).

## 4. Graph-tool use (Plan B §8: "adoption near zero")
- **Adoption 14/129 = 10.9% [6.6, 17.4]**; 15 graph calls in 4,207 tool calls (0.36%).
  fastapi 11/67 = 16.4%, rich 1/48 = 2.1%, requests 2/13. Dev: 2/24.
- Calls: `search_similar_code` 8, `get_code_neighbors` 7. 13 of the 14 tasks made exactly one call.
- Quality when called (n = 15, so wide): resolution precision 11/15 = 73% [48, 89]; yield R1 0/15, R2 7/15, R3 11/15;
  followed within 2 calls 9/13 = 69%; waste (p95 s50) 1/15; async hazard 1/15; hallucinated args 10/4,207 tool calls.
- Adopters vs non-adopters (sub_valid): 3/9 = 33% vs 35/94 = 37%. Uninformative (n = 9, and adoption is self-selected).
- Cell pattern: 8 of 9 valid adopters are d_undef / d2+ (no short static path to the gold), none in d0/d1.

## 5. Scorecard check (`length_check.json`)
- No version is length-confounded (max |rho| 0.23, v5); none exceeds 0.25, so no in-band test is triggered.
- **No version predicts resolution:** AUROC v0 0.51, v1 0.48, v3 0.48, v4 0.49, v5 0.56, v7 (TQS) 0.59 (in-band 0.61).
  Expected: v0-v5 score graph calls, and 115 of 129 tasks have none. v2 / v6 not computable yet (no necessity_v2 labels;
  v6 needs fitting on non-slice tasks of this run).
- Gold-file read 119/129; among sub_valid, not reading the gold file means 0/9 resolved, reading it 38/94.

## 6. Session A comparison (same tasks, seed 1, temperature 0.2)
fastapi_12942: context-exceeded in session A, resolved here (34 calls). rich_3518, rich_3718: resolved in both.
fastapi_15588: unresolved in both. fastapi_15661 resolved in the B0 smoke, context-exceeded here. Same seed does not mean
the same trajectory: seed-to-seed variance (B2_s2, s3) is needed before any arm difference of a few tasks means anything.

## 7. What this means for the next runs
1. B2_s1 is a usable baseline: complete, valid denominators, and dev resolution 11/23 leaves room in both directions.
2. With 0.36% graph calls, B3 (no graph tools) will likely differ from B2 by noise only. That is the expected §8 result,
   and it is what FC / QN are for.
3. Paired dev-slice arms (FC, QN, T0, O8K) on 23 valid tasks can only detect large effects: McNemar needs about 6+ discordant
   pairs in one direction. Expect "no significant difference" unless an arm moves 5+ tasks.
4. Run O8K early: the context wall removes ~16% of valid tasks with certainty. If 8192 output tokens recovers them, decide
   (pre-registration permitting) whether it becomes the condition for later arms.
5. Record in B0.md: validity numbers above, the official-slice exception (fastapi_14356), and the 0-compactions finding.

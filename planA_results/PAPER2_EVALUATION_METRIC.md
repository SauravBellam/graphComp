# Paper 2 positioning: scoring SWE agents by the tests that can tell a fix from no fix

Working title: *Does the test tell a fix from no fix? Discriminating-test resolution for local coding agents.*
Topics: Tasks & Benchmarks (primary), Tuning & Optimization. Award fit: Best New Resource (the per-task annotation set and scorer) or Overall. Written Oct 11, 2026; numbers are from the VM rerun on the real archive (`vm-a2/results/a9/`, built on `a1_v2`, `a2_v3` and the recovered traces; see `ANALYSIS_2026-10-11_VM_RERUN.md`). Anything not read from a file is tagged **[inferred]**.

## 1. The claim in one paragraph

The competition scores a task as resolved when, after the agent's patch, pytest on the test files exits 0 and every test name extracted from `test_patch` passed (`../HARNESS_README.md` section 8). Nothing in that rule asks whether those tests failed *before* the patch. On the 129 public tasks, run before and after the gold patch under the harness's own pytest settings (`vm-a2/results/a2_runs/`), 197 of the 669 named required tests already pass on the base commit, 22 tasks name no test at all so exit 0 is the whole requirement, and 2 tasks are resolved by an empty patch (`a9/a9_summary.json`, `discriminativeness`). We propose scoring on the *discriminating* tests, the ones that fail or cannot be collected before the fix and pass after it, in two forms: a strict pass/fail that an empty patch cannot earn, and a continuous credit that gives a dense signal for training. We ship the per-task annotations needed to apply both, plus two difficulty annotations from the same audit (entry distance and tool answerability), and we validate the scorer on two reference submissions whose answers are known: the gold patch and the empty patch.

## 2. What is already established

- **Pass/fail hides process damage and absorbs failures.** Jang et al. show a 4-bit Gemma-4-31B doubling its out-of-list tool calls with no change in τ²-bench score because the error budget absorbed it (handoff ref. 11). Repeat-run consistency pass^k exists for the same reason (ref. 4).
- **Weak and leaked tests inflate SWE-bench results.** SWE-bench+ documents solution leakage and tests that do not check the fix (ref. 13); models localize Verified files from issue text alone far better than on fresher sets (ref. 14).
- **Training wants denser signal than resolve/not.** The newest survey names fine-grained process rewards and intrinsic validation through tests as open problems (ref. 1); selective loss on high-value trajectory segments beat training on everything (ref. 16); action-level credit lifted small localizers (ref. 40). All of these need a per-task signal finer than one bit.
- **The harness rule itself** (section 8 of `../HARNESS_README.md`): exit 0, JUnit has `passed_tests > 0`, zero failures and errors, every required node passed and was not skipped. Required nodes are `FAIL_TO_PASS`, `PASS_TO_PASS`, or names extracted from `test_patch`. In the public set `FAIL_TO_PASS` is empty for 124 of 129 tasks (`vm-a2/results/a2/a2_run_status.csv`), so extraction is the rule in practice.

What is missing: nobody has measured, for this benchmark or its pipeline, how many of the required tests discriminate, or offered a scoring rule that is immune to the ones that do not. That is a measurement about the benchmark, in the "highlight important properties of existing methods" sense of the Novelty criterion (`../compPhilosophy.md` section 2).

## 3. The three annotations and the scorer

All computed by `scripts/planA/a9_metrics.py` from the Plan A outputs; definitions in its docstring.

**Discriminating tests.** For each task, the effective requirement is every test collected in the target files after the fix (exit 0 needs all of them), with the extracted names flagged. A test is discriminating if it failed, errored or could not be collected before the fix and passes after it. Per task we store the counts and the two shares (over named tests, over all effective tests), and `null_patch_resolvable`: the pre-fix run already satisfies the harness rule.

**Scorer.** Given a submission's post-patch outcomes on a task:

| Metric | Rule | Property |
| --- | --- | --- |
| `harness_resolved` | exit 0, at least one pass, every named test passed | the competition's rule, reproduced |
| `strict_resolved` | `harness_resolved` and at least one discriminating test exists and all of them pass | an empty patch scores 0 by construction |
| `disc_credit` | share of discriminating tests passing | continuous in [0, 1]; undefined when none exist |

Aggregates: the three rates over tasks, the number of tasks with no discriminating test, and the empty-patch floor reported next to every resolution rate.

**Entry distance** `d_entry`: minimum directed `calls` hops in the provided graph from the nodes where the failing tests enter the library to a gold function those tests execute (0: the gold is an entry node). Defined for 84 tasks, against 47 for the issue-text distance used in the original plan (`a9_summary.json`, `entry_distance`; `../HANDOFF.md` B.3).

**Tool answerability**: four booleans per task. Every gold symbol is a node; no gold symbol's natural query resolves to a different node under the harness resolver; at least one executed gold node is reachable from the entry set; all of them are. `answerable` is the conjunction.

## 4. What the public set looks like under these annotations

All numbers below are from the real archive and the final traces (`vm-a2/results/a9/`); the earlier local recount in `local_recount_2026-10-11/` is kept only for comparison.

**Required tests.** 669 named required tests across the 129 tasks: 459 discriminate, 199 pass before the fix, 203 cannot be collected before the fix (their module imports the new code), 6 never appear after it. 22 tasks name no test. Counting all tests in the target files, 1,490 of 7,153 discriminate (`a9_summary.json`). Per task, the named share is 1.0 for 58 tasks, strictly between for 42, and 0 for 3; 62 tasks hinge on exactly one discriminating test, 60 have two or more, 7 have none (`discriminating_per_task_hist`, `tasks_without_any_discriminating_test`). Two tasks, fastapi_14851 and fastapi_14077, are resolved by the empty patch: their pre-fix run exits 0 and every named test passes, the first because the patch is a coverage-only change whose tests already pass, the second because no test name is extracted and the target file is already green (`task_metrics.csv`).

**Reference submissions** (125 tasks with both traces; `scores_gold_vs_null.json`):

| | harness rule | strict | mean credit | floor |
| --- | --- | --- | --- | --- |
| Gold patch | 112 of 125 | 110 of 125 | 1.00 | |
| Empty patch | 2 of 125 | 0 of 125 | 0.00 | 1.6% |

The 13 gold-patch misses under the harness rule are our environment, not the patches: 9 collection errors, 2 import errors, 4 requests tasks with no collectable tests (`vm-a2/results/a2_v3/a2_run_status.csv`). The organizers' cached dependencies would close most of them.

**Entry distance** (91 usable tasks, 72 with an executed gold node): 32 at 0 hops, 9 at 1, 13 at 2, 9 at 3, 6 at 4 to 7, 3 unreachable (`vm-a2/results/a2_v3/a2_summary.json`, `gold_min_hops_from_entry_hist`). In 69 of the 72 tasks every executed gold node is reachable along edge direction; none is reachable only against it.

**Answerability**: 53 of 129 tasks (41%) are answerable by a perfect tool user; 50 fail because a gold symbol is not a node, 21 because a gold query lands on the wrong node, 50 because some executed gold node is unreachable from the entry set, mostly tasks where no gold node was executed at all (`a9_summary.json`, `answerability`). Requests is the most answerable repo (7 of 13), rich next (25 of 48), fastapi the least (21 of 67).

## 5. How the metric is used to test and to improve models

**Testing.** Report four numbers per configuration instead of one: resolution under the harness rule, strict resolution, mean discriminating credit, and the empty-patch floor. Stratify by `d_entry` (0, 1 to 2, 3 or more, unreachable) and by `answerable`. With 129 tasks the strict rate and the harness rate will differ by at most the 2 null-resolvable tasks plus whatever a submission gains from pre-passing named tests, so the headline will barely move; the credit and the strata are where configurations separate. **[inferred]** Plan B's paired comparisons (tools vs no tools, push vs pull) gain power from the credit: a condition that fixes two of three discriminating tests scores 0.67 rather than 0.

**Improving.** Three uses, cheapest first:
1. **Trajectory selection for SFT.** Keep only trajectories whose patch reaches credit 1 on a task with at least two discriminating tests; this removes the free points and the single-test coin flips from the data. Fits the selective-loss finding (ref. 16).
2. **Reward for RL.** Credit is a dense, non-gameable reward on the 60 tasks with two or more discriminating tests; on the 62 single-test tasks it reduces to strict pass/fail. Outcome reward over the harness rule would pay for the 2 null-resolvable tasks and for pre-passing named tests.
3. **Curriculum.** `d_entry` orders tasks by how far the fix sits from where the failing test enters the code, and `answerable` separates tasks the provided tools can in principle solve from those they cannot. Adoption experiments in Plan C should train and report on the answerable stratum first.

**For the benchmark.** A task-acceptance rule follows directly: at least one discriminating test, preferably two, and `null_patch_resolvable` false. The hidden set was built "under a similar pipeline" (`../compPhilosophy.md` section 4); the organizers can run the same two pytest passes.

## 6. Babystep plan

| Step | Adds | Status |
| --- | --- | --- |
| E1 | Annotations and scorer on the 129 public tasks; gold and empty patch as reference submissions | Done on the real archive (`vm-a2/results/a9/`, `ops/run_gaps.sh` step 6) |
| E2 | Score Plan B's B2 and B3 runs under all four numbers; compare condition rankings and paired p-values under harness vs strict vs credit | Needs B0/B2/B3 logs |
| E3 | Stratified payoff surface by `d_entry` and `answerable` | Same logs |
| E4 (stretch) | Credit as the reward or filter in Plan C's adoption LoRA | Plan C |

E1 alone is a standalone write-up under 3,000 words: the measurement in section 4, the scorer, the two reference submissions, and the released annotations. E2 turns it into a result about models.

## 7. Risks and honest limits

- The metric needs a pre-fix run of the tests. Competitors can do it on the public set; only the organizers can on the hidden set. The paper should say the annotations are a benchmark-side artefact, not something an agent computes.
- Our pre-fix runs used per-task venvs with network access, not the official cached dependencies; 9 collection errors and 2 import errors are ours. Discriminativeness for those tasks is undefined, not zero, and is reported as such.
- `uncollected_pre` tests (203) are counted as discriminating because the harness would see a non-zero exit before the fix; a stricter reading that requires an explicit failure drops them to `pre_unknown`. Both counts are in `task_metrics.csv`; the paper should show both.
- With 2 null-resolvable tasks in 129, the strict rule changes the public leaderboard by at most 1.6 points. The argument is not that scores are wrong today; it is that the rule has no guard, the credit gives a training signal the rule cannot, and the hidden set is unchecked.

## 8. Files

- `scripts/planA/a9_metrics.py`: annotations, scorer, aggregates, reference submissions; unit tests in `scripts/tests/test_pipeline.py::test_a9_scoring`.
- `vm-a2/results/a9/`: `task_metrics.csv` (one row per task, all three annotations), `task_tests.jsonl` (one row per required test with pre and post outcome), `a9_summary.json`, `scores_gold_vs_null.json`. Built on `vm-a2/results/a1_v2/`, `a2_v3/` and `a2_runs/`.
- `local_recount_2026-10-11/`: the earlier stand-in computation, kept for comparison only.

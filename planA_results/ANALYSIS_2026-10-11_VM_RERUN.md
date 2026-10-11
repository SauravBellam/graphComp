# Plan A, VM rerun of Oct 11: the numbers that replace the Oct 10 ones

Written Oct 11, 2026 from the outputs committed on branch `planA-gaps` (two analysis passes on vm-a2 over the same archive and traces, plus one recovered task), now merged here. This supersedes the provisional numbers in `FIXES_2026-10-11.md` section 3 and `local_recount_2026-10-11/`. Paths are relative to `planA_results/vm-a2/results/` unless marked otherwise; the latest folder of each step is used (`a1_v2`, `a7`, `a2_v3`, `a5_v3`, `a6_v3`, `a8`, `a9`). Anything not read from a file is tagged **[inferred]**.

## 1. Key findings

1. **Both data gaps are closed, and they were id-scheme effects.** A7 reconciles every provided node of fastapi and rich with a source definition: fastapi 256,383 plain matches, 8,033 nested defs under flattened ids, 1,409 class-dropped ids, 0 unmatched; rich 86,613 plus 5,985 nested, 0 unmatched (`a7/a7_summary.json`, `per_repo`). The class-dropped ids in fastapi are 1,683 inner `Config` classes of pydantic models, listed as `module.Config` without their model (`a7/a7_class_dropped.csv`). Requests has 1,619 class-dropped ids (the `TestRequests` methods, 1,233 of 1,425 with text identical to the method's source) and 268 unmatched nodes over 12 graphs, all methods of helper classes defined inside test functions two levels deep (`tests.test_utils.BoomFile.__len__`). The requests "absent" functions and the `provided_only` nodes were the same population seen from two sides.

2. **Nested definitions are in the graph: 14,483 of 18,178 (79.7%).** The missing fifth are nested async defs (4,428 in fastapi) and a few hundred nested defs in requests (`a1_v2/a1_summary.json`, `omission_direct_files`; `a7/a7_summary.json`, `defs`). The Oct 10 claim of zero nested nodes is withdrawn. Gold symbols present rise from 463 to 470 of 711; tasks with a gold symbol misresolving by bare name fall from 26 to 21.

3. **H-A2 is rejected on the real graphs.** With the failing test's entry nodes as the BFS source, at least one executed gold function is reachable along `calls` edges in 69 of 72 usable tasks (95.8%, CI 88.5 to 98.6), and in every one of those 69 all executed gold functions are reachable; none is reachable only against edge direction (`a2_v3/a2_summary.json`, `usable_tasks`). Minimum hops: 0 for 32 tasks, 1 for 9, 2 for 13, 3 for 9, 4 to 7 for 6, none for 3. The containment edges the local recount lacked are what turned the 16 undirected-only nodes into directed ones.

4. **The `calls` edges are largely not calls.** A8 reads the caller's source for every static edge whose endpoints both ran but whose call was never observed: in 60.5% the callee's name does not appear in the caller's code at all, in 23.1% only in a string or comment, and in 10.9% there is an explicit call site (`a8/a8_summary.json`, `all_edges.unconfirmed`). Captured edges have an explicit call site 92.1% of the time. Classifying the unconfirmed non-test edges structurally (my check against the v2 graphs, 1,500-edge sample): 23.4% go from a method to its own class, 15.3% from a class to a subclass, 14.3% from a class to a base class, 19.1% to some other class, 27.9% other. **[inferred]** The generator emits an edge from a method to its enclosing class and inheritance edges in both directions under the label `calls`; those relations, not spurious call resolution, explain most of the 21% observed precision.

5. **The misses are method calls the generator does not resolve.** Of 18,232 missed runtime calls between existing nodes, 31.5% are `obj.callee()` on another object, 24.0% references without a call (callbacks, decorators, arguments), 18.7% implicit dunder calls, 10.9% `self.callee()`, 7.6% bare-name calls (`a8/a8_summary.json`, `all_edges.missed`). Structurally 45.1% of missed non-test edges run between two methods of the same class (my sample). Recall itself is unchanged by the reruns: 37.1% of representable calls, 65.6% in fastapi, 30.5% in rich, 34.7% in requests (`a2_v3`, `per_repo_usable`).

6. **Free-text search over the real source beats every graph mode by a wide margin.** A5's new mode e, BM25 over the snapshot files scored against every gold symbol, finds the gold file in the top 5 for 85 of 129 tasks, against 43 for the harness's symbol-resolve-then-cosine mode and 63 for BM25 over node text; 47 tasks are found by mode e and not by the harness mode, 5 the other way (exact McNemar p < 0.0001) (`a5_v3/a5_summary.json`, `overall_all_tasks`). The harness mode still returns nothing for 54 of 129 tasks.

7. **Test discriminativeness and the two reference submissions, now on the final traces** (`a9/a9_summary.json`): 459 of 669 named required tests discriminate, 199 pass before the fix, 203 cannot be collected before it; 22 tasks name no test; 2 tasks (fastapi_14851, fastapi_14077) are resolved by an empty patch; 7 tasks have no discriminating test at all. Gold patch: 112 of 125 under the harness rule, 110 strict, credit 1.00. Empty patch: 2, 0, 0.00. Tool answerability holds for 53 of 129 tasks (41%), 49 of 91 usable.

8. **rich_4006 is recovered and nothing else moved.** The pre-fix run hung on 4 escape-character cases in `tests/test_cells.py` that ignore pytest-timeout; deselecting them after the 300 s cap gave a clean run and a 91st usable task (`../../README.md` section 6; `a2_rich_4006_rerun.log`). Observed precision 21.4%, strict tasks 55, harness-resolved-post 92 of 129.

## 2. Hypothesis scorecard, final

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| H-A1 edges high-precision, low-recall; recall lowest in fastapi | Low recall confirmed (37%). Precision inside executed code 21%, and A8 shows why: most unconfirmed edges are containment or inheritance under the `calls` label, not calls. fastapi has the highest recall | `a2_v3/a2_summary.json`, `a8/a8_summary.json` |
| H-A2 gold statically unreachable but dynamically reachable | Rejected: reachable from the test's entry nodes in 69 of 72 tasks, 32 at zero hops | `a2_v3/a2_summary.json` |
| H-A3 vectors reproducible with EmbeddingGemma | Rejected, unchanged | `a3/a3_summary.json`, `a3_gemma2/a3_summary.json` |
| H-A4 symbol-only search retrieves gold far less often than free text | Supported, now against real free text over source: 43 vs 85 file hits of 129 | `a5_v3/a5_summary.json` |

## 3. What the generator appears to do

Putting A7 and A8 together, the provided graph is consistent with a single-pass AST walker that **[inferred]**:

- records every `def` and `class` as a node, async defs excluded, nested defs attached to the nearest enclosing class or module;
- emits a `calls` edge for a bare-name call or a direct instantiation (these are the captured edges: 43% `callee()`, 36% `Class()`);
- also emits, under the same label, method-to-own-class and class-to-base and class-to-subclass edges (the unconfirmed edges);
- does not resolve `self.x()`, `obj.x()`, callbacks or dunder dispatch (the missed edges);
- loses the class scope for a subset of requests test methods and for pydantic `Config` inner classes.

Each bullet is testable against `a8/a8_sample.csv` (80 edges with caller snippets and empty `human_label` columns) and should be labelled by hand before it goes in the paper.

## 4. Where this leaves the two write-ups

**Plan A paper** ("Is the map right?"). The headline stays the async omission (0 of 45,492 defs; 81 gold symbols; 22 tasks) and the resolver that turns omissions into wrong nodes. The second result is now sharper than "low precision": the `calls` relation mixes containment, inheritance and calls, captures 37% of real calls, and misses method dispatch, with the breakdown from A8. H-A2's replacement finding is that the static graph does reach the gold from the failing test, usually within 3 hops, so the graph's problem for an agent is omission and mislabelling, not connectivity. The retrieval result gains mode e: free text over source finds twice as many gold files as the shipped symbol search.

**Paper 2** (discriminating-test scoring, `PAPER2_EVALUATION_METRIC.md`). All numbers in it are now final rather than provisional, with the changes: 459 discriminating named tests (was 461), 199 pre-passing (197), 203 uncollected (209), gold patch 112 and 110 of 125 (111 and 109 of 124), answerable 53 of 129 (45), entry distance defined for 84 tasks (81) with the hop histogram above. The two null-resolvable tasks and the 7 tasks without a discriminating test are unchanged.

## 5. Things still open

- The 268 unmatched requests nodes are methods of classes nested inside test functions; the flatten rule keeps the class but these ids keep the class and drop the test function, which the rule already does. **[inferred]** They fail to match because A7 records nested classes' methods under `test_fn.<locals>.Class.method` and the flatten gives `Class.method`, so they should match; a7 is worth one more look on those rows before the number is quoted.
- `a8/a8_sample.csv` has not been hand-labelled.
- The A6 v3 graphs carry 98 id collisions per fastapi graph (a nested def whose flattened id equals a top-level one); the release card must state the rule.
- Plan B's cell counts are unchanged (`a6_v3/a6_summary.json`): 7 tasks in the far-and-invisible cell on the issue-text distance. The entry-distance strata in `a9/task_metrics.csv` are the better axis.

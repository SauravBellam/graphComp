# Experiments

Every experiment is one git branch (`orx/<slug>`) holding the exact code its runs used. Runs execute `bash vm_job.sh` on a single-L4 GCP VM (the per-branch `JOB` block says what runs). Kaggle notebooks are listed separately in [NOTEBOOKS.md](NOTEBOOKS.md). Listed in creation order; the tree is: E4B root → LoRA data → notebook 20 → external task import → {synthetic, external trajectories → SFT mix → trainer tests → session-2 data rules → OOM recovery → 31B memory}, with the re-grader, validity v2-v4, forced-call search and tool tracing as side branches.

| ID | Experiment | Branch | Result |
| --- | --- | --- | --- |
| `2416cbef` | E4B VM pipeline: deploy rehearsal → action agreement → merged adapter | `orx/e4b-vm-pipeline-deploy-rehearsal-action-agreemen` | Failed (run 4741e5e0, exit 1); E4B at the floor (3/23). Separately found: the VM's PyPI vLLM cannot serve Gemma 4 LoRAs |
| `6d0473fb` | LoRA data: commit trainer v2 + build_rft_data, optional spin filter | `orx/lora-data-commit-trainer-v2-build-rft-data-optio` | Code only: trainer v2 + spin filter committed |
| `c2358a8e` | E2B: forced graph call (FC) vs B2 baseline on all 129 tasks | `orx/e2b-forced-graph-call-fc-vs-b2-baseline-on-all-1` | Cancelled (post-training took the GPU); E2B B2 3/115 resolved |
| `da9699f7` | requests/httpx re-grader: src/ path shim, validity on the 14 tasks | `orx/requests-httpx-re-grader-src-path-shim-validity` | 6/14 requests/httpx tasks valid (from 0) |
| `b2017c91` | re-grader v2: + pytest-httpbin test deps (PYTHONPATH, no env change) on the 14 requests/httpx tasks | `orx/re-grader-v2-pytest-httpbin-test-deps-pythonpath` | 10/14 valid |
| `d4e6af3b` | Tool tracing: which tool calls reach and reveal the fix location, per forced rate | `orx/tool-tracing-which-tool-calls-reach-and-reveal-t` | Not run; 31B baseline trace computed locally |
| `26e7ad1a` | Notebook 20: 45 h iterative RFT campaign (round-1 LoRA sampling, accurate-tool selection, +10 requests/httpx tasks) | `orx/notebook-20-45-h-iterative-rft-campaign-round-1` | Notebook built, not run (superseded by nb21) |
| `394a0d99` | External task import: same-repo SWE tasks -> competition layout (snapshots, graphs, embeddings, validity) | `orx/external-task-import-same-repo-swe-tasks-competi` | 160 tasks imported; graphs 98-100% node recall |
| `eb125260` | Synthetic SWE tasks: procedural bugs at competition snapshots -> 31B pass-rate filter in nb20 | `orx/synthetic-swe-tasks-procedural-bugs-at-competiti` | Paused: VM disk; httpx smoke 12/13 valid |
| `ca1b26ab` | External verified trajectories -> harness tool format (CoderForge, SWE-rebench-OH, SWE-smith, Kwai) | `orx/external-verified-trajectories-harness-tool-form` | 149,500 trajectories kept (CoderForge 124,174; nebius 25,326) |
| `5303fc9a` | SFT mix: dedupe trajectories + external-task overlap + nb21 31B LoRA notebook | `orx/sft-mix-dedupe-trajectories-external-task-overla` | 135,990 kept; 36,442 fit 32k over 13,928 tasks |
| `eb5e85d4` | Diagnose external import: why gold resolves only 8/160 | `orx/diagnose-external-import-why-gold-resolves-only` | Causes found: missing test deps / old Python / no tests |
| `3aab2b0a` | External import validity v2: httpx test deps (trustme) on the sandbox path | `orx/external-import-validity-v2-httpx-test-deps-trus` | 30/160 valid |
| `be2b5398` | VM disk survey (read only) | `orx/vm-disk-survey-read-only` | Failed (run c10faece, exit 1) |
| `b503a927` | External import validity v3: pytest-asyncio for old async httpx tests | `orx/external-import-validity-v3-pytest-asyncio-for-o` | 38/160 valid |
| `8e1a1a79` | External import validity v4: pytest-trio for old httpx [trio] tests | `orx/external-import-validity-v4-pytest-trio-for-old` | Running |
| `fb116af8` | Trainer session features functional test (tiny Gemma-4-31B-architecture model) | `orx/trainer-session-features-functional-test-tiny-ge` | 11/11 checks pass; found the no-reasoning-turn bug |
| `d5e4b2a2` | nb21 session 2+ data rules: replay, competition rotation + visit cap, length-quartile mixing | `orx/nb21-session-2-data-rules-replay-competition-rot` | Code only; checked by an offline 8-session simulation |
| `b70d7942` | Trainer in-process out-of-memory recovery (no model reload) | `orx/trainer-in-process-out-of-memory-recovery-no-mod` | 18/18 checks pass (run 4e2bd751) |
| `b83c0e28` | 31B memory: lighter GPU 3 device map, OOM-leak detection + resume within session, safer windows | `orx/31b-memory-lighter-gpu-3-device-map-oom-leak-det` | 18/18 checks pass (run 4fd70af7); in Kaggle session 3 |

## Full notes (as recorded during the work)

### `2416cbef` E4B VM pipeline: deploy rehearsal → action agreement → merged adapter

- Branch: `orx/e4b-vm-pipeline-deploy-rehearsal-action-agreemen`
- Parent: — (root experiment)
- Latest run: 4741e5e0-1bf4-4a69-a33f-1d3cba354281 (failed, commit bf81e2e, ran 2h32m, updated 4d ago)

Single L4 VM run of three process experiments, in order:
1. Deployed-budget rehearsal: E4B w4a16, dev 24, B2 vs DEP (nb19 budget addendum, 8192 output, 30 calls / 5 min / 80 turns / 240 s). Measures cap hits, wall time, end reasons. Folder ~/gc/runs_dev_e4b_deploy.
2. Next-action agreement with 31B on rft_e4b_v1's 39 val turns: E4B base, E4B + rft_e4b_v1, E2B base (nf4, greedy, thinking on). Output ~/gc/vm_job_e4b_20261015/action_agreement.json.
3. Unquantized E4B base vs rft_e4b_v1 merged into it (bf16, both served by PyPI vLLM), B2 on dev 24. Folders ~/gc/runs_dev_e4b_unq, ~/gc/runs_dev_e4b_merged_rft.
E4B resolution is at the floor (3/23), so process metrics (tool calls, gold-file read, end reasons, time) are the readouts.

### `6d0473fb` LoRA data: commit trainer v2 + build_rft_data, optional spin filter

- Branch: `orx/lora-data-commit-trainer-v2-build-rft-data-optio`
- Parent: 2416cbef-af3b-4419-a2a9-7d3647496866 (branch orx/e4b-vm-pipeline-deploy-rehearsal-action-agreemen)
- Latest run: — (never run)

Commits the Mac copies of train_lora.py (v2, Oct 15) and build_rft_data.py, which were not in the repo (the VM still has train_lora v1). Adds --max-repeat-calls N to build_rft_data.py: drops trajectories with a run of more than N consecutive identical tool calls (spinning). Off by default, so existing data builds are unchanged. Note: train_lora v2 already trains only on turns before the first harness compaction (the compaction summary is not in the traces), so training never uses context the model did not see.

### `c2358a8e` E2B: forced graph call (FC) vs B2 baseline on all 129 tasks

- Branch: `orx/e2b-forced-graph-call-fc-vs-b2-baseline-on-all-1`
- Parent: 2416cbef-af3b-4419-a2a9-7d3647496866 (branch orx/e4b-vm-pipeline-deploy-rehearsal-action-agreemen)
- Latest run: 593ff1c6-6ab9-4e77-982e-31cde66ad8a3 (cancelled, commit a0c1665, ran 2h20m, updated 4d ago)

NOW (commit a0c1665, run 593ff1c6): E4B at max_output_tokens 8192 on the 64 fastapi + rich tasks in neither the dev nor the holdout slice (nor the 3 extra holdout tasks). Plan (user, 2026-10-15): B2 first; then forced graph calls at B2's MEASURED graph-call share (rounded %, every turn's schedule as in fc_rate_arm); if that does not resolve more tasks than B2, forced calls are a bust (fc_search_verdict.txt) and the search stops; otherwise probe 100% and continue the midpoint search. Tool tracking on every probe, B2 included: fc_search_tools.md (tool use vs outcome), per-tool counts in fc_search_tasks.csv, and tool_trace.py per probe in fc_search_trace/<arm>/ (purpose, on/reveals gold code, followed, first gold contact, shortest resolved routes, bigrams). Forcing logic removed from notebook 20 (reverted) until this shows a gain. Earlier runs: d6eab217 (E2B, B2 overflow 33/115, 3/115 resolved), 7c8e413f and 0dad6610 cancelled within minutes for design changes. Caveat: ~8 resolved per probe at E4B's ~13% on 64 tasks; only a large gain beats B2 clearly.

### `da9699f7` requests/httpx re-grader: src/ path shim, validity on the 14 tasks

- Branch: `orx/requests-httpx-re-grader-src-path-shim-validity`
- Parent: 2416cbef-af3b-4419-a2a9-7d3647496866 (branch orx/e4b-vm-pipeline-deploy-rehearsal-action-agreemen)
- Latest run: 3e3c4dd7-a745-4579-bc0c-c95e130b24e6 (done, commit 6de0ab3, ran 1m35s, updated 4d ago)

Hypothesis: requests and httpx (14 tasks) are unscorable in the subprocess sandbox only because both keep their package in src/ and the editable install fails (no pip in the sandbox venv), so tests import the host's installed copy. Baseline evidence (VM ~/gc/runs_dev/validity, no shim): gold fails 13/14; requests_6644 gold passes but the empty patch also passes. Change: run_arm.py puts /workspace/src first on PYTHONPATH for every subprocess-sandbox command (agent tools and the harness's own verify_task; no-op for fastapi/rich, none has src/). Test (CPU only): gold pass on the 14, then empty pass on gold-resolving ones; VALID = gold passes and empty fails. Readout: ~/gc/regrade_src_shim/validity_src_shim.json and the log table. Launch after the E4B root run ends (its install would otherwise swap run_arm.py under that run). Next if it works: port the shim into the Kaggle notebooks' run_arm cell and add the valid tasks to the teacher pool.

RESULT (run 3e3c4dd7, commit 6de0ab3): VALID with the src/ shim: 6/14 (httpx_3672, requests_6644, requests_7205, requests_7309, requests_7315, requests_7427). requests_6644, whose empty patch passed before the shim, now fails it, so the tests import the workspace. The other 8 are gold fails: their tests use the httpbin fixture from the pytest-httpbin plugin, which is not in the competition wheelhouse ('recursive dependency involving fixture httpbin', 187 setup errors on requests_6589). Next: child node with pytest-httpbin + deps on PYTHONPATH.

### `b2017c91` re-grader v2: + pytest-httpbin test deps (PYTHONPATH, no env change) on the 14 requests/httpx tasks

- Branch: `orx/re-grader-v2-pytest-httpbin-test-deps-pythonpath`
- Parent: da9699f7-707a-4581-be6a-eb270f1f451b (branch orx/requests-httpx-re-grader-src-path-shim-validity)
- Latest run: 6fb87ccb-6a9c-4c22-8aa2-b456263adb02 (done, commit e27615d, ran 5m04s, updated 4d ago)

Builds on the src/ shim (6/14 valid). The 8 remaining gold fails all error with 'recursive dependency involving fixture httpbin': pytest-httpbin is missing from the competition wheelhouse. Change: unpack pytest-httpbin 2.1.0 + httpbin 0.10.4 + their 13 deps (pip download, py3.12 wheels) into ~/gc/testdeps_httpbin and put that dir on PYTHONPATH after /workspace/src, only when SWE_EXTRA_PYTHONPATH is set (baked into the sandbox command by run_arm.py; the shared venv is not modified, so concurrent dev_runs package checks are unaffected). Test: gold pass on all 14, empty pass on gold-resolving; VALID iff gold passes and empty fails. Measurement fix, not a modelling hypothesis. Caveat: official scoring uses the competition wheelhouse, which lacks pytest-httpbin; these labels are for our training-data verification.

RESULT (run 6fb87ccb, commit e27615d): VALID 10/14, up from 6/14 with the shim alone. Newly valid: requests_7328, 7433, 7502, 7505. Still gold fails (4), each with exactly 1 failing test while 319-326 pass, for environment reasons unrelated to the fix: requests_6589/6592/6629: test_redirecting_to_bad_url expects InvalidURL for port -1, but the wheelhouse's newer urllib3 raises ValueError; requests_6757: expired test TLS certificate. Not pursued (would need per-task old urllib3 and a backdated cert). Valid set for training: httpx_3672, requests_6644, 7205, 7309, 7315, 7328, 7427, 7433, 7502, 7505.

### `d4e6af3b` Tool tracing: which tool calls reach and reveal the fix location, per forced rate

- Branch: `orx/tool-tracing-which-tool-calls-reach-and-reveal-t`
- Parent: c2358a8e-5920-4210-a231-baa0afe392bb (branch orx/e2b-forced-graph-call-fc-vs-b2-baseline-on-all-1)
- Latest run: — (never run)

Analysis node (CPU only) on the E2B forced-rate search output. Question: what are the tools actually used for, and which uses go with solving the task? tool_trace.py follows every call: purpose (normalized action taxonomy, cf. TRACEPROBE arXiv 2607.06184), on_gold (target names a gold file/function from the gold patch), reveals_gold (output names a gold item not named before), followed (a .py name first seen in its output is targeted within 3 calls); per trajectory: first purpose to reach the gold code, calls until then, gold file edited, whether the issue text names the gold file/function. Run after the search finishes, over every probe's merged run, valid tasks only. Baseline already computed locally on 177 Kaggle 31B trajectories (artifacts/vm-experiments/tool-trace-31b-kaggle/). Literature: TRACEPROBE (2607.06184): function selection and completion behavior localize failure better than file choice; DeFA (2610.01256): dependency-guided attribution; cost-inefficient behaviors (2609.30725).

### `26e7ad1a` Notebook 20: 45 h iterative RFT campaign (round-1 LoRA sampling, accurate-tool selection, +10 requests/httpx tasks)

- Branch: `orx/notebook-20-45-h-iterative-rft-campaign-round-1`
- Parent: 6d0473fb-961e-4ec2-bbfd-bc4127fc6721 (branch orx/lora-data-commit-trainer-v2-build-rft-data-optio)
- Latest run: — (never run)

Kaggle notebook (runs on Kaggle, not orx): continuous resumable sampling for ~45 GPU-hours over ~4 sessions. Each round samples with the current best LoRA (round 1 = nb19 LORA31), never-solved tasks first, then slow/hard solved ones; per-task time cap recomputed from remaining sampling time so every task gets an attempt each pass; 30 tool calls. Every attempt is saved. Training data = verified trajectories with accurate tool use (all source edits on the gold files, a reproduction/test after the last edit), fastest first, max 3 per task; retrain from the base 31B each round; holdout decides; training rounds stop when a round does not beat the best (at least one new round). +10 requests/httpx tasks (src/ shim + pytest-httpbin via SWE_EXTRA_PYTHONPATH, revalidated on the Kaggle image). Literature: SWE-Gym 2412.21139 (iterative RFT plateaus after 2 rounds; on-policy self-training can hurt), ASCENT 2610.05303, ReSAIL 2609.39306, TQS 2610.13152, TRACEPROBE 2607.06184.

### `394a0d99` External task import: same-repo SWE tasks -> competition layout (snapshots, graphs, embeddings, validity)

- Branch: `orx/external-task-import-same-repo-swe-tasks-competi`
- Parent: 26e7ad1a-920c-46c0-a92f-42949f84014a (branch orx/notebook-20-45-h-iterative-rft-campaign-round-1)
- Latest run: 78afa2f4-4f6f-4c70-a83d-df03cfc9b38d (done, commit cd93284, ran 14h11m, updated 3d ago)

Imports external tasks from the four competition repos (SWE-bench requests 44 / fastapi 28; SWE-rebench-V2 rich 28 / httpx 5; SWE-rebench httpx 25+62; SWE-bench-extra httpx 16), drops every task with the same PR as a competition task (no holdout/dev leakage) and duplicates, writes competition-format snapshots, graphs (code_graph.py: validated against the competition graphs, nodes 98-100% / edges 97-99% recall on fastapi+rich, ~80% nodes on requests' giant test file) and embeddings (Qwen3-Embedding-0.6B 256-d + call-graph smoothing; hops chosen by Spearman against the competition vectors), then gold/empty validity on the VM (src/ shim + pytest-httpbin). Output: a Kaggle dataset for notebook 20 (graphcomp_external.json lists valid tasks; nb20 re-checks on its image). Literature for the graph/embedding choices in the experiment log (LocAgent 2503.09089, RepoGraph 2410.14684, CodexGraph 2408.03910, deterministic anchoring 2606.26979, jina-code-embeddings 2508.21290, CoIR 2407.02883).

### `eb125260` Synthetic SWE tasks: procedural bugs at competition snapshots -> 31B pass-rate filter in nb20

- Branch: `orx/synthetic-swe-tasks-procedural-bugs-at-competiti`
- Parent: 394a0d99-2416-43e5-adf1-c97d3721af95 (branch orx/external-task-import-same-repo-swe-tasks-competi)
- Latest run: e324fff0-81da-4e5d-bbee-330d862148b0 (failed, commit 150c83e, ran 3s, updated 3d ago)

Synthetic SWE tasks for the 31B pass-rate filter in notebook 20 (v2 at commit 150c83e, run 701cb087). Scope widened on 2026-10-15 by the user: no Kaggle-quota limit, as much high-quality data as possible for the full 45 h trajectory run, and combined bugs. v1 runs 3549f5a5 / 0a9c2dcb were cancelled before any result: v1 found only 267 fastapi sites because its leakage guard covered all 129 competition tasks (1,419 sites skipped).
Design (scripts/planB/synthetic/make_synthetic.py), per repo at the newest competition snapshot:
(1) the test suite once under coverage with per-test contexts;
(2) single procedural bugs: flip compare / and-or / + - / True-False, negate or drop an if, swap if-else, drop a loop or a re-bound statement, constant +1. One per function, at most 8 per file. Targets: fastapi 300, rich 250, requests 80, httpx 80;
(3) combined bugs, as in SWE-smith B.3: 2-4 kept bugs from one file (at most 3 per file) or 2-3 from different files of one module (at most 6 per module), each re-validated; at least one failing test is hidden per bug;
(4) PR mirror: clean git apply -R of past package commits (at most 150 lines, no file adds or deletes), skipping commits with competition or imported PR numbers.
Guard: functions touched by the 65 dev/holdout/extra-holdout/requests/httpx tasks (scripts/planB/synthetic/guard_tasks.json).
Validity: gold/empty via run_arm, httpx at concurrency 1, others at 6.
Issues: written by 12B (google/gemma-4-12B-it-qat-w4a16-ct; fallback E4B) after the E2B search frees the GPU; leak-filtered.
Output: ~/gc/synthetic/graphcomp-synthetic(.tgz).
Smoke test, v2 on httpx: 13 built (8 single, 4 combined, 1 mirror); 12 of 13 valid (the mirror failed the empty check); httpx's snapshot has only 2 commits, so its mirror yield is tiny.
Literature: SWE-smith 2504.21798, R2E-Gym 2504.07164, SSR 2512.18552, FrogNano 2609.07925, CodeMidas 2609.22068.

QUEUE NOTE (2026-10-15 ~11:00 UTC, session chat_9727b38f, at the user's request): run 701cb087 (commit 150c83e) was cancelled to give the GPU to the E4B forced-rate search (experiment c2358a8e, run 7c8e413f). This experiment is queued to relaunch unchanged when that search ends; session chat_9727b38f will relaunch it. Please do not relaunch onto the busy GPU before then.

### `ca1b26ab` External verified trajectories -> harness tool format (CoderForge, SWE-rebench-OH, SWE-smith, Kwai)

- Branch: `orx/external-verified-trajectories-harness-tool-form`
- Parent: 394a0d99-2416-43e5-adf1-c97d3721af95 (branch orx/external-task-import-same-repo-swe-tasks-competi)
- Latest run: c9b10d84-8779-43b2-88ef-4220677883b8 (done, commit 41c99ce, ran 3h33m, updated 4d ago)

Convert public test-verified SWE agent trajectories into the swegemma harness format by REPLAY (scripts/planB/external/convert_trajs.py).

Scope (user, 2026-10-15): only teachers stronger than Gemma 4 31B -> Qwen3-Coder-480B-A35B only:
- togethercomputer/CoderForge-Preview filtered_reward1 (155k, reward 1; SWE-rebench / SWE-smith / R2E-Gym tasks)
- nebius/SWE-rebench-openhands-trajectories resolved==1 (32k)
Excluded: SWE-smith-trajectories (teacher models borderline vs Gemma 4 31B or weaker), Kwai 66k (teacher unknown). Both are supported by the converter (--sources smith,kwai --models any).

Replay: each task checked out at the source's state (CoderForge SWE-rebench = base commit + test patch; nebius = base commit, no test patch, its issue.md rereads dropped; SWE-smith = mirror + bug patch; R2E = fix commit with non-test files reverted). Editor calls -> read_file (150-line/10k chunks), edit_file (harness apply_replacement), write_file; directory views -> run_command find; bash -> run_command with the source's output; OpenHands soft timeouts merged (waits -> full output, Ctrl-C -> TimeoutExceeded). Every view is compared line-by-line with the source (stale -> reject). Local check: nebius final diffs equal the source model_patch on 7/7.

Labels per row: source_dataset, source_traj_id, source_model, scaffold, base recipe, chunked_views, soft_timeouts_merged, network_or_install_cmds (pip etc., offline at deploy), test_files_written, bash_may_write, dropped_* counts, final_diff_matches_source (nebius).

Output on VM: ~/gc/external_trajs/graphcomp-external-trajs/{train/<src>/*.jsonl.gz, rejects/, tasks.jsonl, manifest.json, convert_trajs.py}. Graphs/embeddings NOT built (no source trajectory calls graph tools; 50k+ snapshots do not fit the disk).
Run 1 (6a161f84) failed: re-clone of a deleted repo in cleanup. Run 2 fb08a5b2: ~90% kept, ~25 s per unit.


v3 (commit 59def26, run 28d54b9e) adds the literature's trajectory filters (SWE-Lego 2601.01426, SWE-HERO 2604.01496): git hacking (commits after the base / other refs; past history allowed since snapshots ship .git), final patch touching test-patch or existing test files, >= 3 editor errors, null source change. Loss-side practice (tool outputs masked, step-level error masking on "status": "error" turns) is already in train_lora v2; curriculum-by-length via n_turns label. First units: 360/462 and 175/231 kept.
Run fb08a5b2 was cancelled on purpose to apply v3; run 6a161f84 failed by itself (deleted repo).


RESULT (run c9b10d84, v4 code, done 19:20 UTC Oct 15): CoderForge kept 124,174 / 155,144 over 32,208 tasks; nebius kept 25,326 / 32,161 over 3,511 tasks (final diff equals the source model_patch on 24,712 = 97.6%). Main rejects: >=3 editor errors (9,966), stale file (8,581), modifies tests (8,614), interactive input (2,762), no finish (2,716), git hacking (1,984). 12,822 rows carry network_or_install_cmds. Output 3.5 GB gz at ~/gc/external_trajs/graphcomp-external-trajs.

### `5303fc9a` SFT mix: dedupe trajectories + external-task overlap + nb21 31B LoRA notebook

- Branch: `orx/sft-mix-dedupe-trajectories-external-task-overla`
- Parent: ca1b26ab-33a4-429b-8272-dfcfce2855d0 (branch orx/external-verified-trajectories-harness-tool-form)
- Latest run: 765665ec-7c68-4333-ba1b-4f248bccc7be (done, commit 997b437, ran 33m00s, updated 4d ago)

build_sft_mix.py merges the converted external trajectories (ca1b26ab output) with our own verified RFT trajectories, drops exact duplicates (same task, same action sequence) and near-duplicates (same task, route similarity >= 0.9; different routes for the same problem are kept), excludes official dev/holdout tasks and rows with offline-violating installs, estimates tokens and writes train_fit.jsonl (fits the 31B training window) + train_all.jsonl.gz + manifest. It also splits the imported same-repo external tasks (394a0d99) into covered / not covered by a trajectory (the not-covered list is the queue for later teacher runs). make_nb21.py builds Kaggle notebook 21 from notebook 19 (setup, resume, budget, 31B LoRA training, BASE31 vs LORA31 holdout, budget-safe submission) without teacher generation or students, training on the attached mix. CPU job on the VM after the conversion run ends.

Status (Oct 15): code committed (658a47a); launch after conversion run 6c594d84 ends and import run 78afa2f4 is done (run_arm.py is replaced by nb20's version on install). Local checks: synthetic exact/near/label cases dropped as intended; on 728 converted rows 3 near-duplicates and 137 install rows dropped. nb21 built (32 cells, all parse).

RESULT (run 765665ec, done 19:54 UTC Oct 15): read 149,517 rows (149,500 converted + 17 own); dropped 12,822 with installs and 705 near-duplicates (0 exact duplicates); kept 135,990 over 32,466 tasks (train_all.jsonl.gz, 3.2 GB); 36,442 fit 32k tokens over 13,928 tasks (train_fit.jsonl, 3.7 GB: 33,562 CoderForge, 2,867 nebius, 13 own). Imported-task overlap not computed yet (import job 78afa2f4 still running). Uploaded as private Kaggle dataset sauravbellam/graphcomp-sft-mix for notebook 21.

### `eb5e85d4` Diagnose external import: why gold resolves only 8/160

- Branch: `orx/diagnose-external-import-why-gold-resolves-only`
- Parent: 394a0d99-2416-43e5-adf1-c97d3721af95 (branch orx/external-task-import-same-repo-swe-tasks-competi)
- Latest run: ea89b5bd-28de-4e62-815f-54c4903bba82 (done, commit ff5d165, ran 4s, updated 3d ago)

Read-only CPU job: group the import's gold-pass failures (~/gc/external/gold) by test exit code, error and the last lines of test output, per source dataset and repo, to find the setup problem behind 8/160.

### `3aab2b0a` External import validity v2: httpx test deps (trustme) on the sandbox path

- Branch: `orx/external-import-validity-v2-httpx-test-deps-trus`
- Parent: 394a0d99-2416-43e5-adf1-c97d3721af95 (branch orx/external-task-import-same-repo-swe-tasks-competi)
- Latest run: efcd7233-777d-4bd4-9008-777601b4b064 (done, commit f3fd097, ran 15m49s, updated 3d ago)

Diagnosis (run ea89b5bd): httpx 69 fail on 'No module named trustme' in conftest; fastapi 28 have no tests in SWE-bench train (unfixable); requests 44 are old versions using collections.MutableMapping (fail on Python 3.12, unfixable on the competition image); rich 11 have Python-version-dependent failing tests. Change: put trustme (and any missing deps) on SWE_EXTRA_PYTHONPATH like pytest-httpbin, rerun the gold and empty passes over the already-built 160 tasks, record validity, repackage, and print the remaining failure causes.

### `be2b5398` VM disk survey (read only)

- Branch: `orx/vm-disk-survey-read-only`
- Parent: 394a0d99-2416-43e5-adf1-c97d3721af95 (branch orx/external-task-import-same-repo-swe-tasks-competi)
- Latest run: c10faece-66c3-403c-af62-ecfdf8b97f83 (failed, commit a92d923, ran 9s, updated 3d ago)

Synthetic run e324fff0 stopped at its preflight: 4 GB free, needs 20 GB. List the largest directories under ~ and ~/gc so the user can choose what to delete.

### `b503a927` External import validity v3: pytest-asyncio for old async httpx tests

- Branch: `orx/external-import-validity-v3-pytest-asyncio-for-o`
- Parent: 3aab2b0a-7c52-44e3-afaf-bbf531e8ab60 (branch orx/external-import-validity-v2-httpx-test-deps-trus)
- Latest run: 3ba94d7b-d163-4aa1-9ea5-626b4ff529ff (done, commit 45c1a7b, ran 7m57s, updated 3d ago)

v2 (run efcd7233): 30/160 valid (httpx 22, rich 8). 35 httpx still fail with 'async def functions are not natively supported' (old tests relied on pytest-asyncio). Change: pytest-asyncio 0.21.2 (--no-deps, so the venv's pytest is not shadowed) on the sandbox test-deps path, then the same gold/empty/record/package steps. Not addressed: 10 httpx needing an older httpcore (CloseError), 2 timeouts; requests/fastapi/rich failures are unfixable (see eb5e85d4).

### `8e1a1a79` External import validity v4: pytest-trio for old httpx [trio] tests

- Branch: `orx/external-import-validity-v4-pytest-trio-for-old`
- Parent: b503a927-0ee8-4408-81a1-0728d56e0edc (branch orx/external-import-validity-v3-pytest-asyncio-for-o)
- Latest run: ed5d3a5b-8d17-4125-a282-0df1cbba5b79 (running, commit d4ba1d9, ran 91h34m, updated 1h ago)

v3 (run 3ba94d7b): 38/160 valid (httpx 30, rich 8). 26 httpx still fail, mostly [trio]-parametrized async tests ('async def functions are not natively supported'). Change: pytest-trio 0.8.0 (--no-deps; trio already on the path) added to the sandbox test-deps dir; same gold/empty/record/package steps.

### `fb116af8` Trainer session features functional test (tiny Gemma-4-31B-architecture model)

- Branch: `orx/trainer-session-features-functional-test-tiny-ge`
- Parent: 5303fc9a-5721-48f2-89dc-b76c394797b2 (branch orx/sft-mix-dedupe-trajectories-external-task-overla)
- Latest run: 928b2cc0-5098-4ee3-8660-0c9f53cab72b (done, commit 4f9730d, ran 10m29s, updated 21h ago)

Before notebook 21's 11.75 h Kaggle session: train_lora.py on a tiny random-weight model built from the 31B's own config (same tokenizer, chat template, layer types incl. k_eq_v full attention), 4-bit, --device-map auto, on 200 real converted trajectories. A: --keep-order --tail-n 3 --lr-floor 0.3 --stop-after-s 20 --save-every-min 0.05: must jump to the tail, stop on the deadline, train every tail trajectory, checkpoint, log tok/s. B: --init-adapter A: must resume and save. Exit 0 only if all checks pass. New VM comp (us-central1-b), GPU.

RESULT (run 928b2cc0, commit 4f9730d, VM comp): 11/11 PASS. A: jumped to the tail ("skipping 198 main samples to train the 4 tail samples"), stopped on the deadline after step 10, saved with trained_ids (all tail trajectories) and tok/s. B: "resumed from adapter", saved. Two earlier runs failed on a real bug the test found: assistant turns with no reasoning text (read_file continuation chunks, Qwen steps without text, final answers) cannot be targets under the thinking-on template -> now context only (2116 of the turns in 200 converted trajectories). Notebook 21 session 1 pushed to Kaggle afterwards with this trainer (kernel sauravbellam/graphcomp-21-sft-mix-lora31-s1).

### `d5e4b2a2` nb21 session 2+ data rules: replay, competition rotation + visit cap, length-quartile mixing

- Branch: `orx/nb21-session-2-data-rules-replay-competition-rot`
- Parent: fb116af8-1b91-4c45-b08b-5869ab5060e8 (branch orx/trainer-session-features-functional-test-tiny-ge)
- Latest run: — (never run)

Literature check (Oct 18): SWE-Lego cumulative curriculum + 2609.33559 (repeated texts make models fragile; replay and fresh solutions prevent it) + SWE-Gym easy-task bias. Session 2+ of notebook 21: REPLAY_FRAC 0.08 of earlier sessions' outside trajectories spread through the slice; competition tail rotated (least-visited half per session, cap 3 visits, every session still ends on competition data); new data round-robin over length quartiles. Session 1 (already running) is unchanged. Verified by an offline 8-session simulation of the data cell.

### `b70d7942` Trainer in-process out-of-memory recovery (no model reload)

- Branch: `orx/trainer-in-process-out-of-memory-recovery-no-mod`
- Parent: d5e4b2a2-336c-43cc-873a-7a4aa1fdb6c9 (branch orx/nb21-session-2-data-rules-replay-competition-rot)
- Latest run: 4e2bd751-0c8a-4cf6-87d9-2a70cb0e9af4 (done, commit 0f592b4, ran 22m38s, updated 19h ago)

Session 1 lost ~28 min per window fallback because an OOM kills the trainer process and the retry reloads the 31B. train_lora.py now catches torch.OutOfMemoryError: discards the group's partial gradients and skips its optimizer step, frees memory, skips that sample and every later one at least as long, continues with the loaded model; skipped trajectories are not counted as trained; val is OOM-safe; gives up after --max-oom-events. Functional test on the VM (tiny 31B-architecture model): A/B as before + C with a simulated OOM above half the max sample length (TRAIN_LORA_FAKE_OOM_ABOVE) must not crash, must recover, keep training and save.

### `b83c0e28` 31B memory: lighter GPU 3 device map, OOM-leak detection + resume within session, safer windows

- Branch: `orx/31b-memory-lighter-gpu-3-device-map-oom-leak-det`
- Parent: b70d7942-870f-4acd-b587-aef7bcc0f9f9 (branch orx/trainer-in-process-out-of-memory-recovery-no-mod)
- Latest run: 4fd70af7-2f2f-4b94-a63b-b71bff01bc95 (done, commit 887bdf1, ran 31m45s, updated 1h ago)

Session 2 (Kaggle, Oct 19): first OOM at a 25.9k-token sample was real (GPU 3 holds lm_head + logits + 15 layers), but the in-process recovery did not get the memory back, so every later sample OOMed (down to 4.9k) and 0 steps ran. Fix: --device-map custom (embeddings + lm_head + last --last-gpu-layers 9 layers on the last GPU, the other 51 spread evenly), leak check after an OOM (allocated memory not back near the pre-loop baseline -> save + exit 75), notebook continues at the next window from the saved adapter on exit 75, windows 24k/20k/16k with 2048/1536/1024 targets, speed from sessions with 0 steps ignored. Notebook 21 uses LAST_GPU_LAYERS = 6 (lighter than the 9 first described). Validated: device map covers every 31B parameter (meta model on the VM) + tiny-model functional test.

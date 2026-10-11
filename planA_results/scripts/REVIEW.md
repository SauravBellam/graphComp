# Code review of the Plan A pipeline (Oct 10, 2026)

Reviewed every file, ran the fixture pipeline + test assertions (Python 3.13 container), and ran
A1 against the **real archive** on 4 tasks (fastapi_14962, rich_4070, requests_7505, httpx_3672).

## Verdict

Sound. One design flaw found and fixed; everything else is notes. The harness-mirroring pieces
(resolve_node_name tiers, pytest.ini/conftest, `-p no:anyio`, edge-type `==` filter) match
swegemma 0.2.7 and the archive's sandbox/setup.py.

## Fixed in this review (a1_census.py, a6_release.py)

**v was computed over graph node text, not the repo.** `v_file_rank` ranked BM25 over node texts
grouped by module — but the graph omits all async defs, so an async-heavy gold file gets a weak
"document" and a depressed rank. That makes v (the Plan-B lexical stratifier) correlated with the
graph defect A1 is measuring, confounding the d×v payoff surface. Fix: added `v_snap_file_rank`
/ `v_snap_file_rank_src` / `v_snap_high` computed by BM25 over the snapshot's real .py file
contents (cached per base commit); a6's `_cell` now stratifies on the snapshot rank, falling back
to the old column only for old A1 outputs. Both metrics are kept — the graph-text rank is itself
evidence ("visibility inside the graph universe"). Real-data check: fastapi_14962 gold file ranks
5 by graph text, 1 by snapshot source text.

Also added `omission_direct_source_files` / `omission_direct_test_files` to the A1 summary
(tests are themselves async-heavy in fastapi; the headline claim should be quotable source-only).

## Verified on real data (4 tasks / 4 graphs)

- async omission is total: **0/667 async defs present** (sync functions 6212/6333 = 98.1%,
  classes 997/997, nested 0/562 by design). Edge types: only lowercase `calls` (11,590 edges).
- fastapi_14962: gold `fastapi.routing.serialize_response` (async) absent; harness-style query
  resolves to `tests.test_serialize_response.Item` (substring tier) — wrong node, test file.
  httpx_3672: 8/8 async gold functions absent, all misresolving.
- Prefix rules learned correctly (`docs_src/` drop seen as `custom_response.tutorial010_py310`).
- `py_compile` clean on Python 3.10.12 (A1/A3/A4/A5/A6 run on 3.10; A2 runner correctly refuses
  <3.11 — run it on 3.13 to match the harness).
- Fixture pipeline (a1,a4,a5,a6 + unit tests): all assertions pass. The only failing test
  assertion was the A2-dependent line in `test_a4_a5_a6`, because PyPI is unreachable in the
  review sandbox; the folder's own README records A2-venv tested end-to-end where it was written.

## Notes, no action needed (ranked)

1. `requirements.txt` lists networkx and tqdm but nothing imports them; scipy IS needed (a3, a4).
2. A2 oracle uses the **pre-fix trace only** for edge attribution (correct: the graph is built at
   the base commit), and both traces for gold execution. "all" scope includes test-patch-added
   test functions as `absent_sync` noise — quote the `nontest` scope in the writeup.
3. Tracer's generator/coroutine first-entry heuristic (`f_lasti <=` probe offset) is measured at
   import time per interpreter — fine on one Python version; don't mix trace files from different
   minor versions.
4. Subprocesses spawned *by tests* are not traced (sys.setprofile is per-process). Edges via
   anyio worker threads ARE traced (threading.setprofile installed at session start).
5. `is_test_module_id` also pattern-matches the final symbol name, so a source function literally
   named `test_*` would be classed as test — rare; shows up only in the nontest splits.
6. a6 `calls_dynamic` edges come from `edges_direct` only (via-library edges excluded) — a
   deliberate precision choice; say so in the v2 graph card.
7. A4 shard summaries are per shard; merge rel_rows CSVs manually if sharded (A1 has `merge`).
8. In the identifier stop list, bare repo names ("fastapi", "requests", "rich", "httpx") are
   dropped — deliberate (they'd anchor to the package root), worth one sentence in the writeup.

## Environment facts for running

- A1 4 tasks ≈ 3–4 min single-process incl. async audit + snapshot BM25 (macOS VM, py3.10).
  Budget roughly 1 core-minute per task; shard across cores.
- A2 must run with Python 3.13 (harness parity): Debian 13 VM (ships 3.13) with the venv
  backend, or any VM + Docker with `planA/a2/Dockerfile.audit`. Needs PyPI network access.
- A3 needs huggingface.co + `HF_TOKEN` (EmbeddingGemma v1 gated) + sentence-transformers/torch.

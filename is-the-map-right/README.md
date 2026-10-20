# Is the Map Right?

**An execution-grounded audit of the code graph and embeddings handed to local SWE agents.**
Gemma 4 Developer Agent Competition, Paper Track.

This repository holds the paper, every script that produced its numbers, the result files those scripts wrote, and a one-command check that the numbers in the paper match the files.

| Start here | |
| --- | --- |
| Paper | [`planA_results/papers/latex/paper1_is_the_map_right.pdf`](planA_results/papers/latex/paper1_is_the_map_right.pdf) |
| Guide for judges (claim-to-file map, reproduction) | [`planA_results/papers/latex/for_judges.pdf`](planA_results/papers/latex/for_judges.pdf) |

## Check the paper's numbers

```
cd planA_results
python3 papers/verify_paper1_numbers.py
```

Standard library only; a few seconds. It recomputes 66 claims from the result files and prints PASS or FAIL with the source of each. Result: 66 PASS, 0 FAIL, 9 INFO (saved in `planA_results/papers/verify_paper1_numbers.out.txt`); INFO lines are values shown only in figures.

## Layout

```
planA_results/
  papers/
    latex/            paper and judges' guide: .tex sources and PDFs
    figures/          the 8 figures (PDF), PNG copies, and the numbers each plots (figure_data.json)
    make_figures.py   draws every figure from the result files
    verify_paper1_numbers.py
  scripts/
    planA/            pipeline: census (a1), dynamic oracle (a2), encoder (a3), similarity (a4),
                      retrieval (a5), release graph (a6), id reconciliation (a7), edge semantics (a8)
    tests/            synthetic-fixture test suite (15 tests)
    ops/              VM setup and the exact launch scripts used
  vm-a2/results/      outputs of the final runs (a1_v2, a2_v3, a2_runs, a3, a5_v3, a6_v3, a7, a8)
  vm-main/results/    similarity probe (a4)
  provenance/         host specs, package versions, script checksums
```

The released typed graphs (one per base commit, 127 files) are in `planA_results/vm-a2/results/a6_v3/graphs_v2/` as gzipped node-link JSON in the harness's format.

## Reproduce

The competition archive (tasks, graphs, embeddings, repository snapshots) is not redistributed here. With it, on Linux with Python 3.13 and network access:

```
cd planA_results/scripts
pip install -r planA/requirements.txt
python3 planA/run_all.py --data <archive> --out <results> --stages a1,a4,a5 --a1-shards 8 --jobs 8
python3 planA/a2/run_task.py --data <archive> --out <results>/a2_runs --backend venv --jobs 16 --keep
bash ops/rerun_fixes.sh <archive> <results> <cache> 8
bash ops/run_gaps.sh <archive> <results> <cache>
```

Redraw the figures from the included results (needs matplotlib):

```
cd planA_results && python3 papers/make_figures.py
```

Run the pipeline's own tests (no competition data needed; needs numpy, scipy, pytest):

```
cd planA_results/scripts && python3 -m pytest tests/test_pipeline.py -q
```

## Notes

- Our test runs used per-task virtual environments with network access instead of the competition's cached dependencies; 15 of 129 tasks did not run cleanly in that setup (9 collection errors, 2 import errors, 4 with no collectable tests).
- One task, rich_4006, hung on four test cases; they were deselected and recorded as failed (`hung_tests` in its `status.json`).
- Log and trace files record absolute paths from the machines the runs used.

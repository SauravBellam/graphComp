# planB_results: index (Oct 14, 2026)

Every run folder holds the raw harness output untouched: `runs/<RUN>/merged_<RUN>/` (task_results.jsonl, traces,
patches, test outputs, logs), `scorecard/<RUN>/` (scorecard 0.3.2), `validity/`, `provenance/` (packages, serving, env),
`submissions/` (the exact agent config), `progress.log`. Resolution is always on that environment's sub_valid tasks.

## Kaggle, 4 x L4, gemma-4-31b-it-qat-w4a16-ct (`kaggle/`)
| Folder | Notebook | Run | Tasks | Wheelhouse build | Headline | Read first |
| --- | --- | --- | --- | --- | --- | --- |
| `nb01_B2_31B_all129_oldbuild/` | 01 | validity + B2 baseline, seed 1 | all 129 | old (2c482f9d8c, Oct 12) | resolved 38/103 = 36.9%; graph adoption 14/129; 20 context-exceeded | `B2_s1_evaluation.md`, `runs/B2_s1/summary_B2_s1.md` |
| `nb14_B2_31B_dev_newbuild/` | 14 | B2 on the rebuilt wheelhouse, seed 1 | dev 24 | new (0c81a00daf, Oct 13) | 10/23 vs 11/23 on the old build; 20/23 agree; validity re-check 24/24 | `runs/B2-WH2_s1/summary_B2-WH2_s1.md`, `validity/recheck/recheck.json` |

| `nb03_FC_31B_dev_newbuild/` | 03 | FC: forced graph call every turn, seed 1 | dev 24 | new (0c81a00daf) | 8/23 vs B2-WH2 10/23 (B2-only 3, FC-only 1, p 0.63); context-exceeded 2 -> 6; hallucination 2/818 -> 19/801; anti-Goodhart: v0-v5 score FC higher (fail), v7 (TQS) lower (pass) | `runs/FC_s1/summary_FC_s1.md` |
| `nb11_B2_12B_dev_newbuild_FAILED-serving-tp4/` | 11 | B2 on gemma-4-12b-it-qat-w4a16-ct, tp 4 | dev 24 | new (0c81a00daf) | **failed at serving** (no agent tasks ran): vLLM 0.19.1 has no native gemma4_unified, the Transformers fallback crashed at tp 4 (`Expected hidden_size to be 3840, but found: 256`); validity re-check 24/24 agree | the `.log` (search `ValueError`), `validity/recheck/` |

New-build Kaggle arms pair with `nb14`; `nb01` pairs only with old-build runs.

## VM, one L4, PyPI stack (`vm/`), pairs only with VM runs
| Folder | Model | Arms | Tasks | Headline | Read first |
| --- | --- | --- | --- | --- | --- |
| `e4b_dev_screen_B2-FC-QN-T0-O8K/` | gemma-4-e4b-it-qat-w4a16-ct | B2, FC, QN, T0, O8K | dev 24 | B2 3/23 (floor); QN graph-call precision 36% -> 52%; O8K context-exceeded 5 -> 0 | `summary_dev.md`, `settings.json` |

## LoRA training data (`lora_data/`)
| Folder | Source | Content |
| --- | --- | --- |
| `rft_31b_B2s1/` | `kaggle/nb01.../runs/B2_s1/merged_B2_s1` | 17 resolved non-slice 31B trajectories; `manifest.json`, `excluded.csv` |

## Setup records and inputs (kept at fixed paths: scripts read them)
| Folder | What |
| --- | --- |
| `b0/` | B0 harness bring-up: session A (Kaggle), gold-patch runs, `B0.md` |
| `b1/` | B1 task profile (`profile.csv`), CPU validity run, copy of the official slices |
| `kaggle_upload/` | Kaggle notebooks 01-17 (`notebooks/`) and the `graphcomp-planb` dataset files (oracle pack) |
| `kaggle_upload/b4_briefs/` | B4 briefs for notebook 16 (`briefs.json` embedded in it, `<id>.md`, `diagnostics.json`); built by `scripts/planB/build_b4_briefs.py` |

New runs: put a downloaded Kaggle zip in `kaggle/nbNN_<arm>_<model>_<scope>_<build>/`, a VM tarball in
`vm/<model>_<scope>_<arms>/`, and add a row here.

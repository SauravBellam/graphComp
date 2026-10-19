# Kaggle notebooks

All notebooks are in [`../notebooks/`](../notebooks/). They run on Kaggle with 4×L4 and Internet off. Each one is
self-contained: scripts, test-dependency wheels and the oracle pack are embedded in the notebook or attached as private
datasets.

The status column says whether results from a notebook are in this repository. "Prepared" means the notebook was built,
but no results from it are recorded here.

## Plan B: graph tools, prompting, context (Oct 12-14)

| # | Notebook | Arm | Status |
| --- | --- | --- | --- |
| 01 | `01_validity_B2_s1.ipynb` | validity check + B2 baseline, 31B, all 129 tasks | **Ran:** 38/103 resolved (`results/kaggle/nb01_B2_31B_all129_oldbuild/`) |
| 02 | `02_B3_s1.ipynb` | B3: graph tools promoted in the prompt | Prepared |
| 03 | `03_forced_call_dev_s1.ipynb` | FC: forced graph call every turn, dev | **Ran:** 8/23 vs B2 10/23 (`results/kaggle/nb03_FC_31B_dev_newbuild/`) |
| 04 | `04_qualified_names_dev_s1.ipynb` | QN: qualified names in graph calls | Prepared (screened on E4B instead) |
| 05-08 | `05_B2_s2` … `08_B3_s3` | seeds 2-3 of B2 / B3 | Prepared |
| 09 | `09_thinking0_dev_s1.ipynb` | T0: thinking off | Prepared (screened on E4B) |
| 10 | `10_outreserve8k_dev_s1.ipynb` | O8K: 8192 output reserve | Prepared (screened on E4B) |
| 11 | `11_B2_12b_dev_s1.ipynb` | B2 on 12B | **Failed at serving** (vLLM tp 4) |
| 12-13 | `12_B2_e4b_dev_s1`, `13_B2_e2b_dev_s1` | B2 on E4B / E2B | Prepared (E4B run on the VM instead) |
| 14 | `14_B2_newstack_dev_s1.ipynb` | B2 on the rebuilt wheelhouse, dev | **Ran:** 10/23; defines the noise bar (`results/kaggle/nb14_B2_31B_dev_newbuild/`) |
| 15 | `15_B2_subset_newbuild_s1.ipynb` | B2 on 43 tasks (paired baseline for 16/17) | Prepared |
| 16 | `16_B4_push_s1.ipynb` | B4: structural brief pushed into the issue | Prepared |
| 17 | `17_O8K_ctxheavy_s1.ipynb` | O8K on 21 context-heavy tasks | Prepared |

## Post-training (Oct 15-19)

| # | Notebook | What | Status |
| --- | --- | --- | --- |
| 19 | `19_teacher_lora31_student_holdout_submission.ipynb` ([README](../notebooks/19_README.md)) | 31B teacher trajectories → 31B LoRA + 12B student → BASE31 vs LORA31 holdout → submission | **Ran Oct 15:** 280 attempts → 45 verified trajectories; both LoRAs failed (accelerate `logits_to_keep` device bug; broken 12B checkpoint); BASE31 14/27 submitted (`results/lora_data/graphcomp_nb19/SUMMARY.md`) |
| 20 | `20_iterative_rft_campaign.ipynb` | 45 h iterative RFT: sample with the current LoRA, keep verified trajectories with accurate tool use, retrain | Built, not run (replaced by 21) |
| 21s | `21s_SMOKE_sft_mix_lora31.ipynb` | smoke test: one trajectory → LoRA → submission | Built |
| 21 | `21_sft_mix_lora31_submission.ipynb` | 31B LoRA on the 36k-trajectory SFT mix, resumable 11.75 h sessions, competition data last, adapter check, submission | **s1** (Oct 18): 0 steps, out of memory. **s2** (Oct 19): 0 steps, OOM leak. **s3** (Oct 19): running with the memory fixes. See README §4.5 |

Kernel metadata for the notebook 21 sessions is in
[`../notebooks/kaggle_kernels/`](../notebooks/kaggle_kernels/). It lists the attached datasets and models, the
`NvidiaL4` machine shape, the pinned Docker image, and `kernel_sources`, which chains each session to the previous one.

The notebooks are generated, so edit the builder rather than the notebook:

- notebooks 19, 20 and 21 come from `scripts/planB/notebook_builder/make_nb19.py`, `make_nb20.py` and `make_nb21.py`;
- notebooks 15-17 come from `make_nb15_17.py`;
- notebooks 01-14 come from `gen_all.py` and `variants.json`, in the same folder.

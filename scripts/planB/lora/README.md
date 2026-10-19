# LoRA pipeline (Plan C, Oct 14): RFT / distillation first, adoption later

1. **Data generation** (VM, never on dev / holdout): `dev_runs.py --model e4b --task-set rest --arms B2 --seeds 1,2,3`
   runs the valid tasks in neither official slice; trajectories land in `~/gc/runs_dev_e4b/runs/B2_s<seed>_rest/merged_*`.
   Adoption data later: `--arms FC --task-set rest`.
2. **Build** (laptop or VM, CPU): `build_rft_data.py --runs <merged dirs> --profile <validity profile> --out <dir>`
   keeps resolved, valid, non-slice trajectories (refuses to write if a dev / holdout task slips in), at most 2 per task.
   Tested Oct 14 on B2_s1 (31B, Kaggle): 17 trajectories, 540 assistant turns -> `planB_results/lora_data/rft_31b_B2s1/`.
3. **Check** (CPU, needs transformers 5.13 + the harness wheels): `train_lora.py --data ... --base google/gemma-4-E4B-it-qat-q4_0-unquantized --out ... --check`.
   Renders every turn with the model's chat template and the harness's own tool schemas (swegemma tools -> ADK
   FunctionTool -> litellm), checks each target continues its prompt, and compares the rendered prompt size with the size
   recorded in the trace. Oct 14 result on rft_31b_B2s1: 0 prefix mismatches; recorded - rendered = 27 tokens (median;
   p10 20, p90 27; ratio 1.003); 13/17 trajectories compacted, so 306 later turns are dropped (the compaction summary is not in
   the trace); 234 usable turns (195 train / 39 val by task), 40k target tokens.
   Findings built in: arguments must be a mapping; earlier turns keep their reasoning (the template keeps it within one user
   turn, and the recorded sizes confirm it); ADK appends `You are an agent. Your internal name is "swe_planb_agent".`;
   the reasoning of the final answer is the text-only step just before it.
4. **Train** (VM, one L4; NOT YET RUN): same command without `--check`. QLoRA (nf4) on the unquantized QAT base, rank 16,
   language-model q/k/v/o/gate/up/down projections, loss on the target turn only, val loss every 20 steps.
   Needs `pip install peft` in ~/gc/venv. Adapter -> `<out>/adapter` (base_model_name_or_path set to the served w4a16 id).
5. **Evaluate** (VM): `dev_runs.py --model e4b --adapter rft_lora=<out>/adapter --arms B2,LORA` -> its own folder
   (`~/gc/runs_dev_e4b_rft_lora`): B2 with the adapter mounted but unused vs LORA routing to it, paired on the dev slice.
   Noise bar from 14 vs B2_s1: about 3/23 tasks flip between two runs of the same condition.

Known limits: post-compaction turns are not trained on (late phases: editing, testing, submitting are under-represented);
the data is small (17 31B trajectories); distillation from 31B into E4B is the main source until data generation adds more.

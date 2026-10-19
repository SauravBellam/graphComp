# Plan B dev-slice runs on a single-L4 VM

Two scripts. `setup_vm.sh` prepares the VM once; `dev_runs.py` runs validity, the B2 baseline and the four dev ablations
(FC forced call, QN qualified names, T0 thinking 0, O8K output reserve 8192) on the 24 dev tasks, then writes a paired
summary and a tarball. Both are resumable: re-run the same command after any interruption.

## What is compared with what
Everything in one `--out` folder is one condition: this VM, its Python environment, one serving setting, one concurrency.
**B2 here is the baseline for FC / QN / T0 / O8K here.** None of these runs is paired with a Kaggle run (different machine
type, different sandbox environment: Plan B pairing rule). For the same reason validity (which tasks can be resolved at
all) is measured on the VM, for all 129 tasks, and resolution is reported on the dev tasks that are valid here.

## 1. Copy the needed files to the VM
From the laptop, inside the `graphComp` folder (replace `VM` and `ZONE`):

    COPYFILE_DISABLE=1 tar czf /tmp/gc_vm.tgz scripts/planB harness_wheels planB_results/b1/profile.csv planB_results/kaggle_upload/graphcomp-planb
    gcloud compute scp /tmp/gc_vm.tgz VM:~ --zone ZONE
    gcloud compute ssh VM --zone ZONE -- 'mkdir -p ~/graphComp && tar xzf ~/gc_vm.tgz -C ~/graphComp'

And your Kaggle token (kaggle.com > Settings > API > Create New Token), never into the repo:

    gcloud compute scp ~/Downloads/kaggle.json VM:~ --zone ZONE
    gcloud compute ssh VM --zone ZONE -- 'mkdir -p ~/.kaggle && mv ~/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json'

## 2. Set up (once, about 30-60 min, mostly downloads)

    gcloud compute ssh VM --zone ZONE
    tmux new -s gc
    cd ~/graphComp && bash scripts/planB/vm/setup_vm.sh 2>&1 | tee -a ~/gc/setup.log

It stops with a clear message if the GPU driver is older than 570 (vLLM 0.19.1 from the competition wheelhouse is built
for CUDA 12.8) or the token is missing. It makes a Python 3.12 venv at `~/gc/venv`, downloads the competition data,
the wheelhouse and the model into `~/gc/data`, installs the wheelhouse and the competition's test-dependency wheels
(including `inline-snapshot` and `dirty-equals`), checks the imports, and saves the package list.

**Required (Oct 13):** `planB_results/kaggle_upload/graphcomp-planb/oracle_pack.json.gz` must be on the VM (it is in `gc_vm.tgz`). `setup_vm.sh` (step 1) and `dev_runs.py` (step 0) stop at once if it is missing or lacks a competition task, so a run never finishes without its scorecard. If a scorecard still fails, `scorecard/SCORECARD_MISSING_<ARM>.txt` is written and the run is kept.

**Size sweep (Oct 13, smaller variants):** 31B stays the default and the Plan B condition. For another variant:
`MODEL_VARIANT=e4b MODEL_VERSION=<version on its Kaggle model page> bash scripts/planB/vm/setup_vm.sh` (downloads to
`~/gc/data/model_e4b`), then `~/gc/venv/bin/python scripts/planB/vm/dev_runs.py --model e4b --arms B2` (results in
`~/gc/runs_dev_e4b`, own fixed settings, sample LoRAs off, this VM's 31B validity reused if present). Choices: 12b, e4b, e2b.
12B is a different architecture (`gemma4_unified`); its first serving attempt is the check.

**Stack (Oct 13):** `setup_vm.sh` defaults to `STACK=pypi`: no Kaggle wheelhouse. Harness wheels come from `harness_wheels/`
(checked to be B2_s1's build), everything else is pinned from PyPI (`stack_pins.txt`). `STACK=wheelhouse` restores the old download.
The model comes from Kaggle, or from Hugging Face if Kaggle refuses. VM results pair only with VM results.

## 3. Run (detach with Ctrl-b d; re-attach with `tmux attach -t gc`)

    cd ~/graphComp && ~/gc/venv/bin/python scripts/planB/vm/dev_runs.py 2>&1 | tee -a ~/gc/dev_runs.log

The dev slice is the **official** one, `scripts/planB/official_slices/dev_slice.txt` (fixed Oct 12, the same file
the Kaggle notebooks use; it travels with `scripts/planB` in step 1). Options: `--arms B2,FC` (subset; B2 should always
be run first), `--dev-slice FILE` (a different slice file; not for Plan B results), `--chunk 4` (tasks per harness call),
`--concurrency N` (first run only; default from vLLM's KV capacity), `--no-loras` (do not mount the sample LoRAs; frees
memory on one L4, but must then stay off for every arm).

Expected time: validity about 40-60 min (CPU). Agent tasks: unmeasured on one L4. Kaggle's 4 x L4 took about 5 min per
task at concurrency 1; one L4 decodes more slowly and will likely run at concurrency 1, so plan on 10-15 min per task,
i.e. about 4-6 h per arm and roughly a day for all five. The first B2 chunks print the real rate (ETA after every chunk).

What it prints: time-stamped START / DONE per step; one line per finished task (resolved, tool calls, graph calls, how
it ended, seconds); a heartbeat every 2 minutes (counts, elapsed, GPU utilisation, vLLM running / waiting); vLLM's log
every 30 s while it starts; the serving ladder's outcome per rung; an ETA after every chunk. All of it also goes to
`~/gc/runs_dev/progress.log`.

Fixed on the first run and enforced afterwards (the script refuses to continue otherwise): the package list, the serving
setting (`gpu_memory_utilization`, `max_model_len`), whether LoRAs are mounted, and the concurrency. The serving ladder
tries (0.90, 32768), (0.95, 32768), then 28672 / 24576 / 20480; a context below 32768 is a condition of every arm here
and is printed in the summary.

## 4. Results
- `~/gc/runs_dev/summary_dev.md`: per arm, tasks run, resolution on valid dev tasks (Wilson 95%), share of tasks with a
  graph call, graph / tool calls, end reasons, median seconds, and arm-only vs B2-only resolved tasks with an exact McNemar p.
- `runs/<ARM>_s1/merged_<ARM>_s1/`: the harness results per arm (traces, logs, patches, test outputs, task_results.jsonl).
- `scorecard/<ARM>_s1/`: scorecard 0.3.2 per arm (all versions v0 to v7, length check, end reasons).
- `validity/`: gold and empty passes, `slices/` (dev, holdout, profile with `sub_valid`, meta).
- `provenance/`: package lists, environment, vLLM logs per serving attempt.
- `~/gc/runs_dev_<time>.tar.gz`: all of it; copy to `planB_results/vm/` on the laptop:

      gcloud compute scp VM:~/gc/runs_dev_*.tar.gz planB_results/vm/ --zone ZONE

## Notes
- `validity/slices/dev_slice.txt` on the VM is make_slices.py's own draw on this VM's validity, kept for reference only;
  the runs use the official slice (logged at step 2 as `OFFICIAL slice`). How the official slice was drawn:
  `scripts/planB/official_slices/README.md`.
- Delete `~/.kaggle/kaggle.json` (or expire the token) when the downloads are done.

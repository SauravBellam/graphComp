#!/bin/bash
# The project's fixed run command: `bash vm_job.sh`, launched on the GCP VM through the SSH backend.
# Runs from the run's source snapshot. It installs this commit's scripts over the VM checkout (~/graphComp, which holds
# the data, harness and other scripts) and then runs the job. Each experiment branch edits JOB below and leaves the rest alone.
# Exit 0 = done; any failure exits non-zero, so the run is marked failed and the agent is woken.
set -euo pipefail
G=~/graphComp; PY=~/gc/venv/bin/python
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
log() { echo "[$(date -u +%H:%M:%S) UTC] $*"; }
trap 'log "FAILED at line $LINENO (exit $?)"' ERR

for f in $(find scripts -type f ! -path "*__pycache__*"); do
  case "$f" in scripts/*) install -D -m 644 "$f" "$G/$f";; esac
done
log "installed this commit's scripts into $G"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader

# ------------------------------------------------------------------ JOB (Oct 15): E4B deploy rehearsal -> action agreement -> merged adapter
HUB=~/.cache/huggingface/hub
E4B=$(ls -d $HUB/models--google--gemma-4-E4B-it-qat-q4_0-unquantized/snapshots/*/ | head -1)
E2B=$(ls -d $HUB/models--google--gemma-4-E2B-it-qat-q4_0-unquantized/snapshots/*/ | head -1)
AD=~/gc/lora/rft_e4b_v1
R=~/gc/vm_job_e4b_20261015; mkdir -p $R

log "STEP 1/3: deployed-budget rehearsal, E4B w4a16, dev 24, arms B2 + DEP"
$PY $G/scripts/planB/vm/dev_runs.py --model e4b --arms B2,DEP --no-lean --out ~/gc/runs_dev_e4b_deploy
log "STEP 1/3 done"; grep -E "^\| (B2|DEP) " ~/gc/runs_dev_e4b_deploy/summary_dev.md

log "STEP 2/3: next-action agreement with 31B on the adapter's 39 val turns (E2B, E4B, E4B + rft_e4b_v1)"
$PY $G/scripts/planB/lora/action_agreement.py --data $G/planB_results/lora_data/rft_31b_B2s1/train.jsonl \
  --val-from $AD/train_meta.json --out $R/action_agreement.json \
  --model e4b=$E4B --model e4b_rft=$E4B+$AD/adapter --model e2b=$E2B
log "STEP 2/3 done"

log "STEP 3/3: unquantized E4B base vs merged rft_e4b_v1, bf16, dev 24, B2"
$PY $G/scripts/planB/vm/merge_lora.py --base $E4B --out ~/gc/data/model_e4b_unq
$PY $G/scripts/planB/vm/dev_runs.py --model e4b --arms B2 --no-lean --model-dir ~/gc/data/model_e4b_unq --out ~/gc/runs_dev_e4b_unq
$PY $G/scripts/planB/vm/merge_lora.py --base $E4B --adapter $AD/adapter --out ~/gc/data/model_e4b_merged_rft
$PY $G/scripts/planB/vm/dev_runs.py --model e4b --arms B2 --no-lean --model-dir ~/gc/data/model_e4b_merged_rft --out ~/gc/runs_dev_e4b_merged_rft
rm -rf ~/gc/data/model_e4b_merged_rft      # 15 GB; re-create with merge_lora.py
for d in unq merged_rft; do grep -E "^\| B2 " ~/gc/runs_dev_e4b_$d/summary_dev.md | sed "s/^/$d: /"; done
log "STEP 3/3 done; ALL DONE"

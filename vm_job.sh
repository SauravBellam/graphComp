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

# ------------------------------------------------------------------ JOB (Oct 18): train_lora.py session features, functional test (GPU)
# Tiny random-weight model with the 31B's architecture; 200 real converted trajectories. See scripts/planB/lora/test_train_resume.py.
export HF_HUB_OFFLINE=0
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
D=~/gc/train_test; mkdir -p $D
head -n 200 ~/gc/sft_mix/graphcomp-sft-mix/train_fit.jsonl > $D/trajs.jsonl
$PY $G/scripts/planB/lora/test_train_resume.py --data $D/trajs.jsonl --work $D
log "ALL DONE"

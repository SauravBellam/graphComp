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

# ------------------------------------------------------------------ JOB (Oct 15): E2B, B2 vs FC (forced graph call) on all 129 tasks
M=~/gc/data/model_e2b
R=~/gc/runs_all129_e2b; mkdir -p $R

log "STEP 1/2: E2B w4a16 (gemma-4-e2b-it-qat-w4a16-ct) from Hugging Face, once (8.3 GB)"
if [ ! -f $M/model.safetensors ]; then
  HF_HUB_OFFLINE=0 $PY -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B-it-qat-w4a16-ct', local_dir=sys.argv[1])" $M
  echo "model gemma-4-e2b-it-qat-w4a16-ct from Hugging Face google/gemma-4-E2B-it-qat-w4a16-ct $(date -u +%FT%TZ)" > ~/gc/logs/model_source_e2b.txt
fi
ls -la $M; df -h ~ | tail -1
log "STEP 1/2 done"

log "STEP 2/2: B2 then FC on all 129 tasks (invalid-here tasks run but are excluded from rates)"
$PY -c "import json,sys; print('\\n'.join(json.loads(l)['instance_id'] for l in open(sys.argv[1]) if l.strip()))" ~/gc/data/comp/tasks.jsonl > $R.slice.txt
test "$(wc -l < $R.slice.txt)" -eq 129
$PY $G/scripts/planB/vm/dev_runs.py --model e2b --arms B2,FC --no-lean --dev-slice $R.slice.txt --out $R
grep -E "^\| (B2|FC) " $R/summary_dev.md
log "STEP 2/2 done; ALL DONE"

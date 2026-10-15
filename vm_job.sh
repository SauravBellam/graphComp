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

# ------------------------------------------------------------------ JOB (Oct 15): external verified trajectories -> harness tool format (CPU only)
# Sources: CoderForge-Preview filtered_reward1 + nebius SWE-rebench-openhands resolved; teacher Qwen3-Coder-480B only (user: only
# models stronger than Gemma 4 31B). Each trajectory is replayed on a checkout of its task (see convert_trajs.py); resumable per unit.
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=0   # the GPU search and the import job keep running
C=~/gc/data/comp
X=~/gc/external_trajs; O=$X/graphcomp-external-trajs; mkdir -p $X
E=$G/scripts/planB/external
FREE=$(df -BG --output=avail ~ | tail -1 | tr -dc 0-9); log "disk free: ${FREE} GB"
[ "$FREE" -ge 12 ] || { log "need >= 12 GB free (outputs ~5 GB gz + 5 GB git cache)"; exit 1; }
nproc; free -g | head -2

log "STEP 1/2: convert (replay on checkouts; git cache capped at 5 GB)"
export CONVERT_TMP=$X/tmp; mkdir -p $CONVERT_TMP
$PY -W ignore $E/convert_trajs.py --out $O --comp-tasks $C/tasks.jsonl --system-prompt $E/deploy_system_prompt.txt \
    --sources coderforge,rebench_oh --workers 6 --cache $X/git_cache --cache-gb 5

log "STEP 2/2: package summary"
cp $E/convert_trajs.py $O/
du -sh $O/train $O/rejects $X/git_cache; df -h ~ | tail -1
log "ALL DONE"

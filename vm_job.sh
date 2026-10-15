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

# ------------------------------------------------------------------ JOB (Oct 15): tool tracing over the E2B forced-rate search runs (CPU only)
R=~/gc/runs_fcsearch_e2b
test -f $R/fc_search.json || { log "no finished search at $R"; exit 1; }
O=$R/tool_trace; mkdir -p $O
$PY -I $G/scripts/planB/tool_trace.py --selftest
log "STEP 1/2: all probes together"
$PY $G/scripts/planB/tool_trace.py $R/runs/*/merged_* --tasks ~/gc/data/comp/tasks.jsonl --valid $R/validity/slices/profile.csv --out $O/all
log "STEP 2/2: one report per probe (forced rate)"
for m in $R/runs/*/merged_*; do
  n=$(basename $(dirname $m))
  $PY $G/scripts/planB/tool_trace.py $m --tasks ~/gc/data/comp/tasks.jsonl --valid $R/validity/slices/profile.csv --out $O/$n > /dev/null
  log "  $n: $(grep -m1 trajectories $O/$n/tool_trace.md)"
done
tar czf $R.tool_trace.tgz -C $R tool_trace; ls -la $R.tool_trace.tgz
log "ALL DONE"

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

# ------------------------------------------------------------------ JOB (Oct 16): disk survey, read only
df -h / ~ | cat
echo "== ~ (top level)"; du -xsh ~/* ~/.cache ~/.local 2>/dev/null | sort -rh | head -25
echo "== ~/gc"; du -xsh ~/gc/* 2>/dev/null | sort -rh | head -30
echo "== ~/gc second level (>= 500 MB)"; du -xh --max-depth=2 ~/gc 2>/dev/null | awk '$1 ~ /G$/ || ($1 ~ /M$/ && $1+0 >= 500)' | sort -rh | head -40
echo "== ~/.cache"; du -xsh ~/.cache/* 2>/dev/null | sort -rh | head -10
echo "== /tmp"; du -xsh /tmp/* 2>/dev/null | sort -rh | head -10
log "ALL DONE"

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

# ------------------------------------------------------------------ JOB (Oct 15): E4B, max_output_tokens 8192, search for the best forced graph-call rate on the 64 fastapi + rich tasks in neither slice
M=~/gc/data/model_e4b
R=~/gc/runs_fcsearch_e4b_o8k_rest; mkdir -p $R

log "STEP 1/2: E4B w4a16 already on the VM (~/gc/data/model_e4b)"
test -f $M/config.json
log "STEP 1/2 done"

log "STEP 2/2: forced-rate search among forced conditions only (7% = E4B's natural graph-call share, and 100%, then midpoints)"
# never dev, holdout or the 3 extra holdout tasks: 64 fastapi + rich tasks remain
$PY - ~/gc/data/comp/tasks.jsonl $G/scripts/planB/official_slices > $R.slice.txt <<'PYEOF'
import json, sys
from pathlib import Path
sl = Path(sys.argv[2]); out = set()
for f in ("dev_slice.txt", "holdout_slice.txt"):
    out |= {l.strip() for l in open(sl / f) if l.strip()}
out |= {"fastapi_15661", "rich_3471", "rich_3718"}
ids = [json.loads(l)["instance_id"] for l in open(sys.argv[1]) if l.strip()]
print("\n".join(t for t in ids if t.split("_")[0] in ("fastapi", "rich") and t not in out))
PYEOF
test "$(wc -l < $R.slice.txt)" -eq 64
$PY -I $G/scripts/planB/vm/fc_search.py --selftest
$PY -I $G/scripts/planB/tool_usage.py --selftest
$PY $G/scripts/planB/vm/fc_search.py --out $R --start 7,100 --max-probes 7 --min-gap 12 -- --model e4b --max-out 8192 --no-lean --auto-concurrency --dev-slice $R.slice.txt
cat $R/fc_search.md
tar czf $R.fc_search.tgz -C $(dirname $R) $(basename $R)/fc_search.md $(basename $R)/fc_search.json $(basename $R)/fc_search_tasks.csv $(basename $R)/fc_search_by_category.csv $(basename $R)/fc_search_tools.md $(basename $R)/fc_search_prompts $(basename $R)/settings.json $(basename $R)/summary_dev.md
ls -la $R.fc_search.tgz
log "STEP 2/2 done; ALL DONE"

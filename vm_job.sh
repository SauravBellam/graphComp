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

# ------------------------------------------------------------------ JOB (Oct 15): requests/httpx re-grader (src/ shim), validity on 14 tasks
C=~/gc/data/comp
R=~/gc/regrade_src_shim; rm -rf $R; mkdir -p $R
IDS=$($PY -c "import json,sys; print(' '.join(t for t in (json.loads(l)['instance_id'] for l in open(sys.argv[1]) if l.strip()) if t.split('_')[0] in ('requests', 'httpx')))" $C/tasks.jsonl)
test "$(echo $IDS | wc -w)" -eq 14
RA="$PY $G/scripts/planB/run_arm.py --tasks $C/tasks.jsonl --snapshots $C/snapshots --submission $C/sample_submission --sandbox subprocess --timeout-seconds 300 --concurrency 2"

log "STEP 1/2: gold-patch pass on the 14 requests/httpx tasks (shim on)"
$RA --mode gold --results $R/gold --task-ids $IDS
GOLD_OK=$($PY -c "import json,sys; print(' '.join(r['instance_id'] for r in map(json.loads, open(sys.argv[1])) if r.get('resolved')))" $R/gold/task_results.jsonl)
log "gold resolves $(echo $GOLD_OK | wc -w)/14: $GOLD_OK"

log "STEP 2/2: empty-patch pass on the gold-resolving tasks"
if [ -n "$GOLD_OK" ]; then $RA --mode empty --results $R/empty --task-ids $GOLD_OK; fi
$PY - $R <<'PYEOF'
import json, sys
from pathlib import Path
R = Path(sys.argv[1])
rd = lambda p: {r["instance_id"]: r for r in map(json.loads, open(p))} if p.exists() else {}
g, e = rd(R / "gold/task_results.jsonl"), rd(R / "empty/task_results.jsonl")
rows = []
for t in sorted(g):
    gp, ep = bool(g[t].get("resolved")), (bool(e[t].get("resolved")) if t in e else None)
    verdict = "VALID" if gp and ep is False else "gold fails" if not gp else "empty passes"
    rows.append({"instance_id": t, "gold_resolved": gp, "empty_resolved": ep, "verdict": verdict})
    print(f"{t:16s} gold={gp!s:5s} empty={ep!s:5s} -> {verdict}")
(R / "validity_src_shim.json").write_text(json.dumps(rows, indent=1))
print(f"VALID with the src/ shim: {sum(r['verdict'] == 'VALID' for r in rows)}/{len(rows)}")
PYEOF
log "ALL DONE"

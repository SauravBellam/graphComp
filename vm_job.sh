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

# ------------------------------------------------------------------ JOB (Oct 15): external task import (CPU only; the GPU may be serving)
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6   # leave CPUs for the running forced-rate search
C=~/gc/data/comp
X=~/gc/external; O=$X/graphcomp-external; mkdir -p $X
E=$G/scripts/planB/external
export HF_HUB_OFFLINE=0

log "STEP 1/5: graph builder vs the competition graphs, and the embedding smoothing depth (Spearman vs competition vectors)"
$PY -I $E/code_graph.py --selftest
$PY -W ignore $E/validate_graph.py $C 1 --embed Qwen/Qwen3-Embedding-0.6B --hops 0,1,2,3 | tee $X/validate_embed.txt
HOPS=$($PY - $X/validate_embed.txt <<'PYEOF'
import re, sys, statistics
s = {}
for l in open(sys.argv[1]):
    for h, v in re.findall(r"hops (\d+): spearman (-?[\d.]+)", l):
        s.setdefault(int(h), []).append(float(v))
best = max(s, key=lambda h: statistics.mean(s[h]))
print(best)
print({h: round(statistics.mean(v), 3) for h, v in sorted(s.items())}, file=sys.stderr)
PYEOF
)
log "STEP 1/5 done: smoothing hops = $HOPS"

log "STEP 2/5: import (read our repos' rows, drop competition PRs and duplicates, snapshots, graphs, embeddings)"
$PY -W ignore $E/import_external.py --out $O --comp-tasks $C/tasks.jsonl --hops $HOPS
ln -sfn $C/wheels $O/wheels
N=$(wc -l < $O/tasks.jsonl); log "STEP 2/5 done: $N tasks built"

log "STEP 3/5: gold-patch pass (src/ shim + pytest-httpbin on PYTHONPATH)"
export SWE_EXTRA_PYTHONPATH=~/gc/testdeps_httpbin
IDS=$($PY -c "import json,sys; print(' '.join(json.loads(l)['instance_id'] for l in open(sys.argv[1]) if l.strip()))" $O/tasks.jsonl)
RA="$PY $G/scripts/planB/run_arm.py --tasks $O/tasks.jsonl --snapshots $O/snapshots --submission $C/sample_submission --sandbox subprocess --timeout-seconds 300 --concurrency 4"
rm -rf $X/gold $X/empty
$RA --mode gold --results $X/gold --task-ids $IDS
GOLD_OK=$($PY -c "import json,sys; print(' '.join(r['instance_id'] for r in map(json.loads, open(sys.argv[1])) if r.get('resolved')))" $X/gold/task_results.jsonl)
log "STEP 3/5 done: gold resolves $(echo $GOLD_OK | wc -w)/$N"

log "STEP 4/5: empty-patch pass on the gold-resolving tasks"
if [ -n "$GOLD_OK" ]; then $RA --mode empty --results $X/empty --task-ids $GOLD_OK; fi
$PY $E/import_external.py --out $O --comp-tasks $C/tasks.jsonl --record-validity $X/gold $X/empty

log "STEP 5/5: package the Kaggle dataset (wheels link excluded)"
tar czf $X/graphcomp-external.tgz -C $X --exclude=graphcomp-external/wheels graphcomp-external
ls -la $X/graphcomp-external.tgz; du -sh $O/snapshots $O/graphs $O/embeddings
log "ALL DONE"

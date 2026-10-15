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

# ------------------------------------------------------------------ JOB (Oct 15): synthetic procedural-bug SWE tasks
# SWE-smith-style bugs (2504.21798) at the newest competition snapshot per repo, in the external-task layout, so notebook 20
# picks them up and the 31B (the policy being trained) is the pass-rate filter (FrogNano 2609.07925). CPU until the issues step.
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4      # the GPU belongs to the running E2B search
export SWE_EXTRA_PYTHONPATH=~/gc/testdeps_httpbin
C=~/gc/data/comp; S=~/gc/synthetic; O=$S/graphcomp-synthetic; T=~/gc/synth_tools
Y=$G/scripts/planB/synthetic/make_synthetic.py
mkdir -p $S
FREE=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); log "disk free: ${FREE} GB"
[ "$FREE" -ge 5 ] || { log "need >= 5 GB free"; exit 1; }

log "STEP 1/6: selftest + coverage tool (pip --target, not in the harness venv)"
$PY -I $Y --selftest
[ -d $T/coverage ] || $PY -m pip install -q --target $T "coverage==7.*"

log "STEP 2/6: build (baseline under coverage, mutate covered lines, keep mutations that fail covering tests)"
rm -rf $O $S/work $S/gold $S/empty
$PY $Y build --comp $C --out $O --work $S/work --tools $T --workers 6
ln -sfn $C/wheels $O/wheels
N=$(wc -l < $O/tasks.jsonl); log "STEP 2/6 done: $N tasks; snapshots $(du -sh $O/snapshots | cut -f1)"

# concurrency 1: httpx's test server binds a fixed port (two tasks at once collide), and scoring is sequential anyway
RA="$PY $G/scripts/planB/run_arm.py --tasks $O/tasks.jsonl --snapshots $O/snapshots --submission $C/sample_submission --sandbox subprocess --timeout-seconds 300 --concurrency 1"
log "STEP 3/6: gold-patch pass"
IDS=$($PY -c "import json,sys; print(' '.join(json.loads(l)['instance_id'] for l in open(sys.argv[1]) if l.strip()))" $O/tasks.jsonl)
$RA --mode gold --results $S/gold --task-ids $IDS
GOLD_OK=$($PY -c "import json,sys; print(' '.join(r['instance_id'] for r in map(json.loads, open(sys.argv[1])) if r.get('resolved')))" $S/gold/task_results.jsonl)
log "STEP 3/6 done: gold resolves $(echo $GOLD_OK | wc -w)/$N"
log "STEP 4/6: empty-patch pass on the gold-resolving tasks"
[ -n "$GOLD_OK" ] || { log "no task resolves with its gold patch"; exit 1; }
$RA --mode empty --results $S/empty --task-ids $GOLD_OK
$PY $G/scripts/planB/external/import_external.py --out $O --comp-tasks $C/tasks.jsonl --record-validity $S/gold $S/empty

log "STEP 5/6: wait for the GPU (no fc_search / dev_runs process and < 1500 MiB used for 5 min; max 12 h), then write issues with E4B"
quiet=0; waited=0
while [ $quiet -lt 5 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if ! pgrep -f "fc_search.py|dev_runs.py" >/dev/null && [ "$used" -lt 1500 ]; then quiet=$((quiet + 1)); else quiet=0; fi
  [ $((waited % 30)) -eq 0 ] && log "  GPU wait ${waited} min: ${used} MiB used, search running: $(pgrep -f 'fc_search.py|dev_runs.py' >/dev/null && echo yes || echo no)"
  [ $waited -lt 720 ] || { log "GPU not free after 12 h"; exit 1; }
  sleep 60; waited=$((waited + 1))
done
unset CUDA_VISIBLE_DEVICES; export HF_HUB_OFFLINE=1
$PY $Y issues --out $O --model ~/gc/data/model_e4b

log "STEP 6/6: summary + package the Kaggle dataset (wheels link excluded)"
$PY - $O <<'PYEOF'
import json, sys, collections
o = sys.argv[1]
meta = {m["instance_id"]: m for m in map(json.loads, open(f"{o}/synthetic_meta.jsonl"))}
g = json.load(open(f"{o}/graphcomp_external.json"))
print(f"built {len(g['built'])}, valid with an accepted issue {len(g['valid'])}, issue rejected {len(g.get('issue_rejected', []))}")
for key in ("op", "source_task"):
    b, v = collections.Counter(meta[t][key] for t in g["built"]), collections.Counter(meta[t][key] for t in g["valid"])
    print(f"by {key}: " + ", ".join(f"{k} {v[k]}/{b[k]}" for k in sorted(b)))
PYEOF
tar czf $S/graphcomp-synthetic.tgz -C $S --exclude=graphcomp-synthetic/wheels graphcomp-synthetic
ls -la $S/graphcomp-synthetic.tgz
log "ALL DONE"

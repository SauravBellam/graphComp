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

# ------------------------------------------------------------------ JOB (Oct 15, v2): synthetic SWE tasks at scale
# SWE-smith (2504.21798) strategies at the newest competition snapshot per repo: single procedural bugs, combined bugs
# (same file / same module), PR mirrors; in the external-task layout, so notebook 20 picks them up and the 31B (the policy
# being trained) is the pass-rate filter (FrogNano 2609.07925). CPU until the issues step (the GPU runs the E2B search).
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SWE_EXTRA_PYTHONPATH=~/gc/testdeps_httpbin
C=~/gc/data/comp; S=~/gc/synthetic; O=$S/graphcomp-synthetic; T=~/gc/synth_tools
SY=$G/scripts/planB/synthetic; Y=$SY/make_synthetic.py
mkdir -p $S
FREE=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); log "disk free: ${FREE} GB"
[ "$FREE" -ge 20 ] || { log "need >= 20 GB free (snapshots + the 12B issue writer)"; exit 1; }

log "STEP 1/6: selftest + coverage tool (pip --target, not in the harness venv)"
$PY -I $Y --selftest
[ -d $T/coverage ] || $PY -m pip install -q --target $T "coverage==7.*"

log "STEP 2/6: build (baseline under coverage -> single bugs -> combined bugs -> PR mirrors)"
rm -rf $O $S/work $S/gold* $S/empty*
$PY $Y build --comp $C --out $O --work $S/work --tools $T --guard $SY/guard_tasks.json \
  --exclude-tasks ~/gc/external/graphcomp-external/tasks.jsonl --workers 8
ln -sfn $C/wheels $O/wheels
N=$(wc -l < $O/tasks.jsonl); log "STEP 2/6 done: $N tasks; snapshots $(du -sh $O/snapshots | cut -f1); disk free $(df -h ~ | tail -1 | awk '{print $4}')"

# httpx's test server binds a fixed port: its tasks run one at a time; the other repos 6 at a time
RA="$PY $G/scripts/planB/run_arm.py --tasks $O/tasks.jsonl --snapshots $O/snapshots --submission $C/sample_submission --sandbox subprocess --timeout-seconds 300"
ids() { $PY -c "import json,sys; print(' '.join(i for i in (json.loads(l)['instance_id'] for l in open(sys.argv[1]) if l.strip()) if (i.startswith('httpx_')) == (sys.argv[2] == 'httpx')))" $O/tasks.jsonl $1; }
resolved() { cat "$@" 2>/dev/null | $PY -c "import json,sys; print(' '.join(r['instance_id'] for r in map(json.loads, sys.stdin) if r.get('resolved')))"; }
log "STEP 3/6: gold-patch pass"
$RA --mode gold --concurrency 6 --results $S/gold_main --task-ids $(ids main)
H=$(ids httpx); [ -z "$H" ] || $RA --mode gold --concurrency 1 --results $S/gold_httpx --task-ids $H
GOLD_OK=$(resolved $S/gold_main/task_results.jsonl $S/gold_httpx/task_results.jsonl)
log "STEP 3/6 done: gold resolves $(echo $GOLD_OK | wc -w)/$N"
[ -n "$GOLD_OK" ] || { log "no task resolves with its gold patch"; exit 1; }
log "STEP 4/6: empty-patch pass on the gold-resolving tasks"
GM=$(echo $GOLD_OK | tr ' ' '\n' | grep -v '^httpx_' || true); GH=$(echo $GOLD_OK | tr ' ' '\n' | grep '^httpx_' || true)
[ -z "$GM" ] || $RA --mode empty --concurrency 6 --results $S/empty_main --task-ids $GM
[ -z "$GH" ] || $RA --mode empty --concurrency 1 --results $S/empty_httpx --task-ids $GH
mkdir -p $S/gold $S/empty
cat $S/gold_*/task_results.jsonl > $S/gold/task_results.jsonl; cat $S/empty_*/task_results.jsonl > $S/empty/task_results.jsonl 2>/dev/null || true
$PY $G/scripts/planB/external/import_external.py --out $O --comp-tasks $C/tasks.jsonl --record-validity $S/gold $S/empty

log "STEP 5/6: issue writer: download the 12B (fallback E4B), wait for the GPU (no fc_search / dev_runs and < 1500 MiB for 5 min; max 12 h)"
M12=$(HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/gemma-4-12B-it-qat-w4a16-ct'))" 2>$S/dl12b.err | tail -1) || M12=""
[ -n "$M12" ] && log "12B at $M12" || { log "12B download failed ($(tail -1 $S/dl12b.err)); using E4B"; }
quiet=0; waited=0
while [ $quiet -lt 5 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if ! pgrep -f "fc_search.py|dev_runs.py" >/dev/null && [ "$used" -lt 1500 ]; then quiet=$((quiet + 1)); else quiet=0; fi
  [ $((waited % 30)) -eq 0 ] && log "  GPU wait ${waited} min: ${used} MiB used, search running: $(pgrep -f 'fc_search.py|dev_runs.py' >/dev/null && echo yes || echo no)"
  [ $waited -lt 720 ] || { log "GPU not free after 12 h"; exit 1; }
  sleep 60; waited=$((waited + 1))
done
unset CUDA_VISIBLE_DEVICES; export HF_HUB_OFFLINE=1
cp $O/tasks.jsonl $S/tasks_before_issues.jsonl; cp $O/graphcomp_external.json $S/meta_before_issues.json
if [ -n "$M12" ] && $PY $Y issues --out $O --model $M12; then log "issues written by the 12B"
else
  log "12B issue step failed; retrying with E4B"
  cp $S/tasks_before_issues.jsonl $O/tasks.jsonl; cp $S/meta_before_issues.json $O/graphcomp_external.json
  $PY $Y issues --out $O --model ~/gc/data/model_e4b
fi

log "STEP 6/6: summary + package the Kaggle dataset (wheels link excluded)"
$PY - $O <<'PYEOF'
import json, sys, collections
o = sys.argv[1]
meta = {m["instance_id"]: m for m in map(json.loads, open(f"{o}/synthetic_meta.jsonl"))}
g = json.load(open(f"{o}/graphcomp_external.json"))
print(f"built {len(g['built'])}, valid with an accepted issue {len(g['valid'])}, issue rejected {len(g.get('issue_rejected', []))}, issue model {g.get('issue_model')}")
for key in ("op", "source_task", "n_bugs"):
    b, v = collections.Counter(meta[t][key] for t in g["built"]), collections.Counter(meta[t][key] for t in g["valid"])
    print(f"by {key}: " + ", ".join(f"{k} {v[k]}/{b[k]}" for k in sorted(b)))
PYEOF
tar czf $S/graphcomp-synthetic.tgz -C $S --exclude=graphcomp-synthetic/wheels graphcomp-synthetic
ls -la $S/graphcomp-synthetic.tgz
log "ALL DONE"

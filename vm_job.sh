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

# ------------------------------------------------------------------ JOB (Oct 16): external import validity v2 (CPU only)
# Missing test-only modules (first: trustme for httpx) go into a --target dir on the sandbox PYTHONPATH, never installed in
# the venv, as pytest-httpbin is. Up to 3 rounds: gold pass -> modules named in "No module named" -> add them -> again.
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
C=~/gc/data/comp; X=~/gc/external; O=$X/graphcomp-external; E=$G/scripts/planB/external
TD=~/gc/testdeps_extra; mkdir -p $TD
export SWE_EXTRA_PYTHONPATH=~/gc/testdeps_httpbin:$TD
IDS=$($PY -c "import json,sys; print(' '.join(json.loads(l)['instance_id'] for l in open(sys.argv[1]) if l.strip()))" $O/tasks.jsonl)
RA="$PY $G/scripts/planB/run_arm.py --tasks $O/tasks.jsonl --snapshots $O/snapshots --submission $C/sample_submission --sandbox subprocess --timeout-seconds 300 --concurrency 4"
WANT="trustme"
for round in 1 2 3; do
  log "round $round: adding test deps: $WANT"
  for m in $WANT; do
    case "$m" in _*|backports) continue;; esac            # private / Python-2 compat names, not packages
    PYTHONPATH=$SWE_EXTRA_PYTHONPATH $PY -c "import $m" 2>/dev/null || $PY -m pip install -q --target $TD "$m" || log "pip: no package $m"
  done
  ls $TD
  rm -rf $X/gold
  $RA --mode gold --results $X/gold --task-ids $IDS
  log "round $round: gold resolves $($PY -c "import json,sys; print(sum(bool(json.loads(l).get('resolved')) for l in open(sys.argv[1])))" $X/gold/task_results.jsonl)/$(echo $IDS | wc -w)"
  WANT=$(grep -rhoE "No module named '[A-Za-z0-9_]+" $X/gold/test_outputs $X/gold/logs 2>/dev/null | sed "s/No module named '//" | sort -u | tr '\n' ' ' || true)
  WANT=$(echo $WANT | tr ' ' '\n' | grep -vE '^(_|backports$)' | tr '\n' ' ' || true)
  [ -z "${WANT// /}" ] && break
done
GOLD_OK=$($PY -c "import json,sys; print(' '.join(r['instance_id'] for r in map(json.loads, open(sys.argv[1])) if r.get('resolved')))" $X/gold/task_results.jsonl)
rm -rf $X/empty; mkdir -p $X/empty; : > $X/empty/task_results.jsonl
if [ -n "$GOLD_OK" ]; then rm -rf $X/empty; $RA --mode empty --results $X/empty --task-ids $GOLD_OK; fi
$PY $E/import_external.py --out $O --comp-tasks $C/tasks.jsonl --record-validity $X/gold $X/empty
log "test deps added: $(ls $TD | tr '\n' ' ')"
tar czf $X/testdeps_extra.tgz -C ~/gc testdeps_extra
tar czf $X/graphcomp-external.tgz -C $X --exclude=graphcomp-external/wheels graphcomp-external
ls -la $X/graphcomp-external.tgz $X/testdeps_extra.tgz
log "remaining gold failures:"
X=$X O=$O
$PY - $X/gold $O/tasks.jsonl <<'PYEOF'
import json, sys, collections
from pathlib import Path
gold, tasks = Path(sys.argv[1]), {}
for l in open(sys.argv[2]):
    if l.strip():
        t = json.loads(l); tasks[t["instance_id"]] = t
rows = [json.loads(l) for l in open(gold / "task_results.jsonl") if l.strip()]
print("fields:", sorted(rows[0]), "| task fields:", sorted(next(iter(tasks.values()))))
src = lambda i: tasks.get(i, {}).get("source") or tasks.get(i, {}).get("dataset") or "?"
def tail(i, n=25):
    for p in list((gold / "test_outputs").glob(f"*{i}*")) + list((gold / "logs").glob(f"*{i}*")):
        if p.is_file():
            return f"--- {p.name}\n" + "\n".join(p.read_text(errors="replace").splitlines()[-n:])
    return "(no test output file)"
c = collections.Counter((i.split("_")[0], src(r["instance_id"]), r.get("resolved") in (True, "True"), str(r.get("test_exit_code")), str(r.get("error"))[:80])
                        for r in rows for i in [r["instance_id"]])
for k, v in sorted(c.items(), key=lambda x: -x[1]):
    print(v, k)
seen = collections.Counter()
for r in rows:
    i = r["instance_id"]; key = (i.split("_")[0], str(r.get("test_exit_code")))
    if r.get("resolved") in (True, "True") or seen[key] >= 2:
        continue
    seen[key] += 1
    t = tasks.get(i, {})
    print(f"\n===== {i} exit {r.get('test_exit_code')} error {str(r.get('error'))[:200]}")
    print("FAIL_TO_PASS:", str(t.get("FAIL_TO_PASS") or t.get("fail_to_pass"))[:300])
    print("test cmd:", str(t.get("test_cmd") or t.get("test_command") or t.get("eval_script"))[:300])
    print(tail(i))
PYEOF
log "ALL DONE"

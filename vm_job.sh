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

# ------------------------------------------------------------------ JOB (Oct 16): why the external import's gold pass resolves 8/160 (read only)
X=~/gc/external; O=$X/graphcomp-external
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

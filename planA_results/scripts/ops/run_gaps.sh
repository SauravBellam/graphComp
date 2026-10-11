#!/bin/bash
# Closes the gaps left after rerun_fixes.sh (see GAPS_2026-10-11.md):
#   1. rich_4006: rerun with per-test checkpoints + hung-test deselection (only step that runs tests)
#   2. A2 analysis -> a2_v3: pre_skipped, truly-added fail-to-pass set, post-trace edges where the pre
#      trace is missing/empty, recovered rich_4006 trace
#   3. A5 -> a5_v3: all-task denominators + mode (e) BM25 over the real snapshot source
#   4. A8 -> a8: every unconfirmed / missed / captured edge checked against the caller's source,
#      plus a stratified sample for hand review
#   5. A6 -> a6_v3: contains-edge fix and the new A2 columns in task_profile.csv
#
# Usage (vm-a2 layout):  bash ops/run_gaps.sh ~/graphComp/archive ~/results ~/cache_v2
# Reads <results>/a1_v2 (from rerun_fixes.sh) and <results>/a2_runs; Oct 10 / v2 outputs stay untouched.
set -euo pipefail
DATA=${1:?archive dir}
RES=${2:?results dir (must contain a2_runs/ and a1_v2/)}
CACHE=${3:-$RES/../cache_v2}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
cd "$HERE"
echo "START $(date)"

# 1. rich_4006: keep the earlier attempt, rerun with checkpointing + deselection of the hung test
if [ -d "$RES/a2_runs/rich_4006" ] && [ ! -d "$RES/a2_runs_killed2/rich_4006" ]; then
  mkdir -p "$RES/a2_runs_killed2" && mv "$RES/a2_runs/rich_4006" "$RES/a2_runs_killed2/"
fi
$PY planA/a2/run_task.py --data "$DATA" --out "$RES/a2_runs" --cache "$RES/../cache" --backend venv --jobs 1 --keep \
    --timeout 300 --test-timeout 60 --deselect-hung --task-ids rich_4006 > "$RES/a2_rich_4006_rerun.log" 2>&1
echo "RICH_4006_EXIT $? $(date)"

# 2. A2 analysis
$PY planA/a2/analyze.py --data "$DATA" --runs "$RES/a2_runs" --a1 "$RES/a1_v2" --out "$RES/a2_v3" \
    --edges-from-post-when-pre-missing > "$RES/a2_v3_analyze.log" 2>&1
echo "A2_ANALYZE_EXIT $? $(date)"

# 3. A5
$PY planA/a5_retrieval.py --data "$DATA" --a1 "$RES/a1_v2" --out "$RES/a5_v3" --cache "$CACHE" > "$RES/a5_v3.log" 2>&1
echo "A5_EXIT $? $(date)"

# 4. A8 (edges from the A2 analysis just written)
$PY planA/a8_edge_audit.py --data "$DATA" --a2 "$RES/a2_v3" --out "$RES/a8" --cache "$CACHE" > "$RES/a8.log" 2>&1
echo "A8_EXIT $? $(date)"

# 5. A6
$PY planA/a6_release.py --data "$DATA" --a1 "$RES/a1_v2" --a2-runs "$RES/a2_runs" --a2-analysis "$RES/a2_v3" \
    --out "$RES/a6_v3" --cache "$CACHE" > "$RES/a6_v3.log" 2>&1
echo "A6_EXIT $? $(date)"

( cd "$HERE" && find planA tests ops -type f \( -name '*.py' -o -name '*.sh' \) -print0 | sort -z | xargs -0 sha256sum ) > "$RES/scripts_sha256_v3.txt"
echo "DONE $(date)"

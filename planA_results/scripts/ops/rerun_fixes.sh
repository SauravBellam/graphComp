#!/bin/bash
# Re-analysis after the Oct 11 fixes. No new A2 test runs are needed: every step below reads the
# archive, the existing A2 traces (results/a2_runs) and recomputes.
#
#   A1  rerun  : presence under the provided graph's id scheme (nested defs flattened), class-dropped counts
#   A7  new    : provided-node <-> source reconciliation (explains requests "absent" and provided_only)
#   A2  analyze: nested mapping, entry-point reachability, fail-to-pass statuses, strict / harness sets
#   A6  rerun  : v2 graphs with nested defs, no dangling test edges, new task-profile columns
#   A5  rerun  : hit rates change only through A1's gold node set (nested gold symbols now resolvable)
#   A8  new    : per-task metrics (entry distance, answerability, discriminating tests) and reference scores
#
# Usage (on the VM that holds the archive and results/a2_runs; vm-a2 in the Oct 10 layout):
#   bash planA/../ops/rerun_fixes.sh ~/graphComp/archive ~/results ~/cache 8
# Outputs go to <results>/<step>_v2 so the Oct 10 files stay untouched for comparison.
set -euo pipefail
DATA=${1:?archive dir}
RES=${2:?results dir (must contain a2_runs/)}
CACHE=${3:-$RES/../cache}
SHARDS=${4:-8}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
cd "$HERE"
echo "START $(date)"

# A1: shards, then merge
for i in $(seq 0 $((SHARDS-1))); do
  $PY planA/a1_census.py --data "$DATA" --out "$RES/a1_v2" --cache "$CACHE" --shard "$i/$SHARDS" > "$RES/a1_v2_shard$i.log" 2>&1 &
done
wait
$PY planA/a1_census.py merge "$RES/a1_v2"
echo "A1_EXIT $? $(date)"

# A7: id reconciliation (one pass per distinct graph; shard the same way)
for i in $(seq 0 $((SHARDS-1))); do
  $PY planA/a7_id_audit.py --data "$DATA" --out "$RES/a7" --cache "$CACHE" --shard "$i/$SHARDS" > "$RES/a7_shard$i.log" 2>&1 &
done
wait
$PY planA/a7_id_audit.py merge "$RES/a7"
echo "A7_EXIT $? $(date)"

# A2 analysis on the existing traces (reads the new A1 gold symbols)
$PY planA/a2/analyze.py --data "$DATA" --runs "$RES/a2_runs" --a1 "$RES/a1_v2" --out "$RES/a2_v2" > "$RES/a2_v2_analyze.log" 2>&1
echo "A2_ANALYZE_EXIT $? $(date)"

# A5 (depends on A1 only)
$PY planA/a5_retrieval.py --data "$DATA" --a1 "$RES/a1_v2" --out "$RES/a5_v2" > "$RES/a5_v2.log" 2>&1
echo "A5_EXIT $? $(date)"

# A6 (graphs + task profile)
$PY planA/a6_release.py --data "$DATA" --a1 "$RES/a1_v2" --a2-runs "$RES/a2_runs" --a2-analysis "$RES/a2_v2" \
    --out "$RES/a6_v2" --cache "$CACHE" > "$RES/a6_v2.log" 2>&1
echo "A6_EXIT $? $(date)"

# A8: entry distance, tool answerability, test discriminativeness + gold/empty-patch reference scores
$PY planA/a8_metrics.py --data "$DATA" --a1 "$RES/a1_v2" --a2-analysis "$RES/a2_v2" --runs "$RES/a2_runs" --out "$RES/a8" > "$RES/a8.log" 2>&1
echo "A8_EXIT $? $(date)"

# provenance
( cd "$HERE" && find planA tests ops -type f \( -name '*.py' -o -name '*.sh' \) -print0 | sort -z | xargs -0 sha256sum ) > "$RES/scripts_sha256_v2.txt"
echo "DONE $(date)"

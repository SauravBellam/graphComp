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

# ------------------------------------------------------------------ JOB (Oct 15): SFT mix = converted external + our RFT trajectories, deduplicated (CPU)
# Runs after the conversion (exp ca1b26ab) finished. Dev and holdout tasks are kept (user, Oct 15). Output = Kaggle dataset
# graphcomp-sft-mix for notebook 21 (make_nb21.py).
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
L=$G/scripts/planB/lora
CONV=~/gc/external_trajs/graphcomp-external-trajs
O=~/gc/sft_mix/graphcomp-sft-mix; mkdir -p $(dirname $O)
[ -f $CONV/manifest.json ] || { log "conversion output incomplete: $CONV/manifest.json missing"; exit 1; }
RFT=$(ls $G/planB_results/lora_data/*/train.jsonl 2>/dev/null | tr '\n' ' ' || true)
IMP=~/gc/external/graphcomp-external/tasks.jsonl
log "inputs: converted $CONV | our RFT: ${RFT:-none} | imported external tasks: $([ -f $IMP ] && wc -l < $IMP || echo none)"

log "STEP 1/2: merge + exact / near-duplicate removal + fit estimate + imported-task overlap"
$PY $L/build_sft_mix.py --out $O --external $CONV ${RFT:+--rft $RFT} $([ -f $IMP ] && echo --external-tasks $IMP) \
    --slices $G/scripts/planB/official_slices --near-dup 0.9 --fit-tokens 32768

log "STEP 2/2: package"
cp $L/build_sft_mix.py $CONV/convert_trajs.py $O/ 2>/dev/null || cp $L/build_sft_mix.py $O/
du -sh $O/*; df -h ~ | tail -1
log "ALL DONE"

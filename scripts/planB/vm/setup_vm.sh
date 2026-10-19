#!/usr/bin/env bash
# One-time setup of a single-L4 VM for the Plan B dev-slice runs (scripts/planB/vm/dev_runs.py).
# Run as your normal user (it uses sudo only for apt), inside tmux, from the graphComp folder:
#     bash scripts/planB/vm/setup_vm.sh 2>&1 | tee -a ~/gc/setup.log
# Needs: one NVIDIA L4, driver >= 570 (the wheelhouse's vLLM 0.19.1 is built for CUDA 12.8), about 80 GB free disk,
# and your Kaggle API token at ~/.kaggle/kaggle.json (chmod 600). Safe to re-run: finished steps are skipped.
set -euo pipefail
GC=${GC:-$HOME/gc}
mkdir -p "$GC/data" "$GC/logs"
T0=$(date +%s)
log() { echo "[$(date -u +%H:%M:%S) UTC | +$(( $(date +%s) - T0 ))s] $*" | tee -a "$GC/logs/setup.log"; }

log "STEP 1/7 checks: GPU, driver, disk, Kaggle token, oracle pack"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$GC/logs/setup.log"
NGPU=$(nvidia-smi -L | grep -c '^GPU ' || true)
[ "$NGPU" -eq 1 ] || log "WARNING: $NGPU GPUs visible; this script is written for one L4 (tp=1)"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRV" -lt 570 ]; then log "STOP: driver $DRV < 570. Install a 570+ driver (CUDA 12.8) and re-run."; exit 1; fi
df -h "$GC" | tee -a "$GC/logs/setup.log"
AVAIL_GB=$(df -BG --output=avail "$GC" | tail -1 | tr -dc 0-9)
[ "$AVAIL_GB" -ge 80 ] || log "WARNING: only ${AVAIL_GB} GB free; the model is about 19 GB, the data and wheels about 15-30 GB"
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_API_TOKEN:-}" ] && [ ! -f "$HOME/.kaggle/access_token" ]; then
  log "STOP: no Kaggle token. export KAGGLE_API_TOKEN=... (kaggle.com > Settings > API > Generate New Token), or put kaggle.json in ~/.kaggle, then re-run."
  exit 1
fi
[ -f "$HOME/.kaggle/kaggle.json" ] && chmod 600 "$HOME/.kaggle/kaggle.json"
PACK="$(cd "$(dirname "$0")/../../.." && pwd)/planB_results/kaggle_upload/graphcomp-planb/oracle_pack.json.gz"
if [ ! -f "$PACK" ]; then
  log "STOP: oracle pack missing at $PACK. Re-copy gc_vm.tgz (it includes planB_results/kaggle_upload/graphcomp-planb) and re-run."
  exit 1
fi
log "oracle pack present: $PACK"

log "STEP 2/7 system packages (git, patch, build tools, tmux, unzip) and uv"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git patch build-essential tmux pigz unzip curl >/dev/null
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version | tee -a "$GC/logs/setup.log"

log "STEP 3/7 Python 3.12 venv at $GC/venv (the competition wheelhouse is cp312)"
uv python install 3.12
[ -x "$GC/venv/bin/python" ] || uv venv --python 3.12 --seed "$GC/venv"
PY="$GC/venv/bin/python"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q "kaggle==2.2.4"
KG="$GC/venv/bin/kaggle"
"$PY" --version | tee -a "$GC/logs/setup.log"

log "STEP 4/7 downloads (competition data, wheelhouse, model): resumable, skipped when present"
if [ -z "$(find "$GC/data/comp" -name tasks.jsonl 2>/dev/null | head -1)" ]; then
  log "  competition data ..."
  "$KG" competitions download -c gemma-4-developer-agent -p "$GC/data/comp_zip"
  mkdir -p "$GC/data/comp" && unzip -q -o "$GC/data/comp_zip/"*.zip -d "$GC/data/comp" && rm -rf "$GC/data/comp_zip"
fi
TASKS=$(find "$GC/data/comp" -name tasks.jsonl | head -1); COMP=$(dirname "$TASKS")
log "  competition data at $COMP: $(wc -l < "$TASKS") tasks; $(ls "$COMP" | tr '\n' ' ')"
# STACK=pypi (default since Oct 13): no Kaggle wheelhouse (its download is refused, and it was rebuilt on Oct 13). Harness
# wheels from graphComp/harness_wheels (= B2_s1's build), everything else pinned from PyPI (scripts/planB/vm/stack_pins.txt).
# STACK=wheelhouse: the old behaviour (download metric/gemma-4-developer-agent-wheelhouse).
STACK=${STACK:-pypi}
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
if [ "$STACK" = wheelhouse ]; then
  if [ -z "$(find "$GC/data/wheelhouse" -name 'vllm-*.whl' 2>/dev/null | head -1)" ]; then
    log "  wheelhouse ..."
    "$KG" datasets download metric/gemma-4-developer-agent-wheelhouse -p "$GC/data/wheelhouse" --unzip
  fi
  WH=$(dirname "$(find "$GC/data/wheelhouse" -name 'vllm-*.whl' | head -1)")
  log "  wheelhouse at $WH: $(ls "$WH"/*.whl | wc -l) wheels"
else
  HW="$REPO/harness_wheels"
  log "  STACK=pypi: harness wheels from $HW, the rest from PyPI pins ($REPO/scripts/planB/vm/stack_pins.txt)"
  "$PY" - "$HW" <<'PYEOF' | tee -a "$GC/logs/setup.log"
import hashlib, sys
from pathlib import Path
want = {"swegemma-0.2.11-py3-none-any.whl": "63f054a3d42eb6c5", "adk_submission-0.2.13-py3-none-any.whl": "69642a1314515104",
        "adk_eval_core-0.1.0-py3-none-any.whl": "ca261146828a3d8a"}     # sha256 prefixes of B2_s1's (Oct 12) build
for n, h in want.items():
    f = Path(sys.argv[1]) / n
    got = hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else "missing"
    print(f"  {n}: {got}", "ok (B2_s1 build)" if got == h else "!!! NOT B2_s1's build")
    if got != h:
        sys.exit(1)
PYEOF
fi
# MODEL_VARIANT=31b (default, the Plan B condition) or 12b / e4b / e2b (size sweep, Oct 13). MODEL_VERSION = the Kaggle model
# version: 2 for 31B; for the others check the version number on kaggle.com/models/google/gemma-4 and pass it.
MODEL_VARIANT=${MODEL_VARIANT:-31b}
case "$MODEL_VARIANT" in
  31b) SLUG=gemma-4-31b-it-qat-w4a16-ct; MDIR="$GC/data/model";            MODEL_VERSION=${MODEL_VERSION:-2} ;;
  12b) SLUG=gemma-4-12b-it-qat-w4a16-ct; MDIR="$GC/data/model_12b" ;;
  e4b) SLUG=gemma-4-e4b-it-qat-w4a16-ct; MDIR="$GC/data/model_e4b" ;;
  e2b) SLUG=gemma-4-e2b-it-qat-w4a16-ct; MDIR="$GC/data/model_e2b" ;;
  *) log "STOP: MODEL_VARIANT must be 31b, 12b, e4b or e2b"; exit 1 ;;
esac
if [ -z "$(find "$MDIR" -name config.json 2>/dev/null | head -1)" ]; then
  if [ -z "${MODEL_VERSION:-}" ]; then
    log "STOP: set MODEL_VERSION for $SLUG (the version number on its Kaggle model page), e.g. MODEL_VERSION=1"; exit 1
  fi
  log "  model $SLUG v$MODEL_VERSION ..."
  if ! "$KG" models instances versions download "google/gemma-4/other/$SLUG/$MODEL_VERSION" -p "$MDIR" --untar; then
    # Kaggle refused (as it did for the wheelhouse): the same checkpoint is public on Hugging Face (google/gemma-4-*-it-qat-w4a16-ct)
    HF_ID="google/$(echo "$SLUG" | sed -E 's/gemma-4-(31b|12b|e4b|e2b)/gemma-4-\U\1/')"
    log "  Kaggle download failed; trying Hugging Face $HF_ID (recorded in logs/model_source.txt)"
    "$PY" -m pip install -q "huggingface-hub==1.11.0"
    "$PY" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "$HF_ID" "$MDIR"
    echo "model $SLUG from Hugging Face $HF_ID $(date -u +%FT%TZ)" > "$GC/logs/model_source.txt"
  else
    echo "model $SLUG from Kaggle v$MODEL_VERSION $(date -u +%FT%TZ)" > "$GC/logs/model_source.txt"
  fi
fi
MODEL=$(dirname "$(find "$MDIR" -name config.json | head -1)")
log "  model at $MODEL ($(du -sh "$MODEL" | cut -f1))"

if [ "$STACK" = pypi ]; then
log "STEP 5/7 install the stack: torch 2.10.0 (cu128), harness wheels (B2_s1 build), pinned PyPI packages"
"$PY" -m pip install -q "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3 | tee -a "$GC/logs/setup.log"
# Two layers, as on Kaggle (a base image, then the wheelhouse installed over it without full resolution). The exact B2_s1
# versions have declared conflicts (google-genai 2.11.0 wants pydantic>=2.12.5 but Kaggle ran 2.12.3; starlette 0.52.1 vs
# fastapi 0.136.1 and two vLLM extras), so one resolver pass cannot install them.
# Layer A: harness wheels + vLLM with dependencies resolved (fills in everything the pins do not name).
log "  layer A: harness wheels + vllm==0.19.1, dependencies resolved"
"$PY" -m pip install -q "$HW"/*.whl "vllm==0.19.1" "torch==2.10.0" --extra-index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -5 | tee -a "$GC/logs/setup.log"
# Layer B: every pinned package at B2_s1's exact version, --no-deps (overrides what layer A chose).
log "  layer B: $(grep -c '==' "$REPO/scripts/planB/vm/stack_pins.txt") pins + pydantic 2.12.3 at B2_s1's versions, --no-deps"
"$PY" -m pip install -q --no-deps -r "$REPO/scripts/planB/vm/stack_pins.txt" "pydantic==2.12.3" "pydantic-core==2.41.4" "typing-extensions==4.15.0" --extra-index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -5 | tee -a "$GC/logs/setup.log"
"$PY" -m pip install -q --no-deps --force-reinstall "$HW"/*.whl 2>&1 | tail -3 | tee -a "$GC/logs/setup.log"
# Layer C: packages that layer B's --no-deps left MISSING (not mismatched), e.g. ipython's backcall / decorator / jedi, which
# the Kaggle image has. Installed with resolution; the pinned versions are re-checked in step 7.
for _i in 1 2 3; do
  MISSING=$("$PY" -m pip check 2>/dev/null | grep 'which is not installed' | sed -E 's/.* requires ([^,]+), which is not installed.*/\1/' | sort -u | tr '\n' ' ')
  [ -z "${MISSING// /}" ] && break
  log "  layer C (pass $_i): installing missing dependencies: $MISSING"
  "$PY" -m pip install -q $MISSING 2>&1 | tail -3 | tee -a "$GC/logs/setup.log"
done
"$PY" -m pip check > "$GC/logs/pip_check.txt" 2>&1 || true
log "  pip check (declared conflicts expected, the same as Kaggle's B2_s1 environment): $(wc -l < "$GC/logs/pip_check.txt") lines -> logs/pip_check.txt"
head -5 "$GC/logs/pip_check.txt" | tee -a "$GC/logs/setup.log"
"$PY" -m pip freeze > "$GC/logs/freeze_after_stack.txt"
echo "STACK=pypi $(date -u +%FT%TZ) harness_wheels=B2_s1 build; pins=scripts/planB/vm/stack_pins.txt" > "$GC/logs/stack.txt"
else
log "STEP 5/7 install the competition wheelhouse (as the sample notebook: +cu128 names restored, cutlass skipped)"
rm -rf /tmp/wheelhouse && mkdir -p /tmp/wheelhouse
for w in "$WH"/*.whl; do
  n=$(basename "$w"); case "$n" in *cutlass*|*CUTLASS*) continue;; esac
  if [[ "$n" == *cu128* && "$n" != *+cu128* ]]; then n=${n/cu128/+cu128}; fi
  ln -sf "$w" "/tmp/wheelhouse/$n"
done
# A bare venv lacks the Kaggle image's base packages, so dependencies are resolved (wheelhouse versions win: they are named).
"$PY" -m pip install -q --find-links /tmp/wheelhouse /tmp/wheelhouse/*.whl 2>&1 | tail -5 | tee -a "$GC/logs/setup.log"
"$PY" -m pip freeze > "$GC/logs/freeze_after_wheelhouse.txt"
fi

log "STEP 6/7 test dependencies from the competition's own wheels/ folder (only distributions not already installed)"
TW="$COMP/wheels"
if [ "$STACK" = pypi ]; then
  # Oct 13: skipped, as on Kaggle (the notebooks install only the three pinned test deps below). The folder holds several
  # versions of the same packages (flask 2.2.5 / 2.3.3 / 3.1.3, five sqlmodel versions): they cannot share one environment.
  log "  STACK=pypi: competition wheels/ not installed (as on Kaggle); only the pinned test deps below"
elif [ -d "$TW" ]; then
  "$PY" - "$TW" <<'PYEOF' | tee -a "$GC/logs/setup.log"
import re, subprocess, sys
from pathlib import Path
import importlib.metadata as md
norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()
have = {norm(d.metadata["Name"]) for d in md.distributions()}
wheels = sorted(Path(sys.argv[1]).glob("*.whl"))
new = [w for w in wheels if norm(w.name.split("-")[0]) not in have]
print(f"{len(wheels)} test-dependency wheels, {len(new)} not yet installed:", [w.name for w in new])
if new:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *map(str, new)], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
PYEOF
else
  log "NOTE: $TW not found"
fi
# The competition's wheels/ folder does not contain these (Kaggle, Oct 12). Same pinned versions as the Kaggle notebooks
# (Plan A's fastapi environments: inline-snapshot 0.36.0 and executing 2.3.0 in 67/67; dirty-equals 0.11).
log "  pinned test deps: inline-snapshot==0.36.0 executing==2.3.0 dirty-equals==0.11"
"$PY" -m pip install -q "inline-snapshot==0.36.0" "executing==2.3.0" "dirty-equals==0.11" 2>&1 | tail -3 | tee -a "$GC/logs/setup.log"

log "STEP 7/7 checks: harness and serving stack import, CUDA visible, test deps import; package list saved"
"$PY" - <<'PYEOF' | tee -a "$GC/logs/setup.log"
import importlib, importlib.metadata as md
for m in ["swegemma", "adk_submission", "adk_eval_core", "google.adk", "vllm", "torch", "pytest", "inline_snapshot", "dirty_equals", "executing"]:
    try:
        importlib.import_module(m); print(f"import {m:16s} ok")
    except Exception as e:
        print(f"import {m:16s} FAILED: {type(e).__name__}: {e}")
import torch
print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
for p in ["swegemma", "adk-submission", "adk-eval-core", "google-adk", "vllm", "torch", "pydantic", "fastapi", "starlette",
          "requests", "rich", "pygments", "ipython", "pytest", "inline-snapshot", "dirty-equals", "executing"]:
    try:
        print(f"  {p:16s} {md.version(p)}")
    except Exception:
        print(f"  {p:16s} -")
PYEOF
"$PY" -m pip freeze > "$GC/logs/freeze_setup.txt"
# paths.env keeps the 31B model path (the Plan B default); dev_runs.py --model 12b/e4b/e2b finds its own folder
PATHS_MODEL=$MODEL
[ "$MODEL_VARIANT" = 31b ] || PATHS_MODEL=$(grep -s "^MODEL=" "$GC/paths.env" | cut -d= -f2- || true)
cat > "$GC/paths.env" <<EOF
GC=$GC
PY=$PY
COMP=$COMP
TASKS=$TASKS
MODEL=$PATHS_MODEL
EOF
log "DONE. Paths in $GC/paths.env. Next (inside tmux):"
log "  $PY scripts/planB/vm/dev_runs.py 2>&1 | tee -a $GC/logs/dev_runs.log"

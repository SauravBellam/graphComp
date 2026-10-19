#!/bin/bash
# Runs on the VM: wait for training -> start the B2,LORA eval -> once vLLM is up, probe the adapter. Status in ~/gc/lora_chain.status
cd ~/graphComp
ST=~/gc/lora_chain.status; AD=~/gc/lora/rft_e4b_v1/adapter; LOG=~/gc/dev_runs_rft.log
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a $ST; }
say "waiting for training to save the adapter"
until [ -f $AD/adapter_config.json ]; do
  pgrep -f train_lora.py >/dev/null || { sleep 30; [ -f $AD/adapter_config.json ] || { say "TRAINING EXITED WITHOUT ADAPTER"; tail -5 ~/gc/lora/rft_e4b_v1/train.out >> $ST; exit 1; }; }
  sleep 60
done
while pgrep -f train_lora.py >/dev/null; do sleep 10; done   # let it finish writing and free the GPU
say "adapter saved; starting eval"
start=$(wc -l < $LOG 2>/dev/null || echo 0)
tmux new-window -d -t lora -n eval "cd ~/graphComp && ~/gc/venv/bin/python scripts/planB/vm/dev_runs.py --model e4b --adapter rft_lora=\$HOME/gc/lora/rft_e4b_v1/adapter --arms B2,LORA 2>&1 | tee -a $LOG"
until tail -n +$((start + 1)) $LOG | grep -q "vLLM: STARTED"; do
  tail -n +$((start + 1)) $LOG | grep -q "vLLM: FAILED" && say "vLLM FAILED at one setting (dev_runs may retry the ladder)"
  pgrep -f dev_runs.py >/dev/null || { say "EVAL EXITED BEFORE vLLM STARTED"; tail -n +$((start + 1)) $LOG | tail -8 >> $ST; exit 1; }
  sleep 20
done
say "vLLM up; probing adapter"
sleep 20
~/gc/venv/bin/python /tmp/adapter_probe.py rft_lora > ~/gc/lora/rft_e4b_v1/adapter_probe.txt 2>&1
say "PROBE DONE: $(grep VERDICT ~/gc/lora/rft_e4b_v1/adapter_probe.txt || tail -1 ~/gc/lora/rft_e4b_v1/adapter_probe.txt)"

#!/bin/bash
# [EVAL-WAITER-2026-10-21-TAU] Paper 1 TAU fixed-test-set evaluation, one pass per seed.
# For each tau_ddr_rainnet_seed{0,1,2}: wait until the training run has landed
# (validated results.json; accepts status=done or a clean early-stop partial),
# wait for that seed's GPU to be free, then run rainnet_nowcast_eval.py --model tau
# ONCE, writing logs/eval_<name>.json and logs/eval_<name>.txt.
# Read-only toward training and all queues: never kills, never starts training,
# never touches the HOLD file. Skips seeds whose eval json already exists.
# Self-expires 2026-10-23 12:00 server time. Log: logs/tau_eval_waiter_20261022.log
B=/data/sdukjsp017/zal/MambaEP_paper_repro_20261015
PY=/data/sdukjsp017/zal/MambaEP_reproduction_20261013/.venv/bin/python
RUNS=$B/runs_tau_20261021/runs
LOG=$B/logs/tau_eval_waiter_20261022.log
PIDF=$B/logs/tau_eval_waiter_20261022.pid
DEADLINE=$(date -d "2026-10-23 12:00" +%s)
echo $$ > "$PIDF"
echo "waiter start $(date '+%F %T') pid=$$" >> "$LOG"
for s in 0 1 2; do
  name=tau_ddr_rainnet_seed$s
  out=$B/logs/eval_${name}.json
  if [ -f "$out" ]; then
    echo "$name eval json exists, skip $(date '+%F %T')" >> "$LOG"
    continue
  fi
  status=""; rc=""; gpu=""
  while true; do
    if [ -f "$RUNS/$name/results.json" ]; then
      status=$(python3 -c "import json;print(json.load(open('$RUNS/$name/results.json')).get('status'))" 2>/dev/null)
      rc=$(python3 -c "import json;print(json.load(open('$RUNS/$name/results.json')).get('returncode'))" 2>/dev/null)
      gpu=$(python3 -c "import json;print(json.load(open('$RUNS/$name/results.json')).get('gpu'))" 2>/dev/null)
      break
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
      echo "$name deadline reached, no results.json, abort $(date '+%F %T')" >> "$LOG"
      exit 1
    fi
    sleep 60
  done
  echo "$name landed status=$status rc=$rc gpu=$gpu $(date '+%F %T')" >> "$LOG"
  ckpt=$B/corrected/checkpoints/${name}.pth
  train_log=$B/corrected/logs/${name}.log
  if ! $PY $B/queue_bundle/tau_result_gate.py "$RUNS/$name/results.json" "$train_log" "$ckpt" >> "$LOG" 2>&1; then
    echo "$name result gate rejected run, skip eval $(date '+%F %T')" >> "$LOG"
    continue
  fi
  case "$gpu" in ''|*[!0-9]*) gpu=0;; esac
  used=999999
  tries=0
  while [ $tries -lt 60 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d " ")
    [ -z "$used" ] && used=999999
    [ "$used" -lt 1000 ] && break
    tries=$((tries+1))
    sleep 30
  done
  echo "$name eval start GPU$gpu mem=${used}MiB $(date '+%F %T')" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$gpu $PY $B/queue_bundle/rainnet_nowcast_eval.py \
    --data-path $B/data/rainnet --model tau \
    --checkpoints $B/corrected/checkpoints/${name}.pth \
    --out "$out" > $B/logs/eval_${name}.txt 2>&1
  erc=$?
  echo "$name eval rc=$erc $(date '+%F %T')" >> "$LOG"
  grep -a "OVERALL" $B/logs/eval_${name}.txt | tail -n 2 >> "$LOG"
  if [ "$erc" -ne 0 ] || [ ! -s "$out" ]; then
    echo "$name evaluator failed or produced no JSON; stop waiter $(date '+%F %T')" >> "$LOG"
    exit 1
  fi
done
echo "waiter all done $(date '+%F %T')" >> "$LOG"

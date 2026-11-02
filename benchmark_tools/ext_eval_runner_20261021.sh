#!/bin/bash
# [EXT-EVAL-RUNNER-2026-10-21] Sequential extended evaluation of every Paper 1 row
# (rainnet_extended_eval.py, one pass per model/tag over the fixed test split).
# GPU3 only; TAU training on GPUs 0-2 is untouched. Resumable: a tag whose npz
# already exists is skipped, so rerunning this script after TAU training lands
# adds only the three tau tags. Sample windows are picked deterministically from
# the obs-only pass (heavy = max observed peak; moderate = max mean rain with
# peak < 20 mm) and dumped only for the seed0 figure models.
B=/data/sdukjsp017/zal/MambaEP_paper_repro_20261015
PY=/data/sdukjsp017/zal/MambaEP_reproduction_20261013/.venv/bin/python
E=$B/queue_bundle/rainnet_extended_eval.py
OUT=$B/logs/ext_eval
LOG=$B/logs/ext_eval_runner_20261021.log
mkdir -p "$OUT"
echo "runner start $(date '+%F %T') pid=$$" >> "$LOG"

run() { # tag model ckpt samples dev
  tag=$1; model=$2; ckpt=$3; samples=$4; dev=$5
  if [ -f "$OUT/ext_$tag.npz" ]; then
    echo "skip $tag (npz exists) $(date '+%F %T')" >> "$LOG"; return
  fi
  if [ -n "$ckpt" ] && [ ! -f "$ckpt" ]; then
    echo "skip $tag (ckpt missing) $(date '+%F %T')" >> "$LOG"; return
  fi
  if [ "$model" = "tau" ]; then
    # tau ckpts appear mid-training (--resume auto saves every epoch), so
    # ckpt existence is NOT enough: require the landed training result
    ok=$(python3 -c "import json;d=json.load(open('$B/runs_tau_20261021/runs/$tag/results.json'));print('yes' if d.get('status')=='done' and d.get('returncode')==0 else 'no')" 2>/dev/null || echo no)
    if [ "$ok" != "yes" ]; then
      echo "skip $tag (tau training not landed) $(date '+%F %T')" >> "$LOG"; return
    fi
  fi
  extra=""
  [ -n "$ckpt" ] && extra="--checkpoints $ckpt"
  [ -n "$samples" ] && extra="$extra --sample-indices $samples"
  echo "start $tag dev=$dev $(date '+%F %T')" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$dev $PY $E --data-path $B/data/rainnet --model $model \
    --tag $tag --out-dir $OUT $extra >> "$LOG" 2>&1
  echo "done $tag rc=$? $(date '+%F %T')" >> "$LOG"
}

if [ ! -f "$OUT/ext_obs_stats.npz" ]; then
  CUDA_VISIBLE_DEVICES="" $PY $E --data-path $B/data/rainnet --obs-only \
    --tag obs_stats --out-dir $OUT >> "$LOG" 2>&1
fi
SAMPLES=$($PY -c "
import numpy as np
d = np.load('$OUT/ext_obs_stats.npz')
mx = d['win_max_obs_mm']; mn = d['win_mean_obs_mm']
heavy = int(np.argmax(mx))
mod = int(np.argmax(np.where(mx < 20.0, mn, -1.0))) if bool((mx < 20.0).any()) else int(np.argmax(mn))
print(f'{heavy},{mod}')")
echo "sample windows (heavy,moderate): $SAMPLES" >> "$LOG"

run identity identity "" "" 3
run persistence persistence "" "$SAMPLES" 3
run optical_flow optical_flow "" "$SAMPLES" ""

for s in 0 1 2; do
  sm=""; [ "$s" = "0" ] && sm="$SAMPLES"
  ck=$B/corrected/checkpoints/convlstm_rainnet_seed$s.pth
  # as-released RainNet ships one checkpoint (no _seed0 suffix)
  [ "$s" = "0" ] && ck=$B/corrected/checkpoints/convlstm_rainnet.pth
  run convlstm_rainnet_seed$s convlstm $ck "$sm" 3
  run convlstm_ed_mse_rainnet_seed$s convlstm_ed $B/corrected/checkpoints/convlstm_ed_mse_rainnet_seed$s.pth "$sm" 3
  run unet_mse_rainnet_seed$s unet $B/corrected/checkpoints/unet_mse_rainnet_seed$s.pth "$sm" 3
  run convlstm_ed_balanced_rainnet_seed$s convlstm_ed $B/corrected/checkpoints/convlstm_ed_balanced_rainnet_seed$s.pth "" 3
  run unet_balanced_rainnet_seed$s unet $B/corrected/checkpoints/unet_balanced_rainnet_seed$s.pth "" 3
  run simvp_mse_rainnet_seed$s simvp $B/corrected/checkpoints/simvp_mse_rainnet_seed$s.pth "$sm" 3
  sm=""; [ "$s" = "0" ] && sm="$SAMPLES"
  run tau_ddr_rainnet_seed$s tau $B/corrected/checkpoints/tau_ddr_rainnet_seed$s.pth "$sm" 3
done
echo "runner all done $(date '+%F %T')" >> "$LOG"

#!/bin/bash
# [PYSTEPS-TUNING-2026-10-21] Validation-only pysteps parameter tuning.
# NEVER touches test windows: --split val is the seeded 750-window carve
# (Random(2026), first 750 of a shuffled train index) from the TRAIN years.
# No number produced here is a paper number; the chosen params go into exactly
# ONE final test-split pass per method (that pass is the only test number).
# Resumable: existing ext_<tag>.npz are skipped, so reruns are cheap.
# Runs on CPU only (CUDA_VISIBLE_DEVICES=""); GPUs 0-2 train TAU, GPU3 runs
# the extended evaluator - neither is touched.
B=/data/sdukjsp017/zal/MambaEP_paper_repro_20261015
PY=/data/sdukjsp017/zal/MambaEP_reproduction_20261013/.venv/bin/python
E=$B/queue_bundle/rainnet_extended_eval.py
OUT=$B/logs/ext_eval_val
LOG=$B/logs/pysteps_tuning_20261021.log
PIDF=$OUT/tuning.pid
mkdir -p "$OUT"
# pid-file double-launch guard (pgrep -f self-matches the bridge exec shell)
if [ -f "$PIDF" ]; then
  oldpid=$(cat "$PIDF")
  if [ -n "$oldpid" ] && ps -p "$oldpid" -o args= 2>/dev/null | grep -q pysteps_tuning; then
    echo "tuning already running pid=$oldpid, exit $(date '+%F %T')" >> "$LOG"
    exit 0
  fi
fi
echo $$ > "$PIDF"
echo "pysteps tuning start $(date '+%F %T') pid=$$" >> "$LOG"

runval() { # tag model pysteps-args extra-args
  tag=$1; model=$2; pargs=$3; extra=$4
  if [ -f "$OUT/ext_$tag.npz" ]; then
    echo "skip $tag (npz exists) $(date '+%F %T')" >> "$LOG"; return
  fi
  echo "start $tag args=$pargs $(date '+%F %T')" >> "$LOG"
  CUDA_VISIBLE_DEVICES="" $PY $E --data-path $B/data/rainnet \
    --split val --model $model --pysteps-args "$pargs" \
    --tag $tag --out-dir $OUT $extra >> "$LOG" 2>&1
  echo "done $tag rc=$? $(date '+%F %T')" >> "$LOG"
}

# phase 0: on-server carve check - identity on the val split must be exactly
# zero and the window set must be the seeded carve (asserted in the evaluator)
runval identity_val identity "{}" ""

# phase A: LK motion-frame sweep (motion field from the last t_flow frames)
for t in 5 10 20; do
  runval lk_t$t pysteps_lk_extrap "{\"t_flow\": $t}" ""
done

# phase B: S-PROG timing probe (25 windows) to size the grid
if [ ! -f "$OUT/ext_sprog_probe.npz" ]; then
  echo "start sprog_probe (25 windows) $(date '+%F %T')" >> "$LOG"
  CUDA_VISIBLE_DEVICES="" $PY $E --data-path $B/data/rainnet --split val \
    --model pysteps_sprog \
    --pysteps-args '{"ar_order": 2, "n_cascade_levels": 6, "precip_thr": 0.1, "t_flow": 10}' \
    --tag sprog_probe --out-dir $OUT --limit 25 >> "$LOG" 2>&1
  echo "done sprog_probe rc=$? $(date '+%F %T')" >> "$LOG"
fi
if [ -f "$OUT/ext_sprog_probe.npz" ]; then
  $PY -c "
import json, numpy as np
d = np.load('$OUT/ext_sprog_probe.npz')
m = json.loads(str(d['meta']))
eta_h = float(m['elapsed_s']) / int(m['n_windows']) * 750.0 / 3600.0
grid = [(ar, c, thr) for ar in (1, 2) for c in (6, 8) for thr in (0.1, 1.0)]
if eta_h > 2.0:   # >2h per config: fix cascade at the pysteps default 6
    grid = [(ar, 6, thr) for ar in (1, 2) for thr in (0.1, 1.0)]
if eta_h > 4.0:   # >4h per config: fix the rain threshold at the default 0.1mm
    grid = [(ar, 6, 0.1) for ar in (1, 2)]
lines = []
for ar, c, thr in grid:
    args = json.dumps({'ar_order': ar, 'n_cascade_levels': c,
                       'precip_thr': thr, 't_flow': 10})
    lines.append('sprog_a%s_c%s_p%s|%s' % (ar, c, thr, args))
open('$OUT/sprog_configs.txt', 'w').write('\n'.join(lines) + '\n')
print('sprog probe %.2f s/window, full-config ETA %.2f h, %d configs'
      % (float(m['elapsed_s']) / int(m['n_windows']), eta_h, len(grid)))
" >> "$LOG" 2>&1
  echo "grid: $(cat $OUT/sprog_configs.txt | tr '\n' ' ')" >> "$LOG"
else
  echo "sprog probe FAILED (no npz) - skipping the sprog grid entirely $(date '+%F %T')" >> "$LOG"
  echo -n > "$OUT/sprog_configs.txt"
fi

# phase C: S-PROG grid on the full val carve
while IFS='|' read -r tag pargs; do
  [ -z "$tag" ] && continue
  runval "$tag" pysteps_sprog "$pargs" ""
done < "$OUT/sprog_configs.txt"

# phase D: aggregate -> pysteps_tuning_summary.json with chosen params
$PY -c "
import glob, json, numpy as np
rows = {}
for p in sorted(glob.glob('$OUT/ext_*.npz')):
    d = np.load(p)
    m = json.loads(str(d['meta']))
    if 'probe' in m['tag'] or 'sanity' in m['tag']:
        continue
    if m.get('model') == 'identity':
        mse = float(d['se_total']) / float(d['n_seen'])
        print('identity_val mse=%.3g (must be 0)' % mse)
        assert mse == 0.0, 'identity on val must be exactly zero'
        continue
    if m.get('model') not in ('pysteps_lk_extrap', 'pysteps_sprog'):
        continue
    mse = float(d['se_total']) / float(d['n_seen'])
    mae = float(d['lead_ae'].sum()) / float(d['n_seen'])
    rows[m['tag']] = {'model': m['model'], 'pysteps_args': m['pysteps_args'],
                      'mse_mm2': mse * m['unit_scale'] ** 2,
                      'mae_mm': mae * m['unit_scale']}
best = {}
for tag, r in sorted(rows.items(), key=lambda kv: kv[1]['mse_mm2']):
    fam = r['model']
    if fam not in best:
        best[fam] = dict(r, tag=tag)
    print('%.4f mm^2  %s' % (r['mse_mm2'], tag))
fams = sorted({r['model'] for r in rows.values()})
if 'pysteps_sprog' not in fams:
    print('MISSING pysteps_sprog rows - tuning INCOMPLETE, '
          'do not use this summary for the test pass')
json.dump({'note': 'validation-only pysteps tuning (seeded 750-window carve '
                   'from train years); NEVER paper numbers - test numbers come '
                   'only from the single final test-split pass',
           'all': rows, 'chosen': best},
          open('$OUT/pysteps_tuning_summary.json', 'w'), indent=2)
for fam, r in best.items():
    print('CHOSEN %s: tag=%s args=%s' % (fam, r['tag'],
                                         json.dumps(r['pysteps_args'])))
" >> "$LOG" 2>&1
echo "pysteps tuning all done $(date '+%F %T')" >> "$LOG"

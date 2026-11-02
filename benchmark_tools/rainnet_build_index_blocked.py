"""[BLOCKED-VAL-2026-10-22] Build a month-blocked validation index for the
Paper 1 sensitivity analysis.

The published protocol carves 750 validation windows at random from the 6000
stride-1 training windows, so 98.0% of validation windows share at least one
frame with a training window. This builder holds out WHOLE calendar months
instead:

- validation months: for each calendar month Jul..Nov, `--months-per-cal`
  years drawn from the training years with a fixed seed (default 2 -> 10 of
  the 80 training-year months, 12.5%, the same share as the random carve);
- buffer: training windows within `--buffer-h` hours of a validation month's
  boundary in the adjacent month of the same year are excluded;
- windows never span files, so month-level holdout already gives zero shared
  frames; the script still checks frame-level disjointness of train / val /
  test explicitly and refuses to write on any overlap;
- the test split and the file list are copied from the source index
  unchanged, so test window i is the same window as in every published row.

Output keeps the source schema plus an explicit 'val' list, which
dataloader_rainnet_fixed.load_data uses as-is.
"""

import argparse
import json
import os
import random

import numpy as np


def frames(windows, length=20):
    return {(f, t0 + k) for f, t0 in windows for k in range(length)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lowres-dir', required=True)
    p.add_argument('--source-index', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n-train', type=int, default=5250)
    p.add_argument('--n-val', type=int, default=750)
    p.add_argument('--months-per-cal', type=int, default=2)
    p.add_argument('--buffer-h', type=int, default=24)
    p.add_argument('--seed', type=int, default=20261022)
    a = p.parse_args()

    src = json.load(open(a.source_index))
    files = src['files']
    lengths = [np.load(os.path.join(a.lowres_dir, n), mmap_mode='r').shape[2] for n in files]
    ym = [(int(n.split('_')[0]), int(n.split('_')[1].split('.')[0])) for n in files]
    train_years = sorted({y for y, _ in ym if y < 2018})

    rng = random.Random(a.seed)
    val_months = set()
    for cal in sorted({m for _, m in ym}):
        years = [y for y in train_years if (y, cal) in ym]
        for y in rng.sample(years, a.months_per_cal):
            val_months.add((y, cal))
    val_files = {i for i, k in enumerate(ym) if k in val_months}

    train_cand, val_cand = [], []
    for i, (y, m) in enumerate(ym):
        if y >= 2018:
            continue
        T = lengths[i]
        for t0 in range(0, T - 19):
            if i in val_files:
                val_cand.append([i, t0])
                continue
            if (y, m - 1) in val_months and t0 < a.buffer_h:
                continue
            if (y, m + 1) in val_months and t0 + 19 > T - 1 - a.buffer_h:
                continue
            train_cand.append([i, t0])

    if len(train_cand) < a.n_train or len(val_cand) < a.n_val:
        raise SystemExit(f'not enough candidates: train {len(train_cand)} val {len(val_cand)}')
    train = rng.sample(train_cand, a.n_train)
    val = rng.sample(val_cand, a.n_val)
    test = src['test']

    ftr, fva, fte = frames(train), frames(val), frames(test)
    overlaps = {'train_val': len(ftr & fva), 'train_test': len(ftr & fte),
                'val_test': len(fva & fte)}
    if any(overlaps.values()):
        raise SystemExit(f'frame overlap detected, refusing to write: {overlaps}')

    out = {'files': files, 'train': train, 'val': val, 'test': test,
           'config': {'seed': a.seed, 'n_train': a.n_train, 'n_val': a.n_val,
                      'months_per_cal': a.months_per_cal, 'buffer_h': a.buffer_h,
                      'val_months': sorted(f'{y}_{m:02d}' for y, m in val_months),
                      'source_index': os.path.basename(a.source_index),
                      'frame_overlap': overlaps,
                      'candidates': {'train': len(train_cand), 'val': len(val_cand)},
                      'note': 'month-blocked validation; test split copied unchanged'}}
    with open(a.out, 'w') as f:
        json.dump(out, f)
    print('val months:', out['config']['val_months'])
    print(f'candidates train={len(train_cand)} val={len(val_cand)}; '
          f'sampled train={len(train)} val={len(val)} test={len(test)}')
    print('frame overlap:', overlaps)
    print('wrote', a.out)


if __name__ == '__main__':
    main()

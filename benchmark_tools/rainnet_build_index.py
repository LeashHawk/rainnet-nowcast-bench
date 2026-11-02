"""Build the fixed RainNet nowcasting window index from per-month lowres npy files.

Per paper Table 1: N_train=6000, N_test=1500, T_in=10, T_out=10 (20-frame hourly windows).
The paper does not state which years were used. Documented choice (leakage-free):
- train: 6000 windows, fixed-seed sampled from 2002-2017
- test:  1500 windows, fixed-seed sampled from 2018
Validation reuses the test windows (the paper reports no val count).

Output JSON: {'files': [...ordered month names...], 'train': [[file_idx, t0], ...],
'test': [[file_idx, t0], ...], 'config': {...}}
"""

import argparse
import glob
import json
import os
import random

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lowres-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-train', type=int, default=6000)
    parser.add_argument('--n-test', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=2026)
    a = parser.parse_args()

    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.lowres_dir, '*.npy')))
    if not files:
        raise SystemExit(f'no lowres npy files under {a.lowres_dir}')

    candidates = {'train': [], 'test': []}
    for file_idx, name in enumerate(files):
        year = int(name.split('_')[0])
        T = np.load(os.path.join(a.lowres_dir, name), mmap_mode='r').shape[2]
        split = 'test' if year >= 2018 else 'train'
        for t0 in range(0, T - 19):
            candidates[split].append([file_idx, t0])

    rng = random.Random(a.seed)
    index = {'files': files,
             'train': rng.sample(candidates['train'], min(a.n_train, len(candidates['train']))),
             'test': rng.sample(candidates['test'], min(a.n_test, len(candidates['test']))),
             'config': {'seed': a.seed, 'n_train': a.n_train, 'n_test': a.n_test,
                        'train_years': '2002-2017', 'test_years': '2018',
                        'window': 20, 'stride': 1,
                        'note': 'chronological split; val reuses test (documented)'}}
    with open(a.out, 'w') as f:
        json.dump(index, f)
    print(f"windows: train={len(index['train'])} (of {len(candidates['train'])} candidates) "
          f"test={len(index['test'])} (of {len(candidates['test'])})", flush=True)
    print(f'wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()

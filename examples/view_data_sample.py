"""Print rain statistics for one data sample and optionally render a panel.

Usage:
  python examples/view_data_sample.py data_samples/lowres_samples/2002_07_w0000.npy
  python examples/view_data_sample.py data_samples/lowres_samples/2002_07_w0000.npy --png panel.png

The .npy holds one 20-frame window (208 x 333) in raw mm. Without --png the
script only prints; the optional panel writes one image per frame (4 x 5
grid) with a shared colour scale.
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Inspect one RainNet data sample.')
    parser.add_argument('npy', help='path to a *_w*.npy sample window')
    parser.add_argument('--png', help='optional output panel image path')
    parser.add_argument('--thr', type=float, default=0.1,
                        help='rain threshold in mm for the fraction column (default 0.1)')
    a = parser.parse_args()

    arr = np.load(a.npy)
    if arr.ndim != 3 or arr.shape[:2] != (208, 333):
        raise SystemExit('unexpected shape %s; expected (208, 333, T)' % (arr.shape,))

    print('%s' % a.npy)
    print('shape %s  dtype %s' % (arr.shape, arr.dtype))
    print('month values: min %.3f mm  max %.3f mm' % (arr.min(), arr.max()))
    print()
    print('frame  role    mean mm  max mm  >=%.1fmm' % a.thr)
    for t in range(arr.shape[2]):
        fr = arr[:, :, t]
        role = 'input' if t < 10 else 'target'
        frac = float((fr >= a.thr).mean())
        print('%5d  %-6s %8.4f %8.2f %7.1f%%' % (t + 1, role, fr.mean(), fr.max(), 100 * frac))

    if a.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = arr.shape[2]
        ncols = 5
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 1.6 * nrows))
        vmax = float(arr.max())
        for t, ax in enumerate(axes.flat):
            if t < n:
                ax.imshow(arr[:, :, t], vmin=0.0, vmax=max(vmax, 1.0),
                          cmap='viridis', origin='lower')
                ax.set_title('t%+d' % (t - 9 if t >= 10 else t - 10), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle('%s  (0 to %.1f mm)' % (a.npy, vmax), fontsize=9)
        fig.tight_layout()
        fig.savefig(a.png, dpi=150)
        print('panel written: %s' % a.png)


if __name__ == '__main__':
    main()

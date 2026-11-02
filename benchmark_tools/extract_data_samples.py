"""Extract a few real 20-frame RainNet windows for data inspection.

Reads one monthly member of RainNet_HDF5.zip (default 2002_07, a month on the
training side of the frozen split), scans every 20-frame window, and saves:

- the N non-overlapping windows with the highest mean target rain rate
  (frames 10..19; greedy scan down the mean ranking, keeping a window only
  if it shares no frame with an already kept one),
- the M windows closest to the month median of that statistic (first window
  in hour order wins ties, distinct start hours).

Values are the archive's raw hourly accumulations in mm, copied without
scaling. The benchmark pipeline (rainnet_build_lowres.py) normalizes with
clip[0, 140] / 140; apply that recipe before feeding these excerpts to any
model. The samples are inspection excerpts, not a split: no evaluation in the
paper uses them, and they do not replace the archive download.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile

import h5py
import numpy as np


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--zip', required=True, help='path to RainNet_HDF5.zip')
    parser.add_argument('--out-dir', required=True, help='output directory')
    parser.add_argument('--member', default='2002_07.hdf5', help='monthly zip member')
    parser.add_argument('--rainy', type=int, default=3, help='number of rainiest windows')
    parser.add_argument('--typical', type=int, default=2,
                        help='number of windows nearest the month median')
    a = parser.parse_args()
    if not a.member.endswith('.hdf5'):
        raise SystemExit('member must be a monthly .hdf5 name, e.g. 2002_07.hdf5')

    stem = a.member[:-5]
    sample_dir = os.path.join(a.out_dir, 'lowres_samples')
    os.makedirs(sample_dir, exist_ok=True)

    tmp = tempfile.mkdtemp(prefix='rainnet_sample_')
    try:
        with zipfile.ZipFile(a.zip) as zf:
            if a.member not in zf.namelist():
                raise SystemExit('member %s not found in the zip' % a.member)
            tmp_path = os.path.join(tmp, a.member)
            with zf.open(a.member) as src, open(tmp_path, 'wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
        with h5py.File(tmp_path, 'r') as hf:
            low = np.asarray(hf['lowres'][:, :, :], dtype=np.float32)  # (208, 333, T)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if low.shape[0] != 208 or low.shape[1] != 333:
        raise SystemExit('unexpected grid shape %s' % (low.shape,))
    n_t = low.shape[2]
    if n_t < 20:
        raise SystemExit('month has fewer than 20 frames')

    stats = []
    for t0 in range(n_t - 19):
        target = low[:, :, t0 + 10:t0 + 20]
        stats.append((float(target.mean()), t0))
    means = np.array([s[0] for s in stats])
    median = float(np.median(means))

    order = sorted(stats, key=lambda s: -s[0])
    rainy = []
    for mean, t0 in order:
        if all(abs(t0 - k) >= 20 for k in rainy):
            rainy.append(t0)
        if len(rainy) == a.rainy:
            break
    by_med = sorted(range(len(stats)), key=lambda i: (abs(means[i] - median), i))
    typical = []
    for i in by_med:
        t0 = stats[i][1]
        if t0 not in rainy and t0 not in typical:
            typical.append(t0)
        if len(typical) == a.typical:
            break

    windows = rainy + typical
    entries = []
    for t0 in windows:
        arr = low[:, :, t0:t0 + 20]
        out = os.path.join(sample_dir, '%s_w%04d.npy' % (stem, t0))
        np.save(out, arr)
        entries.append({
            'file': os.path.relpath(out, a.out_dir).replace(os.sep, '/'),
            'source_member': a.member,
            'month': stem,
            'start_frame': t0,
            'shape': [208, 333, 20],
            'dtype': 'float32',
            'unit': 'mm, raw archive values (no normalization applied)',
            'min_mm': float(arr.min()),
            'max_mm': float(arr.max()),
            'mean_target_rate_mm': float(arr[:, :, 10:].mean()),
            'rule': 'rainiest' if t0 in rainy else 'nearest month median',
            'sha256': sha256_of(out),
        })
        print('%s: t0=%d max=%.2f mm mean_target=%.4f mm/h' %
              (os.path.basename(out), t0, arr.max(), arr[:, :, 10:].mean()), flush=True)

    manifest = {
        'description': 'Inspection excerpts from the RainNet archive, one 20-frame '
                       'window per file (10 input hours + 10 target hours), raw mm values.',
        'source_archive': 'RainNet_HDF5.zip (Chen et al. 2022, distributed under CC BY; '
                         'download from https://github.com/neuralchen/RainNet)',
        'selection_rule': 'scan every 20-frame window of the month; keep the %d '
                          'non-overlapping windows with the highest mean target rain rate '
                          '(frames 10..19; greedy down the ranking) and the %d closest to '
                          'the month median of that statistic (ties resolved by earlier '
                          'start hour)' % (a.rainy, a.typical),
        'normalization_note': 'benchmark inputs are clip[0, 140] mm then /140 '
                              '(rainnet_build_lowres.py); apply before model use',
        'split_note': 'these excerpts are not part of any train/validation/test split and '
                      'are not used by any evaluation in the paper',
        'producing_tool': 'benchmark_tools/extract_data_samples.py',
        'windows': entries,
    }
    with open(os.path.join(a.out_dir, 'samples_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print('done: %d windows -> %s' % (len(entries), a.out_dir), flush=True)


if __name__ == '__main__':
    main()

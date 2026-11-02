"""Extract ONLY the lowres (208x333) precipitation arrays from RainNet_HDF5.zip.

The zip (13.6 GB) holds 85 hdf5 files (years 2002-2018, months 07-11), each with 'highres'
(624x999xT) and 'lowres' (208x333xT) datasets. Full extraction is 322 GB; lowres alone is
~17.5 GB fp32 (~8.8 GB fp16). We stream: extract one member at a time to a temp file,
read 'lowres', normalize exactly like the official dataset_prepare_hdf5.py (clip to
[0, 140] mm then divide by 140), save one float16 .npy per month, delete the temp.

Outputs: <out>/<YYYY>_<MM>.npy with shape (208, 333, T) float16, plus
lowres_manifest.json (per-file shape, value range, source member, sha256 of the output).

The MambaEP paper uses the full-frame 208x333 lowres sequences for nowcasting
(10-in/10-out per Table 1) — the official patch pipeline is for the downscaling task
and is NOT used here.
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--out', required=True)
    a = parser.parse_args()
    os.makedirs(a.out, exist_ok=True)
    manifest = {}
    tmp = tempfile.mkdtemp(prefix='rainnet_tmp_')
    try:
        with zipfile.ZipFile(a.zip) as zf:
            members = sorted(n for n in zf.namelist()
                             if n.lower().endswith(('.hdf5', '.h5')))
            if not members:
                raise SystemExit('no hdf5 members found in the zip')
            for name in members:
                base = os.path.basename(name)
                stem = os.path.splitext(base)[0]              # e.g. 2009_07
                target = os.path.join(a.out, stem + '.npy')
                if os.path.exists(target):
                    print(f'skip {base} (output exists)', flush=True)
                    continue
                tmp_path = os.path.join(tmp, base)
                with zf.open(name) as src, open(tmp_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                with h5py.File(tmp_path, 'r') as hf:
                    low = hf['lowres'][:]                     # (208, 333, T)
                os.remove(tmp_path)
                low = (np.clip(low, 0.0, 140.0) / 140.0).astype(np.float16)
                np.save(target, low)
                manifest[stem] = {'shape': list(low.shape),
                                  'min': float(low.min()), 'max': float(low.max()),
                                  'sha256': sha256_of(target),
                                  'source_member': name}
                print(f'{stem}: shape={low.shape} '
                      f'range=[{low.min():.4f},{low.max():.4f}]', flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(a.out, 'lowres_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'done: {len(manifest)} months -> {a.out}', flush=True)


if __name__ == '__main__':
    main()

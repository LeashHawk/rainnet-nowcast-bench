"""[FIX-RAINNET-2026-10-15] Replacement RainNet loader.

The original dataloader_rainnet.py was a broken copy of the temperature loader: it
pointed at the temperature directory, reshaped data to (12, 1, 32, 64), and contained a
function with an empty body (IndentationError at import). It cannot load RainNet data;
no MambaEP entry for RainNet existed either.

This loader implements the paper's RainNet task (Table 1): lowres 208x333, 10 frames in
/ 10 frames out, N_train=6000, N_test=1500, reading the per-month lowres npy files and
the window index produced by rainnet_build_lowres.py / rainnet_build_index.py.

Expected root layout (root = --data_path):
  <root>/lowres/<YYYY>_<MM>.npy     # (208, 333, T) float16, normalized 0..1 (clip/140)
  <root>/rainnet_index.json
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class RainNetNowcastDataset(Dataset):
    def __init__(self, lowres_dir, files, windows):
        self.files = files
        self.windows = [tuple(w) for w in windows]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_idx, t0 = self.windows[idx]
        arr = np.load(self.files[file_idx], mmap_mode='r')        # (208, 333, T) fp16
        seq = np.array(arr[:, :, t0:t0 + 20], dtype=np.float32)  # (208, 333, 20)
        seq = np.transpose(seq, (2, 0, 1))[:, None, :, :]        # (20, 1, 208, 333)
        inp = torch.from_numpy(seq[:10].copy()).contiguous().float()
        out = torch.from_numpy(seq[10:].copy()).contiguous().float()
        return inp, out


def load_data(root, batch_size=4, num_workers=0):
    lowres_dir = os.path.join(root, 'lowres')
    index_path = os.path.join(root, 'rainnet_index.json')
    with open(index_path) as f:
        index = json.load(f)
    files = [os.path.join(lowres_dir, name) for name in index['files']]

    # [FIX-RAINNET-VAL-2026-10-15] Audit finding: the first version of this loader set
    # val_set = test_set. train_function.py selects the best checkpoint on the EVAL
    # loader, so val=test means model selection runs on the test set — an optimistic
    # bias no clean protocol should carry into a reported metric. Carve a disjoint
    # 750-window val (12.5% of the 6000 train windows, the same proportion as the MM
    # loader fix) with a SEEDED shuffle: train 5250 / val 750 / test 1500, all
    # disjoint; the test split (year 2018, built by rainnet_build_index.py) is
    # untouched. N_train 6000->5250 deviation from the paper's Table 1 is documented
    # in COMPAT_LOG.md.
    # [BLOCKED-VAL-2026-10-22] An index that carries an explicit 'val' list (built
    # by rainnet_build_index_blocked.py: whole held-out months, zero shared frames
    # with train) is used as-is. Indexes without it keep the seeded carve below
    # bit-for-bit, so every published row is unaffected.
    if 'val' in index:
        train_windows = [list(w) for w in index['train']]
        val_windows = [list(w) for w in index['val']]
    else:
        order = list(range(len(index['train'])))
        random.Random(2026).shuffle(order)
        val_pos = set(order[:750])
        train_windows = [index['train'][i] for i in order if i not in val_pos]
        val_windows = [index['train'][i] for i in sorted(val_pos)]

    train_set = RainNetNowcastDataset(lowres_dir, files, train_windows)
    val_set = RainNetNowcastDataset(lowres_dir, files, val_windows)
    test_set = RainNetNowcastDataset(lowres_dir, files, index['test'])

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return dataloader_train, dataloader_validation, dataloader_test, 0, 1


if __name__ == '__main__':
    tr, va, te, _, _ = load_data(root='.')
    print(len(tr), len(va), len(te))

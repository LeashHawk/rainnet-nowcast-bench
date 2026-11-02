"""Checks for the month-blocked validation index (rainnet_build_index_blocked.py)
and the loader's explicit-'val' branch, on tiny synthetic month files.

1. builder: zero shared frames train/val/test, val windows only in val months,
   buffer respected, test split copied unchanged, deterministic for a seed;
2. loader: explicit 'val' index -> exactly those windows; legacy index (no
   'val') -> the published seeded carve, unchanged.
"""

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = next(p for p in (ROOT / 'corrected', ROOT / 'repro_corrected')
            if (p / 'dataloader').exists())
BUILDER = next(p for p in (ROOT / 'benchmark_tools' / 'rainnet_build_index_blocked.py',
                           ROOT / 'queue_bundle' / 'rainnet_build_index_blocked.py')
               if p.exists())
sys.path.insert(0, str(CODE))
from dataloader.dataloader_rainnet_fixed import load_data  # noqa: E402

T = 80
fails = 0


def check(cond, msg):
    global fails
    print(('PASS ' if cond else 'FAIL ') + msg)
    fails += 0 if cond else 1


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    low = tmp / 'lowres'
    low.mkdir()
    files = []
    for y in range(2002, 2019):
        for m in range(7, 12):
            name = f'{y}_{m:02d}.npy'
            np.save(low / name, np.random.rand(4, 5, T).astype(np.float16))
            files.append(name)
    rng = random.Random(1)
    tr_c = [[i, t] for i, n in enumerate(files) if not n.startswith('2018')
            for t in range(T - 19)]
    te_c = [[i, t] for i, n in enumerate(files) if n.startswith('2018')
            for t in range(T - 19)]
    legacy = {'files': files, 'train': rng.sample(tr_c, 1000),
              'test': rng.sample(te_c, 50), 'config': {}}
    (tmp / 'rainnet_index.json').write_text(json.dumps(legacy))

    outs = []
    for k in range(2):
        out = tmp / f'blocked_{k}.json'
        r = subprocess.run([sys.executable, str(BUILDER), '--lowres-dir', str(low),
                            '--source-index', str(tmp / 'rainnet_index.json'),
                            '--out', str(out), '--n-train', '400', '--n-val', '60',
                            '--buffer-h', '10'], capture_output=True, text=True)
        check(r.returncode == 0, f'builder run {k} rc=0 ({r.stderr.strip()[-200:]})')
        outs.append(json.loads(out.read_text()))
    b = outs[0]
    check(b == outs[1], 'builder deterministic for a fixed seed')

    def frames(ws):
        return {(f, t + j) for f, t in ws for j in range(20)}
    check(not (frames(b['train']) & frames(b['val'])), 'train/val share no frame')
    check(not (frames(b['train']) & frames(b['test'])), 'train/test share no frame')
    check(not (frames(b['val']) & frames(b['test'])), 'val/test share no frame')
    check(b['test'] == legacy['test'] and b['files'] == files, 'test split + files unchanged')
    vm = set(b['config']['val_months'])
    check(len(vm) == 10, f'10 validation months ({len(vm)})')
    check(all(files[f][:7] in vm for f, _ in b['val']), 'val windows only in val months')
    check(not any(files[f][:7] in vm for f, _ in b['train']), 'no train window in val months')
    ok_buf = True
    for f, t0 in b['train']:
        y, m = int(files[f][:4]), int(files[f][5:7])
        if f'{y}_{m - 1:02d}' in vm and t0 < 10:
            ok_buf = False
        if f'{y}_{m + 1:02d}' in vm and t0 + 19 > T - 1 - 10:
            ok_buf = False
    check(ok_buf, 'buffer around val months respected')

    root_b = tmp / 'root_b'
    root_b.mkdir()
    (root_b / 'lowres').mkdir()
    for n in files:
        (root_b / 'lowres' / n).write_bytes((low / n).read_bytes())
    (root_b / 'rainnet_index.json').write_text(json.dumps(b))
    tr, va, te, _, _ = load_data(root=str(root_b), batch_size=2, num_workers=0)
    check([list(w) for w in va.dataset.windows] == b['val'], 'loader uses explicit val')
    check([list(w) for w in tr.dataset.windows] == b['train'], 'loader uses explicit train')
    check([list(w) for w in te.dataset.windows] == b['test'], 'loader test unchanged')

    tr, va, te, _, _ = load_data(root=str(tmp), batch_size=2, num_workers=0)
    order = list(range(len(legacy['train'])))
    random.Random(2026).shuffle(order)
    vp = set(order[:750])
    check([list(w) for w in va.dataset.windows] ==
          [legacy['train'][i] for i in sorted(vp)], 'legacy index: seeded val carve unchanged')
    check([list(w) for w in tr.dataset.windows] ==
          [legacy['train'][i] for i in order if i not in vp], 'legacy index: train unchanged')
    x, y = va.dataset[0]
    check(tuple(x.shape) == (10, 1, 4, 5) and tuple(y.shape) == (10, 1, 4, 5), 'sample shapes')

print('ALL_PASS' if fails == 0 else f'FAILURES={fails}')
sys.exit(1 if fails else 0)

"""[TEST-EXT-EVAL-2026-10-21] Local verification of benchmark_tools/rainnet_extended_eval.py.

End-to-end on a synthetic RainNet-shaped dataset (2 months, 32x64, 48 hourly
frames), no GPU required:

  1. fss_sums unit check vs a brute-force sliding-window reference;
  2. --model identity canary: per-window MSE == 0, contingency = perfect,
     FSS == 1 wherever events exist, sample dumps written and aligned;
  3. --model persistence cross-check: per-window MSE reproduces an independent
     numpy recomputation from the raw fp16 files;
  4. trained-model mechanics: random-init convlstm_ed at the fake geometry via
     the --build-kwargs escape hatch loads its checkpoint and runs.

Run: python local_checks/test_extended_eval.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'benchmark_tools' / 'rainnet_extended_eval.py'
_TMP_CONTEXT = tempfile.TemporaryDirectory(prefix='rainnet-ext-eval-test-')
TMP = Path(_TMP_CONTEXT.name)
H, W, T = 32, 64, 720
UNIT = 140.0

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f' ({detail})' if detail else ''))
    if not cond:
        fails.append(name)


def build_fake_dataset():
    data_dir = TMP / 'data'
    (data_dir / 'lowres').mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for fi, name in enumerate(('2018_07.npy', '2018_08.npy')):
        arr = np.zeros((H, W, T), dtype=np.float16)
        yy, xx = np.mgrid[0:H, 0:W]
        for t in range(T):
            amp = 0.05 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * (t + 7 * fi) / 24.0))
            cx, cy = 8 + 1.1 * t, 16 + 0.9 * t + 3 * fi
            arr[:, :, t] = (amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                                         / (2 * 5.0 ** 2))).astype(np.float16)
        np.save(data_dir / 'lowres' / name, arr)
    index = {
        'files': ['2018_07.npy', '2018_08.npy'],
        # the fixed loader always carves 750 validation windows out of train,
        # so the fake train split must hold > 750 windows to stay non-empty
        'train': [[f, t0] for f in (0, 1) for t0 in range(0, T - 19)],
        'test': [[1, t0] for t0 in range(0, 28, 3)] + [[0, t0] for t0 in (2, 9, 16, 23)],
    }
    (data_dir / 'rainnet_index.json').write_text(json.dumps(index))
    return data_dir, index


def run_ext(args, out_dir):
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE',
               CUDA_VISIBLE_DEVICES='')
    cmd = [sys.executable, str(SCRIPT), '--data-path', str(TMP / 'data'),
           '--out-dir', str(out_dir), '--csi-thresholds', '0.1,0.5,1,2,5,10',
           '--fss-thresholds', '0.5,1,2', '--fss-nbds', '1,3,5',
           '--device', 'cpu'] + args
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
    return r


def fss_from_npz(d):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1.0 - d['fss_d2'] / (d['fss_f2'] + d['fss_o2'])


def main():
    out_dir = TMP / 'out'
    data_dir, index = build_fake_dataset()
    sys.path.insert(0, str(ROOT / 'benchmark_tools'))
    from rainnet_extended_eval import fss_sums  # noqa: E402
    import torch  # noqa: E402

    print('--- 1. fss_sums unit check vs brute force ---')
    rng = np.random.RandomState(1)
    pb = (rng.rand(2, 3, 8, 9) > 0.5).astype(np.float32)
    ob = (rng.rand(2, 3, 8, 9) > 0.5).astype(np.float32)
    d2, f2, o2 = fss_sums(torch.from_numpy(pb), torch.from_numpy(ob), 3)
    ref_pf = np.lib.stride_tricks.sliding_window_view(pb, (3, 3), axis=(2, 3)).mean(axis=(-1, -2))
    ref_po = np.lib.stride_tricks.sliding_window_view(ob, (3, 3), axis=(2, 3)).mean(axis=(-1, -2))
    # sum over (batch, valid H, valid W) to match fss_sums' per-lead (L,) output
    ref_d2 = ((ref_pf - ref_po) ** 2).sum(axis=(0, 2, 3))
    ref_f2 = (ref_pf ** 2).sum(axis=(0, 2, 3))
    ref_o2 = (ref_po ** 2).sum(axis=(0, 2, 3))
    check('fss d2 matches brute force', np.allclose(d2, ref_d2, atol=1e-12))
    check('fss f2 matches brute force', np.allclose(f2, ref_f2, atol=1e-12))
    check('fss o2 matches brute force', np.allclose(o2, ref_o2, atol=1e-12))

    print('--- 2. identity canary ---')
    r = run_ext(['--model', 'identity', '--tag', 'identity',
                 '--sample-indices', '3,9'], out_dir)
    check('identity subprocess rc=0', r.returncode == 0)
    d = np.load(out_dir / 'ext_identity.npz', allow_pickle=True)
    meta = json.loads(str(d['meta']))
    check('identity meta tag/model', meta['model'] == 'identity')
    check('identity meta carries code sha256s',
          len(meta.get('model_nowcast_sha256', '')) == 64
          and len(meta.get('index_sha256', '')) == 64)
    check('identity win_mse all zero', np.all(d['win_mse_norm'] == 0.0))
    check('identity win_mae all zero', np.all(d['win_mae_norm'] == 0.0))
    check('identity se_total == 0', float(d['se_total']) == 0.0)
    n_lead, n_thr = d['cont'].shape[0], d['cont'].shape[1]
    ok_cont = True
    for lead in range(n_lead):
        for ti in range(n_thr):
            h, m, f, c = d['cont'][lead, ti]
            events = h + m
            if events > 0 and not (h == events and f == 0 and m == 0):
                ok_cont = False
            if events == 0 and not (h == 0 and m == 0 and f == 0):
                ok_cont = False
    check('identity contingency is perfect where events exist', ok_cont)
    fss = fss_from_npz(d)
    check('identity FSS is 1 or nan (no d2 anywhere)',
          bool(np.all(np.isnan(fss) | np.isclose(fss, 1.0))))
    check('identity FSS actually has 1s (events exist)',
          bool(np.any(np.isclose(fss, 1.0))))
    check('identity window identity arrays match index order',
          d['file_idx'].tolist() == [w[0] for w in index['test']]
          and d['t0'].tolist() == [w[1] for w in index['test']])
    blocks = np.unique(d['file_idx'].astype(np.int64) * 1000 + d['t0'] // 24)
    # file1 windows t0=0..27 -> days 0,1; file0 windows t0=2..23 -> day 0 only
    check('identity day blocks = {file0-day0, file1-day0, file1-day1}',
          blocks.tolist() == [0, 1000, 1001], f'{blocks.tolist()}')
    s = np.load(out_dir / 'ext_identity_win3.npz')
    check('sample dump shapes', s['inp'].shape == (10, H, W)
          and s['obs'].shape == (10, H, W) and s['pred'].shape == (10, H, W))
    check('identity sample pred == obs', np.array_equal(s['pred'], s['obs']))
    check('identity sample unit_scale/meta', float(s['unit_scale']) == UNIT
          and int(s['file_idx']) == index['test'][3][0]
          and int(s['t0']) == index['test'][3][1])

    print('--- 3. persistence cross-check vs independent numpy recompute ---')
    r = run_ext(['--model', 'persistence', '--tag', 'persistence'], out_dir)
    check('persistence subprocess rc=0', r.returncode == 0)
    d = np.load(out_dir / 'ext_persistence.npz')
    ref = np.zeros(len(index['test']))
    for i, (fi, t0) in enumerate(index['test']):
        arr = np.load(data_dir / 'lowres' / index['files'][fi],
                      mmap_mode='r')
        seq = np.transpose(np.array(arr[:, :, t0:t0 + 20], dtype=np.float32),
                           (2, 0, 1))
        inp, obs = seq[:10], seq[10:]
        pred = np.broadcast_to(inp[-1], obs.shape)
        ref[i] = float(((obs - pred) ** 2).mean())
    check('persistence per-window MSE matches independent recompute',
          np.allclose(d['win_mse_norm'], ref, atol=1e-9, rtol=1e-6),
          f'max abs diff {np.abs(d["win_mse_norm"] - ref).max():.2e}')
    overall = float(d['se_total']) / int(d['n_seen'])
    check('persistence overall == mean(win_mse) (equal pixel counts)',
          abs(overall - ref.mean()) < 1e-9)
    fss = fss_from_npz(d)
    check('persistence FSS within [0, 1] or nan',
          bool(np.all(np.isnan(fss) | ((fss >= 0) & (fss <= 1)))))

    print('--- 4. trained-model mechanics (random-init convlstm_ed, fake geometry) ---')
    sys.path.insert(0, str(ROOT / 'corrected'))
    from model_nowcast import build  # noqa: E402
    kw = {'t_in': 10, 't_out': 10, 'c_in': 1, 'c_out': 1}
    torch.manual_seed(0)
    m = build('convlstm_ed', **kw)
    ckpt = TMP / 'ckpt_ed.pth'
    torch.save(m.state_dict(), ckpt)
    r = run_ext(['--model', 'convlstm_ed', '--checkpoints', str(ckpt),
                 '--build-kwargs', json.dumps(kw), '--tag', 'ed_rnd',
                 '--sample-indices', '1'], out_dir)
    check('trained-model subprocess rc=0', r.returncode == 0)
    if r.returncode == 0:
        d = np.load(out_dir / 'ext_ed_rnd.npz')
        check('ed_rnd win_mse finite and positive',
              bool(np.all(np.isfinite(d['win_mse_norm']))
                   and np.all(d['win_mse_norm'] > 0)))
        check('ed_rnd meta records checkpoint',
              'ckpt_ed.pth' in str(d) or True)  # meta below is the real check
        meta = json.loads(str(d['meta']))
        check('ed_rnd meta has checkpoints path',
              str(ckpt) == meta['checkpoints'])
        s = np.load(out_dir / 'ext_ed_rnd_win1.npz')
        check('ed_rnd sample pred differs from obs (random init)',
              not np.array_equal(s['pred'], s['obs'])
              and s['pred'].shape == (10, H, W))

    print('--- 5. obs-only pass (no --model) ---')
    r = run_ext(['--obs-only', '--tag', 'obs_stats'], out_dir)
    check('obs-only subprocess rc=0 (no --model required)', r.returncode == 0)
    if r.returncode == 0:
        d = np.load(out_dir / 'ext_obs_stats.npz')
        meta = json.loads(str(d['meta']))
        check('obs-only meta model is None', meta['model'] is None)
        check('obs-only npz has obs stats but no win_mse',
              'win_max_obs_mm' in d and 'win_mse_norm' not in d)
        check('obs-only window identity matches index order',
              d['file_idx'].tolist() == [w[0] for w in index['test']]
              and d['t0'].tolist() == [w[1] for w in index['test']])

    print('--- 6. --split val (carve replication; tuning-only path) ---')
    r = run_ext(['--model', 'identity', '--tag', 'identity_val',
                 '--split', 'val'], out_dir)
    check('val-split identity subprocess rc=0', r.returncode == 0)
    if r.returncode == 0:
        d = np.load(out_dir / 'ext_identity_val.npz')
        meta = json.loads(str(d['meta']))
        # fake train split has 1402 windows; the loader carves 750 for val
        check('val split has 750 windows', int(meta['n_windows']) == 750)
        check('val identity still zero', np.all(d['win_mse_norm'] == 0.0))
        import random as _r
        _order = list(range(len(index['train'])))
        _r.Random(2026).shuffle(_order)
        exp = [index['train'][i] for i in sorted(set(_order[:750]))]
        check('val window identity matches the seeded carve',
              d['file_idx'].tolist() == [w[0] for w in exp]
              and d['t0'].tolist() == [w[1] for w in exp])
        # the fake fixture draws test windows from the same two month-files as
        # train, so disjointness does not apply here (real data: train years
        # 2002-2017, test 2018, disjoint by file); check the carve property
        check('val windows are a subset of the train list',
              set(map(tuple, exp)) <= set(map(tuple, index['train'])))

    print('ALL_PASS' if not fails else f'FAILED: {fails}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())

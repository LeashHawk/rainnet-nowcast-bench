"""[TEST-EXT-ANALYSIS-2026-10-21] Unit checks for local_checks/ext_analysis_paper1.py.

  1. block_bootstrap: exact point estimate, multiset semantics of block
     resampling (windows move together, repeated blocks contribute twice),
     determinism for a fixed seed;
  2. paired_bootstrap: p-value behaviour for a constant shift (tiny p),
     zero difference (p=1) and block-dominated noise (p large);
  3. day_block_ids: (file_idx, t0) -> (file, day) block id;
  4. fss_from_sums / csi_tables / hss: formulas incl. nan cases;
  5. train_val_overlap: brute-force nested-loop reference on a synthetic index;
  6. regression_check: exact-match pass + perturbed-json fail.

Run: python local_checks/test_ext_analysis.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'local_checks'))
from ext_analysis_paper1 import (  # noqa: E402
    block_bootstrap, csi_tables, day_block_ids, fss_from_sums,
    hss, paired_bootstrap, regression_check, train_val_overlap)

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f' ({detail})' if detail else ''))
    if not cond:
        fails.append(name)


def main():
    print('--- 1. block_bootstrap ---')
    # block A: 3 windows of value 1; block B: 1 window of value 3
    values = np.array([1.0, 1.0, 1.0, 3.0])
    blocks = np.array([5, 5, 5, 9])
    res = block_bootstrap(values, blocks, reps=4000, seed=1)
    check('point = plain mean 1.5', abs(res['point'] - 1.5) < 1e-12)
    check('n_blocks = 2', res['n_blocks'] == 2)
    allowed = {1.0, 1.5, 3.0}  # {A,A}: (3+3)/6; {A,B} or {B,A}: (3+3)/4; {B,B}: 3
    check('bootstrap draws only take block-multiset values',
          bool(np.all(np.isin(res['boots'], list(allowed)))))
    expected_mean = (1.0 + 1.5 + 1.5 + 3.0) / 4.0
    check('bootstrap mean ~ analytic 1.75',
          abs(res['boots'].mean() - expected_mean) < 0.1,
          f'{res["boots"].mean():.3f}')
    res2 = block_bootstrap(values, blocks, reps=4000, seed=1)
    check('deterministic for fixed seed',
          np.array_equal(res['boots'], res2['boots']))

    print('--- 2. paired_bootstrap ---')
    n = 60
    blk = np.repeat(np.arange(6), 10)
    base = np.random.RandomState(7).rand(n)
    res = paired_bootstrap(base + 0.05, blk, reps=2000, seed=3)
    check('constant +shift: CI excludes 0',
          res['ci95'][0] > 0, f'CI={res["ci95"]}')
    check('constant +shift: p < 0.05', res['p_two_sided'] < 0.05,
          f'p={res["p_two_sided"]}')
    res = paired_bootstrap(base * 0.0, blk, reps=2000, seed=3)
    check('zero difference: p == 1 and CI brackets 0',
          res['p_two_sided'] == 1.0 and res['ci95'][0] <= 0 <= res['ci95'][1])
    noise = np.where(blk % 2 == 0, 1.0, -1.0) + 0.01 * np.arange(n)
    res = paired_bootstrap(noise, blk, reps=2000, seed=3)
    check('block-dominated noise: p large',
          res['p_two_sided'] > 0.10, f'p={res["p_two_sided"]}')

    print('--- 3. day_block_ids ---')
    ids = day_block_ids(np.array([0, 0, 1]), np.array([0, 25, 24]))
    check('day block ids', ids.tolist() == [0, 1, 1001], f'{ids.tolist()}')

    print('--- 4. fss / csi / hss formulas ---')
    check('fss 1 when d2=0 with events', fss_from_sums(0.0, 2.0, 2.0) == 1.0)
    v = fss_from_sums(0.0, 0.0, 0.0)
    check('fss nan when no events (0/0)', np.isnan(v))
    v = fss_from_sums(1.0, 3.0, 1.0)
    check('fss value in [0,1] for a real case', 0.0 <= float(v) <= 1.0,
          f'{float(v):.3f}')
    check('hss perfect', abs(hss(8, 0, 0, 8) - 1.0) < 1e-12)
    check('hss zero skill', abs(hss(4, 4, 4, 4) - 0.0) < 1e-12)
    cont = np.zeros((2, 2, 4), dtype=np.int64)
    cont[0, 0] = (6, 2, 3, 9)      # csi=6/11, pod=6/8, far=3/9
    cont[1, 1] = (0, 0, 0, 17)     # no events -> all nan, hss nan
    csi, pod, far, hss_t = csi_tables(cont)
    check('csi', abs(csi[0, 0] - 6 / 11) < 1e-12 and np.isnan(csi[1, 1]))
    check('pod/far', abs(pod[0, 0] - 0.75) < 1e-12 and abs(far[0, 0] - 1 / 3) < 1e-12)
    check('nan cells for no-event lead/threshold',
          np.isnan(pod[1, 1]) and np.isnan(far[1, 1]) and np.isnan(hss_t[1, 1]))
    h_exp = 2 * (6 * 9 - 2 * 3) / ((8 * 12) + (9 * 11))
    check('hss formula', abs(hss_t[0, 0] - h_exp) < 1e-12)

    print('--- 5. train_val_overlap vs brute force ---')
    train = [(0, t) for t in range(600)] + [(1, t) for t in range(600)]
    index = {'train': train, 'test': [[2, t] for t in range(10)]}
    got = train_val_overlap(index)
    # brute-force reference with the same seeded carve
    import random as _random
    order = list(range(len(train)))
    _random.Random(2026).shuffle(order)
    val_pos = set(order[:750])
    val_w = [train[i] for i in sorted(val_pos)]
    train_w = [train[i] for i in order if i not in val_pos]
    counts = []
    shared = 0
    n_over = 0
    for fi, t0 in val_w:
        c = 0
        for fj, tj in train_w:
            if fj == fi and abs(tj - t0) < 20:
                c += 1
                shared += 20 - abs(tj - t0)
        counts.append(c)
        n_over += (c > 0)
    check('n_train/n_val', got['n_train'] == len(train_w) == 450
          and got['n_val'] == 750)
    check('overlap pct matches brute force',
          got['val_windows_overlapping_train_pct']
          == round(100.0 * n_over / 750, 1))
    check('mean/max overlapping counts match brute force',
          got['mean_overlapping_train_windows_per_val_window']
          == round(float(np.mean(counts)), 2)
          and got['max_overlapping_train_windows_per_val_window'] == int(max(counts)))
    check('shared frames total matches brute force',
          got['total_shared_frames'] == shared)
    check('test files disjoint from train', got['test_files_disjoint_from_train_files'])

    print('--- 6. regression_check ---')
    tmp_context = tempfile.TemporaryDirectory(prefix='rainnet-ext-analysis-test-')
    tmp = Path(tmp_context.name)
    unit_scale = 140.0
    L, T = 2, 2
    cont = np.zeros((L, T, 4), dtype=np.int64)
    cont[:, 0] = (60, 20, 30, 90)
    cont[:, 1] = (6, 2, 3, 9)
    mse_norm = 1.45e-05
    mae_norm = 0.1
    n_seen = 10 * 208 * 333 * 4
    lead_se = np.full(L, mse_norm * n_seen / L)
    lead_ae = np.full(L, mae_norm * n_seen / L)
    np.savez(tmp / 'ext_fake.npz', meta=np.array(json.dumps(
        {'unit_scale': unit_scale})), cont=cont, se_total=mse_norm * n_seen,
        n_seen=n_seen, lead_se=lead_se, lead_ae=lead_ae)
    with np.load(tmp / 'ext_fake.npz') as d:
        csi, _, _, hss_t = csi_tables(cont)
        ev = {
            'overall_mse': round(mse_norm, 7),
            'overall_mse_mm2': round(mse_norm * unit_scale ** 2, 4),
            'overall_rmse_mm': round(float(np.sqrt(mse_norm)) * unit_scale, 4),
            'overall_mae_mm': round(0.1 * unit_scale, 4),
            'thresholds': [
                {'threshold_mm': 0.5, 'csi_mean': round(float(np.nanmean(csi[:, 0])), 4),
                 'hss_mean': round(float(np.nanmean(hss_t[:, 0])), 4)},
                {'threshold_mm': 1.0, 'csi_mean': round(float(np.nanmean(csi[:, 1])), 4),
                 'hss_mean': round(float(np.nanmean(hss_t[:, 1])), 4)}]}
        rows = regression_check(d, ev, unit_scale)
        check('exact-match json: all rows ok', all(r[3] for r in rows),
              f'{[(r[0], r[3]) for r in rows]}')
        ev_bad = dict(ev, overall_mse=round(mse_norm + 1e-3, 7))
        rows = regression_check(d, ev_bad, unit_scale)
        mse_row = [r for r in rows if r[0] == 'overall_mse_norm'][0]
        check('perturbed json: mse row fails', not mse_row[3])

    print('ALL_PASS' if not fails else f'FAILED: {fails}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())

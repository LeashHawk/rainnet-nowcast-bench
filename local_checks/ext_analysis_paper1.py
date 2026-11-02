"""[EXT-ANALYSIS-2026-10-21] Paper 1 extended-eval analysis.

Consumes the ext_*.npz passes produced by queue_bundle/rainnet_extended_eval.py
(one per model/seed, aligned per-window by construction) and the pre-existing
eval_*.json files, and produces:

  1. REGRESSION: every tag with an eval_*.json must reproduce its overall
     MSE/RMSE/MAE and per-threshold CSI/HSS means within rounding tolerances
     (same accumulation order, same GPU model class => near bit-identical).
  2. DAY-BLOCK BOOTSTRAP: 95% CIs for per-model mean per-window MSE (mm^2),
     resampling calendar days (test windows are stride-1 and heavily
     autocorrelated within a day; blocks = (month file, t0 // 24)).
  3. PAIRED TESTS: per-window paired differences for model pairs, same block
     bootstrap; two-sided bootstrap p.
  4. FSS: 1 - d2/(f2+o2) per lead/threshold/neighbourhood; lead-mean table.
  5. HEAVY RAIN: CSI@5mm and CSI@10mm per lead from the contingency counts.
  6. TRAIN/VAL OVERLAP: reproduces the loader's seeded 750-window validation
     carve and quantifies how many validation windows share frames with train
     windows (temporal overlap of the selection set; disclosed in the paper).
  7. Sample-window rain stats for the case-study figure caption.

Everything lands in artifacts/ext_eval/: EXT_ANALYSIS.md (human tables) +
ext_summary.json (numbers for the draft). Deterministic: bootstrap seed fixed.

Run: python local_checks/ext_analysis_paper1.py \
        --ext-dir artifacts/ext_eval --eval-dir artifacts/ext_eval/eval_jsons \
        --index-json artifacts/ext_eval/rainnet_index.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

FAMILIES = [
    # key, label, tags, has independent standard-eval json (regression gate)
    ('persistence', 'Persistence (Eulerian)', ['persistence'], True),
    ('optical_flow', 'Optical flow (Lagrangian)', ['optical_flow'], True),
    # pysteps baselines: single pass; no standard-evaluator JSON exists
    ('pysteps_lk', 'pysteps LK extrapolation', ['pysteps_lk_extrap'], False),
    ('pysteps_sprog', 'pysteps S-PROG', ['pysteps_sprog'], False),
    ('as_released', 'ConvLSTM (as released)',
     [f'convlstm_rainnet_seed{s}' for s in range(3)], True),
    ('ed_mse', 'ConvLSTM-ED (MSE)',
     [f'convlstm_ed_mse_rainnet_seed{s}' for s in range(3)], True),
    ('unet_mse', 'U-Net (MSE)',
     [f'unet_mse_rainnet_seed{s}' for s in range(3)], True),
    ('ed_balanced', 'ConvLSTM-ED (balanced)',
     [f'convlstm_ed_balanced_rainnet_seed{s}' for s in range(3)], True),
    ('unet_balanced', 'U-Net (balanced)',
     [f'unet_balanced_rainnet_seed{s}' for s in range(3)], True),
    ('simvp', 'SimVP v1 (MSE)',
     [f'simvp_mse_rainnet_seed{s}' for s in range(3)], True),
    ('tau', 'TAU (SimVP v2 + attention)',
     [f'tau_ddr_rainnet_seed{s}' for s in range(3)], True),
]
REGRESSION_EXCLUDE = {'identity', 'obs_stats'}
# the as-released seed0 eval predates the per-seed naming convention
EVAL_JSON_ALIAS = {'convlstm_rainnet_seed0': 'eval_convlstm.json'}

PAIRS = [
    ('simvp', 'ed_mse'), ('simvp', 'unet_mse'), ('ed_mse', 'unet_mse'),
    ('ed_mse', 'as_released'), ('ed_mse', 'optical_flow'),
    ('ed_mse', 'ed_balanced'), ('unet_mse', 'unet_balanced'),
    ('tau', 'simvp'), ('tau', 'ed_mse'),
    # appended after the original pairs so their per-pair bootstrap seeds
    # (a.seed + 7919 * index) stay unchanged
    ('pysteps_sprog', 'pysteps_lk'), ('pysteps_sprog', 'optical_flow'),
    ('simvp', 'pysteps_sprog'),
]


def hss(h, m, f, c):
    denom = ((h + m) * (f + c) + (h + f) * (m + c))
    return 2.0 * (h * c - m * f) / denom if denom else float('nan')


def day_block_ids(file_idx, t0):
    return (file_idx.astype(np.int64) * 1000 + t0 // 24)


def block_bootstrap(values, blocks, reps=2000, seed=20261021):
    """Resample day blocks with replacement; returns stats of the mean.

    Windows inside a resampled block move together (stride-1 windows are
    strongly autocorrelated within a day). Repeated blocks contribute their
    windows once per draw. Deterministic via the fixed seed."""
    values = np.asarray(values, dtype=np.float64)
    uniq, inverse = np.unique(np.asarray(blocks), return_inverse=True)
    n_blocks = len(uniq)
    # window indices grouped by block, for fast multiset construction
    order = np.argsort(inverse, kind='stable')
    bounds = np.searchsorted(inverse[order], np.arange(n_blocks + 1))
    win_of_block = [order[bounds[i]:bounds[i + 1]] for i in range(n_blocks)]
    block_means = np.array([values[w].mean() for w in win_of_block])
    block_sizes = np.array([len(w) for w in win_of_block], dtype=np.float64)
    rng = np.random.RandomState(seed)
    point = float(values.mean())
    boots = np.empty(reps)
    for r in range(reps):
        pick = rng.randint(0, n_blocks, n_blocks)
        boots[r] = np.average(block_means[pick], weights=block_sizes[pick])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'point': point, 'ci95': (float(lo), float(hi)),
            'n_blocks': int(n_blocks), 'boots': boots}


def paired_bootstrap(diff, blocks, reps=2000, seed=20261021):
    """Same block bootstrap on a paired per-window difference array; two-sided
    bootstrap p from the share of resampled means on each side of zero."""
    res = block_bootstrap(diff, blocks, reps=reps, seed=seed)
    boots = res['boots']
    frac_le = float(np.mean(boots <= 0.0))
    frac_ge = float(np.mean(boots >= 0.0))
    p = 2.0 * min(frac_le, frac_ge)
    res['p_two_sided'] = float(min(1.0, max(1.0 / len(boots), p)))
    return res


def fss_from_sums(d2, f2, o2):
    # asarray so scalar inputs also take the numpy divide path (nan, no raise)
    d2 = np.asarray(d2, dtype=np.float64)
    f2 = np.asarray(f2, dtype=np.float64)
    o2 = np.asarray(o2, dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1.0 - d2 / (f2 + o2)


def csi_tables(cont):
    """cont[L, T, 4] -> per-(lead, thr) CSI/POD/FAR/HSS (nan-aware), the exact
    formulas of rainnet_nowcast_eval.py."""
    n_lead, n_thr, _ = cont.shape
    csi = np.full((n_lead, n_thr), np.nan)
    pod = np.full((n_lead, n_thr), np.nan)
    far = np.full((n_lead, n_thr), np.nan)
    hss_t = np.full((n_lead, n_thr), np.nan)
    for l in range(n_lead):
        for t in range(n_thr):
            h, m, f, c = cont[l, t]
            if h + m + f:
                csi[l, t] = h / (h + m + f)
            if h + m:
                pod[l, t] = h / (h + m)
            if h + f:
                far[l, t] = f / (h + f)
            hss_t[l, t] = hss(h, m, f, c)
    return csi, pod, far, hss_t


def load_family(ext_dir, tags, require_all=True):
    out = []
    for tag in tags:
        p = Path(ext_dir) / f'ext_{tag}.npz'
        if not p.exists():
            if require_all:
                return None
            continue
        d = np.load(p, allow_pickle=False)
        out.append(d)
    return out or None


def regression_check(d, ev, unit_scale):
    """Compare one ext pass against its pre-existing eval json."""
    rows = []
    mse_norm = float(d['se_total']) / int(d['n_seen'])
    rows.append(('overall_mse_norm', mse_norm, float(ev['overall_mse']),
                 abs(mse_norm - float(ev['overall_mse'])) <= 2e-7))
    mse_mm2 = mse_norm * unit_scale ** 2
    rows.append(('overall_mse_mm2', mse_mm2, float(ev['overall_mse_mm2']),
                 abs(mse_mm2 - float(ev['overall_mse_mm2'])) <= 2e-3))
    rmse = float(np.sqrt(mse_norm)) * unit_scale
    rows.append(('overall_rmse_mm', rmse, float(ev['overall_rmse_mm']),
                 abs(rmse - float(ev['overall_rmse_mm'])) <= 2e-3))
    mae_mm = float(d['lead_ae'].sum()) / int(d['n_seen']) * unit_scale
    rows.append(('overall_mae_mm', mae_mm, float(ev['overall_mae_mm']),
                 abs(mae_mm - float(ev['overall_mae_mm'])) <= 2e-3))
    csi, _, _, hss_t = csi_tables(d['cont'])
    for t, erow in zip(range(len(ev['thresholds'])), ev['thresholds']):
        ext_csi = float(np.nanmean(csi[:, t]))
        rows.append((f"csi_mean@{erow['threshold_mm']}mm", ext_csi,
                     float(erow['csi_mean']),
                     abs(ext_csi - float(erow['csi_mean'])) <= 1.1e-3))
        ext_hss = float(np.nanmean(hss_t[:, t]))
        rows.append((f"hss_mean@{erow['threshold_mm']}mm", ext_hss,
                     float(erow['hss_mean']),
                     abs(ext_hss - float(erow['hss_mean'])) <= 1.1e-3))
    return rows


def train_val_overlap(index):
    """Reproduce the loader's seeded val carve and quantify temporal overlap
    between the selection (val) set and the training set. Returns a dict."""
    train = [tuple(w) for w in index['train']]
    order = list(range(len(train)))
    random.Random(2026).shuffle(order)
    val_pos = set(order[:750])
    val_windows = [train[i] for i in sorted(val_pos)]
    train_windows = [train[i] for i in order if i not in val_pos]
    by_file = {}
    for fi, t0 in train_windows:
        by_file.setdefault(fi, []).append(t0)
    for fi in by_file:
        by_file[fi] = np.sort(np.asarray(by_file[fi]))
    n_overlap, overlap_counts, shared_frames = 0, [], 0
    for fi, t0 in val_windows:
        tset = by_file.get(fi)
        if tset is None:
            overlap_counts.append(0)
            continue
        near = tset[np.abs(tset - t0) < 20]
        overlap_counts.append(len(near))
        shared_frames += int(np.sum(20 - np.abs(near - t0)))
        if len(near):
            n_overlap += 1
    test_files = {w[0] for w in index['test']}
    train_files = {w[0] for w in train_windows}
    return {
        'n_train': len(train_windows), 'n_val': len(val_windows),
        'val_windows_overlapping_train_pct':
            round(100.0 * n_overlap / len(val_windows), 1),
        'mean_overlapping_train_windows_per_val_window':
            round(float(np.mean(overlap_counts)), 2),
        'max_overlapping_train_windows_per_val_window':
            int(np.max(overlap_counts)),
        'test_files_disjoint_from_train_files':
            bool(test_files.isdisjoint(train_files)),
        'total_shared_frames': shared_frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ext-dir', default='artifacts/ext_eval')
    ap.add_argument('--eval-dir', default='artifacts/ext_eval/eval_jsons')
    ap.add_argument('--index-json', default='artifacts/ext_eval/rainnet_index.json')
    ap.add_argument('--out-dir', default='artifacts/ext_eval')
    ap.add_argument('--reps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--sample-indices', default='667,865',
                    help='heavy,moderate window positions picked by the runner')
    a = ap.parse_args()
    sample_indices = [int(i) for i in a.sample_indices.split(',') if i.strip()]
    ext_dir, eval_dir, out_dir = Path(a.ext_dir), Path(a.eval_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = []
    summary = {'reps': a.reps, 'seed': a.seed,
               'regression': [], 'families': {}, 'pairs': {}, 'fss': {},
               'heavy_rain': {}, 'sample_windows': {}}

    # ---- alignment anchor + observed-rain stats -------------------------
    obs = np.load(ext_dir / 'ext_obs_stats.npz')
    file_idx, t0 = obs['file_idx'], obs['t0']
    blocks = day_block_ids(file_idx, t0)
    sizes = np.bincount(
        blocks - blocks.min())[np.bincount(blocks - blocks.min()) > 0]
    md.append(f"Test split: {len(file_idx)} windows, "
              f"{len(np.unique(blocks))} day blocks "
              f"(windows/block min {sizes.min()}, median "
              f"{int(np.median(sizes))}, max {sizes.max()}).")
    for i in sample_indices:
        summary['sample_windows'][str(i)] = {
            'max_obs_mm': round(float(obs['win_max_obs_mm'][i]), 2),
            'mean_obs_mm': round(float(obs['win_mean_obs_mm'][i]), 3),
            'file_idx': int(file_idx[i]), 't0': int(t0[i])}

    fam_data = {}
    for key, label, tags, _ in FAMILIES:
        fam = load_family(ext_dir, tags, require_all=False)
        if fam is None:
            md.append(f"family {label}: MISSING (not all tags present)")
            continue
        if len(fam) < len(tags):
            md.append(f"family {label}: PARTIAL "
                      f"({len(fam)}/{len(tags)} seeds present; CIs below use "
                      f"the available seeds only)")
        per_win = np.stack([d['win_mse_norm'] for d in fam])
        per_mae = np.stack([d['win_mae_norm'] for d in fam])
        for d in fam:
            assert np.array_equal(d['file_idx'], file_idx), \
                f'{tags}: window order mismatch'
            assert np.array_equal(d['t0'], t0), f'{tags}: t0 mismatch'
        unit_scale = float(json.loads(str(fam[0]['meta']))['unit_scale'])
        overall_mm2 = np.array(
            [float(d['se_total']) / int(d['n_seen']) for d in fam]) \
            * unit_scale ** 2
        fam_data[key] = {
            'label': label, 'tags': tags, 'per_win': per_win,
            'per_mae': per_mae, 'overall_mm2': overall_mm2,
            'unit_scale': unit_scale,
            'cont': np.sum([d['cont'] for d in fam], axis=0),
            'fss': [fss_from_sums(d['fss_d2'], d['fss_f2'], d['fss_o2'])
                    for d in fam],
            'n_seeds': len(fam),
            'meta': [json.loads(str(d['meta'])) for d in fam],
        }

    # ---- 1. regression vs pre-existing eval JSONs ------------------------
    md.append('\n## 1. Regression vs existing eval JSONs\n')
    md.append('| tag | metric | ext | eval json | ok |')
    md.append('|---|---|---|---|---|')
    n_fail = 0
    for key, label, tags, _ in FAMILIES:
        for tag in tags:
            p = ext_dir / f'ext_{tag}.npz'
            if not p.exists():
                continue
            if tag in REGRESSION_EXCLUDE:
                continue
            ej = eval_dir / EVAL_JSON_ALIAS.get(
                tag, f'eval_{tag}.json') if tag in EVAL_JSON_ALIAS \
                else eval_dir / f'eval_{tag}.json'
            if not ej.exists():
                md.append(f'| {tag} | (no eval json) | - | - | SKIPPED |')
                continue
            d = np.load(p)
            ev = json.loads(ej.read_text())
            rows = regression_check(d, ev,
                                    float(json.loads(str(d['meta']))['unit_scale']))
            for name, ext_v, json_v, ok in rows:
                if not ok:
                    n_fail += 1
                md.append(f'| {tag} | {name} | {ext_v:.7g} | {json_v:.7g} '
                          f'| {"PASS" if ok else "**FAIL**"} |')
            summary['regression'].append(
                {'tag': tag, 'ok': all(r[3] for r in rows),
                 'max_dev': max(abs(r[1] - r[2]) for r in rows)})
    md.append(f'\nRegression summary: '
              f'{sum(1 for r in summary["regression"] if r["ok"])} tags PASS, '
              f'{n_fail} individual metric FAILs.')

    # ---- 2. day-block bootstrap CIs --------------------------------------
    md.append('\n## 2. Day-block bootstrap CIs (family = seed-mean '
              'per-window MSE, mm^2)\n')
    md.append('| model | seeds | overall MSE mm^2 (per seed) | point | 95% CI |')
    md.append('|---|---|---|---|---|')
    for key, fam in fam_data.items():
        win_mm2 = fam['per_win'].mean(axis=0) * fam['unit_scale'] ** 2
        boot = block_bootstrap(win_mm2, blocks, a.reps, a.seed)
        per_seed = ', '.join(f'{x:.4f}' for x in fam['overall_mm2'])
        md.append(f"| {fam['label']} | {fam['n_seeds']} | {per_seed} "
                  f"| {boot['point']:.4f} "
                  f"| [{boot['ci95'][0]:.4f}, {boot['ci95'][1]:.4f}] |")
        summary['families'][key] = {
            'label': fam['label'], 'n_seeds': fam['n_seeds'],
            'per_seed_overall_mm2': [round(float(x), 4)
                                     for x in fam['overall_mm2']],
            'seed_mean_point_mm2': round(boot['point'], 4),
            'ci95_mm2': [round(boot['ci95'][0], 4), round(boot['ci95'][1], 4)],
            'n_blocks': boot['n_blocks']}

    # ---- 3. paired tests ---------------------------------------------------
    md.append('\n## 3. Paired per-window tests (A - B, mm^2, day-block '
              'bootstrap)\n')
    md.append('| A | B | mean diff | 95% CI | p (two-sided) |')
    md.append('|---|---|---|---|---|')
    for pair_i, (a_key, b_key) in enumerate(PAIRS):
        if a_key not in fam_data or b_key not in fam_data:
            continue
        fa, fb = fam_data[a_key], fam_data[b_key]
        diff = (fa['per_win'].mean(axis=0) - fb['per_win'].mean(axis=0)) \
            * fa['unit_scale'] ** 2
        # deterministic per-pair seed (hash() is not stable across runs)
        res = paired_bootstrap(diff, blocks, a.reps, a.seed + 7919 * pair_i)
        md.append(f"| {fa['label']} | {fb['label']} | {res['point']:+.4f} "
                  f"| [{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}] "
                  f"| {res['p_two_sided']:.4f} |")
        summary['pairs'][f'{a_key}_vs_{b_key}'] = {
            'A': fa['label'], 'B': fb['label'],
            'mean_diff_mm2': round(res['point'], 4),
            'ci95_mm2': [round(res['ci95'][0], 4), round(res['ci95'][1], 4)],
            'p_two_sided': round(res['p_two_sided'], 4),
            'significant_at_0.05': bool(res['ci95'][0] * res['ci95'][1] > 0)}

    # ---- 4. FSS ------------------------------------------------------------
    md.append('\n## 4. FSS (lead-mean; thresholds mm, neighbourhoods px)\n')
    meta0 = fam_data[list(fam_data)[0]]['meta'][0]
    fss_thr = meta0['fss_thresholds']
    fss_nbds = meta0['fss_nbds']
    md.append('| model | ' + ' | '.join(
        f'FSS@{thr}mm nbd{k}px' for thr in fss_thr for k in fss_nbds) + ' |')
    md.append('|---|' + '---|' * (len(fss_thr) * len(fss_nbds)))
    for key, fam in fam_data.items():
        fss_lead_mean = np.mean(fam['fss'], axis=0)  # over seeds AND leads
        vals = []
        for ti in range(len(fss_thr)):
            for ki in range(len(fss_nbds)):
                v = float(np.nanmean(fss_lead_mean[:, ti, ki]))
                vals.append(f'{v:.3f}')
                summary['fss'].setdefault(key, {})[
                    f'thr{fss_thr[ti]}mm_nbd{fss_nbds[ki]}px'] = round(v, 3)
        md.append(f"| {fam['label']} | " + ' | '.join(vals) + ' |')

    # ---- 5. heavy rain -----------------------------------------------------
    md.append('\n## 5. Heavy-rain CSI (5 mm / 10 mm), lead-mean and per '
              'lead (seed-summed counts)\n')
    md.append('| model | CSI@5mm mean | CSI@10mm mean | CSI@5mm +1h..+10h |')
    md.append('|---|---|---|---|')
    thr_list = meta0['csi_thresholds']
    for key, fam in fam_data.items():
        csi, _, _, _ = csi_tables(fam['cont'])
        t5 = thr_list.index(5.0) if 5.0 in thr_list else None
        t10 = thr_list.index(10.0) if 10.0 in thr_list else None
        c5 = csi[:, t5]
        c10 = csi[:, t10]
        md.append(f"| {fam['label']} | {np.nanmean(c5):.4f} "
                  f"| {np.nanmean(c10):.4f} "
                  f"| {', '.join(f'{x:.3f}' for x in c5)} |")
        summary['heavy_rain'][key] = {
            'csi5_mean': round(float(np.nanmean(c5)), 4),
            'csi10_mean': round(float(np.nanmean(c10)), 4),
            'csi5_per_lead': [round(float(x), 4) for x in c5],
            'csi10_per_lead': [round(float(x), 4) for x in c10]}

    # ---- 6. train/val overlap ---------------------------------------------
    if Path(a.index_json).exists():
        ov = train_val_overlap(json.loads(Path(a.index_json).read_text()))
        summary['train_val_overlap'] = ov
        md.append('\n## 6. Train/val temporal overlap (selection-set '
                  'disclosure)\n')
        md.append(f"- validation windows that share at least one frame with "
                  f"a train window: **{ov['val_windows_overlapping_train_pct']}%**")
        md.append(f"- mean overlapping train windows per val window: "
                  f"{ov['mean_overlapping_train_windows_per_val_window']}; "
                  f"max {ov['max_overlapping_train_windows_per_val_window']}")
        md.append(f"- test files disjoint from train files: "
                  f"{ov['test_files_disjoint_from_train_files']} "
                  f"(year-2018 test split is untouched by the carve)")

    (out_dir / 'EXT_ANALYSIS.md').write_text('\n'.join(md) + '\n',
                                             encoding='utf-8')
    (out_dir / 'ext_summary.json').write_text(
        json.dumps(summary, indent=1, default=str), encoding='utf-8')
    print('\n'.join(md))
    print(f"\nWROTE {out_dir / 'EXT_ANALYSIS.md'}")
    print(f"WROTE {out_dir / 'ext_summary.json'}")
    fails = [r for r in summary['regression'] if not r['ok']]
    print('REGRESSION_' + ('PASS' if not fails else f'FAIL: {fails}'))


if __name__ == '__main__':
    main()

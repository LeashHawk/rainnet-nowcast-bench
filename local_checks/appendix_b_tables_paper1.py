"""Render Paper 1 Appendix B tables (Markdown) from the final ext_summary.json.

Run: python local_checks/appendix_b_tables_paper1.py \
        artifacts/final_20261022/analysis/ext_summary.json > appendix_b.md
"""
import json
import sys
from pathlib import Path

s = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
fam = s['families']
order = list(fam)
out = []

out.append('**Table B1. Lead-mean FSS** (seed-mean for deep rows; neighborhoods '
           'are square side lengths in pixels).\n')
for thr in ('0.5', '1.0', '2.0'):
    nbds = [1, 3, 5, 9, 17]
    out.append(f'\n*Threshold {thr} mm*\n')
    out.append('| Model | ' + ' | '.join(f'{k} px' for k in nbds) + ' |')
    out.append('|---|' + '---:|' * len(nbds))
    cols = {k: [s['fss'][m][f'thr{thr}mm_nbd{k}px'] for m in order] for k in nbds}
    for m in order:
        cells = []
        for k in nbds:
            v = s['fss'][m][f'thr{thr}mm_nbd{k}px']
            cells.append(f'**{v:.3f}**' if v == max(cols[k]) else f'{v:.3f}')
        out.append(f"| {fam[m]['label']} | " + ' | '.join(cells) + ' |')

out.append('\n**Table B2. Heavy-rain CSI** (counts summed over seeds; lead-mean '
           'and per lead +1 h…+10 h).\n')
out.append('| Model | CSI@5 mm mean | CSI@10 mm mean | CSI@5 mm by lead | CSI@10 mm by lead |')
out.append('|---|---:|---:|---|---|')
for m in order:
    h = s['heavy_rain'][m]
    out.append(f"| {fam[m]['label']} | {h['csi5_mean']:.4f} | {h['csi10_mean']:.4f} "
               f"| {', '.join(f'{x:.3f}' for x in h['csi5_per_lead'])} "
               f"| {', '.join(f'{x:.3f}' for x in h['csi10_per_lead'])} |")

out.append(f"\n**Table B3a. Mean per-window MSE with day-block bootstrap 95% "
           f"intervals** ({s['reps']} resamples, seed {s['seed']}; deep rows use "
           f"the seed-mean per-window MSE).\n")
out.append('| Model | seeds | MSE (mm²) | 95% CI |')
out.append('|---|---:|---:|---|')
for m in order:
    f = fam[m]
    out.append(f"| {f['label']} | {f['n_seeds']} | {f['seed_mean_point_mm2']:.4f} "
               f"| [{f['ci95_mm2'][0]:.4f}, {f['ci95_mm2'][1]:.4f}] |")

out.append('\n**Table B3b. Paired per-window comparisons** (A − B, mm²; same '
           'block resampling; p < 0.001 is the bootstrap resolution limit).\n')
out.append('| A | B | mean diff | 95% CI | p | CI excludes 0 |')
out.append('|---|---|---:|---|---:|---|')
for p in s['pairs'].values():
    pv = '< 0.001' if p['p_two_sided'] <= 0.0005 else f"{p['p_two_sided']:.3f}"
    out.append(f"| {p['A']} | {p['B']} | {p['mean_diff_mm2']:+.4f} "
               f"| [{p['ci95_mm2'][0]:+.4f}, {p['ci95_mm2'][1]:+.4f}] | {pv} "
               f"| {'yes' if p['significant_at_0.05'] else 'no'} |")
print('\n'.join(out))

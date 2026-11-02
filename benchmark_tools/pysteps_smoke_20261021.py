"""[PYSTEPS-SMOKE-2026-10-21] Functional install verification (not science).

Synthetic Gaussian blob moving right at 1 px/step, 64x64, 5 frames:
  - LK motion field must point right (~ +1 px/step) where it rains;
  - semilagrangian extrapolation must advect the blob right;
  - S-PROG must produce a finite 3-step forecast.
Run with the server venv python; CPU only, a few seconds.
"""
import numpy as np

H = W = 64
T = 5
yy, xx = np.mgrid[0:H, 0:W]
precip = np.zeros((T, H, W), dtype=np.float32)
for t in range(T):
    cx = 10 + 1.0 * t
    precip[t] = 5.0 * np.exp(-((xx - cx) ** 2 + (yy - 20.0) ** 2)
                             / (2 * 4.0 ** 2))
precip[precip < 0.1] = 0.0

import pysteps  # noqa: E402
from pysteps import extrapolation, motion, nowcasts  # noqa: E402

print('pysteps at', pysteps.__file__)
assert precip.max() > 1.0

lk = motion.get_method('LK')
vfield = lk(precip)
assert vfield.shape == (2, H, W), vfield.shape
# pysteps convention (motion.lucaskanade docstring): out[0] = x-components,
# out[1] = y-components, in pixels/timestep
vx_blob = float(np.mean(vfield[0][precip[-1] > 1.0]))
vy_blob = float(np.mean(vfield[1][precip[-1] > 1.0]))
print(f'LK mean (vx, vy) at blob: ({vx_blob:+.3f}, {vy_blob:+.3f}) '
      f'(expect ~ +1, ~0)')
assert 0.4 < vx_blob < 2.2, 'LK x-motion wrong (pysteps out[0] is vx)'
assert abs(vy_blob) < 0.5, 'LK y-motion should be ~0 for horizontal motion'

semilag = extrapolation.get_method('semilagrangian')
fcast = semilag(precip[-1], vfield, 3)
assert fcast.shape == (3, H, W), fcast.shape
# outval=NaN fills border pixels whose parcels originate outside the grid,
# so the peak search must be NaN-aware (argmax would just pick the NaN)
t3 = fcast[2]
peak_row, peak_col = np.unravel_index(np.nanargmax(t3), t3.shape)
t0_peak_col = int(np.argmax(precip[-1].max(axis=0)))
print(f'semilag peak column t+0: {t0_peak_col} -> t+3: {peak_col} '
      f'(expect +3); t+3 max {np.nanmax(t3):.2f}')
assert peak_col >= t0_peak_col + 2, 'semilagrangian did not advect right'

sprog = nowcasts.get_method('sprog')
sf = sprog(precip, vfield, 3, n_cascade_levels=3, ar_order=2,
           precip_thr=0.1, num_workers=1)
assert sf.shape == (3, H, W), sf.shape
frac_finite = float(np.isfinite(sf).mean())
print(f'S-PROG t+3 max: {np.nanmax(sf[2]):.2f}; finite fraction '
      f'{frac_finite:.1%} (advection borders are NaN by design)')
assert frac_finite > 0.9 and np.isfinite(np.nanmax(sf)), 'S-PROG output bad'
print('PYSTEPS_SMOKE_ALL_PASS')

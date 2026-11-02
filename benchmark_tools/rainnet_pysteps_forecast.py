"""[PYSTEPS-FORECAST-2026-10-21] RainNet-window pysteps baselines.

Two deterministic baselines (user-selected scope; no STEPS/CRPS):
  lk_extrap_forecast  Lucas-Kanade feature flow (cv2, pysteps 'LK') computed
                      from the last t_flow observed hourly frames, then
                      semi-Lagrangian extrapolation of the last frame
                      (Lagrangian persistence: advects rain, no growth/decay);
  sprog_forecast      the same LK motion driving pysteps' deterministic S-PROG
                      (Gaussian band-pass cascade + lag-AR innovation decay,
                      CDF-matched to the last observed field).

Interface: (B, T, H, W) float32 fields in mm (RainNet hourly accumulation =
normalized * 140) -> (B, n_steps, H, W) float32 in mm.

pysteps 1.21.5 conventions (verified against the installed sources and the
server smoke test, queue_bundle/pysteps_smoke_20261021.py):
  - motion.get_method('LK') -> (2, H, W) with out[0] = x-component and
    out[1] = y-component, pixels per timestep;
  - nowcasts 'sprog' consumes exactly ar_order+1 trailing frames;
  - semi-Lagrangian extrapolation fills border pixels (parcels originating
    from outside the grid) with outval; lk_extrap_forecast passes outval=0.0
    (no rain advected in, the pysteps-example convention), so its forecasts
    stay finite everywhere;
  - S-PROG (pysteps 1.21.5) fixes outval='min' positionally in its internal
    Lagrangian alignment (nowcasts/sprog.py:246), so outval MUST NOT be
    forwarded via extrap_kwargs (TypeError: multiple values); its final
    forecast loop then leaves border pixels NaN by design, and we fill them
    with 0.0 after the call - the same no-rain-advected-in convention;
  - S-PROG band-pass output can go negative; negative rain is unphysical, so
    sprog forecasts are clipped at zero (pysteps postprocessing convention).

pysteps is imported lazily inside both functions so this module (and its
importers) load fine on machines without pysteps; local unit tests never call
these functions.
"""

import numpy as np


def _flow(seq, fd_kwargs):
    from pysteps import motion
    return motion.get_method('LK')(seq, fd_kwargs=fd_kwargs)


def lk_extrap_forecast(inputs_mm, n_steps, params):
    """params: {'t_flow': int >= 2, 'fd_kwargs': dict or None (pysteps
    defaults)}. Returns (B, n_steps, H, W) mm, finite (outval=0.0)."""
    from pysteps import extrapolation
    b = inputs_mm.shape[0]
    out = np.empty((b, n_steps, *inputs_mm.shape[2:]), dtype=np.float32)
    t_flow = max(2, int(params.get('t_flow', 10)))
    fd_kwargs = params.get('fd_kwargs')
    for i in range(b):
        seq = np.ascontiguousarray(inputs_mm[i, -t_flow:], dtype=np.float32)
        v = _flow(seq, fd_kwargs)
        out[i] = extrapolation.get_method('semilagrangian')(
            seq[-1], v, n_steps, outval=0.0)
    return out


def sprog_forecast(inputs_mm, n_steps, params):
    """params: LK entries as above plus {'ar_order': 1|2,
    'n_cascade_levels': int, 'precip_thr': mm}. Returns (B, n_steps, H, W) mm,
    finite, clipped at zero."""
    from pysteps import nowcasts
    b = inputs_mm.shape[0]
    out = np.empty((b, n_steps, *inputs_mm.shape[2:]), dtype=np.float32)
    t_flow = max(2, int(params.get('t_flow', 10)))
    fd_kwargs = params.get('fd_kwargs')
    ar_order = int(params.get('ar_order', 2))
    n_cascade = int(params.get('n_cascade_levels', 6))
    precip_thr = float(params.get('precip_thr', 0.1))
    for i in range(b):
        seq = np.ascontiguousarray(inputs_mm[i, -t_flow:], dtype=np.float32)
        v = _flow(seq, fd_kwargs)
        # outval is NOT forwarded: pysteps 1.21.5 sprog passes outval='min'
        # positionally in its Lagrangian alignment (sprog.py:246) and the
        # final loop's semilagrangian default leaves border pixels NaN; fill
        # them with 0.0 here - the same convention as lk_extrap_forecast.
        sf = nowcasts.get_method('sprog')(
            seq[-(ar_order + 1):], v, n_steps,
            n_cascade_levels=n_cascade, ar_order=ar_order,
            precip_thr=precip_thr, num_workers=1)
        out[i] = np.maximum(np.nan_to_num(sf, nan=0.0), 0.0)
    return out

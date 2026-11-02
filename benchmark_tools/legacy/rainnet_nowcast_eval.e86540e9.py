"""[EVAL-2026-10-16] Paper 1 (RainNet precipitation nowcasting) evaluation.

Computes, on the held-out test split (year 2018, 1500 windows):
  - MSE / MAE overall and per lead time. [FIX-LEAD-UNIT-2026-10-17] The RainNet
    lowres maps are HOURLY (720 per month), so the 10 forecast steps are
    +1h..+10h; earlier drafts mislabeled them "+10..+50 min at 5-min frames"
    and printed only the first 5 leads;
  - physical units alongside the normalized values: mm = normalized *
    unit_scale, so MSE_mm2 = normalized MSE * unit_scale^2 and
    RMSE_mm = sqrt(normalized MSE) * unit_scale;
  - categorical nowcasting scores CSI / POD / FAR / HSS at rain-rate thresholds
    (mm, converted from the loader's normalized values via --unit-scale, default 140
    = the build recipe clip[0,140]/140);
  - data-range report so the unit conversion can be verified before thresholds are
    trusted (run --model persistence --report-only first).

Models:
  mamep         PaddedModel, loads a checkpoint;
  convlstm      family config from main_convLSTM_rainnet.py, loads a checkpoint;
  simvp         [NEW-2026-10-19] SimVP v1 (IncepU), loads a checkpoint;
  persistence   Eulerian persistence: last input frame repeated across the horizon;
  optical_flow  Lagrangian persistence: pure-numpy single-level Lucas-Kanade flow
                averaged over the last 3 observed frame pairs + semi-Lagrangian
                extrapolation of the last frame
                (deliberately skill-free: no growth/decay model, flow computed on
                sqrt-brightened fields, |v| clipped at 8 px per hourly step).
                CPU only, no external dependencies (pysteps-free by design).

Multi-channel outputs (the ConvLSTM family ends in 12 channels) are reduced by a
channel-dim mean before the categorical scores, so CSI etc. see a single rain field.

CPU-runnable; uses CUDA when available and idle for the checkpoint models. Output:
JSON file + printed markdown.

Usage (from anywhere; sys.path bootstrap resolves the sibling corrected/ tree):
  python rainnet_nowcast_eval.py --data-path $B/data/rainnet --model mamep \
      --checkpoints $B/corrected/checkpoints/mamep_rainnet.pth --out $B/logs/eval_mamep.json
  python rainnet_nowcast_eval.py --data-path $B/data/rainnet --model optical_flow \
      --out $B/logs/eval_optical_flow.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'corrected'))

from dataloader.dataloader_rainnet_fixed import load_data  # noqa: E402


# ---------------------------------------------------------------------------
# [EVAL-OFLOW-2026-10-16] pure-numpy optical-flow nowcast (Lagrangian persistence)
# ---------------------------------------------------------------------------
def _box(a, r):
    """(2r+1)x(2r+1) mean filter with edge padding via an integral image; output
    shape == input shape. Only a leading zero row/col may be added to the
    integral image: a trailing one shifts the window sums (v2 draft bug)."""
    k = 2 * r + 1
    pad = np.pad(a.astype(np.float64), r, mode='edge')
    c = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1))
    c[1:, 1:] = pad.cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / float(k * k)


def _bilinear_sample(img, ys, xs):
    """Sample img (H, W) at float coordinates (H', W') grids with edge clamping."""
    h, w = img.shape
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    wy = ys - y0
    wx = xs - x0
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    return (img[y0, x0c] * (1 - wy) * (1 - wx) + img[y0, x1c] * (1 - wy) * wx
            + img[y1, x0c] * wy * (1 - wx) + img[y1, x1c] * wy * wx)


def lucas_kanade_flow(prev, nxt, win=7, max_v=8.0):
    """Single-level Lucas-Kanade optical flow (px per step) on sqrt-brightened
    radar fields. prev/nxt: (H, W) normalized rain fields."""
    a = np.sqrt(np.maximum(prev, 0.0))
    b = np.sqrt(np.maximum(nxt, 0.0))
    a = _box(a, 1)
    b = _box(b, 1)
    # gradients of the mid-frame; np.gradient is one-sided at borders
    # (np.roll would wrap the opposite edge into the derivative)
    iy, ix = np.gradient(0.5 * (a + b))
    it = b - a
    sxx = _box(ix * ix, win)
    syy = _box(iy * iy, win)
    sxy = _box(ix * iy, win)
    sxt = _box(ix * it, win)
    syt = _box(iy * it, win)
    det = sxx * syy - sxy * sxy
    # Scale-invariant validity: windowed means of normalized rain fields are tiny,
    # so an absolute det cutoff zeroes every vector. Require texture (trace) and a
    # non-degenerate structure tensor (no aperture problem) relative to its trace.
    trace = sxx + syy
    valid = (trace > 1e-12) & (det > 1e-3 * trace * trace)
    det = np.where(valid, det, np.nan)
    vx = -(syy * sxt - sxy * syt) / det
    vy = -(sxx * syt - sxy * sxt) / det
    vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
    vy = np.nan_to_num(vy, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.hypot(vx, vy)
    scale = np.where(speed > max_v, max_v / np.maximum(speed, 1e-9), 1.0)
    return vy * scale, vx * scale


def optical_flow_nowcast(inputs, steps=10):
    """inputs: (B, T, 1, H, W) normalized. Returns (B, steps, 1, H, W) numpy:
    Lagrangian persistence — LK flow on the last observed pair, semi-Lagrangian
    backward extrapolation of the last frame."""
    b, t, _, h, w = inputs.shape
    out = np.zeros((b, steps, 1, h, w), dtype=np.float32)
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing='ij')
    for i in range(b):
        f = inputs[i, :, 0]
        # flow averaged over the last 3 observed pairs (single-pair LK is noisy)
        pairs = [lucas_kanade_flow(f[j - 1], f[j]) for j in range(t - 3, t)]
        vy = np.mean([p[0] for p in pairs], axis=0)
        vx = np.mean([p[1] for p in pairs], axis=0)
        last = f[t - 1]
        for s in range(1, steps + 1):
            out[i, s - 1, 0] = _bilinear_sample(
                last, ys - vy * s, xs - vx * s)
    return out


def build_model(kind):
    if kind == 'mamep':
        from main_mamep_rainnet import PaddedModel
        return PaddedModel((10, 1, 208, 333))
    if kind == 'convlstm':
        from model_ConvLSTM import ConvLSTM
        return ConvLSTM(input_dim=1, hidden_dim=[32, 128, 64, 12],
                       kernel_size=(3, 3), num_layers=4, batch_first=True, bias=True,
                       return_all_layers=False)
    if kind in ('convlstm_ed', 'unet', 'simvp', 'tau'):
        from model_nowcast import build
        return build(kind)
    return None


def reduce_channels(x):
    """(B, T, C, H, W) -> (B, T, H, W) with a channel-mean for C > 1."""
    if x.dim() == 5:
        if x.size(2) == 1:
            return x[:, :, 0]
        return x.mean(dim=2)
    return x


def hss(hits, misses, false, correct_neg):
    denom = ((hits + misses) * (false + correct_neg)
             + (hits + false) * (misses + correct_neg))
    if denom == 0:
        return float('nan')
    return 2.0 * (hits * correct_neg - misses * false) / denom


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--model', required=True,
                   choices=['mamep', 'convlstm', 'convlstm_ed', 'unet', 'simvp', 'tau',
                            'persistence', 'optical_flow'])
    p.add_argument('--checkpoints', default=None)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--unit-scale', type=float, default=140.0,
                   help='mm = normalized value * scale (build recipe: clip[0,140]/140)')
    p.add_argument('--thresholds', default='0.1,0.5,1,2,5,10',
                   help='rain-rate thresholds in mm')
    p.add_argument('--report-only', action='store_true',
                   help='stop after the data-range report')
    p.add_argument('--out', default=None)
    a = p.parse_args()

    thresholds = [float(t) for t in a.thresholds.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, test_loader, _, _ = load_data(root=a.data_path, batch_size=a.batch_size)

    model = None
    if a.model in ('mamep', 'convlstm', 'convlstm_ed', 'unet', 'simvp', 'tau'):
        if not a.checkpoints:
            raise SystemExit('--checkpoints required for trained models')
        model = build_model(a.model).to(device)
        model.load_state_dict(torch.load(a.checkpoints, map_location=device))
        model.eval()

    n_lead = None
    se, n_seen = 0.0, 0
    lead_se, lead_ae = None, None
    obs_max_mm = 0.0
    # contingency per (lead, threshold): hits, misses, false alarms, correct negatives
    cont = {}

    with torch.no_grad():
        for inputs, targets in test_loader:
            targets = targets.to(device)
            if a.model == 'persistence':
                inputs = inputs.to(device)
                outputs = inputs[:, -1:, :, :, :].expand(-1, targets.size(1), -1, -1, -1)
            elif a.model == 'optical_flow':
                outputs = torch.from_numpy(
                    optical_flow_nowcast(inputs.to('cpu', torch.float32).numpy()))
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
            pred = reduce_channels(outputs).to('cpu', torch.float32)
            obs = reduce_channels(targets).to('cpu', torch.float32)

            if n_lead is None:
                n_lead = obs.size(1)
                lead_se = np.zeros(n_lead)
                lead_ae = np.zeros(n_lead)
            pred_mm = pred.numpy() * a.unit_scale
            obs_mm = obs.numpy() * a.unit_scale
            obs_max_mm = max(obs_max_mm, float(obs_mm.max()))

            se += float(((obs - pred) ** 2).sum())
            lead_se += ((obs - pred) ** 2).sum(dim=(0, 2, 3)).numpy()
            lead_ae += (obs - pred).abs().sum(dim=(0, 2, 3)).numpy()
            n_seen += obs.numel()

            for t_idx, thr in enumerate(thresholds):
                po = pred_mm >= thr
                oo = obs_mm >= thr
                hits = np.logical_and(po, oo)
                for lead in range(n_lead):
                    key = (lead, t_idx)
                    h, m, f, c = cont.get(key, (0, 0, 0, 0))
                    h += int(hits[:, lead].sum())
                    m += int(np.logical_and(~po[:, lead], oo[:, lead]).sum())
                    f += int(np.logical_and(po[:, lead], ~oo[:, lead]).sum())
                    c += int(np.logical_and(~po[:, lead], ~oo[:, lead]).sum())
                    cont[key] = (h, m, f, c)

    if n_lead is None:
        print('empty test split; nothing to report')
        return
    # [FIX-LEADMSE-2026-10-17] per-lead divisor = all windows x H x W.
    # lead_se/lead_ae accumulate across batches, so dividing by a single
    # batch's pixel count (the v1/v2 draft) inflated lead_mse by ~#batches.
    pixels = n_seen / n_lead
    print(f'MODEL={a.model} windows={len(test_loader.dataset)} device={device}')
    print(f'DATA_RANGE observed max = {obs_max_mm:.2f} mm (unit_scale={a.unit_scale})')
    print('NOTE: verify the observed max against the expected rain-rate range of '
          'RainNet lowres before trusting threshold scores.')
    if a.report_only:
        return

    mse = se / n_seen
    mae = float(lead_ae.sum()) / n_seen
    scale2 = a.unit_scale ** 2
    mse_mm2 = mse * scale2
    rmse_mm = float(np.sqrt(mse)) * a.unit_scale
    lead_mse = [float(x / pixels) for x in lead_se]
    lead_mse_mm2 = [x * scale2 for x in lead_mse]
    print(f'OVERALL MSE={mse:.7f} MAE={mae:.7f}  (normalized, unit_scale={a.unit_scale})')
    print(f'OVERALL MSE={mse_mm2:.4f} mm^2  RMSE={rmse_mm:.4f} mm  '
          f'MAE={mae * a.unit_scale:.4f} mm')
    print('LEADS  ' + ' | '.join(f'+{i + 1}h' for i in range(n_lead)))
    print('MSE    ' + ' | '.join(f'{x:.3f}' for x in lead_mse_mm2) + '   (mm^2)')
    print('RMSE   ' + ' | '.join(f'{np.sqrt(x):.3f}' for x in lead_mse_mm2) + '   (mm)')

    rows = []
    for t_idx, thr in enumerate(thresholds):
        csi_l, pod_l, far_l, hss_l = [], [], [], []
        for lead in range(n_lead):
            h, m, f, c = cont[(lead, t_idx)]
            csi_l.append(h / (h + m + f) if (h + m + f) else float('nan'))
            pod_l.append(h / (h + m) if (h + m) else float('nan'))
            far_l.append(f / (h + f) if (h + f) else float('nan'))
            hss_l.append(hss(h, m, f, c))
        rows.append({
            'threshold_mm': thr,
            'csi_per_lead': [round(x, 4) for x in csi_l],
            'csi_mean': round(float(np.nanmean(csi_l)), 4),
            'pod_mean': round(float(np.nanmean(pod_l)), 4),
            'far_mean': round(float(np.nanmean(far_l)), 4),
            'hss_mean': round(float(np.nanmean(hss_l)), 4),
        })

    print()
    print('| thr(mm) | CSI(mean) | POD | FAR | HSS | CSI per lead: '
          + ' | '.join(f'+{i + 1}h' for i in range(n_lead)) + ' |')
    print('|---|' + '---|' * (4 + n_lead))
    for r in rows:
        lead_all = ', '.join(f'{x:.3f}' for x in r['csi_per_lead'])
        print(f"| {r['threshold_mm']} | {r['csi_mean']} | {r['pod_mean']} | "
              f"{r['far_mean']} | {r['hss_mean']} | {lead_all} |")

    result = {
        'model': a.model, 'checkpoints': a.checkpoints, 'unit_scale': a.unit_scale,
        'windows': len(test_loader.dataset), 'device': device,
        'observed_max_mm': round(obs_max_mm, 2),
        'overall_mse': round(mse, 7),
        'overall_mae': round(mae, 7),
        'overall_mse_mm2': round(mse_mm2, 4),
        'overall_rmse_mm': round(rmse_mm, 4),
        'overall_mae_mm': round(mae * a.unit_scale, 4),
        'lead_hours': list(range(1, n_lead + 1)),
        'lead_mse': [round(x, 7) for x in lead_mse],
        'lead_mae': [round(float(x / pixels), 7) for x in lead_ae],
        'lead_mse_mm2': [round(x, 4) for x in lead_mse_mm2],
        'lead_rmse_mm': [round(float(np.sqrt(x)), 4) for x in lead_mse_mm2],
        'thresholds': rows,
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=1))
        print(f'\nWROTE {a.out}')


if __name__ == '__main__':
    main()

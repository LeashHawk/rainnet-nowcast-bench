"""[EXT-EVAL-2026-10-21] Paper 1 (RainNet) extended evaluator.

One model per invocation, ONE pass over the fixed test split in the loader's
canonical order (shuffle=False), so window i is the SAME window for every
model and paired per-window comparisons across models are valid.

Per pass it records:
  per-window metrics: win_mse_norm, win_mae_norm (normalized; mm = x * unit_scale)
                      win_max_obs_mm, win_mean_obs_mm
  window identity:    file_idx, t0  (day block = (file_idx, t0 // 24));
                      t0 is the first of the 20 hourly frames, 720 per month
  per-lead pixel sums: lead_se, lead_ae (same accumulation order as
                      rainnet_nowcast_eval.py, for regression parity)
  contingency:        cont[L, T, 4] = hits, misses, false alarms, correct negatives
  FSS accumulators:   fss_d2, fss_f2, fss_o2 [L, TF, NB]; FSS = 1 - d2 / (f2 + o2)
                      with a valid-area (no padding) uniform filter per nbd
  sample dumps:       ext_<tag>_win<i>.npz with inp / pred / obs fields
  scalars:            se, n_seen, observed_max_mm + a meta json blob

Overall printed numbers use the same formulas as rainnet_nowcast_eval.py, so the
existing eval JSONs double as a regression target (same checkpoint + batch size
=> overall_mse must agree to <=1e-9 normalized / ~1e-4 mm^2 cross-GPU noise).

Verification hooks (used by local_checks/test_extended_eval.py):
  --model identity   outputs == targets: MSE=0, MAE=0, CSI/POD/HSS=1 where
                     events exist, FSS=1, win_mse=0. Never a paper row.
  --build-kwargs '{"t_in":10,...}'  escape hatch to build a wired arch at a
                     non-RainNet geometry for unit tests. Production never
                     passes it; the default build IS the published geometry.

Usage:
  python rainnet_extended_eval.py --data-path $B/data/rainnet --model simvp \
      --checkpoints .../simvp_mse_rainnet_seed0.pth --tag simvp_mse_rainnet_seed0
  python rainnet_extended_eval.py --data-path $B/data/rainnet --obs-only --tag obs_stats
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_CODE_CANDIDATES = [_HERE.parent / 'corrected', _HERE.parent / 'repro_corrected', _HERE]
for _c in _CODE_CANDIDATES:
    if (_c / 'dataloader' / 'dataloader_rainnet_fixed.py').exists():
        sys.path.insert(0, str(_c))
        _CODE_DIR = _c
        break
else:
    raise SystemExit('cannot locate the corrected/ code tree next to this script')

from dataloader.dataloader_rainnet_fixed import load_data  # noqa: E402
from rainnet_nowcast_eval import optical_flow_nowcast, reduce_channels  # noqa: E402

CSI_THRESHOLDS_DOC = '0.1,0.5,1,2,5,10'  # same defaults as rainnet_nowcast_eval.py


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def fss_sums(pred_bin, obs_bin, k):
    """Valid-area uniform-filter FSS sums for one batch.

    pred_bin/obs_bin: (B, L, H, W) float tensors (0/1). Returns (d2, f2, o2),
    each (L,) float64 summed over windows and valid neighborhood centers:
      FSS = 1 - sum((P_f - P_o)^2) / (sum(P_f^2) + sum(P_o^2)).
    Valid area only (conv2d without padding), so every neighborhood center sees
    exactly k*k pixels; no border correction is needed."""
    b, l, h, w = pred_bin.shape
    kern = torch.full((1, 1, k, k), 1.0 / (k * k), dtype=pred_bin.dtype,
                      device=pred_bin.device)
    pf = F.conv2d(pred_bin.reshape(b * l, 1, h, w), kern)
    po = F.conv2d(obs_bin.reshape(b * l, 1, h, w), kern)
    d2 = ((pf - po) ** 2).sum(dim=(1, 2, 3)).reshape(b, l).sum(0)
    f2 = (pf ** 2).sum(dim=(1, 2, 3)).reshape(b, l).sum(0)
    o2 = (po ** 2).sum(dim=(1, 2, 3)).reshape(b, l).sum(0)
    return (d2.double().cpu().numpy(), f2.double().cpu().numpy(),
            o2.double().cpu().numpy())


def build_ext_model(kind, build_kwargs):
    """Same dispatch as rainnet_nowcast_eval.build_model, plus the --build-kwargs
    test escape hatch. mamep is deliberately absent: no MambaEP checkpoint exists
    for RainNet (deferred as infeasible; cost row only)."""
    if kind in ('convlstm_ed', 'unet', 'simvp', 'tau'):
        from model_nowcast import build
        return build(kind, **build_kwargs)
    if kind == 'convlstm':
        from model_ConvLSTM import ConvLSTM
        return ConvLSTM(input_dim=1, hidden_dim=[32, 128, 64, 12],
                        kernel_size=(3, 3), num_layers=4, batch_first=True, bias=True,
                        return_all_layers=False)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--model', default=None,
                   choices=['convlstm', 'convlstm_ed', 'unet', 'simvp', 'tau',
                            'persistence', 'optical_flow', 'identity',
                            'pysteps_lk_extrap', 'pysteps_sprog'])
    p.add_argument('--checkpoints', default=None)
    p.add_argument('--build-kwargs', default='{}',
                   help='test-only: extra build() kwargs for the wired archs')
    p.add_argument('--tag', required=True)
    p.add_argument('--out-dir', default=str(_HERE.parent / 'logs' / 'ext_eval'))
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--unit-scale', type=float, default=140.0)
    p.add_argument('--csi-thresholds', default=CSI_THRESHOLDS_DOC)
    p.add_argument('--fss-thresholds', default='0.5,1,2')
    p.add_argument('--fss-nbds', default='1,3,5,9,17')
    p.add_argument('--sample-indices', default='',
                   help='comma-separated test-window positions to dump fields for')
    p.add_argument('--limit', type=int, default=0,
                   help='debug: only the first N test windows')
    p.add_argument('--obs-only', action='store_true',
                   help='no model: record per-window observed-rain stats only')
    p.add_argument('--split', default='test', choices=['test', 'val'],
                   help='test = the fixed year-2018 test split (all paper '
                        'numbers); val = the seeded 750-window validation '
                        'carve (pysteps PARAMETER TUNING ONLY, never a paper '
                        'number)')
    p.add_argument('--pysteps-args', default='{}',
                   help='params JSON passed to the pysteps baselines '
                        '(see queue_bundle/rainnet_pysteps_forecast.py)')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                   help='auto = cuda when available (never for obs-only/'
                        'optical_flow); cpu keeps unit tests hermetic')
    a = p.parse_args()
    if not a.obs_only and not a.model:
        p.error('--model is required unless --obs-only is given')
    ps_params = json.loads(a.pysteps_args)

    csi_thr = [float(t) for t in a.csi_thresholds.split(',')]
    fss_thr = [float(t) for t in a.fss_thresholds.split(',')]
    nbds = [int(k) for k in a.fss_nbds.split(',')]
    sample_idx = sorted({int(i) for i in a.sample_indices.split(',') if i.strip()})

    _, val_loader, test_loader, _, _ = load_data(root=a.data_path,
                                                 batch_size=a.batch_size)
    index_path = Path(a.data_path) / 'rainnet_index.json'
    index = json.loads(index_path.read_text())
    if a.split == 'val':
        test_loader = val_loader
        if 'val' in index:
            # blocked-validation index: explicit held-out-month windows
            full_windows = [tuple(w) for w in index['val']]
        else:
            # replicate the loader's seeded carve to recover per-window identity
            # (asserted against the dataset object below; any divergence fails loud)
            order = list(range(len(index['train'])))
            random.Random(2026).shuffle(order)
            full_windows = [tuple(index['train'][i])
                            for i in sorted(set(order[:750]))]
    else:
        full_windows = [tuple(w) for w in index['test']]
    test_set = test_loader.dataset
    if a.limit:
        test_set.windows = test_set.windows[:a.limit]
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=a.batch_size,
                                                   shuffle=False)
    used_windows = [tuple(w) for w in test_set.windows]
    assert used_windows == full_windows[:len(used_windows)], \
        'loader order diverges from the replicated index order'
    n_win = len(used_windows)
    file_idx = np.array([w[0] for w in used_windows], dtype=np.int32)
    t0s = np.array([w[1] for w in used_windows], dtype=np.int32)
    n_lead = None
    obs_max_mm = 0.0

    if a.device == 'cpu':
        device = 'cpu'
    elif a.device == 'cuda':
        device = 'cuda'
    else:
        device = 'cuda' if (torch.cuda.is_available() and not a.obs_only
                            and a.model != 'optical_flow'
                            and a.model not in ('pysteps_lk_extrap',
                                                'pysteps_sprog')) else 'cpu'
    model = None
    if not a.obs_only:
        if a.model in ('convlstm', 'convlstm_ed', 'unet', 'simvp', 'tau'):
            if not a.checkpoints:
                raise SystemExit('--checkpoints required for trained models')
            model = build_ext_model(a.model, json.loads(a.build_kwargs))
            model.load_state_dict(torch.load(a.checkpoints, map_location=device))
            model.eval().to(device)

    win_mse = np.zeros(n_win)
    win_mae = np.zeros(n_win)
    win_max_obs = np.zeros(n_win)
    win_mean_obs = np.zeros(n_win)
    se_total, n_seen = 0.0, 0
    lead_se = lead_ae = None
    cont = None
    fss_d2 = fss_f2 = fss_o2 = None
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    gidx = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            if n_lead is None:
                n_lead = targets.size(1)
                lead_se = np.zeros(n_lead)
                lead_ae = np.zeros(n_lead)
                cont = np.zeros((n_lead, len(csi_thr), 4), dtype=np.int64)
                fss_d2 = np.zeros((n_lead, len(fss_thr), len(nbds)))
                fss_f2 = np.zeros_like(fss_d2)
                fss_o2 = np.zeros_like(fss_d2)
            if a.obs_only:
                obs = reduce_channels(targets).to(device, torch.float32)
                pred = None
            else:
                targets_d = targets.to(device)
                if a.model == 'persistence':
                    inputs_d = inputs.to(device)
                    outputs = inputs_d[:, -1:, :, :, :].expand(
                        -1, targets_d.size(1), -1, -1, -1)
                elif a.model == 'optical_flow':
                    outputs = torch.from_numpy(
                        optical_flow_nowcast(inputs.to('cpu', torch.float32).numpy()))
                elif a.model == 'identity':
                    outputs = targets_d
                elif a.model in ('pysteps_lk_extrap', 'pysteps_sprog'):
                    from rainnet_pysteps_forecast import (lk_extrap_forecast,
                                                          sprog_forecast)
                    x_mm = inputs[:, :, 0].to('cpu', torch.float32).numpy() \
                        * a.unit_scale
                    fn = (lk_extrap_forecast if a.model == 'pysteps_lk_extrap'
                          else sprog_forecast)
                    y_mm = fn(x_mm, targets_d.size(1), ps_params)
                    outputs = torch.from_numpy(
                        y_mm / a.unit_scale)[:, :, None].to(device)
                else:
                    inputs_d = inputs.to(device)
                    if not json.loads(a.build_kwargs) \
                            and inputs_d.shape[-2:] != (208, 333):
                        raise SystemExit(
                            f'wired arch {a.model} was built at the published '
                            f'208x333 geometry but got input {tuple(inputs_d.shape)}')
                    outputs = model(inputs_d)
                pred = reduce_channels(outputs).to(device, torch.float32)
                obs = reduce_channels(targets_d).to(device, torch.float32)

            b = obs.size(0)
            obs_mm = obs * a.unit_scale
            win_max_obs[gidx:gidx + b] = obs_mm.amax(dim=(1, 2, 3)).cpu().numpy()
            win_mean_obs[gidx:gidx + b] = obs_mm.mean(dim=(1, 2, 3)).cpu().numpy()
            obs_max_mm = max(obs_max_mm, float(obs_mm.max()))

            if not a.obs_only:
                diff2 = (obs - pred) ** 2
                diffa = (obs - pred).abs()
                win_mse[gidx:gidx + b] = diff2.sum(dim=(1, 2, 3)).cpu().numpy() \
                    / obs[0].numel()
                win_mae[gidx:gidx + b] = diffa.sum(dim=(1, 2, 3)).cpu().numpy() \
                    / obs[0].numel()
                # same accumulation order as rainnet_nowcast_eval.py (regression parity)
                se_total += float(diff2.sum())
                lead_se += diff2.sum(dim=(0, 2, 3)).cpu().numpy()
                lead_ae += diffa.sum(dim=(0, 2, 3)).cpu().numpy()
                n_seen += obs.numel()

                pred_mm = pred * a.unit_scale
                for ti, thr in enumerate(csi_thr):
                    po = pred_mm >= thr
                    oo = obs_mm >= thr
                    hits = po & oo
                    cont[:, ti, 0] += hits.sum(dim=(0, 2, 3)).cpu().numpy()
                    cont[:, ti, 1] += (~po & oo).sum(dim=(0, 2, 3)).cpu().numpy()
                    cont[:, ti, 2] += (po & ~oo).sum(dim=(0, 2, 3)).cpu().numpy()
                    cont[:, ti, 3] += (~po & ~oo).sum(dim=(0, 2, 3)).cpu().numpy()
                for ti, thr in enumerate(fss_thr):
                    pb = (pred_mm >= thr).to(torch.float32)
                    ob = (obs_mm >= thr).to(torch.float32)
                    for ki, k in enumerate(nbds):
                        d2, f2, o2 = fss_sums(pb, ob, k)
                        fss_d2[:, ti, ki] += d2
                        fss_f2[:, ti, ki] += f2
                        fss_o2[:, ti, ki] += o2

                for i in sample_idx:
                    if gidx <= i < gidx + b:
                        j = i - gidx
                        np.savez(
                            out_dir / f'ext_{a.tag}_win{i}.npz',
                            inp=inputs[j, :, 0].to('cpu', torch.float32).numpy(),
                            pred=pred[j].to('cpu', torch.float32).numpy(),
                            obs=obs[j].to('cpu', torch.float32).numpy(),
                            unit_scale=a.unit_scale,
                            file_idx=file_idx[i], t0=t0s[i])
            gidx += b

    if n_lead is None:
        raise SystemExit('empty test split')

    if device == 'cuda':
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_name = 'unavailable'
    else:
        cuda_name = None
    meta = {
        'tag': a.tag, 'model': a.model, 'checkpoints': a.checkpoints,
        'obs_only': bool(a.obs_only), 'limit': a.limit, 'split': a.split,
        'pysteps_args': ps_params if a.model in ('pysteps_lk_extrap',
                                                 'pysteps_sprog') else None,
        'batch_size': a.batch_size, 'unit_scale': a.unit_scale,
        'csi_thresholds': csi_thr, 'fss_thresholds': fss_thr, 'fss_nbds': nbds,
        'sample_indices': sample_idx, 'n_windows': int(n_win),
        'n_lead': int(n_lead), 'device': device,
        'torch': torch.__version__,
        'cuda_device': cuda_name,
        'index_sha256': sha256_file(index_path),
        'code_dir': str(_CODE_DIR),
        'model_nowcast_sha256': sha256_file(_CODE_DIR / 'model_nowcast.py'),
        'eval_script_sha256': sha256_file(_HERE / 'rainnet_nowcast_eval.py'),
        'this_script_sha256': sha256_file(_HERE / 'rainnet_extended_eval.py'),
        'files': list(index['files']),
        'elapsed_s': round(time.time() - t_start, 1),
    }
    out = out_dir / f'ext_{a.tag}.npz'
    payload = dict(
        meta=np.array(json.dumps(meta)),
        file_idx=file_idx, t0=t0s,
        win_max_obs_mm=win_max_obs, win_mean_obs_mm=win_mean_obs)
    if not a.obs_only:
        payload.update(win_mse_norm=win_mse, win_mae_norm=win_mae,
                       se_total=se_total, n_seen=n_seen,
                       lead_se=lead_se, lead_ae=lead_ae,
                       cont=cont, fss_d2=fss_d2, fss_f2=fss_f2, fss_o2=fss_o2)
        mse = se_total / n_seen
        mae = float(lead_ae.sum()) / n_seen
        pixels = n_seen / n_lead
        print(f'MODEL={a.model} tag={a.tag} windows={n_win} device={device} '
              f'elapsed={meta["elapsed_s"]}s')
        print(f'OVERALL MSE={mse:.7f} MAE={mae:.7f}  (normalized, '
              f'unit_scale={a.unit_scale})')
        print(f'OVERALL MSE={mse * a.unit_scale ** 2:.4f} mm^2  '
              f'RMSE={float(np.sqrt(mse)) * a.unit_scale:.4f} mm  '
              f'MAE={mae * a.unit_scale:.4f} mm')
        print(f'DATA_RANGE observed max = {obs_max_mm:.2f} mm')
        # day-block summary (bootstrap grouping sanity check)
        blocks = np.unique(file_idx.astype(np.int64) * 1000 + t0s // 24)
        print(f'BLOCKS day blocks={len(blocks)} windows={n_win} '
              f'mean win/block={n_win / len(blocks):.1f}')
    np.savez_compressed(out, **payload)
    print(f'WROTE {out}')


if __name__ == '__main__':
    main()

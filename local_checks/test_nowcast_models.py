"""CPU checks for repro_corrected/model_nowcast.py.

1. shape: (B,10,1,H,W) -> (B,10,1,H,W) at an odd width (like RainNet's 333);
2. causality probe: perturbing the LAST input frame must change the +1h forecast
   of a real forecaster; the family ConvLSTM's +1h output is provably unchanged
   (it only ever saw input frame 1);
3. balanced-loss weights 1/2/5/10/30;
4. one optimizer step decreases nothing NaN;
5. [NEW-2026-10-19] simvp: same shape/causality/step checks at the odd test size,
   plus one full-resolution (1,10,1,208,333) forward to pin the exact RainNet
   geometry (encoder /4 + output crop) and a parameter-count print;
6. [GEN-2026-10-20] generic geometry gate for the WeatherBench / Moving-MNIST
   rows: (B,12,C,32,64) with C=1 and C=2, (B,10,1,64,64) for MM, causality at
   T=12/C=2, plus pinned RainNet parameter counts so the generalization cannot
   silently change the already-published RainNet rows.

Run: python local_checks/test_nowcast_models.py [--code-dir repro_corrected]
"""

import argparse
import sys
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir', default=str(Path(__file__).resolve().parent.parent / 'corrected'))
args = ap.parse_args()
sys.path.insert(0, args.code_dir)

from model_ConvLSTM import ConvLSTM  # noqa: E402
from model_nowcast import BalancedLoss, build  # noqa: E402

torch.manual_seed(0)
fails = []


def check(name, cond):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond:
        fails.append(name)


x = torch.rand(2, 10, 1, 26, 41) * 0.05
x2 = x.clone()
x2[:, -1] += 0.5

for arch in ('convlstm_ed', 'unet', 'simvp'):
    m = build(arch).eval()
    with torch.no_grad():
        y = m(x)
        y2 = m(x2)
    check(f'{arch} shape {tuple(y.shape)}', tuple(y.shape) == (2, 10, 1, 26, 41))
    d1 = (y2[:, 0] - y[:, 0]).abs().max().item()
    check(f'{arch} +1h forecast depends on last input frame (max diff {d1:.2e})', d1 > 1e-6)

simvp = build('simvp').eval()
with torch.no_grad():
    full = simvp(torch.rand(1, 10, 1, 208, 333) * 0.05)
check(f'simvp full-resolution RainNet geometry {tuple(full.shape)}',
      tuple(full.shape) == (1, 10, 1, 208, 333))
n_params = sum(p.numel() for p in simvp.parameters())
print(f'simvp parameters: {n_params / 1e6:.2f}M')
check('simvp parameter count plausible (1M-60M)', 1e6 < n_params < 6e7)

legacy = ConvLSTM(input_dim=1, hidden_dim=[32, 128, 64, 12], kernel_size=(3, 3),
                  num_layers=4, batch_first=True, bias=True, return_all_layers=False).eval()
with torch.no_grad():
    ly = legacy(x)
    ly2 = legacy(x2)
d_first = (ly2[:, 0] - ly[:, 0]).abs().max().item()
d_last = (ly2[:, -1] - ly[:, -1]).abs().max().item()
print(f'legacy family ConvLSTM output {tuple(ly.shape)}; +1h diff {d_first:.2e}, +10h diff {d_last:.2e}')
check('legacy +1h output is blind to the last input frame (defect reproduced)', d_first == 0.0)
check('legacy +10h output does see it', d_last > 0.0)

bl = BalancedLoss()
t_mm = torch.tensor([0.5, 2.0, 5.0, 10.0, 30.0, 70.0])
check('balanced weights', bl.weight_map(t_mm).tolist() == [1.0, 2.0, 5.0, 10.0, 30.0, 30.0])

for arch in ('convlstm_ed', 'unet', 'simvp'):
    m = build(arch)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    target = torch.rand(2, 10, 1, 26, 41) * 0.05
    for loss_fn in (torch.nn.MSELoss(), bl):
        loss = loss_fn(m(x), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        check(f'{arch} {type(loss_fn).__name__} step finite ({loss.item():.3e})',
              torch.isfinite(loss).item())

# ---------------------------------------------------------------------------
# [GEN-2026-10-20] generic geometry gate: WeatherBench (T=12, 32x64, C=1/2) and
# Moving MNIST (T=10, 64x64, C=1), i.e. the shapes the new baseline rows use.
# ---------------------------------------------------------------------------
print('\n--- generic geometry (WeatherBench / Moving MNIST) ---')

RAINNET_PARAMS = {'convlstm_ed': 1_066_369, 'unet': 7_762_410, 'simvp': 30_401_153}
for arch, expected in RAINNET_PARAMS.items():
    got = sum(p.numel() for p in build(arch).parameters())
    # pinned so the generalization cannot silently change the RainNet rows;
    # exact equality is expected (the generic kwargs default to the RainNet values)
    check(f'RainNet {arch} params unchanged ({got:,} vs pinned {expected:,})',
          got == expected)

GEOMS = [
    ('temperature', 12, 1, 32, 64),
    ('cloud_cover', 12, 1, 32, 64),
    ('component_of_wind', 12, 2, 32, 64),
    ('moving_mnist', 10, 1, 64, 64),
]
for name, t, c, h, w in GEOMS:
    xg = torch.rand(2, t, c, h, w) * 0.05
    xg2 = xg.clone()
    xg2[:, -1] += 0.5
    for arch in ('convlstm_ed', 'unet', 'simvp'):
        m = build(arch, t_in=t, t_out=t, c_in=c, c_out=c).eval()
        with torch.no_grad():
            y = m(xg)
            y2 = m(xg2)
        check(f'{name} {arch} shape {tuple(y.shape)}',
              tuple(y.shape) == (2, t, c, h, w))
        d1 = (y2[:, 0] - y[:, 0]).abs().max().item()
        check(f'{name} {arch} +1 forecast depends on last input frame ({d1:.2e})',
              d1 > 1e-6)

# training step at the largest new geometry (wind, C=2) for every arch
for arch in ('convlstm_ed', 'unet', 'simvp'):
    m = build(arch, t_in=12, t_out=12, c_in=2, c_out=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    xw = torch.rand(2, 12, 2, 32, 64)
    tw = torch.rand(2, 12, 2, 32, 64)
    loss = torch.nn.MSELoss()(m(xw), tw)
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    opt.step()
    check(f'wind {arch} step finite ({loss.item():.3e})', torch.isfinite(loss).item())
    check(f'wind {arch} all params receive gradient '
          f'({len(grads)}/{len(list(m.parameters()))})',
          len(grads) == len(list(m.parameters())))

for arch in ('convlstm_ed', 'unet', 'simvp'):
    n = sum(p.numel() for p in build(arch, t_in=12, t_out=12, c_in=1).parameters())
    print(f'{arch} weatherbench T=12 C=1 params: {n / 1e6:.2f}M')

# ---------------------------------------------------------------------------
# [NEW-2026-10-21] TAU (SimVP v2 backbone + temporal attention, OpenSTL config)
# ---------------------------------------------------------------------------
print('\n--- TAU ---')
from model_nowcast import TAULoss  # noqa: E402

tau = build('tau').eval()
with torch.no_grad():
    y = tau(x)
    y2 = tau(x2)
check(f'tau shape {tuple(y.shape)}', tuple(y.shape) == (2, 10, 1, 26, 41))
d1 = (y2[:, 0] - y[:, 0]).abs().max().item()
check(f'tau +1h forecast depends on last input frame (max diff {d1:.2e})', d1 > 1e-6)
x_first = x.clone()
x_first[:, 0] += 0.5
with torch.no_grad():
    d0 = (tau(x_first)[:, 0] - y[:, 0]).abs().max().item()
check(f'tau +1h forecast also depends on first input frame ({d0:.2e})', d0 > 1e-6)
with torch.no_grad():
    full = tau(torch.rand(1, 10, 1, 208, 333) * 0.05)
check(f'tau full-resolution RainNet geometry {tuple(full.shape)}',
      tuple(full.shape) == (1, 10, 1, 208, 333))
n_tau = sum(p.numel() for p in tau.parameters())
print(f'tau parameters: {n_tau / 1e6:.2f}M')
check('tau parameter count plausible (5M-120M)', 5e6 < n_tau < 1.2e8)

tl = TAULoss()
p = torch.rand(2, 10, 1, 26, 41)
check('tau DDR is ~0 when prediction == target',
      abs(tl.ddr(p, p.clone()).item()) < 1e-6)
q = torch.rand(2, 10, 1, 26, 41)
check('tau loss >= plain MSE (DDR is a KL, non-negative)',
      tl(p, q).item() >= torch.nn.MSELoss()(p, q).item() - 1e-9)

m = build('tau')
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
target = torch.rand(2, 10, 1, 26, 41) * 0.05
loss = tl(m(x), target)
opt.zero_grad()
loss.backward()
grads = [p_ for p_ in m.parameters() if p_.grad is not None]
opt.step()
check(f'tau TAULoss step finite ({loss.item():.3e})', torch.isfinite(loss).item())
check(f'tau all params receive gradient ({len(grads)}/{len(list(m.parameters()))})',
      len(grads) == len(list(m.parameters())))

print('ALL_PASS' if not fails else f'FAILED: {fails}')
sys.exit(1 if fails else 0)

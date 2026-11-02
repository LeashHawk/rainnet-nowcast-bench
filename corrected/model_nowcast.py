"""[NEW-2026-10-18] Properly wired nowcasting baselines for RainNet (Paper 1).

Why these exist: the family ConvLSTM entry (model_ConvLSTM.ConvLSTM, as shipped
for MM / temperature / cloud cover and ported to RainNet) returns the hidden state
of INPUT step t as the forecast for lead t. Lead +1h therefore sees only the first
input frame (10 h before the target) and only lead +10h sees all inputs: it is a
sequence-aligned mapping, not a forecaster. local_checks/test_nowcast_models.py
demonstrates this with a causality probe.

All models map (B, T, C, H, W) -> (B, T, C, H, W) for any H, W. [GEN-2026-10-20]
The builders are parameterized over T (10 or 12) and C (1 or 2) so the same
implementations serve RainNet (T=10, C=1, 208x333), WeatherBench
(T=12, C=1 or 2, 32x64) and Moving MNIST (T=10, C=1, 64x64). The no-kwargs
defaults reproduce the RainNet configuration exactly.

ConvLSTMED  encoder-forecaster (Shi et al. 2015): the encoder reads all 10 frames,
            the forecaster starts from the encoder states and rolls out 10 steps,
            feeding back its own previous prediction (the last observed frame for
            the first step).
UNet        fully convolutional direct multi-lead model (RainNet / Agrawal et al.
            2019 family): 10 input frames as channels -> 10 output leads.
SimVP       [NEW-2026-10-19] SimVP v1 (Gao et al., CVPR 2022), official default
            config: per-frame encoder -> U-shaped Inception translator (N_T=8)
            -> per-frame decoder; the modern simple video-prediction baseline.
            Padded to a multiple of 4 and cropped, so odd widths (333) are exact.
TAU         [NEW-2026-10-21] TAU (Tan et al., CVPR 2023) on the SimVP v2 backbone,
            OpenSTL config; trained with MSE + differential divergence reg.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_ConvLSTM import ConvLSTMCell


class ConvLSTMED(nn.Module):
    def __init__(self, in_ch=1, feat=32, hidden=64, t_out=10):
        super().__init__()
        self.t_out = t_out
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, feat, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
        k = (3, 3)
        self.enc1 = ConvLSTMCell(feat, hidden, k, True)
        self.enc2 = ConvLSTMCell(hidden, hidden, k, True)
        self.dec1 = ConvLSTMCell(feat, hidden, k, True)
        self.dec2 = ConvLSTMCell(hidden, hidden, k, True)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden, feat, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, in_ch, 1))

    def forward(self, x):
        b, t_in, _, h, w = x.shape
        z0 = self.down(x[:, 0])
        hs = z0.shape[-2:]
        s1 = self.enc1.init_hidden(b, hs)
        s2 = self.enc2.init_hidden(b, hs)
        for t in range(t_in):
            z = z0 if t == 0 else self.down(x[:, t])
            s1 = self.enc1(z, s1)
            s2 = self.enc2(s1[0], s2)
        prev = x[:, -1]
        outs = []
        for _ in range(self.t_out):
            s1 = self.dec1(self.down(prev), s1)
            s2 = self.dec2(s1[0], s2)
            y = self.up(s2[0])[..., :h, :w]
            outs.append(y)
            prev = y
        return torch.stack(outs, dim=1)


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """[GEN-2026-10-20] Generalized to any (B, T, C, H, W): the T frames are
    stacked into T*C input channels and the head emits t_out*c_out channels,
    reshaped back to (B, t_out, c_out, H, W). Defaults (T=10, C=1) reproduce the
    RainNet geometry byte-for-byte."""

    def __init__(self, t_in=10, t_out=10, base=32, levels=4, c_in=1, c_out=None):
        super().__init__()
        self.t_out = t_out
        self.c_out = c_in if c_out is None else c_out
        self.levels = levels
        widths = [base * 2 ** i for i in range(levels + 1)]
        self.downs = nn.ModuleList()
        cin = t_in * c_in
        for wdt in widths[:-1]:
            self.downs.append(_block(cin, wdt))
            cin = wdt
        self.mid = _block(widths[-2], widths[-1])
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(levels - 1, -1, -1):
            self.ups.append(nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2))
            self.dec.append(_block(widths[i] * 2, widths[i]))
        self.head = nn.Conv2d(widths[0], t_out * self.c_out, 1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        z = x.reshape(b, t * c, h, w)
        m = 2 ** self.levels
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            z = F.pad(z, (0, pw, 0, ph), mode='replicate')
        skips = []
        for blk in self.downs:
            z = blk(z)
            skips.append(z)
            z = F.max_pool2d(z, 2)
        z = self.mid(z)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            z = dec(torch.cat([up(z), skip], dim=1))
        y = self.head(z)[..., :h, :w]
        return y.reshape(b, self.t_out, self.c_out, h, w)


# SimVP v1 (Gao et al., CVPR 2022), transcribed from the authors' release
# (chengtan9907/SimVP-Simpler-yet-Better-Video-Prediction, modules.py/model.py)
# with its default hyper-parameters hid_S=64, hid_T=256, N_S=4, N_T=8,
# incep_ker=3/5/7/11, groups=4. The per-layer GroupNorm is load-bearing: without
# it the 16 stacked Inception blocks shrink the input signal to ~1e-9 at init.

class _BasicConv2d(nn.Module):
    def __init__(self, c_in, c_out, k, stride, padding, transpose=False, act_norm=False):
        super().__init__()
        self.act_norm = act_norm
        if transpose:
            self.conv = nn.ConvTranspose2d(c_in, c_out, k, stride, padding,
                                           output_padding=stride // 2)
        else:
            self.conv = nn.Conv2d(c_in, c_out, k, stride, padding)
        self.norm = nn.GroupNorm(2, c_out)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        return self.act(self.norm(y)) if self.act_norm else y


class _ConvSC(nn.Module):
    def __init__(self, c_in, c_out, stride, transpose=False, act_norm=True):
        super().__init__()
        self.conv = _BasicConv2d(c_in, c_out, 3, stride, 1,
                                 transpose=transpose and stride != 1, act_norm=act_norm)

    def forward(self, x):
        return self.conv(x)


class _GroupConv2d(nn.Module):
    def __init__(self, c_in, c_out, k, groups):
        super().__init__()
        if c_in % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(c_in, c_out, k, 1, k // 2, groups=groups)
        self.norm = nn.GroupNorm(groups, c_out)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class IncepU(nn.Module):
    """SimVP-v1 Inception block: 1x1 reduction, then the sum of grouped convs
    with kernels 3/5/7/11 (each GroupNorm + LeakyReLU)."""

    def __init__(self, c_in, c_hid, c_out, kers=(3, 5, 7, 11), groups=4):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_hid, 1)
        self.layers = nn.ModuleList([_GroupConv2d(c_hid, c_out, k, groups) for k in kers])

    def forward(self, x):
        x = self.conv1(x)
        return sum(layer(x) for layer in self.layers)


class SimVP(nn.Module):
    """Per-frame spatial encoder (strides 1,2,1,2 -> /4), U-shaped Inception
    translator over the frame-stacked latent (T*hid_S channels), per-frame
    decoder with a full-resolution skip from the first encoder layer. Inputs
    are zero-padded to a multiple of 4 (333 -> 336; zero = no rain) and the
    output is cropped back, so any H, W works."""

    def __init__(self, t=10, hid_s=64, hid_t=256, n_s=4, n_t=8, groups=4,
                 c_in=1, c_out=None):
        super().__init__()
        self.c_out = c_in if c_out is None else c_out
        strides = [1, 2] * 10
        enc_s = strides[:n_s]
        dec_s = list(reversed(enc_s))
        self.down = 2 ** sum(s == 2 for s in enc_s)
        self.enc = nn.ModuleList([_ConvSC(c_in, hid_s, enc_s[0])] +
                                 [_ConvSC(hid_s, hid_s, s) for s in enc_s[1:]])
        self.dec = nn.ModuleList([_ConvSC(hid_s, hid_s, s, transpose=True) for s in dec_s[:-1]] +
                                 [_ConvSC(2 * hid_s, hid_s, dec_s[-1], transpose=True)])
        self.readout = nn.Conv2d(hid_s, self.c_out, 1)
        c_in, half = t * hid_s, hid_t // 2
        self.n_t = n_t
        self.t_enc = nn.ModuleList([IncepU(c_in, half, hid_t, groups=groups)] +
                                   [IncepU(hid_t, half, hid_t, groups=groups) for _ in range(n_t - 1)])
        self.t_dec = nn.ModuleList([IncepU(hid_t, half, hid_t, groups=groups)] +
                                   [IncepU(2 * hid_t, half, hid_t, groups=groups) for _ in range(n_t - 2)] +
                                   [IncepU(2 * hid_t, half, c_in, groups=groups)])

    def forward(self, x):
        b, t, c, h, w = x.shape
        ph, pw = (-h) % self.down, (-w) % self.down
        z = F.pad(x.reshape(b * t, c, h, w), (0, pw, 0, ph))
        skip = self.enc[0](z)
        z = skip
        for layer in self.enc[1:]:
            z = layer(z)
        _, c_, h_, w_ = z.shape
        z = z.reshape(b, t * c_, h_, w_)
        skips = []
        for i, layer in enumerate(self.t_enc):
            z = layer(z)
            if i < self.n_t - 1:
                skips.append(z)
        z = self.t_dec[0](z)
        for i in range(1, self.n_t):
            z = self.t_dec[i](torch.cat([z, skips[-i]], dim=1))
        z = z.reshape(b * t, c_, h_, w_)
        for layer in self.dec[:-1]:
            z = layer(z)
        y = self.readout(self.dec[-1](torch.cat([z, skip], dim=1)))
        return y[..., :h, :w].reshape(b, t, self.c_out, h, w)


# TAU (Tan et al., CVPR 2023, "Temporal Attention Unit"), transcribed from the
# OpenSTL release (openstl/models/simvp_model.py + modules/simvp_modules.py,
# model_type='tau') with the OpenSTL Moving-MNIST TAU config: hid_S=64,
# hid_T=512, N_S=4, N_T=8, mlp_ratio=8, drop_path=0. This is the SimVP v2
# backbone: SiLU ConvSC encoder/decoder with PixelShuffle upsampling and an
# additive (not concatenated) full-resolution skip, and a stack of TAU
# meta-blocks over the frame-stacked latent.

def _trunc_init_conv(m):
    if isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class _BasicConv2dV2(nn.Module):
    def __init__(self, c_in, c_out, k, stride, padding, upsampling=False, act_norm=True):
        super().__init__()
        self.act_norm = act_norm
        if upsampling:
            self.conv = nn.Sequential(nn.Conv2d(c_in, c_out * 4, k, 1, padding),
                                      nn.PixelShuffle(2))
        else:
            self.conv = nn.Conv2d(c_in, c_out, k, stride, padding)
        self.norm = nn.GroupNorm(2, c_out)
        self.act = nn.SiLU()
        self.apply(_trunc_init_conv)

    def forward(self, x):
        y = self.conv(x)
        return self.act(self.norm(y)) if self.act_norm else y


class _ConvSCV2(nn.Module):
    def __init__(self, c_in, c_out, k=3, downsampling=False, upsampling=False):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (k - stride + 1) // 2
        self.conv = _BasicConv2dV2(c_in, c_out, k, stride, padding, upsampling=upsampling)

    def forward(self, x):
        return self.conv(x)


class _DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class _TemporalAttentionModule(nn.Module):
    def __init__(self, dim, kernel_size=21, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = dilation * (dd_k - 1) // 2
        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p,
                                      groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), nn.Sigmoid())

    def forward(self, x):
        u = x.clone()
        f_x = self.conv1(self.conv_spatial(self.conv0(x)))
        b, c = x.shape[:2]
        se = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return se * f_x * u


class _TemporalAttention(nn.Module):
    def __init__(self, d_model, kernel_size=21):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = _TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_2(self.spatial_gating_unit(self.activation(self.proj_1(x))))
        return x + shortcut


class _MixMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        return self.fc2(self.act(self.dwconv(self.fc1(x))))


class _TAUSubBlock(nn.Module):
    def __init__(self, dim, kernel_size=21, mlp_ratio=8.0, drop_path=0.0, init_value=1e-2):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = _TemporalAttention(dim, kernel_size)
        self.drop_path = _DropPath(drop_path)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = _MixMlp(dim, int(dim * mlp_ratio))
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(dim))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1[:, None, None] * self.attn(self.norm1(x)))
        return x + self.drop_path(self.layer_scale_2[:, None, None] * self.mlp(self.norm2(x)))


class _MetaBlock(nn.Module):
    def __init__(self, c_in, c_out, mlp_ratio, drop_path):
        super().__init__()
        self.block = _TAUSubBlock(c_in, 21, mlp_ratio, drop_path)
        self.reduction = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        z = self.block(x)
        return z if self.reduction is None else self.reduction(z)


class TAU(nn.Module):
    """SimVP-v2 encoder (samplings False,True,False,True -> /4) -> N_T TAU
    meta-blocks over the (T*hid_S)-channel latent (first block reduces to hid_T,
    last restores T*hid_S) -> decoder with an additive skip from the first
    encoder layer. Zero-padded to a multiple of 4 and cropped, like SimVP."""

    def __init__(self, t=10, hid_s=64, hid_t=512, n_s=4, n_t=8, mlp_ratio=8.0,
                 drop_path=0.0, c_in=1, c_out=None):
        super().__init__()
        assert n_t >= 2
        self.c_out = c_in if c_out is None else c_out
        samp = ([False, True] * (n_s // 2))[:n_s]
        self.down = 2 ** sum(samp)
        self.enc = nn.ModuleList([_ConvSCV2(c_in, hid_s, downsampling=samp[0])] +
                                 [_ConvSCV2(hid_s, hid_s, downsampling=s) for s in samp[1:]])
        rsamp = list(reversed(samp))
        self.dec = nn.ModuleList([_ConvSCV2(hid_s, hid_s, upsampling=s) for s in rsamp])
        self.readout = nn.Conv2d(hid_s, self.c_out, 1)
        ch = t * hid_s
        dpr = [v.item() for v in torch.linspace(1e-2, drop_path, n_t)]
        blocks = [_MetaBlock(ch, hid_t, mlp_ratio, dpr[0] if drop_path > 0 else 0.0)]
        blocks += [_MetaBlock(hid_t, hid_t, mlp_ratio, dpr[i] if drop_path > 0 else 0.0)
                   for i in range(1, n_t - 1)]
        blocks += [_MetaBlock(hid_t, ch, mlp_ratio, drop_path)]
        self.hid = nn.ModuleList(blocks)

    def forward(self, x):
        b, t, c, h, w = x.shape
        ph, pw = (-h) % self.down, (-w) % self.down
        z = F.pad(x.reshape(b * t, c, h, w), (0, pw, 0, ph))
        skip = self.enc[0](z)
        z = skip
        for layer in self.enc[1:]:
            z = layer(z)
        _, c_, h_, w_ = z.shape
        z = z.reshape(b, t * c_, h_, w_)
        for blk in self.hid:
            z = blk(z)
        z = z.reshape(b * t, c_, h_, w_)
        for layer in self.dec[:-1]:
            z = layer(z)
        y = self.readout(self.dec[-1](z + skip))
        return y[..., :h, :w].reshape(b, t, self.c_out, h, w)


class TAULoss(nn.Module):
    """MSE + alpha * differential divergence regularization (TAU paper / OpenSTL
    methods/tau.py, alpha=0.1, tau=0.1). Checkpoint selection and the TEST line
    stay plain MSE in train_function, like every other row."""

    def __init__(self, alpha=0.1, tau=0.1, eps=1e-12):
        super().__init__()
        self.alpha, self.tau, self.eps = alpha, tau, eps
        self.mse = nn.MSELoss()

    def ddr(self, pred, target):
        b, t = pred.shape[:2]
        if t <= 2:
            return pred.new_zeros(())
        gp = (pred[:, 1:] - pred[:, :-1]).reshape(b, t - 1, -1)
        gb = (target[:, 1:] - target[:, :-1]).reshape(b, t - 1, -1)
        sp = F.softmax(gp / self.tau, -1)
        sb = F.softmax(gb / self.tau, -1)
        return (sp * torch.log(sp / (sb + self.eps) + self.eps)).mean()

    def forward(self, pred, target):
        return self.mse(pred, target) + self.alpha * self.ddr(pred, target)


class BalancedLoss(nn.Module):
    """B-MSE + B-MAE of Shi et al. (2017), weights 1/2/5/10/30 at rain-rate
    thresholds 2/5/10/30 mm, computed in mm (inputs are normalized by unit_scale)."""

    def __init__(self, unit_scale=140.0, thresholds=(2.0, 5.0, 10.0, 30.0),
                 weights=(1.0, 2.0, 5.0, 10.0, 30.0)):
        super().__init__()
        self.unit_scale = unit_scale
        self.thresholds = thresholds
        self.weights = weights

    def weight_map(self, target_mm):
        w = torch.full_like(target_mm, self.weights[0])
        for thr, wt in zip(self.thresholds, self.weights[1:]):
            w = torch.where(target_mm >= thr, torch.full_like(w, wt), w)
        return w

    def forward(self, pred, target):
        p = pred * self.unit_scale
        t = target * self.unit_scale
        w = self.weight_map(t)
        d = p - t
        return (w * d * d).mean() + (w * d.abs()).mean()


def build(arch, t_in=10, t_out=10, c_in=1, c_out=None, **kw):
    """[GEN-2026-10-20] Uniform keyword interface for the three baselines: callers
    pass t_in/t_out/c_in/c_out and the builder maps them onto each module's own
    argument names. With the defaults (T=10, C=1) this reproduces the RainNet
    configuration exactly. `**kw` forwards arch-specific extras (e.g. hid_s)."""
    c_out = c_in if c_out is None else c_out
    if arch == 'convlstm_ed':
        return ConvLSTMED(in_ch=c_in, t_out=t_out, **kw)
    if arch == 'unet':
        return UNet(t_in=t_in, t_out=t_out, c_in=c_in, c_out=c_out, **kw)
    if arch == 'simvp':
        return SimVP(t=t_in, c_in=c_in, c_out=c_out, **kw)
    if arch == 'tau':
        return TAU(t=t_in, c_in=c_in, c_out=c_out, **kw)
    raise ValueError(arch)

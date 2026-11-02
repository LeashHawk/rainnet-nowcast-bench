"""Paper 1 V2 figures (RainNet hourly nowcasting benchmark), 2026-10-27.

Writes Figures 1-10 and Appendix Figure E1 into artifacts/figures_paper1_v2/
(or --out-dir), each as a 300 dpi PNG and a PDF with vector text/curves
and embedded raster map fields.

    python local_checks/make_figures_paper1_v2.py [--out-dir DIR]

Provenance (nothing is recomputed beyond plotting):
  fig01  schematic; embedded fields from ext_*_win667.npz (SimVP: seed 0)
  fig02  artifacts/paper1_v2/derived_stats.json + derived_spatial_2018.npz
         + ext_obs_stats.npz            (2026-10-27 ledger entry)
  fig03  ext_*_win667.npz field dumps   (deep models: seed 0)
  fig04  ext_*_win667.npz field dumps
  fig05  ext_*_win667/865.npz field dumps
  fig06  per-window win_mse_norm from ext_*.npz (seed-mean for deep rows)
  fig07-10  eval_*.json + ext_pysteps_*.npz (same sources as make_figures_paper1.py)
  figE1  ext_*_win865.npz field dumps

House style: colour-blind-safe palette (Tol muted + greys), no in-figure
titles (captions carry them), fonts >= 7 pt, width <= 190 mm, maps carry
lon/lat ticks interpolated over the stated archive bounds, 105-65 W /
25-50 N. These labels orient the maps; they are not native geolocation.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--data-dir",
                 default=str(HERE.parent / "artifacts" / "final_20261022" / "full"))
_ap.add_argument("--ext-dir", default=None, help="default: <data-dir>/ext_eval")
_ap.add_argument("--v2-dir", default=str(HERE.parent / "artifacts" / "paper1_v2"))
_ap.add_argument("--out-dir", default=str(HERE.parent / "artifacts" / "figures_paper1_v2"))
_ap.add_argument("--only", default=None, help="render one figure function, e.g. fig01_schematic")
_args = _ap.parse_args()
DATA = Path(_args.data_dir)
EXT = Path(_args.ext_dir) if _args.ext_dir else DATA / "ext_eval"
V2 = Path(_args.v2_dir)
OUT = Path(_args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

UNIT = 140.0
MM = 1 / 25.4  # mm -> inch

# ---------------------------------------------------------------- palette
# Tol-muted family colours: greys = persistence, blues = pysteps,
# greens = ConvLSTM-ED family, yellows = U-Net family, wines = SimVP family,
# vermillion = the miswired as-released row (deliberately alarming).
PAL = {
    "euler": "#999999", "oflow": "#4d4d4d",
    "ps_lk": "#88CCEE", "ps_sprog": "#332288",
    "legacy": "#CC6677", "ed_mse": "#117733", "ed_bal": "#44AA99",
    "unet_mse": "#DDCC77", "unet_bal": "#999933",
    "simvp": "#882255", "tau": "#AA4499",
}
ROWS = [
    ("euler", "Eulerian persistence", ["eval_persistence.json"]),
    ("oflow", "Lagrangian persistence", ["eval_optical_flow.json"]),
    ("ps_lk", "pysteps LK extrapolation", ["ext_pysteps_lk_extrap.npz"]),
    ("ps_sprog", "pysteps S-PROG", ["ext_pysteps_sprog.npz"]),
    ("legacy", "as-released ConvLSTM",
     ["eval_convlstm.json", "eval_convlstm_rainnet_seed1.json",
      "eval_convlstm_rainnet_seed2.json"]),
    ("ed_mse", "ConvLSTM-ED (MSE)",
     ["eval_convlstm_ed_mse_rainnet_seed0.json",
      "eval_convlstm_ed_mse_rainnet_seed1.json",
      "eval_convlstm_ed_mse_rainnet_seed2.json"]),
    ("unet_mse", "U-Net (MSE)",
     ["eval_unet_mse_rainnet_seed0.json", "eval_unet_mse_rainnet_seed1.json",
      "eval_unet_mse_rainnet_seed2.json"]),
    ("ed_bal", "ConvLSTM-ED (balanced)",
     ["eval_convlstm_ed_balanced_rainnet_seed0.json",
      "eval_convlstm_ed_balanced_rainnet_seed1.json",
      "eval_convlstm_ed_balanced_rainnet_seed2.json"]),
    ("unet_bal", "U-Net (balanced)",
     ["eval_unet_balanced_rainnet_seed0.json",
      "eval_unet_balanced_rainnet_seed1.json",
      "eval_unet_balanced_rainnet_seed2.json"]),
    ("simvp", "SimVP (MSE)",
     ["eval_simvp_mse_rainnet_seed0.json", "eval_simvp_mse_rainnet_seed1.json",
      "eval_simvp_mse_rainnet_seed2.json"]),
    ("tau", "TAU (DDR)",
     ["eval_tau_ddr_rainnet_seed0.json", "eval_tau_ddr_rainnet_seed1.json",
      "eval_tau_ddr_rainnet_seed2.json"]),
]
# case-study row order and display labels (persistence row omitted: it repeats
# the last input frame)
CASE_MODELS = [
    ("oflow", "Lagr. persistence", "ext_optical_flow"),
    ("ps_lk", "pysteps LK", "ext_pysteps_lk_extrap"),
    ("ps_sprog", "pysteps S-PROG", "ext_pysteps_sprog"),
    ("legacy", "as-released", "ext_convlstm_rainnet_seed0"),
    ("ed_mse", "ConvLSTM-ED", "ext_convlstm_ed_mse_rainnet_seed0"),
    ("unet_mse", "U-Net", "ext_unet_mse_rainnet_seed0"),
    ("simvp", "SimVP", "ext_simvp_mse_rainnet_seed0"),
    ("tau", "TAU", "ext_tau_ddr_rainnet_seed0"),
]
CASE_LEADS = [0, 2, 5, 9]  # +1, +3, +6, +10 h

# precipitation colour scale aligned with the verification thresholds
PRECIP_LEVELS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 75.0]
_pc = list(plt.cm.cividis(np.linspace(0.06, 1.0, len(PRECIP_LEVELS) - 1)))
_pc[0] = (0.95, 0.95, 0.95, 1.0)  # near-white for the <0.1 mm bin
PRECIP_CMAP = mcolors.ListedColormap(_pc)
PRECIP_NORM = mcolors.BoundaryNorm(PRECIP_LEVELS, PRECIP_CMAP.N)
ERR_CMAP = plt.cm.RdBu_r
ERR_LIM = 20.0
# hit / miss / false alarm (colour-blind safe)
TRI = {"hit": "#009E73", "miss": "#D55E00", "fa": "#0072B2", "bg": "#f0f0f0"}

LON0, LON1, LAT0, LAT1 = -105.0, -65.0, 25.0, 50.0
EXTENT = [LON0, LON1, LAT0, LAT1]  # row 0 = 50 N sits on top with origin="upper"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 8,
    "font.family": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
    "axes.titlesize": 8, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def save(fig, name):
    # Apply the minimum at export, including panel-specific tick overrides.
    from matplotlib.text import Text
    for artist in fig.findobj(Text):
        if artist.get_fontsize() < 7:
            artist.set_fontsize(7)
    fig.canvas.draw()
    width_mm = (fig.get_tightbbox(fig.canvas.get_renderer()).width + 0.2) * 25.4
    if width_mm > 190:
        raise ValueError(f"{name}: exported width {width_mm:.2f} mm exceeds 190")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def map_axes(ax, ticks=True, lat_ticks=True):
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    if ticks:
        ax.set_xticks([-105, -95, -85, -75, -65])
        ax.set_xticklabels(["105°W", "95°W", "85°W", "75°W", "65°W"])
        ax.set_yticks([25, 30, 35, 40, 45, 50])
        if lat_ticks:
            ax.set_yticklabels(["25°N", "30°N", "35°N", "40°N", "45°N",
                                "50°N"])
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


# ---------------------------------------------------------------- fig01
def fig01_schematic():
    # Axes units are layout millimetres, printed at 184.5/189 so that the
    # exported page (tight bbox plus padding) stays within 190 mm.
    from matplotlib.patches import Ellipse
    W, H = 189.0, 152.0
    k = 184.5 / W
    fig = plt.figure(figsize=(W * k * MM, H * k * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ink, sub = "#1A1A1A", "#4D4D4D"
    dev, test, val_c = "#4477AA", "#BB5522", "#DDAA33"
    fills = {
        "blue": ("#DCE8F5", "#4477AA"), "green": ("#DDEFDD", "#2E7D4F"),
        "yellow": ("#FFF2CC", "#B8942A"), "orange": ("#FBE1D2", "#BB5522"),
        "grey": ("#E9E9E9", "#6F6F6F"),
    }
    checks = []  # (text artist, left limit, right limit) in layout mm
    MATH = 8.6   # mathtext scripts shrink by 0.7, so 8.6 pt keeps them >= 6 pt

    def text(x, y, s, size=7, lim=None, **kw):
        kw.setdefault("color", ink)
        kw.setdefault("va", "center")
        t = ax.text(x, y, s, fontsize=size, zorder=6, **kw)
        if lim is not None:
            checks.append((t, *lim))
        return t

    def rbox(x, y, w, h, fc, ec, lw=0.8, ls="-", z=2, r=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

    def panel(x0, y0, x1, y1, fc, ec, title, ref):
        rbox(x0, y0, x1 - x0, y1 - y0, fc, ec, lw=0.9, ls=(0, (4, 2)), z=0,
             r=2.0)
        text(x0 + 2.0, y1 - 3.4, title, 8.5, (x0, x1), fontweight="bold")
        text(x1 - 2.0, y1 - 3.4, ref, 7, (x0, x1), color=sub,
             style="italic", ha="right")

    def arrow(p0, p1, color=ink, ls="-", lw=0.9):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                     mutation_scale=8, color=color, lw=lw,
                                     ls=ls, shrinkA=0, shrinkB=0, zorder=7))

    def thumb(field, x, y, w, h, cmap=PRECIP_CMAP, norm=PRECIP_NORM, z=4):
        ax.imshow(field, extent=(x, x + w, y, y + h), origin="upper",
                  aspect="auto", cmap=cmap, norm=norm,
                  interpolation="nearest", zorder=z)
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#5A5A5A",
                               lw=0.5, zorder=z + 0.1))

    def stack(frames, x, y, w, h, step=1.7):
        # back frames first; each sits up and to the left of the next one
        n = len(frames)
        for i, f in enumerate(frames):
            off = (n - 1 - i) * step
            ax.add_patch(Rectangle((x - off - 0.35, y + off - 0.35), w + 0.7,
                                   h + 0.7, fc="white", ec="none",
                                   zorder=3 + i))
            thumb(f, x - off, y + off, w, h, z=3.2 + i)

    def box(x, y, w, h, kind, head, lines, top_pad=3.6, step=3.5):
        fc, ec = fills[kind]
        rbox(x, y, w, h, fc, ec)
        text(x + w / 2, y + h - top_pad, head, 7.5, (x, x + w),
             ha="center", fontweight="bold")
        yy = y + h - top_pad - 4.4
        for s, style in lines:
            if style == "rule":
                ax.plot([x + 1.5, x + w - 1.5], [yy + 1.6, yy + 1.6],
                        color=ec, lw=0.5, zorder=5)
                yy -= 1.4
                continue
            size = MATH if style == "math" else 7
            text(x + w / 2, yy, s, size, (x, x + w), ha="center",
                 fontweight="bold" if style == "bold" else "normal",
                 style="italic" if style == "italic" else "normal")
            yy -= step + (1.2 if style == "math" else 0.0)

    cf = case_fields(667)
    inp = np.load(EXT / "ext_persistence_win667.npz")["inp"].astype(
        np.float64) * UNIT
    obs = cf["obs"]

    # ------------------------------------------------ (a) data and samples
    panel(1.5, 96.5, 187.5, 150.0, "#F1F5F9", "#6E8FB0",
          "(a) Data and sample construction", "Sections 3.1–3.2")
    # archive cylinder
    cx0, cx1, cy0, cy1, ry = 3.5, 32.5, 115.0, 134.0, 2.4
    cfc, cec = fills["blue"]
    ax.add_patch(Ellipse(((cx0 + cx1) / 2, cy0), cx1 - cx0, 2 * ry, fc=cfc,
                         ec=cec, lw=0.8, zorder=2))
    ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fc=cfc,
                           ec="none", zorder=2))
    ax.plot([cx0, cx0], [cy0, cy1], color=cec, lw=0.8, zorder=2.1)
    ax.plot([cx1, cx1], [cy0, cy1], color=cec, lw=0.8, zorder=2.1)
    ax.add_patch(Ellipse(((cx0 + cx1) / 2, cy1), cx1 - cx0, 2 * ry,
                         fc="#EEF4FB", ec=cec, lw=0.8, zorder=2.2))
    for y, s, wt in ((128.0, "RainNet archive", "bold"),
                     (123.8, "85 monthly HDF5 files", "normal"),
                     (120.3, "Jul–Nov, 2002–2018", "normal")):
        text((cx0 + cx1) / 2, y, s, 7.5 if wt == "bold" else 7, (cx0, cx1),
             ha="center", fontweight=wt)
    text((cx0 + cx1) / 2, 111.0, "NLDAS low-resolution", 7, (cx0, cx1),
         ha="center", color=sub)
    arrow((cx1 + 0.8, 125.0), (37.6, 125.0))

    # one real hourly field: last input hour of test window 667
    mx, my, mw = 38.5, 115.5, 31.0
    mh = mw * 208 / 333
    thumb(inp[-1], mx, my, mw, mh)
    text(mx + mw / 2, my + mh + 2.6, "hourly analysis $I_t$", MATH,
         (mx - 1, mx + mw + 1), ha="center")
    text(mx + mw / 2, 111.0, "0.125°, 208 × 333 grid", 7,
         (mx - 1, mx + mw + 1), ha="center", color=sub)
    arrow((mx + mw + 0.8, 125.0), (74.6, 125.0))

    box(75.5, 116.5, 27.0, 17.5, "yellow", "Archive scaling",
        [(r"$\tilde{I}=\min(I,\,140)\,/\,140$", "math"),
         ("errors reported in mm", "italic")])
    arrow((103.3, 125.0), (106.0, 125.0))

    # 20 h window: three of the ten frames of each half are drawn
    tw, th = 23.0, 23.0 * 208 / 333
    ix, tx, sy = 110.0, 142.5, 116.5
    stack([inp[7], inp[8], inp[9]], ix, sy, tw, th)
    stack([obs[0], obs[1], obs[2]][::-1], tx, sy, tw, th)
    text(ix + tw / 2 - 1.7, 139.2, "input $X$: 10 h", 7.5, (104.0, 136.0),
         ha="center")
    text(tx + tw / 2 - 1.7, 139.2, "target $Y$: +1 to +10 h", 7.5, (135.0, 168.0),
         ha="center")
    text(ix + tw / 2 - 1.7, 111.6, r"$I_{t-9},\,\ldots,\,I_{t}$", MATH,
         (104.0, 136.0), ha="center")
    text(tx + tw / 2 - 1.7, 111.6, r"$I_{t+1},\,\ldots,\,I_{t+10}$", MATH,
         (135.0, 168.0), ha="center")
    ax.plot([104.6, 104.6, 166.0, 166.0], [141.6, 142.6, 142.6, 141.6],
            color=sub, lw=0.6, zorder=5)
    text(135.3, 144.4, "one 20 h sample window", 7, (104.0, 168.0),
         ha="center", color=sub, style="italic")

    # shared precipitation colour scale for every field in the figure
    bx, by, bh = 171.5, 112.0, 25.0
    nb = len(PRECIP_LEVELS) - 1
    for i in range(nb):
        ax.add_patch(Rectangle((bx, by + i * bh / nb), 2.6, bh / nb,
                               fc=PRECIP_CMAP(i), ec="none", zorder=4))
    ax.add_patch(Rectangle((bx, by), 2.6, bh, fill=False, ec="#5A5A5A",
                           lw=0.5, zorder=4.1))
    for i, lev in enumerate(PRECIP_LEVELS):
        text(bx + 3.4, by + i * bh / nb, f"{lev:g}", 7, (bx, 187.0))
    text(bx + 1.3, by + bh + 2.6, "mm h⁻¹", 7, (166.0, 187.0),
         ha="center")

    # year strip of the frozen window index
    text(3.5, 103.4, "Frozen window index", 7.5, (1.5, 33.0),
         fontweight="bold")
    text(3.5, 99.9, "sha256 released", 7, (1.5, 33.0), color=sub,
         style="italic")
    yx0, yx1, t0x, t1x, ys, yh = 34.0, 170.0, 172.5, 186.0, 98.8, 5.0
    cw = (yx1 - yx0) / 16
    for i, yr in enumerate(range(2002, 2018)):
        ax.add_patch(Rectangle((yx0 + i * cw + 0.15, ys), cw - 0.3, yh,
                               fc=dev, ec="none", zorder=3))
        text(yx0 + (i + 0.5) * cw, ys + yh / 2, str(yr), 7, None,
             color="white", ha="center")
    ax.add_patch(Rectangle((t0x, ys), t1x - t0x, yh, fc=test, ec="none",
                           zorder=3))
    text((t0x + t1x) / 2, ys + yh / 2, "2018", 7, None, color="white",
         ha="center", fontweight="bold")
    for a, b in ((yx0, yx1), (t0x, t1x)):
        ax.plot([a + 0.2, a + 0.2, b - 0.2, b - 0.2],
                [ys + yh + 0.6, ys + yh + 1.4, ys + yh + 1.4, ys + yh + 0.6],
                color=sub, lw=0.6, zorder=5)
    text((yx0 + yx1) / 2, ys + yh + 3.3,
         "development: 5250 training + 750 validation windows "
         "(seeded random carve)", 7, (yx0, yx1), ha="center", color=sub)
    text((t0x + t1x) / 2, ys + yh + 3.3, "1500 test", 7, (t0x - 0.5, t1x + 0.5),
         ha="center", color=test, fontweight="bold")

    # ------------------------------------------------ (b) selection
    panel(1.5, 36.0, 116.5, 93.5, "#EEF4FA", dev,
          "(b) Validation-only selection, 2002–2017", "Sections 4, 5.1–5.2")
    box(4.0, 48.0, 35.0, 38.0, "blue", "Classical baselines (4)", [
        ("Eulerian persistence", ""), ("Lagrangian persistence", ""),
        ("pysteps LK extrapolation", ""), ("pysteps S-PROG", ""),
        ("", "rule"), ("grid search on the 750", ""),
        ("validation windows:", ""), ("motion history, AR order,", ""),
        ("cascade levels, threshold", "")])
    box(42.0, 48.0, 37.0, 38.0, "green", "Deep baselines (7), 3 seeds", [
        ("as-released ConvLSTM", ""), ("ConvLSTM-ED (MSE, balanced)", ""),
        ("U-Net (MSE, balanced)", ""), ("SimVP", ""), ("TAU", ""),
        ("", "rule"), ("fit on 5250 training windows", ""),
        ("Adam, learning rate 10⁻³", ""), ("≤ 50 epochs, patience 10", "")])
    box(82.0, 48.0, 32.0, 38.0, "yellow", "Selection rule", [
        (r"$\theta^{*}=\arg\min_{\theta}\ \mathrm{MSE}_{\mathrm{val}}(\theta)$",
         "math"),
        ("one checkpoint per run,", ""), ("whatever the training loss", ""),
        ("one configuration per", ""), ("pysteps baseline", ""),
        ("", "rule"), ("no test window used", "bold")])
    arrow((79.2, 70.0), (81.8, 70.0))
    ax.plot([21.5, 21.5, 98.0], [47.8, 43.0, 43.0], color=ink, lw=0.9,
            zorder=7)
    arrow((98.0, 43.0), (98.0, 47.8))

    # ------------------------------------------------ test isolation
    bar = 119.0
    ax.plot([bar, bar], [36.0, 93.5], color=test, lw=1.0, ls=(0, (3, 2)),
            zorder=1)
    arrow((114.3, 55.0), (123.6, 55.0), lw=1.0)
    ax.add_patch(Rectangle((bar - 1.6, 57.4), 3.2, 2.6, fc=test, ec="none",
                           zorder=8))
    ax.add_patch(Arc((bar, 60.0), 2.1, 2.6, theta1=0, theta2=180,
                     color=test, lw=1.0, zorder=8))
    text(bar, 63.2, "choices frozen", 7, None, color=test, ha="center",
         va="bottom", rotation=90, bbox=dict(fc="white", ec="none", pad=0.6))

    # ------------------------------------------------ (c) locked test
    panel(121.5, 36.0, 187.5, 93.5, "#FCF1EA", test,
          "(c) Locked test, 2018", "Section 5.3")
    pw = 19.5
    ph = pw * 208 / 333
    px = [124.0, 145.0, 166.0]
    py = 72.6
    thumb(obs[0], px[0], py, pw, ph)
    thumb(cf["simvp"][0], px[1], py, pw, ph)
    thumb(tri_map(cf["simvp"][0], obs[0], 5.0), px[2], py, pw, ph,
          cmap=TRI_CMAP, norm=TRI_NORM)
    for x, s in zip(px, ("observed, +1 h", "SimVP, +1 h", "≥ 5 mm h⁻¹")):
        text(x + pw / 2, py + ph + 2.2, s, 7, (x - 1.0, x + pw + 1.0),
             ha="center")
    text(124.0, 69.2, "window 667, 6 July 2018", 7, (122.0, 152.0),
         color=sub, style="italic")
    lx = 154.0
    for key, lab in (("hit", "hit"), ("miss", "miss"), ("fa", "false alarm")):
        ax.add_patch(Rectangle((lx, 68.1), 2.2, 2.2, fc=TRI[key], ec="none",
                               zorder=6))
        t = text(lx + 2.8, 69.2, lab, 7, (152.0, 187.0))
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
        lx = ax.transData.inverted().transform([[bb.x1, bb.y1]])[0][0] + 1.6

    mfc, mec = fills["orange"]
    rbox(124.0, 38.5, 61.5, 28.0, mfc, mec)
    text(154.75, 63.4, "one pass per method on 1500 windows", 7.5,
         (124.0, 185.5), ha="center", fontweight="bold")
    lab_x, val_x, lim = 126.0, 145.5, (124.0, 185.5)
    for y, lab in ((59.7, "Continuous"), (56.1, "Categorical"),
                   (46.7, "Neighbourhood"), (40.4, "Uncertainty")):
        text(lab_x, y, lab, 7, lim, fontweight="bold", color=mec)
    text(val_x, 59.7, "MSE, RMSE, MAE, also per lead", 7, lim)
    text(val_x, 56.1, "CSI = H/(H+M+F); POD, FAR, HSS", 7, lim)
    text(val_x, 53.0, "at 0.1, 0.5, 1, 2, 5, 10 mm h⁻¹", 7, lim)
    text(val_x, 46.7,
         r"$\mathrm{FSS}=1-\dfrac{\sum(P_f-P_o)^2}{\sum(P_f^{\,2}+P_o^{\,2})}$",
         MATH, lim)
    text(val_x, 40.4, "day-block bootstrap, 2000 resamples", 7, lim)

    # ------------------------------------------------ (d) sensitivity
    panel(1.5, 2.0, 187.5, 32.5, "#F4F4F4", "#8A8A8A",
          "(d) Pre-registered sensitivity analysis: month-blocked validation",
          "Section 5.4")
    # validation months of the blocked index (results ledger, 2026-10-22)
    vmonths = {(2002, 9), (2002, 11), (2005, 10), (2007, 9), (2009, 10),
               (2009, 11), (2011, 7), (2011, 8), (2013, 7), (2015, 8)}
    gx0, gcw, gch, gy1 = 13.0, 4.25, 3.3, 25.6
    for j, (mon, name) in enumerate(zip(range(7, 12),
                                        ("Jul", "Aug", "Sep", "Oct", "Nov"))):
        y = gy1 - (j + 1) * gch
        text(gx0 - 1.2, y + gch / 2, name, 7, (1.5, gx0), ha="right")
        for i, yr in enumerate(range(2002, 2019)):
            x = gx0 + i * gcw + (1.6 if yr == 2018 else 0.0)
            c = test if yr == 2018 else (val_c if (yr, mon) in vmonths else dev)
            ax.add_patch(Rectangle((x + 0.15, y + 0.15), gcw - 0.3, gch - 0.3,
                                   fc=c, ec="none", zorder=3))
    for yr in (2002, 2006, 2010, 2014, 2018):
        i = yr - 2002
        x = gx0 + (i + 0.5) * gcw + (1.6 if yr == 2018 else 0.0)
        text(x, gy1 - 5 * gch - 2.2, str(yr), 7, None, ha="center",
             color=sub)
    gx1 = gx0 + 17 * gcw + 1.6
    lx0 = gx1 + 3.0
    for y, c, lab in ((22.6, dev, "training month"),
                      (18.9, val_c, "validation month (10)"),
                      (15.2, test, "test month (2018)")):
        ax.add_patch(Rectangle((lx0, y - 1.1), 2.2, 2.2, fc=c, ec="none",
                               zorder=6))
        text(lx0 + 3.0, y, lab, 7, (lx0, 123.0))
    text(lx0, 10.6, "24 h buffer at month edges", 7, (lx0 - 1.0, 123.0),
         color=sub, style="italic")
    text(lx0, 7.0, "no frame shared between splits", 7, (lx0 - 1.0, 123.0),
         color=sub, style="italic")
    arrow((120.6, 15.0), (123.4, 15.0))
    box(124.0, 4.5, 29.5, 21.5, "grey", "Re-train and re-tune", [
        ("ConvLSTM-ED, U-Net,", ""), ("SimVP, TAU: 3 seeds each", ""),
        ("pysteps re-tuned", ""), ("same 1500 test windows", "")],
        top_pad=3.2, step=3.4)
    arrow((153.7, 15.0), (156.3, 15.0))
    box(156.5, 4.5, 29.5, 21.5, "grey", "Acceptance criteria", [
        ("(i) SimVP lowest, TAU", ""), ("highest mean test MSE", ""),
        ("(ii) both paired differences", ""),
        ("keep sign, 95% CI excl. 0", "")],
        top_pad=3.2, step=3.4)

    fig.canvas.draw()
    inv = ax.transData.inverted()
    rend = fig.canvas.get_renderer()
    over = []
    for t, x0, x1 in checks:
        bb = t.get_window_extent(rend)
        left = inv.transform([[bb.x0, bb.y0]])[0][0]
        right = inv.transform([[bb.x1, bb.y1]])[0][0]
        if left < x0 + 0.8 or right > x1 - 0.8:
            over.append((t.get_text(), round(float(left - x0), 2),
                         round(float(x1 - right), 2)))
    if over:
        raise ValueError(f"fig01 text outside its frame (mm): {over}")
    save(fig, "fig01_schematic")


# ---------------------------------------------------------------- fig02
def fig02_testyear():
    stats = json.loads((V2 / "derived_stats.json").read_text())
    sp = np.load(V2 / "derived_spatial_2018.npz")
    obs = np.load(EXT / "ext_obs_stats.npz")
    ty = stats["test_year"]
    edges = np.array(ty["hist_edges_mm"], dtype=float)
    counts = np.array(ty["hist_counts"], dtype=float)
    frac = counts / ty["pixels"]

    fig = plt.figure(figsize=(180 * MM, 122 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 0.92],
                          wspace=0.28, hspace=0.42,
                          left=0.08, right=0.97, top=0.94, bottom=0.10)

    # (a) pixel-level intensity distribution
    axa = fig.add_subplot(gs[0, 0])
    ctr = np.sqrt(edges[:-1] * edges[1:])
    axa.bar(np.arange(len(frac)), frac, width=0.62, color="#4477AA",
            edgecolor="none")
    axa.set_yscale("log")
    axa.set_xticks(np.arange(len(frac)))
    axa.set_xticklabels([f"{lo:g}–{hi:g}" for lo, hi in
                         zip(edges[:-1], edges[1:])], rotation=45,
                        ha="right", fontsize=7)
    axa.set_ylabel("fraction of pixel-hours")
    axa.set_xlabel("hourly accumulation (mm)")
    axa.text(0.03, 0.86, "92.1% of pixel-hours\nbelow 0.1 mm",
             transform=axa.transAxes, fontsize=7.5)
    axa.set_ylim(1e-7, 0.2)
    axa.text(-0.16, 1.04, "(a)", transform=axa.transAxes, fontsize=9,
             fontweight="bold")
    del ctr

    # (b) per-window distributions
    axb = fig.add_subplot(gs[0, 1])
    wmax = obs["win_max_obs_mm"]
    wmean = obs["win_mean_obs_mm"]
    b1 = np.logspace(np.log10(3), np.log10(80), 25)
    axb.hist(wmax, bins=b1, histtype="stepfilled", alpha=0.75,
             color="#332288", label="window max (mm)")
    axb.set_xscale("log")
    axb.set_xlabel("per-window target-hour maximum (mm)", color="#332288")
    axb.set_ylabel("windows", color="#332288")
    axb.set_xticks([5, 10, 20, 40, 70])
    axb.set_xticklabels(["5", "10", "20", "40", "70"])
    axb.minorticks_off()
    axb.tick_params(axis="x", colors="#332288")
    axb.tick_params(axis="y", colors="#332288")
    axb2 = axb.twiny()
    b2 = np.logspace(np.log10(0.003), np.log10(0.32), 25)
    axb2.hist(wmean, bins=b2, histtype="step", lw=1.4, color="#CC6677",
              label="window mean (mm)")
    axb2.set_xscale("log")
    axb2.set_xlabel("per-window target-hour mean (mm)", color="#CC6677",
                    fontsize=8)
    axb2.set_xticks([0.005, 0.01, 0.05, 0.1, 0.3])
    axb2.set_xticklabels(["0.005", "0.01", "0.05", "0.1", "0.3"])
    axb2.minorticks_off()
    axb2.tick_params(axis="x", colors="#CC6677", labelsize=7.5)
    axb2.spines["top"].set_visible(True)
    axb.text(-0.16, 1.04, "(b)", transform=axb.transAxes, fontsize=9,
             fontweight="bold")

    # Empirical support, not an independently verified geographic land mask.
    land = sp["mean_mm_per_h"] > 0
    oc = mcolors.LinearSegmentedColormap.from_list("cividis_w",
                                                   plt.cm.cividis(np.linspace(
                                                       0, 1, 256)))
    oc.set_bad("white")

    # (c) mean accumulation map
    axc = fig.add_subplot(gs[1, 0])
    im = axc.imshow(np.where(land, sp["mean_mm_per_h"], np.nan),
                    origin="upper", extent=EXTENT, cmap=oc, vmin=0, vmax=0.30,
                    interpolation="nearest")
    map_axes(axc)
    cb = fig.colorbar(im, ax=axc, pad=0.02, fraction=0.046)
    cb.set_label("mean (mm h$^{-1}$)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    axc.text(-0.16, 1.04, "(c)", transform=axc.transAxes, fontsize=9,
             fontweight="bold")

    # (d) rain-hour frequency map
    axd = fig.add_subplot(gs[1, 1])
    im = axd.imshow(np.where(land, sp["rain_freq"], np.nan), origin="upper",
                    extent=EXTENT, cmap=oc, vmin=0, vmax=0.32,
                    interpolation="nearest")
    map_axes(axd)
    cb = fig.colorbar(im, ax=axd, pad=0.02, fraction=0.046)
    cb.set_label("rain-hour frequency", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    axd.text(-0.16, 1.04, "(d)", transform=axd.transAxes, fontsize=9,
             fontweight="bold")
    save(fig, "fig02_testyear")


# ---------------------------------------------------------------- case data
def case_fields(win):
    d0 = np.load(EXT / f"ext_persistence_win{win}.npz")
    out = {"obs": d0["obs"].astype(np.float64) * UNIT}
    for key, _label, stem in CASE_MODELS:
        d = np.load(EXT / f"{stem}_win{win}.npz")
        out[key] = d["pred"].astype(np.float64) * UNIT
    return out


def fig03_case(win, name):
    cf = case_fields(win)
    rows = [("obs", "Observed")] + [(k, lab) for k, lab, _ in CASE_MODELS]
    n_r, n_c = len(rows), len(CASE_LEADS)
    pw, ph = 40.0, 22.0  # compact rows; imshow retains the field aspect ratio
    fig = plt.figure(figsize=((pw * n_c + 22) * MM, (ph * n_r + 14) * MM))
    gs = fig.add_gridspec(n_r, n_c, left=0.115, right=0.985, top=0.985,
                          bottom=0.055, wspace=0.04, hspace=0.05)
    for i, (key, label) in enumerate(rows):
        for j, L in enumerate(CASE_LEADS):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(cf[key][L], origin="upper", extent=EXTENT,
                      cmap=PRECIP_CMAP, norm=PRECIP_NORM,
                      interpolation="nearest")
            if i == n_r - 1:
                map_axes(ax, ticks=True, lat_ticks=(j == 0))
                # inset ticks so edge labels of adjacent panels do not touch
                ax.set_xticks([-100, -90, -80, -70])
                ax.set_xticklabels(["100°W", "90°W", "80°W", "70°W"])
                ax.tick_params(labelsize=6.8, pad=1)
            else:
                map_axes(ax, ticks=False)
            if j == 0:
                # two latitude ticks per row: six labels on a 22 mm row collide
                ax.set_yticks([30, 40])
                ax.set_yticklabels(["30°N", "40°N"])
                ax.tick_params(axis="y", labelsize=6.8, pad=1)
            else:
                ax.set_yticks([])
            if i == 0:
                ax.text(0.5, 1.03, f"+{L + 1} h", transform=ax.transAxes,
                        ha="center", fontsize=8)
            if j == 0:
                ax.text(-0.25, 0.5, label, transform=ax.transAxes, ha="right",
                        va="center", rotation=90, fontsize=7.2)
    cbax = fig.add_axes([0.30, 0.010, 0.42, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=PRECIP_NORM, cmap=PRECIP_CMAP),
                      cax=cbax, orientation="horizontal",
                      ticks=PRECIP_LEVELS)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("mm h$^{-1}$", fontsize=7.5)
    save(fig, name)


def fig04_error():
    cf = case_fields(667)
    obs = cf["obs"]
    core = np.unravel_index(np.argmax(obs[0]), obs[0].shape)  # 71.44 mm pixel
    r, c = int(core[0]), int(core[1])
    dpx = 20
    r0, r1 = max(0, r - dpx), min(207, r + dpx)
    c0, c1 = max(0, c - dpx), min(332, c + dpx)
    # pixel -> lon/lat for the rectangle
    lon_a = LON0 + c0 / 333 * 40
    lon_b = LON0 + c1 / 333 * 40
    lat_a = LAT0 + (207 - r1) / 208 * 25
    lat_b = LAT0 + (207 - r0) / 208 * 25

    show = [("legacy", "as-released ConvLSTM"), ("simvp", "SimVP (MSE)"),
            ("ps_sprog", "pysteps S-PROG")]
    leads = [0, 2]
    fig = plt.figure(figsize=(150 * MM, 128 * MM))
    gs = fig.add_gridspec(3, 2, left=0.13, right=0.96, top=0.95, bottom=0.09,
                          wspace=0.06, hspace=0.16)
    for i, (key, label) in enumerate(show):
        for j, L in enumerate(leads):
            ax = fig.add_subplot(gs[i, j])
            err = cf[key][L] - obs[L]
            im = ax.imshow(err, origin="upper", extent=EXTENT, cmap=ERR_CMAP,
                           vmin=-ERR_LIM, vmax=ERR_LIM, interpolation="nearest")
            ax.add_patch(Rectangle((lon_a, lat_a), lon_b - lon_a, lat_b - lat_a,
                                   fill=False, ec="#111111", lw=1.0))
            map_axes(ax, ticks=(i == 2 or j == 0), lat_ticks=(j == 0))
            if i == 2 or j == 0:
                # inset ticks so edge labels of adjacent panels do not touch
                ax.set_xticks([-100, -90, -80, -70])
                ax.set_xticklabels(["100°W", "90°W", "80°W", "70°W"]
                                   if i == 2 else [])
            ax.tick_params(labelsize=7, pad=1)
            if i == 0:
                ax.text(0.5, 1.04, f"+{L + 1} h", transform=ax.transAxes,
                        ha="center", fontsize=8.5)
            if j == 0:
                ax.text(-0.19, 0.5, label, transform=ax.transAxes,
                        ha="right", va="center", rotation=90, fontsize=8)
    cbax = fig.add_axes([0.30, 0.02, 0.42, 0.022])
    cb = fig.colorbar(im, cax=cbax, orientation="horizontal", extend="both")
    cb.set_label("forecast − observed (mm h$^{-1}$)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    save(fig, "fig04_error667")


def tri_map(pred, obs, thr):
    """3-colour exceedance map: hit / miss / false alarm."""
    o = obs >= thr
    p = pred >= thr
    m = np.full(obs.shape, 0.0)  # background
    m[o & p] = 1.0   # hit
    m[o & ~p] = 2.0  # miss
    m[~o & p] = 3.0  # false alarm
    return m


TRI_CMAP = mcolors.ListedColormap([TRI["bg"], TRI["hit"], TRI["miss"],
                                   TRI["fa"]])
TRI_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], TRI_CMAP.N)


def fig05_threshold():
    cf7 = case_fields(667)
    cf5 = case_fields(865)
    show = [("ps_sprog", "S-PROG"), ("simvp", "SimVP"),
            ("legacy", "as-released")]
    blocks = [("667", cf7, [0, 5]), ("865", cf5, [0, 9])]
    fig = plt.figure(figsize=(190 * MM, 178 * MM))
    gs = fig.add_gridspec(7, 4, left=0.075, right=0.955, top=0.975,
                          bottom=0.075, wspace=0.05, hspace=0.08,
                          height_ratios=[1, 1, 1, 0.22, 1, 1, 1])
    for bi, (win, cf, leads) in enumerate(blocks):
        rbase = bi * 4
        ax = fig.add_subplot(gs[rbase + (0 if bi == 0 else -1), 0])
        ax.axis("off")
        ax.text(0.0, 0.5, f"window {win}" + (" (heavy)" if win == "667"
                else " (moderate)"), fontsize=8.5, va="center")
        for mi, (key, label) in enumerate(show):
            for j, (L, thr) in enumerate([(leads[0], 5.0), (leads[0], 10.0),
                                          (leads[1], 5.0), (leads[1], 10.0)]):
                ax = fig.add_subplot(gs[rbase + mi, j])
                ax.imshow(tri_map(cf[key][L], cf["obs"][L], thr),
                          origin="upper", extent=EXTENT, cmap=TRI_CMAP,
                          norm=TRI_NORM, interpolation="nearest")
                last_row = (bi == 1 and mi == 2)
                map_axes(ax, ticks=(last_row or j == 0), lat_ticks=(j == 0))
                # interior lat ticks avoid label collisions between rows
                ax.set_yticks([30, 40])
                if j == 0:
                    ax.set_yticklabels(["30°N", "40°N"])
                else:
                    ax.set_yticks([])
                if not last_row:
                    ax.set_xticklabels([])
                else:
                    # inset ticks so edge labels of adjacent panels do not touch
                    ax.set_xticks([-100, -90, -80, -70])
                    ax.set_xticklabels(["100°W", "90°W", "80°W", "70°W"])
                    ax.tick_params(labelsize=6.8, pad=1)
                if mi == 0:
                    ax.set_title(f"+{L + 1} h, ≥{thr:g} mm", fontsize=7.8,
                                 pad=2)
                if j == 3:
                    ax.text(1.06, 0.5, label, transform=ax.transAxes,
                            ha="left", va="center", rotation=90,
                            fontsize=7.4)
                if j == 0:
                    ax.tick_params(axis="y", labelsize=6.8, pad=1)
    handles = [Rectangle((0, 0), 1, 1, fc=TRI["hit"], ec="none"),
               Rectangle((0, 0), 1, 1, fc=TRI["miss"], ec="none"),
               Rectangle((0, 0), 1, 1, fc=TRI["fa"], ec="none")]
    fig.legend(handles, ["hit", "miss", "false alarm"], loc="lower center",
               ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.53, 0.012))
    save(fig, "fig05_threshold")


def fig06_scatter():
    def seed_mean(stems):
        return np.mean([np.load(EXT / s)["win_mse_norm"].astype(np.float64)
                        for s in stems], axis=0) * UNIT ** 2
    simvp = seed_mean([f"ext_simvp_mse_rainnet_seed{i}.npz" for i in range(3)])
    ed = seed_mean([f"ext_convlstm_ed_mse_rainnet_seed{i}.npz"
                    for i in range(3)])
    sprog = np.load(EXT / "ext_pysteps_sprog.npz")["win_mse_norm"] * UNIT ** 2
    wmax = np.load(EXT / "ext_obs_stats.npz")["win_max_obs_mm"]

    fig = plt.figure(figsize=(178 * MM, 88 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], left=0.085,
                          right=0.975, top=0.93, bottom=0.14, wspace=0.22)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    pairs = [(ed, "ConvLSTM-ED (MSE)", "64.3%"), (sprog, "pysteps S-PROG",
                                                 "98.4%")]
    lim = [2e-3, 40]
    for ax, (other, label, wr) in zip(axes, pairs):
        sc = ax.scatter(other, simvp, s=4, c=wmax, cmap="cividis", alpha=0.55,
                        linewidths=0, norm=mcolors.LogNorm(vmin=3, vmax=75))
        ax.plot(lim, lim, color="#333333", lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"per-window MSE, {label} (mm$^2$)")
        ax.text(0.04, 0.92, f"SimVP better on {wr}\nof windows",
                transform=ax.transAxes, fontsize=8)
        ax.grid(True, alpha=0.25, lw=0.6)
    axes[0].set_ylabel("per-window MSE, SimVP (mm$^2$)")
    cb = fig.colorbar(sc, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label("observed window maximum (mm)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    save(fig, "fig06_scatter")


# ---------------------------------------------------------------- fig07-10
def npz_as_eval(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    us = float(meta["unit_scale"])
    n_lead = int(meta["n_lead"])
    per_lead_n = int(d["n_seen"]) / n_lead
    cont = d["cont"].astype(np.float64)
    h, m, f, c = (cont[..., i] for i in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        csi = h / (h + m + f)
        pod = h / (h + m)
        far = f / (h + f)
        hss = 2 * (h * c - m * f) / ((h + m) * (f + c) + (h + f) * (m + c))
    thresholds = [
        {"threshold_mm": float(t), "csi_mean": float(np.nanmean(csi[:, i])),
         "csi_per_lead": csi[:, i].tolist(),
         "pod_mean": float(np.nanmean(pod[:, i])),
         "far_mean": float(np.nanmean(far[:, i])),
         "hss_mean": float(np.nanmean(hss[:, i]))}
        for i, t in enumerate(meta["csi_thresholds"])]
    return {"lead_mse_mm2": (d["lead_se"] / per_lead_n * us ** 2).tolist(),
            "overall_mse_mm2": float(d["se_total"]) / int(d["n_seen"]) * us ** 2,
            "thresholds": thresholds}


def load(name):
    path = (EXT if name.endswith(".npz") else DATA) / name
    if not path.exists():
        return None
    if path.suffix == ".npz":
        return npz_as_eval(path)
    return json.loads(path.read_text())


def collect():
    out = {}
    for key, label, files in ROWS:
        js = [j for j in (load(f) for f in files) if j is not None]
        if not js:
            continue
        lead = np.array([j["lead_mse_mm2"] for j in js], dtype=float)
        thr = {}
        for j in js:
            for t in j["thresholds"]:
                thr.setdefault(t["threshold_mm"], []).append(t)
        out[key] = {"label": label, "color": PAL[key], "n_seeds": len(js),
                    "lead_mse_mm2": lead.mean(axis=0), "thr": thr,
                    "overall": float(np.mean([j["overall_mse_mm2"]
                                              for j in js]))}
    return out


def fig07_lead_mse(rows):
    fig, ax = plt.subplots(figsize=(150 * MM, 88 * MM))
    leads = np.arange(1, 11)
    for key, r in rows.items():
        ax.plot(leads, r["lead_mse_mm2"], marker="o", ms=3.0, lw=1.4,
                color=r["color"], label=r["label"])
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("MSE (mm$^2$)")
    ax.set_xticks(leads)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              handlelength=1.6, borderaxespad=0)
    fig.tight_layout()
    save(fig, "fig07_lead_mse")


def fig08_csi_threshold(rows):
    fig, ax = plt.subplots(figsize=(150 * MM, 88 * MM))
    all_thr = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    for key, r in rows.items():
        thrs = [x for x in all_thr if x in r["thr"]]
        means = [float(np.mean([t["csi_mean"] for t in r["thr"][x]]))
                 for x in thrs]
        ax.plot(thrs, means, marker="s", ms=3.4, lw=1.4, color=r["color"],
                label=r["label"])
    if "ed_mse" in rows:
        lo, hi = [], []
        for x in all_thr:
            per_lead = np.array([t["csi_per_lead"]
                                 for t in rows["ed_mse"]["thr"][x]])
            m = per_lead.mean(axis=0)
            lo.append(m.min())
            hi.append(m.max())
        ax.fill_between(all_thr, lo, hi, color=PAL["ed_mse"], alpha=0.13,
                        label="ConvLSTM-ED (MSE): per-lead range")
    ax.set_xscale("log")
    ax.set_xticks(list(all_thr))
    ax.set_xticklabels(["0.1", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("rain-rate threshold (mm h$^{-1}$)")
    ax.set_ylabel("CSI (lead-averaged)")
    ax.set_ylim(0, 0.44)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              handlelength=1.6, borderaxespad=0)
    fig.tight_layout()
    save(fig, "fig08_csi_threshold")


def fig09_pod_far(rows):
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 80 * MM))
    leads = np.arange(1, 11)
    keys = [k for k in ("legacy", "ed_mse", "unet_mse") if k in rows]
    style = {"legacy": "-", "ed_mse": "--", "unet_mse": ":"}
    for key in keys:
        r = rows[key]
        per_lead = np.array([t["csi_per_lead"] for t in r["thr"][1.0]])
        axes[0].plot(leads, per_lead.mean(axis=0), marker="o", ms=3.0,
                     lw=1.5, ls=style[key], color=r["color"],
                     label=r["label"])
    axes[0].set_xlabel("lead time (h)")
    axes[0].set_ylabel("CSI at 1 mm h$^{-1}$")
    axes[0].set_xticks(leads)
    axes[0].set_ylim(-0.01, 0.40)
    axes[0].grid(True, alpha=0.25, lw=0.6)
    axes[0].legend(frameon=False)
    axes[0].text(-0.14, 1.03, "(a)", transform=axes[0].transAxes, fontsize=9,
                 fontweight="bold")

    x = np.arange(len(keys))
    pod = [float(np.mean([t["pod_mean"] for t in rows[k]["thr"][1.0]]))
           for k in keys]
    far = [float(np.mean([t["far_mean"] for t in rows[k]["thr"][1.0]]))
           for k in keys]
    axes[1].bar(x - 0.19, pod, 0.36, color=[rows[k]["color"] for k in keys],
                label="POD at 1 mm h$^{-1}$")
    axes[1].bar(x + 0.19, far, 0.36, color=[rows[k]["color"] for k in keys],
                alpha=0.45, hatch="//", label="FAR at 1 mm h$^{-1}$")
    for xi, (p, f) in enumerate(zip(pod, far)):
        axes[1].text(xi - 0.19, p + 0.012, f"{p:.3f}", ha="center", fontsize=7)
        axes[1].text(xi + 0.19, f + 0.012, f"{f:.3f}", ha="center", fontsize=7)
    axes[1].set_xticks(x)
    short = {"legacy": "as-released\nConvLSTM", "ed_mse": "ConvLSTM-ED\n(MSE)",
             "unet_mse": "U-Net\n(MSE)"}
    axes[1].set_xticklabels([short[k] for k in keys], fontsize=7.5)
    axes[1].set_ylim(0, 0.92)
    axes[1].set_ylabel("score (lead-averaged)")
    axes[1].grid(True, alpha=0.25, lw=0.6, axis="y")
    axes[1].legend(frameon=False, loc="upper right", fontsize=7.5)
    axes[1].text(-0.10, 1.03, "(b)", transform=axes[1].transAxes, fontsize=9,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "fig09_pod_far")


def fig10_cost():
    cost = [("U-Net", 0.0305, 0.78, (10, 8)),
            ("ConvLSTM-ED", 0.2465, 6.5, (10, 6)),
            ("as-released ConvLSTM", 0.634, 12.43, (-8, -24)),
            ("SimVP", 0.894, 13.8, (4, 12)),
            ("TAU", 0.509, 22.25, (-8, -6)),
            ("archived Mamba-based\narchitecture", 87.919, 18.6, (-10, -30))]
    fig, ax = plt.subplots(figsize=(150 * MM, 88 * MM))
    for label, s_it, vram, off in cost:
        infeasible = s_it > 10
        ax.scatter(s_it, vram, s=140 if infeasible else 55,
                   color=PAL["legacy"] if infeasible else PAL["unet_mse"],
                   marker="*" if infeasible else "o", zorder=3)
        ax.annotate(label, (s_it, vram), textcoords="offset points",
                    xytext=off, fontsize=7.5,
                    ha="right" if off[0] < 0 else "left")
    ax.set_xscale("log")
    ax.set_xlim(0.02, 400)
    ax.set_ylim(0, 26)
    ax.set_xlabel("seconds per iteration (log scale)")
    ax.set_ylabel("peak VRAM (GB)")
    ax.axvspan(10, 400, color=PAL["legacy"], alpha=0.07)
    ax.text(60, 2.0, "infeasible under our budget\n(20 epochs ≈ 2564 GPU-h)",
            fontsize=7.5, color="#8c3a31", ha="center")
    ax.grid(True, alpha=0.25, lw=0.6)
    fig.tight_layout()
    save(fig, "fig10_cost")


def main():
    fig01_schematic()
    fig02_testyear()
    fig03_case(667, "fig03_case667")
    fig04_error()
    fig05_threshold()
    fig06_scatter()
    rows = collect()
    print("rows loaded:", {k: (v["label"], v["n_seeds"], round(v["overall"], 4))
                           for k, v in rows.items()})
    fig07_lead_mse(rows)
    fig08_csi_threshold(rows)
    fig09_pod_far(rows)
    fig10_cost()
    fig03_case(865, "figE1_case865")


if __name__ == "__main__":
    if _args.only:
        globals()[_args.only]()
    else:
        main()

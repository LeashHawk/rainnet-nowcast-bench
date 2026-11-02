"""Paper 1 figures (RainNet hourly nowcasting benchmark).

Reads the verified eval JSONs (and the pysteps extended-eval NPZs) and writes
figures into artifacts/figures_paper1/ (or --out-dir).

    python local_checks/make_figures_paper1.py \
        [--data-dir DIR_WITH_eval_JSONS] [--ext-dir DIR_WITH_ext_NPZS] [--out-dir DIR]

Nothing is recomputed: every number comes from an eval JSON whose provenance is
recorded in RESULTS_LEDGER.md.

Figures
  figA_lead_mse.png     per-lead MSE (mm^2) for every row; the as-released
                        configuration is flat in lead time, the wired models decay
  figB_csi_threshold.png CSI vs rain-rate threshold, lead-averaged, with the
                        per-lead spread of the wired models as a band
  figC_cost.png         measured cost (s/iter vs peak VRAM, log x) with the
                        MambaEP point annotated as infeasible
  figD_pod_far.png      POD and FAR at 1 mm by lead: the defect row vs the wired rows

Each figure is written twice: a 300 dpi PNG and a vector PDF with the same basename.
Output canvas widths stay within the Elsevier double-column limit of 190 mm, and no
text is smaller than 7 pt.

Default inputs are the final synced artifacts in artifacts/final_20261022/full/
(eval_*.json for every trained/classical row; <ext-dir>/ext_pysteps_*.npz for the
two pysteps rows, which have no standard-evaluator JSON and are converted to
the same fields from their accumulated sums). Rows whose inputs are missing
are skipped.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--data-dir", default=str(HERE.parent / "artifacts" / "final_20261022" / "full"))
_ap.add_argument("--ext-dir", default=None, help="default: <data-dir>/ext_eval")
_ap.add_argument("--out-dir", default=str(HERE.parent / "artifacts" / "figures_paper1"))
_args = _ap.parse_args()
DATA = Path(_args.data_dir)
EXT = Path(_args.ext_dir) if _args.ext_dir else DATA / "ext_eval"
OUT = Path(_args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

# row key -> (label, colour, [eval json names]) ; a list = seed replicates
ROWS = [
    ("euler", "Eulerian persistence", "#7f7f7f", ["eval_persistence.json"]),
    ("oflow", "Lagrangian persistence", "#4c4c4c", ["eval_optical_flow.json"]),
    ("legacy", "as-released ConvLSTM (defect)", "#b04a3f",
     ["eval_convlstm.json", "eval_convlstm_rainnet_seed1.json",
      "eval_convlstm_rainnet_seed2.json"]),
    ("ed_mse", "ConvLSTM-ED (mse)", "#2f6f4e",
     ["eval_convlstm_ed_mse_rainnet_seed0.json",
      "eval_convlstm_ed_mse_rainnet_seed1.json",
      "eval_convlstm_ed_mse_rainnet_seed2.json"]),
    ("unet_mse", "U-Net (mse)", "#2f6f9e",
     ["eval_unet_mse_rainnet_seed0.json", "eval_unet_mse_rainnet_seed1.json",
      "eval_unet_mse_rainnet_seed2.json"]),
    ("ed_bal", "ConvLSTM-ED (balanced)", "#8ab27a",
     ["eval_convlstm_ed_balanced_rainnet_seed0.json",
      "eval_convlstm_ed_balanced_rainnet_seed1.json",
      "eval_convlstm_ed_balanced_rainnet_seed2.json"]),
    ("unet_bal", "U-Net (balanced)", "#8aa8c8",
     ["eval_unet_balanced_rainnet_seed0.json",
      "eval_unet_balanced_rainnet_seed1.json",
      "eval_unet_balanced_rainnet_seed2.json"]),
    ("simvp", "SimVP v1 (mse)", "#8a6f2f",
     ["eval_simvp_mse_rainnet_seed0.json", "eval_simvp_mse_rainnet_seed1.json",
      "eval_simvp_mse_rainnet_seed2.json"]),
    ("tau", "TAU (ddr)", "#d08c2a",
     ["eval_tau_ddr_rainnet_seed0.json", "eval_tau_ddr_rainnet_seed1.json",
      "eval_tau_ddr_rainnet_seed2.json"]),
    ("ps_lk", "pysteps LK extrap", "#9467bd", ["ext_pysteps_lk_extrap.npz"]),
    ("ps_sprog", "pysteps S-PROG", "#5b3b8c", ["ext_pysteps_sprog.npz"]),
]
FIG_B_THRESHOLDS = (0.1, 0.5, 1.0, 2.0)

# measured cost (Paper 1 Section 7): label, s/iter, VRAM GB, annotation offset
COST = [
    ("U-Net\n(0.03 s/it)", 0.0305, 0.78, (10, 8)),
    ("ConvLSTM-ED\n(0.25 s/it)", 0.2465, 6.5, (10, 6)),
    ("as-released ConvLSTM\n(0.63 s/it)", 0.634, 12.43, (-8, -22)),
    ("SimVP v1\n(0.89 s/it)", 0.894, 13.8, (4, 12)),
    ("TAU\n(0.51 s/it)", 0.509, 22.25, (-8, -4)),
    ("MambaEP (released arch)\n(87.9 s/it)", 87.919, 18.6, (-10, -24)),
]

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
})


def npz_as_eval(path):
    """Eval-JSON-shaped dict from an extended-eval npz (same accumulated sums
    the standard evaluator uses; regression-checked in ext_analysis_paper1)."""
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
         "csi_per_lead": csi[:, i].tolist(), "pod_mean": float(np.nanmean(pod[:, i])),
         "far_mean": float(np.nanmean(far[:, i])), "hss_mean": float(np.nanmean(hss[:, i]))}
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
    """Returns {key: {'label','color','n_seeds','lead_mse_mm2','thr'}} for rows present."""
    out = {}
    for key, label, colour, files in ROWS:
        js = [load(f) for f in files]
        js = [j for j in js if j is not None]
        if not js:
            continue
        lead = np.array([j["lead_mse_mm2"] for j in js], dtype=float)
        thr = {}
        for j in js:
            for t in j["thresholds"]:
                thr.setdefault(t["threshold_mm"], []).append(t)
        out[key] = {
            "label": label, "color": colour, "n_seeds": len(js),
            "lead_mse_mm2": lead.mean(axis=0), "lead_mse_mm2_all": lead,
            "thr": thr,
            "overall": float(np.mean([j["overall_mse_mm2"] for j in js])),
        }
    return out


def stamp(fig, text):
    fig.text(0.01, 0.01, text, fontsize=7, color="#666666", ha="left", va="bottom")


def fig_lead_mse(rows):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    leads = np.arange(1, 11)
    for key, r in rows.items():
        ax.plot(leads, r["lead_mse_mm2"], marker="o", ms=3.2, lw=1.5,
                color=r["color"], label=f'{r["label"]}')
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("MSE (mm$^2$)")
    ax.set_xticks(leads)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    n = max(r["n_seeds"] for r in rows.values())
    stamp(fig, f"source: eval_*.json (pysteps: ext_pysteps_*.npz), 1500 disjoint 2018 test windows; "
               f"3-seed means where replicates exist (max {n} seeds).")
    fig.savefig(OUT / "figA_lead_mse.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figA_lead_mse.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_csi_threshold(rows):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for key, r in rows.items():
        thrs = [x for x in sorted(r["thr"]) if x in FIG_B_THRESHOLDS]
        means = [float(np.mean([t["csi_mean"] for t in r["thr"][x]])) for x in thrs]
        ax.plot(thrs, means, marker="s", ms=4, lw=1.6, color=r["color"],
                label=r["label"])
    # per-lead band for the best wired model, to show the lead spread
    if "ed_mse" in rows:
        thrs = [x for x in sorted(rows["ed_mse"]["thr"]) if x in FIG_B_THRESHOLDS]
        lo, hi = [], []
        for x in thrs:
            per_lead = np.array([t["csi_per_lead"] for t in rows["ed_mse"]["thr"][x]])
            m = per_lead.mean(axis=0)
            lo.append(m.min())
            hi.append(m.max())
        ax.fill_between(thrs, lo, hi, color="#2f6f4e", alpha=0.13,
                        label="ConvLSTM-ED (mse): per-lead range")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.5, 1, 2])
    ax.set_xticklabels(["0.1", "0.5", "1", "2"])
    ax.set_xlabel("rain-rate threshold (mm)")
    ax.set_ylabel("CSI (lead-averaged)")
    top = max([0.42] + [float(np.max(c.get_paths()[0].vertices[:, 1]))
                        for c in ax.collections if c.get_paths()])
    ax.set_ylim(0, top + 0.02)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    stamp(fig, "source: eval_*.json thresholds block (pysteps: ext npz); 3-seed means where "
               "replicates exist; band = ConvLSTM-ED per-lead min/max")
    fig.savefig(OUT / "figB_csi_threshold.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figB_csi_threshold.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cost():
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for label, s_it, vram, off in COST:
        infeasible = s_it > 10
        ax.scatter(s_it, vram, s=150 if infeasible else 55,
                   color="#b04a3f" if infeasible else "#2f6f9e",
                   marker="*" if infeasible else "o", zorder=3)
        ax.annotate(label, (s_it, vram), textcoords="offset points",
                    xytext=off, fontsize=7.4,
                    ha="right" if off[0] < 0 else "left")
    ax.set_xscale("log")
    ax.set_xlim(0.02, 400)
    ax.set_ylim(0, 24)
    ax.set_xlabel("seconds per iteration (log scale)")
    ax.set_ylabel("peak VRAM (GB)")
    ax.axvspan(10, 400, color="#b04a3f", alpha=0.07)
    ax.text(60, 2.0, "infeasible under any\nshared-GPU budget\n"
                     "(20 ep $\\approx$ 2564 GPU-h)",
            fontsize=7.4, color="#8c3a31", ha="center")
    fig.tight_layout()
    stamp(fig, "source: measured s/it and peak VRAM per model; MambaEP at batch 1, "
               "others at their training batch")
    fig.savefig(OUT / "figC_cost.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figC_cost.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_pod_far(rows):
    """Left: CSI at 1 mm per lead (per-lead data exists). Right: lead-averaged
    POD and FAR at 1 mm as grouped bars (the eval JSON stores only lead means
    for POD/FAR, so a per-lead line would be fabricated)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.3))
    leads = np.arange(1, 11)
    keys = [k for k in ("legacy", "ed_mse", "unet_mse") if k in rows]
    style = {"legacy": "-", "ed_mse": "--", "unet_mse": "-"}

    for key in keys:
        r = rows[key]
        per_lead = np.array([t["csi_per_lead"] for t in r["thr"][1.0]])
        axes[0].plot(leads, per_lead.mean(axis=0), marker="o", ms=3.2, lw=1.6,
                     ls=style[key], color=r["color"], label=r["label"])
    axes[0].set_xlabel("lead time (h)")
    axes[0].set_ylabel("CSI @ 1 mm")
    axes[0].set_xticks(leads)
    csi_top = max(float(np.max(np.array([t["csi_per_lead"] for t in rows[k]["thr"][1.0]])
                                .mean(axis=0))) for k in keys)
    axes[0].set_ylim(-0.01, max(0.40, csi_top + 0.03))
    axes[0].set_title("CSI at 1 mm by lead", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.2)

    x = np.arange(len(keys))
    pod = [float(np.mean([t["pod_mean"] for t in rows[k]["thr"][1.0]])) for k in keys]
    far = [float(np.mean([t["far_mean"] for t in rows[k]["thr"][1.0]])) for k in keys]
    axes[1].bar(x - 0.19, pod, 0.36, color=[rows[k]["color"] for k in keys],
                label="POD @ 1 mm")
    axes[1].bar(x + 0.19, far, 0.36, color=[rows[k]["color"] for k in keys],
                alpha=0.45, hatch="//", label="FAR @ 1 mm")
    for xi, (p, f) in enumerate(zip(pod, far)):
        axes[1].text(xi - 0.19, p + 0.012, f"{p:.3f}", ha="center", fontsize=7)
        axes[1].text(xi + 0.19, f + 0.012, f"{f:.3f}", ha="center", fontsize=7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([rows[k]["label"].replace(" (", "\n(") for k in keys],
                            fontsize=7.4)
    axes[1].set_ylim(0, max(0.72, max(pod + far) + 0.08))
    axes[1].set_ylabel("score (lead-averaged)")
    axes[1].set_title("POD / FAR at 1 mm", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7.2, ncol=2)

    fig.tight_layout()
    stamp(fig, "source: logs/eval_*.json thresholds block at 1 mm; 3-seed means.\n"
               "Per-lead POD/FAR are not stored by the evaluator, so the right panel "
               "is lead-averaged.")
    fig.savefig(OUT / "figD_pod_far.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figD_pod_far.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    rows = collect()
    print("rows loaded:", {k: (v["label"], v["n_seeds"], round(v["overall"], 4))
                           for k, v in rows.items()})
    fig_lead_mse(rows)
    fig_csi_threshold(rows)
    fig_cost()
    fig_pod_far(rows)
    for p in sorted(OUT.glob("*.png")):
        print(f"{p.name}  {p.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()

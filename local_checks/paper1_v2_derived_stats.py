"""Paper 1 V2 derived statistics (2026-10-27).

Computes the new numbers that the V2 manuscript adds on top of the locked
main-protocol tables. Every output is derived from already-released or
already-archived artifacts; no model is retrained and no test window is used
for any selection (these are descriptive statistics of the test year and of
the frozen test-set predictions).

Inputs
  - datasets_rainnet/RainNet_HDF5.zip : the official archive (85 monthly HDF5
    files). Only the five 2018 members are read, one at a time, via a
    temporary extraction that is deleted after each month.
  - artifacts/final_20261022/full/ext_eval/ext_*.npz : per-window metrics and
    the two sample-window field dumps (windows 667 and 865).

Outputs
  - artifacts/paper1_v2/derived_stats.json    : all scalar results (ledgered)
  - artifacts/paper1_v2/derived_spatial_2018.npz : spatial mean map and rain
    frequency map of the test year, for Figure 2

Statistics produced
  1. Test-year pixel-level rain-rate distribution: total pixels, per-threshold
     exceedance fractions at 0.1/0.5/1/2/5/10 mm, and a histogram of positive
     hourly accumulations.
  2. Test-year spatial climatology: mean hourly accumulation per pixel and
     rainy-hour frequency (fraction of hours with >= 0.1 mm).
  3. Per-window paired win rates: share of the 1500 test windows on which the
     seed-mean per-window MSE of SimVP is lower than that of ConvLSTM-ED (MSE)
     and of pysteps S-PROG, overall and conditioned on the window's observed
     maximum rain rate (>= 5 mm, >= 10 mm).
  4. Case-study per-lead MSE (mm^2) of every dumped model on windows 667 and
     865, plus the observed per-lead mean and maximum.

Run:  python local_checks/paper1_v2_derived_stats.py
"""

import json
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ZIP = ROOT / "datasets_rainnet" / "RainNet_HDF5.zip"
EXT = ROOT / "artifacts" / "final_20261022" / "full" / "ext_eval"
OUT = ROOT / "artifacts" / "paper1_v2"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
HIST_EDGES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 75.0, 200.0]
MONTHS_2018 = ["2018_07.hdf5", "2018_08.hdf5", "2018_09.hdf5",
               "2018_10.hdf5", "2018_11.hdf5"]
UNIT = 140.0  # normalisation scale used by the archive recipe

# per-window npz tags for the paired win-rate analysis
TAG_SEEDS = {
    "simvp": ["ext_simvp_mse_rainnet_seed0.npz",
              "ext_simvp_mse_rainnet_seed1.npz",
              "ext_simvp_mse_rainnet_seed2.npz"],
    "ed_mse": ["ext_convlstm_ed_mse_rainnet_seed0.npz",
               "ext_convlstm_ed_mse_rainnet_seed1.npz",
               "ext_convlstm_ed_mse_rainnet_seed2.npz"],
    "sprog": ["ext_pysteps_sprog.npz"],
}
CASE_MODELS = [
    ("persistence", "ext_persistence"),
    ("optical_flow", "ext_optical_flow"),
    ("pysteps_lk", "ext_pysteps_lk_extrap"),
    ("pysteps_sprog", "ext_pysteps_sprog"),
    ("as_released", "ext_convlstm_rainnet_seed0"),
    ("ed_mse", "ext_convlstm_ed_mse_rainnet_seed0"),
    ("unet_mse", "ext_unet_mse_rainnet_seed0"),
    ("simvp", "ext_simvp_mse_rainnet_seed0"),
    ("tau", "ext_tau_ddr_rainnet_seed0"),
]
CASE_WINDOWS = [667, 865]


def month_accumulators(arr):
    """Statistics of one monthly lowres array (208, 333, T) in mm."""
    n_pix = arr.shape[0] * arr.shape[1]
    t = arr.shape[2]
    flat = arr.reshape(n_pix, t)
    acc = {
        "frames": t,
        "pixels": n_pix * t,
        "exceed": {str(th): int((flat >= th).sum()) for th in THRESHOLDS},
        "hist": [int(((flat >= lo) & (flat < hi)).sum())
                 for lo, hi in zip(HIST_EDGES[:-1], HIST_EDGES[1:])],
        "sum_map": flat.sum(axis=1).reshape(arr.shape[0], arr.shape[1]),
        "rain_map": (flat >= 0.1).sum(axis=1).reshape(arr.shape[0], arr.shape[1]),
        "max": float(flat.max()),
    }
    return acc


def test_year_stats():
    totals = {"frames": 0, "pixels": 0,
              "exceed": {str(th): 0 for th in THRESHOLDS},
              "hist": [0] * (len(HIST_EDGES) - 1), "max": 0.0}
    sum_map = None
    rain_map = None
    per_month = {}
    with zipfile.ZipFile(ZIP) as zf:
        for name in MONTHS_2018:
            with tempfile.TemporaryDirectory(dir=OUT) as td:
                tmp = zf.extract(name, path=td)
                with h5py.File(tmp, "r") as hf:
                    arr = hf["lowres"][:].astype(np.float64)
            acc = month_accumulators(arr)
            per_month[name] = {k: v for k, v in acc.items()
                               if k not in ("sum_map", "rain_map")}
            totals["frames"] += acc["frames"]
            totals["pixels"] += acc["pixels"]
            for th in THRESHOLDS:
                totals["exceed"][str(th)] += acc["exceed"][str(th)]
            totals["hist"] = [a + b for a, b in zip(totals["hist"], acc["hist"])]
            totals["max"] = max(totals["max"], acc["max"])
            sum_map = acc["sum_map"] if sum_map is None else sum_map + acc["sum_map"]
            rain_map = acc["rain_map"] if rain_map is None else rain_map + acc["rain_map"]
            print(f"  {name}: frames={acc['frames']} max={acc['max']:.2f} mm")
            del arr
    mean_map = sum_map / totals["frames"]
    freq_map = rain_map / totals["frames"]
    np.savez(OUT / "derived_spatial_2018.npz",
             mean_mm_per_h=mean_map.astype(np.float32),
             rain_freq=freq_map.astype(np.float32),
             frames=totals["frames"])
    fracs = {th: totals["exceed"][th] / totals["pixels"] for th in
             [str(t) for t in THRESHOLDS]}
    return {"frames": totals["frames"], "pixels": totals["pixels"],
            "max_mm": round(totals["max"], 2),
            "exceedance_fraction": {k: round(v, 6) for k, v in fracs.items()},
            "hist_edges_mm": HIST_EDGES, "hist_counts": totals["hist"],
            "per_month": per_month}


def seed_mean_win_mse(tags):
    arrs = []
    for t in tags:
        d = np.load(EXT / t)
        arrs.append(d["win_mse_norm"].astype(np.float64))
    return np.mean(arrs, axis=0) * UNIT ** 2  # mm^2


def win_rates():
    simvp = seed_mean_win_mse(TAG_SEEDS["simvp"])
    ed = seed_mean_win_mse(TAG_SEEDS["ed_mse"])
    sprog = seed_mean_win_mse(TAG_SEEDS["sprog"])
    obs = np.load(EXT / "ext_obs_stats.npz")
    wmax = obs["win_max_obs_mm"]

    def rate(a, b, mask):
        d = (a - b)[mask]
        return {"n_windows": int(mask.sum()),
                "win_rate": round(float(np.mean(d < 0)), 4),
                "median_diff_mm2": round(float(np.median(d)), 5),
                "mean_diff_mm2": round(float(np.mean(d)), 5)}

    masks = {"all": np.ones_like(wmax, dtype=bool),
             "obs_max_ge_5mm": wmax >= 5.0,
             "obs_max_ge_10mm": wmax >= 10.0}
    out = {}
    for name, (a, b) in {"simvp_minus_ed_mse": (simvp, ed),
                         "simvp_minus_sprog": (simvp, sprog)}.items():
        out[name] = {m: rate(a, b, masks[m]) for m in masks}
    return out


def window_stats():
    """Distribution of the per-window observed intensity over the 1500 test
    windows (max and mean over each window's ten target hours)."""
    obs = np.load(EXT / "ext_obs_stats.npz")
    mx = obs["win_max_obs_mm"].astype(np.float64)
    mn = obs["win_mean_obs_mm"].astype(np.float64)
    qs = (5, 25, 50, 75, 95, 99)
    return {
        "n_windows": int(mx.size),
        "win_max_obs_mm_quantiles": {f"q{p}": round(float(np.percentile(mx, p)), 3)
                                     for p in qs},
        "win_mean_obs_mm_quantiles": {f"q{p}": round(float(np.percentile(mn, p)), 3)
                                      for p in qs},
        "n_win_max_ge_5mm": int((mx >= 5.0).sum()),
        "n_win_max_ge_10mm": int((mx >= 10.0).sum()),
        "n_win_max_ge_20mm": int((mx >= 20.0).sum()),
    }


def case_study():
    out = {}
    for w in CASE_WINDOWS:
        entry = {}
        for model, stem in CASE_MODELS:
            d = np.load(EXT / f"{stem}_win{w}.npz")
            pred = d["pred"].astype(np.float64) * UNIT
            ob = d["obs"].astype(np.float64) * UNIT
            mse_lead = ((pred - ob) ** 2).mean(axis=(1, 2))
            entry[model] = {
                "mse_per_lead_mm2": [round(float(x), 4) for x in mse_lead],
                "pred_max_per_lead_mm": [round(float(x), 2)
                                         for x in pred.max(axis=(1, 2))],
                "file_idx": int(d["file_idx"]), "t0": int(d["t0"])}
        d0 = np.load(EXT / f"{CASE_MODELS[0][1]}_win{w}.npz")
        ob = d0["obs"].astype(np.float64) * UNIT
        entry["observed"] = {
            "mean_per_lead_mm": [round(float(x), 4) for x in ob.mean(axis=(1, 2))],
            "max_per_lead_mm": [round(float(x), 2) for x in ob.max(axis=(1, 2))]}
        out[str(w)] = entry
    return out


def regional_stats():
    """Land-mask and regional means from the saved 2018 spatial accumulators.

    Grid convention (verified against the coastline in the mean map):
    row 0 = 50 deg N, rows run south (25 deg N at row 207); col 0 = 105 deg W,
    cols run east (65 deg W at col 332). Spacing 25/208 deg per row and
    40/333 deg per col. Ocean pixels of the NLDAS land analysis are zero in
    every frame and are excluded by the land mask."""
    s = np.load(OUT / "derived_spatial_2018.npz")
    mm = s["mean_mm_per_h"].astype(np.float64)
    fq = s["rain_freq"].astype(np.float64)
    land = mm > 0
    row = lambda la: int(round((50.0 - la) * 208 / 25.0))
    col = lambda lo: int(round((lo + 105.0) * 333 / 40.0))

    def box(lat_hi, lat_lo, lon_w, lon_e):
        sub = mm[row(lat_hi):row(lat_lo), col(lon_w):col(lon_e)]
        m = sub > 0
        return round(float(sub[m].mean()), 4) if m.any() else None

    def latlon(idx):
        return [round(50.0 - idx[0] * 25.0 / 208.0, 2),
                round(-105.0 + idx[1] * 40.0 / 333.0, 2)]

    return {
        "land_pixels": int(land.sum()), "grid_pixels": int(mm.size),
        "land_fraction": round(float(land.mean()), 4),
        "land_mean_mm_per_h": round(float(mm[land].mean()), 4),
        "land_rain_freq": round(float(fq[land].mean()), 4),
        "max_mean_mm_per_h": {"value": round(float(mm.max()), 4),
                              "lat_lon": latlon(np.unravel_index(mm.argmax(), mm.shape))},
        "max_rain_freq": {"value": round(float(fq.max()), 4),
                          "lat_lon": latlon(np.unravel_index(fq.argmax(), fq.shape))},
        "regional_land_mean_mm_per_h": {
            "southeast_atlantic_coast_32_35N_81_76W": box(35, 32, -81, -76),
            "gulf_coast_29_31N_95_85W": box(31, 29, -95, -85),
            "northeast_40_47N_80_70W": box(47, 40, -80, -70),
            "great_plains_35_45N_105_97W": box(45, 35, -105, -97)},
    }


def main():
    import argparse
    global ZIP, EXT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ZIP)
    ap.add_argument("--ext-dir", type=Path, default=EXT)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--reuse-test-year", action="store_true",
                    help="keep test_year/win_rates/window_stats/spatial blocks from "
                         "the existing derived_stats.json and recompute only the "
                         "case-study block (used when extending case-study fields; "
                         "a full run without this flag reproduces everything)")
    a = ap.parse_args()
    ZIP, EXT, OUT = a.archive.resolve(), a.ext_dir.resolve(), a.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    prior = None
    if a.reuse_test_year:
        prior = json.loads((OUT / "derived_stats.json").read_text())
    if prior is None:
        print("test-year pixel statistics from the 2018 archive members ...")
        ty = test_year_stats()
        print("paired per-window win rates ...")
        wr = win_rates()
        print("per-window intensity distribution ...")
        ws = window_stats()
        print("regional land statistics ...")
        rg = regional_stats()
    else:
        print("reusing test-year, win-rate, window and spatial blocks from the "
              "existing derived_stats.json")
        ty, wr, ws, rg = (prior["test_year"], prior["win_rates"],
                          prior["window_stats"], prior["spatial"])
    print("case-study per-lead MSE ...")
    cs = case_study()
    result = {
        "provenance": {
            "script": "local_checks/paper1_v2_derived_stats.py",
            "date": "2026-10-27",
            "zip": str(ZIP.name),
            "ext_dir": str(EXT),
            "note": ("descriptive statistics only; no selection, no retraining; "
                     "pixel stats from the five 2018 archive members, win rates "
                     "from the frozen per-window test outputs")},
        "test_year": ty,
        "win_rates": wr,
        "window_stats": ws,
        "case_study": cs,
        "spatial": rg,
    }
    with open(OUT / "derived_stats.json", "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", OUT / "derived_stats.json")
    print("wrote", OUT / "derived_spatial_2018.npz")


if __name__ == "__main__":
    main()

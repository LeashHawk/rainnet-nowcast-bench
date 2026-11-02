"""Run a CPU-only evaluator smoke test without downloading RainNet.

The demo creates two small synthetic monthly arrays in a temporary directory,
runs the evaluator's identity canary on eight windows, and checks that the
resulting error is zero. The identity canary copies targets by design; this is
not a forecast and its metrics are not scientific results.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HEIGHT, WIDTH, FRAMES = 32, 64, 720


def make_synthetic_data(root):
    """Create a RainNet-shaped directory and an index for the smoke test."""
    lowres = root / "lowres"
    lowres.mkdir(parents=True)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]

    filenames = ["2018_07.npy", "2018_08.npy"]
    for file_index, filename in enumerate(filenames):
        fields = np.zeros((HEIGHT, WIDTH, FRAMES), dtype=np.float16)
        for frame in range(FRAMES):
            amplitude = 0.05 + 0.45 * (
                0.5 + 0.5 * np.sin(2 * np.pi * (frame + 7 * file_index) / 24.0)
            )
            center_x = 8 + 1.1 * frame
            center_y = 16 + 0.9 * frame + 3 * file_index
            field = amplitude * np.exp(
                -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * 5.0**2)
            )
            fields[:, :, frame] = field.astype(np.float16)
        np.save(lowres / filename, fields)

    index = {
        "files": filenames,
        "train": [
            [file_index, start]
            for file_index in range(2)
            for start in range(FRAMES - 19)
        ],
        "test": (
            [[1, start] for start in range(0, 28, 3)]
            + [[0, start] for start in (2, 9, 16, 23)]
        ),
    }
    (root / "rainnet_index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def main():
    with tempfile.TemporaryDirectory(prefix="rainnet-synthetic-demo-") as temp:
        work = Path(temp)
        data_dir = work / "data"
        output_dir = work / "output"
        make_synthetic_data(data_dir)

        command = [
            sys.executable,
            str(ROOT / "benchmark_tools" / "rainnet_extended_eval.py"),
            "--data-path",
            str(data_dir),
            "--model",
            "identity",
            "--tag",
            "synthetic_demo",
            "--out-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--limit",
            "8",
            "--sample-indices",
            "0",
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(command, check=True, env=environment)

        result_path = output_dir / "ext_synthetic_demo.npz"
        with np.load(result_path, allow_pickle=False) as result:
            meta = json.loads(str(result["meta"]))
            mse_mm2 = (
                float(result["se_total"])
                / int(result["n_seen"])
                * float(meta["unit_scale"]) ** 2
            )
            windows = int(meta["n_windows"])
            if not np.all(result["win_mse_norm"] == 0.0):
                raise RuntimeError("identity canary did not produce zero error")

        print(
            f"SYNTHETIC_DEMO_PASS windows={windows} "
            f"identity_mse_mm2={mse_mm2:.6f}"
        )
        print("Synthetic smoke-test output is temporary, not a paper result.")


if __name__ == "__main__":
    main()

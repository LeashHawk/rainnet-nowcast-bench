"""Strict JSON serialization checks for undefined categorical metrics."""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark_tools"))

from rainnet_nowcast_eval import json_safe  # noqa: E402


def main():
    original = {
        "finite": 1.25,
        "thresholds": [
            {"far_mean": float("nan"), "csi_mean": 0.0},
            {"pod_mean": float("inf"), "hss_mean": -math.inf},
        ],
    }
    safe = json_safe(original)
    serialized = json.dumps(safe, allow_nan=False)
    parsed = json.loads(serialized)
    assert parsed["finite"] == 1.25
    assert parsed["thresholds"][0] == {"far_mean": None, "csi_mean": 0.0}
    assert parsed["thresholds"][1] == {
        "pod_mean": None, "hss_mean": None,
    }
    print("ALL_PASS")


if __name__ == "__main__":
    main()

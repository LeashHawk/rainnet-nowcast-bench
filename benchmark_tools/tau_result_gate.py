"""Validate a completed TAU run before evaluation.

The queue labels an early-stopped job with finite TEST metrics as ``partial``
because it did not reach the planned epoch count.  That label alone is not
enough to decide whether an evaluation is safe.  This gate accepts either a
full-budget clean run or a clean early-stop terminal record with a finite best
validation score, matching finite TEST log/result metrics, and a nonempty best
checkpoint.
"""

import argparse
import json
import math
import re
from pathlib import Path


EARLY_STOP_RE = re.compile(
    r"EARLY_STOP:::after epoch\s+(\d+):.*?best epoch\s+(\d+)\s+val\s+([^\s]+)"
)
TEST_RE = re.compile(r"TEST:::MSE:\s*([^,]+),\s*MAE:\s*([^\s]+)")


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def validate_tau_result(result_path, log_path, checkpoint_path):
    """Return ``(accepted, reason)`` without modifying any input."""
    result_path = Path(result_path)
    log_path = Path(log_path)
    checkpoint_path = Path(checkpoint_path)

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cannot read result JSON: {exc}"

    if result.get("returncode") != 0:
        return False, f"returncode is not 0: {result.get('returncode')!r}"
    if result.get("status") not in ("done", "partial"):
        return False, f"unexpected terminal status: {result.get('status')!r}"
    if not _finite(result.get("test_mse")) or not _finite(result.get("test_mae")):
        return False, "result TEST MSE/MAE is missing or non-finite"

    try:
        epochs = int(result["epochs_completed"])
        planned = int(result["planned_epochs"])
    except (KeyError, TypeError, ValueError):
        return False, "epoch counts are missing or invalid"
    if epochs <= 0 or planned <= 0:
        return False, "epoch counts must be positive"

    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        return False, f"best checkpoint is missing or empty: {checkpoint_path}"

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return False, f"cannot read training log: {exc}"

    if result["status"] == "done":
        if epochs < planned:
            return False, f"done status but only {epochs}/{planned} epochs completed"
    else:
        if epochs >= planned:
            return False, f"partial status but epoch budget is complete ({epochs}/{planned})"
        stop_matches = list(EARLY_STOP_RE.finditer(log_text))
        if not stop_matches:
            return False, "partial status has no explicit early-stop record"
        stop_epoch, best_epoch, best_val = stop_matches[-1].groups()
        if int(stop_epoch) != epochs:
            return False, f"early-stop epoch {stop_epoch} disagrees with results ({epochs})"
        if int(best_epoch) <= 0 or int(best_epoch) > epochs or not _finite(best_val):
            return False, "best validation epoch/score is invalid or non-finite"

    test_matches = list(TEST_RE.finditer(log_text))
    if not test_matches:
        return False, "training log has no terminal TEST:::MSE/MAE record"
    log_mse, log_mae = test_matches[-1].groups()
    if not _finite(log_mse) or not _finite(log_mae):
        return False, "terminal TEST:::MSE/MAE log values are non-finite"
    if not math.isclose(float(log_mse), float(result["test_mse"]),
                        rel_tol=1e-6, abs_tol=1e-12):
        return False, "TEST MSE in log disagrees with results JSON"
    if not math.isclose(float(log_mae), float(result["test_mae"]),
                        rel_tol=1e-6, abs_tol=1e-12):
        return False, "TEST MAE in log disagrees with results JSON"

    return True, (
        f"accepted {result['status']} run ({epochs}/{planned} epochs), "
        "finite best validation score, TEST metrics, and checkpoint"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json")
    parser.add_argument("train_log")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    accepted, reason = validate_tau_result(
        args.results_json, args.train_log, args.checkpoint
    )
    print(("ACCEPT: " if accepted else "REJECT: ") + reason)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

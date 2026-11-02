"""Focused tests for the TAU early-stop evaluation gate."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark_tools"))
from tau_result_gate import validate_tau_result  # noqa: E402


def write_case(root, *, status="partial", rc=0, epochs=17, planned=50,
               test_mse=1.58e-5, test_mae=0.0009538,
               early_stop=True, best_val="0.0000133", log_test=True,
               log_mse=None, log_mae=None, checkpoint=True):
    root.mkdir(parents=True, exist_ok=True)
    results = root / "results.json"
    log = root / "train.log"
    ckpt = root / "best.pth"
    results.write_text(json.dumps({
        "status": status, "returncode": rc, "epochs_completed": epochs,
        "planned_epochs": planned, "test_mse": test_mse, "test_mae": test_mae,
    }), encoding="utf-8")
    lines = []
    if early_stop:
        lines.append(
            f"EARLY_STOP:::after epoch {epochs}: no val improvement for 10 "
            f"epochs (patience 10); best epoch 7 val {best_val}"
        )
    if log_test:
        logged_mse = test_mse if log_mse is None else log_mse
        logged_mae = test_mae if log_mae is None else log_mae
        lines.append(f"TEST:::MSE: {logged_mse}, MAE: {logged_mae}")
    log.write_text("\n".join(lines), encoding="utf-8")
    if checkpoint:
        ckpt.write_bytes(b"nonempty checkpoint")
    return results, log, ckpt


def run():
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        paths = write_case(root / "valid_partial")
        ok, why = validate_tau_result(*paths)
        if not ok:
            failures.append(f"valid early-stop rejected: {why}")

        paths = write_case(root / "valid_done", status="done", epochs=50,
                           early_stop=False)
        ok, why = validate_tau_result(*paths)
        if not ok:
            failures.append(f"valid full-budget run rejected: {why}")

        paths = write_case(root / "no_early_stop", early_stop=False)
        if validate_tau_result(*paths)[0]:
            failures.append("partial run without early-stop marker was accepted")

        paths = write_case(root / "nan_best", best_val="nan")
        if validate_tau_result(*paths)[0]:
            failures.append("non-finite best validation score was accepted")

        paths = write_case(root / "nan_test", test_mse=float("nan"))
        if validate_tau_result(*paths)[0]:
            failures.append("non-finite TEST score was accepted")

        paths = write_case(root / "missing_ckpt", checkpoint=False)
        if validate_tau_result(*paths)[0]:
            failures.append("missing checkpoint was accepted")

        paths = write_case(root / "no_test", log_test=False)
        if validate_tau_result(*paths)[0]:
            failures.append("run without TEST log record was accepted")

        paths = write_case(root / "mismatch", test_mse=1.6e-5,
                           log_mse=1.58e-5)
        if validate_tau_result(*paths)[0]:
            failures.append("mismatched TEST log and results were accepted")

    print("ALL_PASS" if not failures else "FAILED")
    for failure in failures:
        print(" - " + failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())

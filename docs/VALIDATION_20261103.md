# Local validation — 2026-11-03

Environment: Windows, Python 3.10.20, PyTorch 2.1.2+cu121, NumPy 1.26.4.
The paper environment is specified in `requirements.txt`. The GitHub Actions
workflow in `.github/workflows/checks.yml` specifies Python 3.10 and its
validation dependencies.

Passed:
- Original archive: all 124 manifest entries verified; edited documentation
  verified against saved originals under docs/provenance/.
- Repository manifest, Python syntax, 24 strict evaluation JSONs.
- examples/synthetic_demo.py (software canary, no forecast-skill claim).
- local_checks/test_nowcast_models.py
- local_checks/test_extended_eval.py
- local_checks/test_ext_analysis.py
- local_checks/test_tau_result_gate.py
- local_checks/test_eval_json_safety.py
- local_checks/test_blocked_index.py
- scripts/train_seeded.py --help.

Git includes the 46 shipped NPZ artifacts. The checks above validate the
repository files, evaluator behaviour, model interfaces, and index builder.

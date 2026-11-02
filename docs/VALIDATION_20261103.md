# Local validation — 2026-11-03

Environment: Windows, Python 3.10.20, PyTorch 2.1.2+cu121, NumPy 1.26.4.
This differs from the pinned paper environment; the GitHub CI workflow has been
prepared but has not run on a remote service.

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
- scripts/train_seeded.py --help (training itself was not rerun).

Git includes the 46 shipped NPZ artifacts. No full data download, model
training, checkpoint verification, or missing blocked-protocol experiment
reconstruction was performed. No remote is configured. No initial commit was
created because this machine has no configured Git user.name/user.email;
staged files are ready for review. No author identity was invented.

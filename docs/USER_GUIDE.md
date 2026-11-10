# User guide

This guide covers the released RainNet hourly precipitation nowcasting
benchmark. It describes data preparation, model training, evaluation, and the
validation rules needed to avoid using the test set for model selection.

## What is included

- `corrected/`: benchmark model definitions and training entry point.
- `benchmark_tools/`: RainNet data preparation and evaluation scripts.
- `local_checks/`: tests, analysis, and figure/table generation.
- `results/`: frozen main split, evaluation JSON files, and paper analysis.
- `data_samples/`: five real 20-frame inspection windows with their own
  README and manifest; they are not a split and feed no evaluation.
- `CHECKPOINTS.sha256`: recorded hashes for verifying model weights.

The archive ships only those five small excerpts. Everything else requires
downloading the [official RainNet repository](https://github.com/neuralchen/RainNet)
data; cite Chen et al. (2022). The dataset is distributed under CC BY; follow
the dataset's terms when using or redistributing it.

## Requirements

Use Python 3.8–3.10 (Python 3.10 is recommended). Install the pinned packages:

```bash
python -m pip install -r requirements.txt
```

Training benefits from a CUDA GPU. Evaluation and the synthetic demo can run
on CPU. The optical-flow baselines require the optional `pysteps` and OpenCV
packages listed in `requirements.txt`. Keep the pinned OpenCV 4.10 release:
OpenCV 5.x can require NumPy 2.x, which is incompatible with the pinned
PyTorch 1.12.1 environment.

## Prepare the RainNet data

1. Download `RainNet_HDF5.zip` from the official repository's linked data
   archive.
2. Extract and normalize only the low-resolution NLDAS product:

   ```bash
   python benchmark_tools/rainnet_build_lowres.py \
     --zip /path/to/RainNet_HDF5.zip \
     --out data/rainnet/lowres
   ```

   The output contains one normalized float16 `.npy` file per month. Values
   follow the dataset recipe, `clip[0, 140] / 140`.
3. Build the fixed window index:

   ```bash
   python benchmark_tools/rainnet_build_index.py \
     --lowres-dir data/rainnet/lowres \
     --out data/rainnet/rainnet_index.json
   ```

   For the paper's exact split, use the frozen
   `results/rainnet_index.json` instead. Keep this index unchanged when
   reproducing the published evaluation.

The loader selects 750 validation windows from the 6000 indexed training
windows using its fixed seed, leaving 5250 training windows. The 1500 test
windows come from 2018 and remain separate. Use validation data for every
checkpoint, architecture, and parameter choice. Use the test split only for
final evaluation.

## Train a model

Create output directories, then launch one model:

```bash
mkdir -p logs checkpoints
python corrected/main_nowcast_rainnet.py \
  --arch simvp \
  --loss mse \
  --data_path data/rainnet \
  --batch_size 4 \
  --lr 1e-3 \
  --lr_schedule cosine \
  --num_epochs 50 \
  --early_stop_patience 10 \
  --log_path logs/simvp_mse.log \
  --checkpoints checkpoints/simvp_mse.pth
```

Available architectures are `convlstm_ed`, `unet`, `simvp`, and `tau`.
Available losses are `mse`, `balanced`, and `tau_ddr`; not every loss is
intended for every architecture. Checkpoint selection uses validation MSE.
The training command writes its selected checkpoint to the path supplied
with `--checkpoints`. Use `CHECKPOINTS.sha256` to compare checkpoint hashes
and the archived evaluation JSON files to identify the reported runs.

## Evaluate

The standard evaluator writes paper metrics to JSON:

```bash
python benchmark_tools/rainnet_nowcast_eval.py \
  --data-path data/rainnet \
  --model simvp \
  --checkpoints checkpoints/simvp_mse.pth \
  --out results/eval_simvp.json
```

The extended evaluator writes per-window metrics and FSS accumulators:

```bash
python benchmark_tools/rainnet_extended_eval.py \
  --data-path data/rainnet \
  --model simvp \
  --checkpoints checkpoints/simvp_mse.pth \
  --tag simvp_mse \
  --out-dir artifacts/ext_eval
```

Use `--split val` only for validation diagnostics or parameter tuning. Never
use validation output as a paper test result. The `identity` model is a
test-only canary that copies targets; it is not a forecasting baseline.

For the deterministic pysteps baselines, tune parameters on validation first.
The paper's selected configurations and validation-only tuning procedure are
documented in `results/analysis/` and in the paper appendices. Run each chosen
configuration on the test split only once.

## Run tests and analysis

Run the self-contained regression tests from the repository root:

```bash
python local_checks/test_nowcast_models.py
python local_checks/test_extended_eval.py
python local_checks/test_ext_analysis.py
python local_checks/test_tau_result_gate.py
python local_checks/test_eval_json_safety.py
```

The extended per-window `ext_*.npz` files of every benchmark row are shipped
in `results/ext_eval/`. To recompute the bootstrap intervals, run:

```bash
python local_checks/ext_analysis_paper1.py \
  --ext-dir results/ext_eval \
  --eval-dir results/eval_jsons \
  --index-json results/rainnet_index.json
```

The shipped `results/analysis/ext_summary.json` can be used to render the
Appendix B tables without rerunning the bootstrap:

```bash
python local_checks/appendix_b_tables_paper1.py \
  results/analysis/ext_summary.json \
  > results/analysis/appendix_b_tables.generated.md
```

Use the input artifacts and provenance fields recorded in each evaluation
JSON when reconstructing the reported results.

## Synthetic demo

To check that the extended evaluator runs without downloading RainNet:

```bash
python examples/synthetic_demo.py
```

The script generates a small artificial dataset in a temporary directory,
runs an identity canary on CPU, prints its zero-error check, and removes only
the temporary directory it created. This is a software smoke test, not a
forecast-skill result.

## Data and code citation

The RainNet archive is a separate dataset. Cite Chen et al. (2022) and obtain
the data from the [official repository](https://github.com/neuralchen/RainNet).
For the benchmark code, cite the [public repository](https://github.com/LeashHawk/rainnet-nowcast-bench)
and identify the [v1.0 release](https://github.com/LeashHawk/rainnet-nowcast-bench/tree/v1.0).

## Local audit on 2026-11-05

The frozen main-protocol metrics were reconstructed, and model/seed window
identities were checked. A post-hoc 1/2/3/5/7-day MSE block-length audit
preserved the signs of three selected differences. These checks use the
stored main-protocol outputs. Stored categorical counts are aggregated
over windows.

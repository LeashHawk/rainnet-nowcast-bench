# Tutorial: run one RainNet nowcast

This tutorial trains and evaluates one SimVP configuration. It uses the
RainNet low-resolution hourly product. For a no-download smoke test, run
[`../examples/synthetic_demo.py`](../examples/synthetic_demo.py) instead.

## 1. Install the environment

From the repository root, use Python 3.8–3.10 (3.10 recommended):

```bash
python -m pip install -r requirements.txt
```

The example can train on a CUDA GPU. CPU is sufficient for evaluation.

## 2. Download and prepare data

Download `RainNet_HDF5.zip` from the data link in the
[official RainNet repository](https://github.com/neuralchen/RainNet). Keep the
archive outside Git. Extract the low-resolution product:

```bash
python benchmark_tools/rainnet_build_lowres.py \
  --zip /data/RainNet_HDF5.zip \
  --out data/rainnet/lowres
```

For an exact reproduction of the paper, use the frozen index shipped at
`results/rainnet_index.json`. To build a fresh index for exploratory runs:

```bash
python benchmark_tools/rainnet_build_index.py \
  --lowres-dir data/rainnet/lowres \
  --out data/rainnet/rainnet_index.json
```

The loader makes its validation carve from the indexed training windows. Do
not use the test set to tune a model or select parameters.

## 3. Train SimVP

Create the output directories and train:

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
  --log_path logs/simvp_mse_seed0.log \
  --checkpoints checkpoints/simvp_mse_seed0.pth
```

The run monitors validation MSE and writes the selected checkpoint to the
path supplied with `--checkpoints`.

## 4. Evaluate on the fixed test split

Write the standard score file:

```bash
python benchmark_tools/rainnet_nowcast_eval.py \
  --data-path data/rainnet \
  --model simvp \
  --checkpoints checkpoints/simvp_mse_seed0.pth \
  --out results/eval_simvp_mse_seed0.json
```

For window-level paired analysis and spatial verification accumulators, run:

```bash
python benchmark_tools/rainnet_extended_eval.py \
  --data-path data/rainnet \
  --model simvp \
  --checkpoints checkpoints/simvp_mse_seed0.pth \
  --tag simvp_mse_seed0 \
  --out-dir artifacts/ext_eval
```

The files in `results/eval_jsons/` and `results/analysis/` are the provenance
for the published numbers. A newly trained checkpoint produces a new run; do
not overwrite those archived results.

## 5. Check the setup without RainNet

Before downloading the archive, verify the evaluation code with synthetic
inputs:

```bash
python examples/synthetic_demo.py
```

This invokes the `identity` canary, which returns the synthetic targets
unchanged. A zero error confirms that the data layout and evaluator work; it
does not indicate forecast skill.

To inspect real archive windows without the download, print the per-frame
statistics of a shipped sample (add `--png` for a 20-frame panel):

```bash
python examples/view_data_sample.py \
  data_samples/lowres_samples/2002_07_w0489.npy
```

The five samples under `data_samples/` are inspection excerpts from one
training-side month; see `data_samples/README.md`.

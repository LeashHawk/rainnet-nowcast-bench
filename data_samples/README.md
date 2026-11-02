# Data samples

Five real 20-frame windows (208 × 333, hourly, raw mm) from the RainNet
low-resolution NLDAS product, extracted from one training-side month
(2002_07). They let you inspect the actual data before committing to the
13.6 GB download.

- `lowres_samples/2002_07_w*.npy` — one window per file: 10 input hours +
  10 target hours, float32, archive values copied without scaling.
- `samples_manifest.json` — source member, start hour, value range, sha256,
  and the selection rule for every file.
- `examples/view_data_sample.py` (repository root) prints per-frame
  statistics and can render a 20-frame panel:
  `python examples/view_data_sample.py data_samples/lowres_samples/2002_07_w0489.npy`

Selection: scanning every 20-frame window of the month, the three
non-overlapping windows with the highest mean target rain rate and the two
closest to the month median of that statistic were kept. The rule is fixed
in `samples_manifest.json`; the extracting tool is
`benchmark_tools/extract_data_samples.py`.

These are inspection excerpts, not data. They are not part of any
train/validation/test split, no evaluation in the paper uses them, and they
cannot replace the archive download for any experiment. The benchmark
pipeline normalizes with `clip[0, 140] mm / 140` before model use; apply that
recipe to these values first.

The RainNet archive is Chen et al. (2022), distributed under CC BY from
https://github.com/neuralchen/RainNet. Cite the dataset paper when you use
or redistribute these excerpts.

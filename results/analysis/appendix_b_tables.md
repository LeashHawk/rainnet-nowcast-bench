**Table B1. Lead-mean FSS** (seed-mean for deep rows; neighborhoods are square side lengths in pixels).


*Threshold 0.5 mm*

| Model | 1 px | 3 px | 5 px | 9 px | 17 px |
|---|---:|---:|---:|---:|---:|
| Persistence (Eulerian) | 0.327 | 0.385 | 0.422 | 0.481 | 0.567 |
| Optical flow (Lagrangian) | 0.364 | 0.438 | 0.483 | 0.549 | 0.637 |
| pysteps LK extrapolation | 0.370 | 0.439 | 0.485 | 0.557 | 0.653 |
| pysteps S-PROG | 0.399 | 0.451 | 0.485 | 0.537 | 0.610 |
| ConvLSTM (as released) | 0.106 | 0.128 | 0.144 | 0.173 | 0.222 |
| ConvLSTM-ED (MSE) | 0.455 | 0.514 | 0.548 | 0.598 | 0.663 |
| U-Net (MSE) | 0.453 | 0.510 | 0.544 | 0.595 | 0.662 |
| ConvLSTM-ED (balanced) | 0.342 | 0.390 | 0.420 | 0.468 | 0.538 |
| U-Net (balanced) | 0.387 | 0.440 | 0.473 | 0.522 | 0.588 |
| SimVP v1 (MSE) | **0.467** | **0.523** | **0.557** | **0.608** | **0.674** |
| TAU (SimVP v2 + attention) | 0.416 | 0.473 | 0.509 | 0.564 | 0.638 |

*Threshold 1.0 mm*

| Model | 1 px | 3 px | 5 px | 9 px | 17 px |
|---|---:|---:|---:|---:|---:|
| Persistence (Eulerian) | 0.259 | 0.314 | 0.351 | 0.412 | 0.504 |
| Optical flow (Lagrangian) | 0.299 | 0.372 | 0.418 | 0.487 | 0.581 |
| pysteps LK extrapolation | 0.307 | 0.375 | 0.422 | 0.497 | 0.602 |
| pysteps S-PROG | 0.342 | 0.395 | 0.430 | 0.485 | 0.562 |
| ConvLSTM (as released) | 0.009 | 0.012 | 0.013 | 0.017 | 0.022 |
| ConvLSTM-ED (MSE) | 0.361 | 0.425 | 0.464 | 0.523 | 0.602 |
| U-Net (MSE) | 0.361 | 0.422 | 0.460 | 0.519 | 0.598 |
| ConvLSTM-ED (balanced) | 0.279 | 0.327 | 0.358 | 0.407 | 0.482 |
| U-Net (balanced) | 0.331 | 0.385 | 0.419 | 0.472 | 0.545 |
| SimVP v1 (MSE) | **0.376** | **0.438** | **0.477** | **0.537** | **0.617** |
| TAU (SimVP v2 + attention) | 0.317 | 0.373 | 0.410 | 0.468 | 0.546 |

*Threshold 2.0 mm*

| Model | 1 px | 3 px | 5 px | 9 px | 17 px |
|---|---:|---:|---:|---:|---:|
| Persistence (Eulerian) | 0.183 | 0.235 | 0.272 | 0.335 | 0.433 |
| Optical flow (Lagrangian) | 0.223 | 0.292 | 0.338 | 0.410 | 0.511 |
| pysteps LK extrapolation | 0.232 | 0.299 | 0.347 | **0.425** | **0.537** |
| pysteps S-PROG | **0.266** | **0.319** | 0.355 | 0.412 | 0.495 |
| ConvLSTM (as released) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ConvLSTM-ED (MSE) | 0.237 | 0.295 | 0.333 | 0.390 | 0.463 |
| U-Net (MSE) | 0.246 | 0.304 | 0.341 | 0.400 | 0.476 |
| ConvLSTM-ED (balanced) | 0.201 | 0.245 | 0.275 | 0.323 | 0.393 |
| U-Net (balanced) | 0.260 | 0.315 | 0.351 | 0.407 | 0.485 |
| SimVP v1 (MSE) | 0.256 | 0.316 | **0.356** | 0.417 | 0.494 |
| TAU (SimVP v2 + attention) | 0.209 | 0.259 | 0.293 | 0.345 | 0.412 |

**Table B2. Heavy-rain CSI** (counts summed over seeds; lead-mean and per lead +1 h…+10 h).

| Model | CSI@5 mm mean | CSI@10 mm mean | CSI@5 mm by lead | CSI@10 mm by lead |
|---|---:|---:|---|---|
| Persistence (Eulerian) | 0.0526 | 0.0233 | 0.210, 0.094, 0.060, 0.043, 0.033, 0.025, 0.019, 0.016, 0.014, 0.013 | 0.113, 0.041, 0.024, 0.016, 0.011, 0.008, 0.006, 0.005, 0.004, 0.005 |
| Optical flow (Lagrangian) | 0.0735 | 0.0370 | 0.276, 0.142, 0.088, 0.061, 0.045, 0.036, 0.029, 0.023, 0.019, 0.016 | 0.169, 0.071, 0.040, 0.025, 0.018, 0.015, 0.011, 0.008, 0.007, 0.006 |
| pysteps LK extrapolation | 0.0747 | 0.0358 | 0.267, 0.139, 0.090, 0.063, 0.049, 0.039, 0.032, 0.026, 0.023, 0.019 | 0.158, 0.066, 0.037, 0.026, 0.020, 0.016, 0.011, 0.009, 0.008, 0.007 |
| pysteps S-PROG | 0.0890 | 0.0425 | 0.282, 0.166, 0.112, 0.081, 0.064, 0.051, 0.042, 0.035, 0.031, 0.027 | 0.168, 0.077, 0.046, 0.034, 0.027, 0.022, 0.017, 0.014, 0.011, 0.009 |
| ConvLSTM (as released) | 0.0000 | 0.0000 | 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 |
| ConvLSTM-ED (MSE) | 0.0566 | 0.0161 | 0.265, 0.131, 0.070, 0.038, 0.022, 0.015, 0.010, 0.007, 0.005, 0.004 | 0.114, 0.030, 0.010, 0.004, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000 |
| U-Net (MSE) | 0.0589 | 0.0175 | 0.275, 0.139, 0.069, 0.040, 0.025, 0.016, 0.011, 0.007, 0.004, 0.003 | 0.129, 0.032, 0.008, 0.003, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000 |
| ConvLSTM-ED (balanced) | 0.0542 | 0.0122 | 0.256, 0.142, 0.078, 0.039, 0.018, 0.007, 0.002, 0.000, 0.000, 0.000 | 0.100, 0.019, 0.003, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 |
| U-Net (balanced) | 0.0803 | 0.0235 | 0.284, 0.176, 0.120, 0.078, 0.052, 0.035, 0.024, 0.016, 0.011, 0.007 | 0.145, 0.055, 0.018, 0.008, 0.004, 0.002, 0.001, 0.001, 0.001, 0.000 |
| SimVP v1 (MSE) | 0.0625 | 0.0161 | 0.271, 0.149, 0.084, 0.046, 0.027, 0.016, 0.011, 0.009, 0.007, 0.005 | 0.113, 0.037, 0.009, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 |
| TAU (SimVP v2 + attention) | 0.0505 | 0.0126 | 0.246, 0.126, 0.060, 0.028, 0.015, 0.010, 0.007, 0.005, 0.004, 0.003 | 0.094, 0.025, 0.005, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000 |

**Table B3a. Mean per-window MSE with day-block bootstrap 95% intervals** (2000 resamples, seed 20261021; deep rows use the seed-mean per-window MSE).

| Model | seeds | MSE (mm²) | 95% CI |
|---|---:|---:|---|
| Persistence (Eulerian) | 1 | 0.5764 | [0.5106, 0.6386] |
| Optical flow (Lagrangian) | 1 | 0.5325 | [0.4731, 0.5875] |
| pysteps LK extrapolation | 1 | 0.5096 | [0.4524, 0.5643] |
| pysteps S-PROG | 1 | 0.4897 | [0.4351, 0.5408] |
| ConvLSTM (as released) | 3 | 0.3586 | [0.3167, 0.3980] |
| ConvLSTM-ED (MSE) | 3 | 0.2889 | [0.2562, 0.3197] |
| U-Net (MSE) | 3 | 0.2897 | [0.2575, 0.3204] |
| ConvLSTM-ED (balanced) | 3 | 0.3389 | [0.3016, 0.3742] |
| U-Net (balanced) | 3 | 0.3356 | [0.2982, 0.3705] |
| SimVP v1 (MSE) | 3 | 0.2838 | [0.2521, 0.3139] |
| TAU (SimVP v2 + attention) | 3 | 0.2999 | [0.2664, 0.3316] |

**Table B3b. Paired per-window comparisons** (A − B, mm²; same block resampling; p < 0.001 is the bootstrap resolution limit).

| A | B | mean diff | 95% CI | p | CI excludes 0 |
|---|---|---:|---|---:|---|
| SimVP v1 (MSE) | ConvLSTM-ED (MSE) | -0.0051 | [-0.0065, -0.0038] | < 0.001 | yes |
| SimVP v1 (MSE) | U-Net (MSE) | -0.0058 | [-0.0072, -0.0044] | < 0.001 | yes |
| ConvLSTM-ED (MSE) | U-Net (MSE) | -0.0007 | [-0.0021, +0.0006] | 0.259 | no |
| ConvLSTM-ED (MSE) | ConvLSTM (as released) | -0.0696 | [-0.0816, -0.0587] | < 0.001 | yes |
| ConvLSTM-ED (MSE) | Optical flow (Lagrangian) | -0.2436 | [-0.2708, -0.2181] | < 0.001 | yes |
| ConvLSTM-ED (MSE) | ConvLSTM-ED (balanced) | -0.0500 | [-0.0566, -0.0440] | < 0.001 | yes |
| U-Net (MSE) | U-Net (balanced) | -0.0459 | [-0.0519, -0.0401] | < 0.001 | yes |
| TAU (SimVP v2 + attention) | SimVP v1 (MSE) | +0.0161 | [+0.0138, +0.0187] | < 0.001 | yes |
| TAU (SimVP v2 + attention) | ConvLSTM-ED (MSE) | +0.0110 | [+0.0092, +0.0130] | < 0.001 | yes |
| pysteps S-PROG | pysteps LK extrapolation | -0.0199 | [-0.0269, -0.0139] | < 0.001 | yes |
| pysteps S-PROG | Optical flow (Lagrangian) | -0.0428 | [-0.0540, -0.0319] | < 0.001 | yes |
| SimVP v1 (MSE) | pysteps S-PROG | -0.2059 | [-0.2291, -0.1832] | < 0.001 | yes |

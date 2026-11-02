Test split: 1500 windows, 153 day blocks (windows/block min 1, median 10, max 18).

## 1. Regression vs existing eval JSONs

| tag | metric | ext | eval json | ok |
|---|---|---|---|---|
| persistence | overall_mse_norm | 2.940868e-05 | 2.94e-05 | PASS |
| persistence | overall_mse_mm2 | 0.5764101 | 0.5764 | PASS |
| persistence | overall_rmse_mm | 0.7592168 | 0.7592 | PASS |
| persistence | overall_mae_mm | 0.1309861 | 0.131 | PASS |
| persistence | csi_mean@0.1mm | 0.3133857 | 0.3134 | PASS |
| persistence | hss_mean@0.1mm | 0.416684 | 0.4167 | PASS |
| persistence | csi_mean@0.5mm | 0.2079541 | 0.208 | PASS |
| persistence | hss_mean@0.5mm | 0.301113 | 0.3011 | PASS |
| persistence | csi_mean@1.0mm | 0.1590012 | 0.159 | PASS |
| persistence | hss_mean@1.0mm | 0.2413874 | 0.2414 | PASS |
| persistence | csi_mean@2.0mm | 0.1082036 | 0.1082 | PASS |
| persistence | hss_mean@2.0mm | 0.1739127 | 0.1739 | PASS |
| persistence | csi_mean@5.0mm | 0.05264276 | 0.0526 | PASS |
| persistence | hss_mean@5.0mm | 0.0921074 | 0.0921 | PASS |
| persistence | csi_mean@10.0mm | 0.02326231 | 0.0233 | PASS |
| persistence | hss_mean@10.0mm | 0.04309134 | 0.0431 | PASS |
| optical_flow | overall_mse_norm | 2.716828e-05 | 2.72e-05 | PASS |
| optical_flow | overall_mse_mm2 | 0.5324983 | 0.5325 | PASS |
| optical_flow | overall_rmse_mm | 0.7297248 | 0.7297 | PASS |
| optical_flow | overall_mae_mm | 0.127539 | 0.1275 | PASS |
| optical_flow | csi_mean@0.1mm | 0.3283124 | 0.3283 | PASS |
| optical_flow | hss_mean@0.1mm | 0.4338333 | 0.4338 | PASS |
| optical_flow | csi_mean@0.5mm | 0.2359259 | 0.2359 | PASS |
| optical_flow | hss_mean@0.5mm | 0.3386786 | 0.3387 | PASS |
| optical_flow | csi_mean@1.0mm | 0.1880179 | 0.188 | PASS |
| optical_flow | hss_mean@1.0mm | 0.2824266 | 0.2824 | PASS |
| optical_flow | csi_mean@2.0mm | 0.134965 | 0.135 | PASS |
| optical_flow | hss_mean@2.0mm | 0.2136501 | 0.2137 | PASS |
| optical_flow | csi_mean@5.0mm | 0.07354423 | 0.0735 | PASS |
| optical_flow | hss_mean@5.0mm | 0.1258828 | 0.1259 | PASS |
| optical_flow | csi_mean@10.0mm | 0.03696248 | 0.037 | PASS |
| optical_flow | hss_mean@10.0mm | 0.0669131 | 0.0669 | PASS |
| pysteps_lk_extrap | (no eval json) | - | - | SKIPPED |
| pysteps_sprog | (no eval json) | - | - | SKIPPED |
| convlstm_rainnet_seed0 | overall_mse_norm | 1.829085e-05 | 1.83e-05 | PASS |
| convlstm_rainnet_seed0 | overall_mse_mm2 | 0.3585006 | 0.3585 | PASS |
| convlstm_rainnet_seed0 | overall_rmse_mm | 0.5987492 | 0.5987 | PASS |
| convlstm_rainnet_seed0 | overall_mae_mm | 0.1430926 | 0.1431 | PASS |
| convlstm_rainnet_seed0 | csi_mean@0.1mm | 0.1769684 | 0.177 | PASS |
| convlstm_rainnet_seed0 | hss_mean@0.1mm | 0.2262616 | 0.2263 | PASS |
| convlstm_rainnet_seed0 | csi_mean@0.5mm | 0.04127326 | 0.0413 | PASS |
| convlstm_rainnet_seed0 | hss_mean@0.5mm | 0.06893422 | 0.0689 | PASS |
| convlstm_rainnet_seed0 | csi_mean@1.0mm | 0.002156689 | 0.0022 | PASS |
| convlstm_rainnet_seed0 | hss_mean@1.0mm | 0.004019907 | 0.004 | PASS |
| convlstm_rainnet_seed0 | csi_mean@2.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed0 | hss_mean@2.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed0 | csi_mean@5.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed0 | hss_mean@5.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed0 | csi_mean@10.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed0 | hss_mean@10.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed1 | overall_mse_norm | 1.830045e-05 | 1.83e-05 | PASS |
| convlstm_rainnet_seed1 | overall_mse_mm2 | 0.3586889 | 0.3587 | PASS |
| convlstm_rainnet_seed1 | overall_rmse_mm | 0.5989064 | 0.5989 | PASS |
| convlstm_rainnet_seed1 | overall_mae_mm | 0.1620985 | 0.1621 | PASS |
| convlstm_rainnet_seed1 | csi_mean@0.1mm | 0.1690842 | 0.1691 | PASS |
| convlstm_rainnet_seed1 | hss_mean@0.1mm | 0.217264 | 0.2173 | PASS |
| convlstm_rainnet_seed1 | csi_mean@0.5mm | 0.0676607 | 0.0677 | PASS |
| convlstm_rainnet_seed1 | hss_mean@0.5mm | 0.1084512 | 0.1085 | PASS |
| convlstm_rainnet_seed1 | csi_mean@1.0mm | 0.00786738 | 0.0079 | PASS |
| convlstm_rainnet_seed1 | hss_mean@1.0mm | 0.0142818 | 0.0143 | PASS |
| convlstm_rainnet_seed1 | csi_mean@2.0mm | 6.738721e-05 | 0.0001 | PASS |
| convlstm_rainnet_seed1 | hss_mean@2.0mm | 0.0001288599 | 0.0001 | PASS |
| convlstm_rainnet_seed1 | csi_mean@5.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed1 | hss_mean@5.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed1 | csi_mean@10.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed1 | hss_mean@10.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed2 | overall_mse_norm | 1.829239e-05 | 1.83e-05 | PASS |
| convlstm_rainnet_seed2 | overall_mse_mm2 | 0.3585308 | 0.3585 | PASS |
| convlstm_rainnet_seed2 | overall_rmse_mm | 0.5987744 | 0.5988 | PASS |
| convlstm_rainnet_seed2 | overall_mae_mm | 0.1465342 | 0.1465 | PASS |
| convlstm_rainnet_seed2 | csi_mean@0.1mm | 0.173822 | 0.1738 | PASS |
| convlstm_rainnet_seed2 | hss_mean@0.1mm | 0.2225524 | 0.2226 | PASS |
| convlstm_rainnet_seed2 | csi_mean@0.5mm | 0.05985892 | 0.0599 | PASS |
| convlstm_rainnet_seed2 | hss_mean@0.5mm | 0.09688993 | 0.0969 | PASS |
| convlstm_rainnet_seed2 | csi_mean@1.0mm | 0.004055636 | 0.0041 | PASS |
| convlstm_rainnet_seed2 | hss_mean@1.0mm | 0.007478541 | 0.0075 | PASS |
| convlstm_rainnet_seed2 | csi_mean@2.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed2 | hss_mean@2.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed2 | csi_mean@5.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed2 | hss_mean@5.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed2 | csi_mean@10.0mm | 0 | 0 | PASS |
| convlstm_rainnet_seed2 | hss_mean@10.0mm | 0 | 0 | PASS |
| convlstm_ed_mse_rainnet_seed0 | overall_mse_norm | 1.476494e-05 | 1.48e-05 | PASS |
| convlstm_ed_mse_rainnet_seed0 | overall_mse_mm2 | 0.2893928 | 0.2894 | PASS |
| convlstm_ed_mse_rainnet_seed0 | overall_rmse_mm | 0.5379525 | 0.538 | PASS |
| convlstm_ed_mse_rainnet_seed0 | overall_mae_mm | 0.1149036 | 0.1149 | PASS |
| convlstm_ed_mse_rainnet_seed0 | csi_mean@0.1mm | 0.3266667 | 0.3267 | PASS |
| convlstm_ed_mse_rainnet_seed0 | hss_mean@0.1mm | 0.4399572 | 0.44 | PASS |
| convlstm_ed_mse_rainnet_seed0 | csi_mean@0.5mm | 0.3038081 | 0.3038 | PASS |
| convlstm_ed_mse_rainnet_seed0 | hss_mean@0.5mm | 0.433003 | 0.433 | PASS |
| convlstm_ed_mse_rainnet_seed0 | csi_mean@1.0mm | 0.235171 | 0.2352 | PASS |
| convlstm_ed_mse_rainnet_seed0 | hss_mean@1.0mm | 0.3541641 | 0.3542 | PASS |
| convlstm_ed_mse_rainnet_seed0 | csi_mean@2.0mm | 0.1475842 | 0.1476 | PASS |
| convlstm_ed_mse_rainnet_seed0 | hss_mean@2.0mm | 0.2349881 | 0.235 | PASS |
| convlstm_ed_mse_rainnet_seed0 | csi_mean@5.0mm | 0.05844097 | 0.0584 | PASS |
| convlstm_ed_mse_rainnet_seed0 | hss_mean@5.0mm | 0.1005351 | 0.1005 | PASS |
| convlstm_ed_mse_rainnet_seed0 | csi_mean@10.0mm | 0.01667837 | 0.0167 | PASS |
| convlstm_ed_mse_rainnet_seed0 | hss_mean@10.0mm | 0.03064383 | 0.0306 | PASS |
| convlstm_ed_mse_rainnet_seed1 | overall_mse_norm | 1.472198e-05 | 1.47e-05 | PASS |
| convlstm_ed_mse_rainnet_seed1 | overall_mse_mm2 | 0.2885508 | 0.2886 | PASS |
| convlstm_ed_mse_rainnet_seed1 | overall_rmse_mm | 0.5371692 | 0.5372 | PASS |
| convlstm_ed_mse_rainnet_seed1 | overall_mae_mm | 0.1115016 | 0.1115 | PASS |
| convlstm_ed_mse_rainnet_seed1 | csi_mean@0.1mm | 0.3335493 | 0.3335 | PASS |
| convlstm_ed_mse_rainnet_seed1 | hss_mean@0.1mm | 0.4482871 | 0.4483 | PASS |
| convlstm_ed_mse_rainnet_seed1 | csi_mean@0.5mm | 0.3021116 | 0.3021 | PASS |
| convlstm_ed_mse_rainnet_seed1 | hss_mean@0.5mm | 0.4311526 | 0.4312 | PASS |
| convlstm_ed_mse_rainnet_seed1 | csi_mean@1.0mm | 0.2300199 | 0.23 | PASS |
| convlstm_ed_mse_rainnet_seed1 | hss_mean@1.0mm | 0.3469367 | 0.3469 | PASS |
| convlstm_ed_mse_rainnet_seed1 | csi_mean@2.0mm | 0.1441946 | 0.1442 | PASS |
| convlstm_ed_mse_rainnet_seed1 | hss_mean@2.0mm | 0.2297788 | 0.2298 | PASS |
| convlstm_ed_mse_rainnet_seed1 | csi_mean@5.0mm | 0.05549219 | 0.0555 | PASS |
| convlstm_ed_mse_rainnet_seed1 | hss_mean@5.0mm | 0.09549665 | 0.0955 | PASS |
| convlstm_ed_mse_rainnet_seed1 | csi_mean@10.0mm | 0.0161367 | 0.0161 | PASS |
| convlstm_ed_mse_rainnet_seed1 | hss_mean@10.0mm | 0.02968297 | 0.0297 | PASS |
| convlstm_ed_mse_rainnet_seed2 | overall_mse_norm | 1.473752e-05 | 1.47e-05 | PASS |
| convlstm_ed_mse_rainnet_seed2 | overall_mse_mm2 | 0.2888554 | 0.2889 | PASS |
| convlstm_ed_mse_rainnet_seed2 | overall_rmse_mm | 0.5374526 | 0.5375 | PASS |
| convlstm_ed_mse_rainnet_seed2 | overall_mae_mm | 0.1133494 | 0.1133 | PASS |
| convlstm_ed_mse_rainnet_seed2 | csi_mean@0.1mm | 0.3305112 | 0.3305 | PASS |
| convlstm_ed_mse_rainnet_seed2 | hss_mean@0.1mm | 0.4444249 | 0.4444 | PASS |
| convlstm_ed_mse_rainnet_seed2 | csi_mean@0.5mm | 0.3052409 | 0.3052 | PASS |
| convlstm_ed_mse_rainnet_seed2 | hss_mean@0.5mm | 0.4346447 | 0.4346 | PASS |
| convlstm_ed_mse_rainnet_seed2 | csi_mean@1.0mm | 0.2334746 | 0.2335 | PASS |
| convlstm_ed_mse_rainnet_seed2 | hss_mean@1.0mm | 0.352035 | 0.352 | PASS |
| convlstm_ed_mse_rainnet_seed2 | csi_mean@2.0mm | 0.1488303 | 0.1488 | PASS |
| convlstm_ed_mse_rainnet_seed2 | hss_mean@2.0mm | 0.2374148 | 0.2374 | PASS |
| convlstm_ed_mse_rainnet_seed2 | csi_mean@5.0mm | 0.05580964 | 0.0558 | PASS |
| convlstm_ed_mse_rainnet_seed2 | hss_mean@5.0mm | 0.09599183 | 0.096 | PASS |
| convlstm_ed_mse_rainnet_seed2 | csi_mean@10.0mm | 0.01539718 | 0.0154 | PASS |
| convlstm_ed_mse_rainnet_seed2 | hss_mean@10.0mm | 0.02838083 | 0.0284 | PASS |
| unet_mse_rainnet_seed0 | overall_mse_norm | 1.475642e-05 | 1.48e-05 | PASS |
| unet_mse_rainnet_seed0 | overall_mse_mm2 | 0.2892258 | 0.2892 | PASS |
| unet_mse_rainnet_seed0 | overall_rmse_mm | 0.5377971 | 0.5378 | PASS |
| unet_mse_rainnet_seed0 | overall_mae_mm | 0.1254603 | 0.1255 | PASS |
| unet_mse_rainnet_seed0 | csi_mean@0.1mm | 0.3336427 | 0.3336 | PASS |
| unet_mse_rainnet_seed0 | hss_mean@0.1mm | 0.4477032 | 0.4477 | PASS |
| unet_mse_rainnet_seed0 | csi_mean@0.5mm | 0.3049676 | 0.305 | PASS |
| unet_mse_rainnet_seed0 | hss_mean@0.5mm | 0.4338719 | 0.4339 | PASS |
| unet_mse_rainnet_seed0 | csi_mean@1.0mm | 0.2372607 | 0.2373 | PASS |
| unet_mse_rainnet_seed0 | hss_mean@1.0mm | 0.3562992 | 0.3563 | PASS |
| unet_mse_rainnet_seed0 | csi_mean@2.0mm | 0.1538636 | 0.1539 | PASS |
| unet_mse_rainnet_seed0 | hss_mean@2.0mm | 0.2435405 | 0.2435 | PASS |
| unet_mse_rainnet_seed0 | csi_mean@5.0mm | 0.05758989 | 0.0576 | PASS |
| unet_mse_rainnet_seed0 | hss_mean@5.0mm | 0.09740282 | 0.0974 | PASS |
| unet_mse_rainnet_seed0 | csi_mean@10.0mm | 0.01833886 | 0.0183 | PASS |
| unet_mse_rainnet_seed0 | hss_mean@10.0mm | 0.03299953 | 0.033 | PASS |
| unet_mse_rainnet_seed1 | overall_mse_norm | 1.465223e-05 | 1.47e-05 | PASS |
| unet_mse_rainnet_seed1 | overall_mse_mm2 | 0.2871838 | 0.2872 | PASS |
| unet_mse_rainnet_seed1 | overall_rmse_mm | 0.5358953 | 0.5359 | PASS |
| unet_mse_rainnet_seed1 | overall_mae_mm | 0.109919 | 0.1099 | PASS |
| unet_mse_rainnet_seed1 | csi_mean@0.1mm | 0.3548485 | 0.3548 | PASS |
| unet_mse_rainnet_seed1 | hss_mean@0.1mm | 0.4732488 | 0.4732 | PASS |
| unet_mse_rainnet_seed1 | csi_mean@0.5mm | 0.3066678 | 0.3067 | PASS |
| unet_mse_rainnet_seed1 | hss_mean@0.5mm | 0.4372574 | 0.4373 | PASS |
| unet_mse_rainnet_seed1 | csi_mean@1.0mm | 0.2374063 | 0.2374 | PASS |
| unet_mse_rainnet_seed1 | hss_mean@1.0mm | 0.3567367 | 0.3567 | PASS |
| unet_mse_rainnet_seed1 | csi_mean@2.0mm | 0.1555524 | 0.1556 | PASS |
| unet_mse_rainnet_seed1 | hss_mean@2.0mm | 0.2475966 | 0.2476 | PASS |
| unet_mse_rainnet_seed1 | csi_mean@5.0mm | 0.05881762 | 0.0588 | PASS |
| unet_mse_rainnet_seed1 | hss_mean@5.0mm | 0.101467 | 0.1015 | PASS |
| unet_mse_rainnet_seed1 | csi_mean@10.0mm | 0.01504511 | 0.015 | PASS |
| unet_mse_rainnet_seed1 | hss_mean@10.0mm | 0.02764082 | 0.0276 | PASS |
| unet_mse_rainnet_seed2 | overall_mse_norm | 1.492984e-05 | 1.49e-05 | PASS |
| unet_mse_rainnet_seed2 | overall_mse_mm2 | 0.2926249 | 0.2926 | PASS |
| unet_mse_rainnet_seed2 | overall_rmse_mm | 0.5409482 | 0.5409 | PASS |
| unet_mse_rainnet_seed2 | overall_mae_mm | 0.1118463 | 0.1118 | PASS |
| unet_mse_rainnet_seed2 | csi_mean@0.1mm | 0.3524676 | 0.3525 | PASS |
| unet_mse_rainnet_seed2 | hss_mean@0.1mm | 0.4702632 | 0.4703 | PASS |
| unet_mse_rainnet_seed2 | csi_mean@0.5mm | 0.2958771 | 0.2959 | PASS |
| unet_mse_rainnet_seed2 | hss_mean@0.5mm | 0.4242895 | 0.4243 | PASS |
| unet_mse_rainnet_seed2 | csi_mean@1.0mm | 0.225011 | 0.225 | PASS |
| unet_mse_rainnet_seed2 | hss_mean@1.0mm | 0.3404163 | 0.3404 | PASS |
| unet_mse_rainnet_seed2 | csi_mean@2.0mm | 0.1482723 | 0.1483 | PASS |
| unet_mse_rainnet_seed2 | hss_mean@2.0mm | 0.2366397 | 0.2366 | PASS |
| unet_mse_rainnet_seed2 | csi_mean@5.0mm | 0.06020325 | 0.0602 | PASS |
| unet_mse_rainnet_seed2 | hss_mean@5.0mm | 0.1031286 | 0.1031 | PASS |
| unet_mse_rainnet_seed2 | csi_mean@10.0mm | 0.0190787 | 0.0191 | PASS |
| unet_mse_rainnet_seed2 | hss_mean@10.0mm | 0.03464388 | 0.0346 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | overall_mse_norm | 1.760554e-05 | 1.76e-05 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | overall_mse_mm2 | 0.3450685 | 0.3451 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | overall_rmse_mm | 0.5874253 | 0.5874 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | overall_mae_mm | 0.1707908 | 0.1708 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | csi_mean@0.1mm | 0.1890119 | 0.189 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | hss_mean@0.1mm | 0.2320813 | 0.2321 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | csi_mean@0.5mm | 0.2067083 | 0.2067 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | hss_mean@0.5mm | 0.3015277 | 0.3015 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | csi_mean@1.0mm | 0.1621741 | 0.1622 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | hss_mean@1.0mm | 0.247411 | 0.2474 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | csi_mean@2.0mm | 0.1147207 | 0.1147 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | hss_mean@2.0mm | 0.1838059 | 0.1838 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | csi_mean@5.0mm | 0.05121162 | 0.0512 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | hss_mean@5.0mm | 0.08720816 | 0.0872 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | csi_mean@10.0mm | 0.01093323 | 0.0109 | PASS |
| convlstm_ed_balanced_rainnet_seed0 | hss_mean@10.0mm | 0.02040049 | 0.0204 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | overall_mse_norm | 1.737649e-05 | 1.74e-05 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | overall_mse_mm2 | 0.3405793 | 0.3406 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | overall_rmse_mm | 0.5835917 | 0.5836 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | overall_mae_mm | 0.138235 | 0.1382 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | csi_mean@0.1mm | 0.2896219 | 0.2896 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | hss_mean@0.1mm | 0.386645 | 0.3866 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | csi_mean@0.5mm | 0.2243735 | 0.2244 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | hss_mean@0.5mm | 0.3275966 | 0.3276 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | csi_mean@1.0mm | 0.1808704 | 0.1809 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | hss_mean@1.0mm | 0.2767724 | 0.2768 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | csi_mean@2.0mm | 0.1297876 | 0.1298 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | hss_mean@2.0mm | 0.2089011 | 0.2089 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | csi_mean@5.0mm | 0.0610999 | 0.0611 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | hss_mean@5.0mm | 0.1043109 | 0.1043 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | csi_mean@10.0mm | 0.01649073 | 0.0165 | PASS |
| convlstm_ed_balanced_rainnet_seed1 | hss_mean@10.0mm | 0.02968296 | 0.0297 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | overall_mse_norm | 1.689343e-05 | 1.69e-05 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | overall_mse_mm2 | 0.3311112 | 0.3311 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | overall_rmse_mm | 0.5754226 | 0.5754 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | overall_mae_mm | 0.121012 | 0.121 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | csi_mean@0.1mm | 0.274322 | 0.2743 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | hss_mean@0.1mm | 0.3700713 | 0.3701 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | csi_mean@0.5mm | 0.2198382 | 0.2198 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | hss_mean@0.5mm | 0.3219948 | 0.322 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | csi_mean@1.0mm | 0.1777994 | 0.1778 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | hss_mean@1.0mm | 0.2723087 | 0.2723 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | csi_mean@2.0mm | 0.1214909 | 0.1215 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | hss_mean@2.0mm | 0.1943487 | 0.1943 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | csi_mean@5.0mm | 0.04946929 | 0.0495 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | hss_mean@5.0mm | 0.08380241 | 0.0838 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | csi_mean@10.0mm | 0.008359898 | 0.0084 | PASS |
| convlstm_ed_balanced_rainnet_seed2 | hss_mean@10.0mm | 0.01564016 | 0.0156 | PASS |
| unet_balanced_rainnet_seed0 | overall_mse_norm | 1.632371e-05 | 1.63e-05 | PASS |
| unet_balanced_rainnet_seed0 | overall_mse_mm2 | 0.3199446 | 0.3199 | PASS |
| unet_balanced_rainnet_seed0 | overall_rmse_mm | 0.5656365 | 0.5656 | PASS |
| unet_balanced_rainnet_seed0 | overall_mae_mm | 0.1201724 | 0.1202 | PASS |
| unet_balanced_rainnet_seed0 | csi_mean@0.1mm | 0.3154265 | 0.3154 | PASS |
| unet_balanced_rainnet_seed0 | hss_mean@0.1mm | 0.4219812 | 0.422 | PASS |
| unet_balanced_rainnet_seed0 | csi_mean@0.5mm | 0.2530307 | 0.253 | PASS |
| unet_balanced_rainnet_seed0 | hss_mean@0.5mm | 0.3660167 | 0.366 | PASS |
| unet_balanced_rainnet_seed0 | csi_mean@1.0mm | 0.2058065 | 0.2058 | PASS |
| unet_balanced_rainnet_seed0 | hss_mean@1.0mm | 0.3122802 | 0.3123 | PASS |
| unet_balanced_rainnet_seed0 | csi_mean@2.0mm | 0.1515295 | 0.1515 | PASS |
| unet_balanced_rainnet_seed0 | hss_mean@2.0mm | 0.2423547 | 0.2424 | PASS |
| unet_balanced_rainnet_seed0 | csi_mean@5.0mm | 0.07155721 | 0.0716 | PASS |
| unet_balanced_rainnet_seed0 | hss_mean@5.0mm | 0.1218684 | 0.1219 | PASS |
| unet_balanced_rainnet_seed0 | csi_mean@10.0mm | 0.0202908 | 0.0203 | PASS |
| unet_balanced_rainnet_seed0 | hss_mean@10.0mm | 0.03700701 | 0.037 | PASS |
| unet_balanced_rainnet_seed1 | overall_mse_norm | 1.786356e-05 | 1.79e-05 | PASS |
| unet_balanced_rainnet_seed1 | overall_mse_mm2 | 0.3501259 | 0.3501 | PASS |
| unet_balanced_rainnet_seed1 | overall_rmse_mm | 0.5917143 | 0.5917 | PASS |
| unet_balanced_rainnet_seed1 | overall_mae_mm | 0.1362751 | 0.1363 | PASS |
| unet_balanced_rainnet_seed1 | csi_mean@0.1mm | 0.3080877 | 0.3081 | PASS |
| unet_balanced_rainnet_seed1 | hss_mean@0.1mm | 0.4138577 | 0.4139 | PASS |
| unet_balanced_rainnet_seed1 | csi_mean@0.5mm | 0.250976 | 0.251 | PASS |
| unet_balanced_rainnet_seed1 | hss_mean@0.5mm | 0.363236 | 0.3632 | PASS |
| unet_balanced_rainnet_seed1 | csi_mean@1.0mm | 0.211521 | 0.2115 | PASS |
| unet_balanced_rainnet_seed1 | hss_mean@1.0mm | 0.3200316 | 0.32 | PASS |
| unet_balanced_rainnet_seed1 | csi_mean@2.0mm | 0.1655378 | 0.1655 | PASS |
| unet_balanced_rainnet_seed1 | hss_mean@2.0mm | 0.2637288 | 0.2637 | PASS |
| unet_balanced_rainnet_seed1 | csi_mean@5.0mm | 0.0879891 | 0.088 | PASS |
| unet_balanced_rainnet_seed1 | hss_mean@5.0mm | 0.1499574 | 0.15 | PASS |
| unet_balanced_rainnet_seed1 | csi_mean@10.0mm | 0.02693682 | 0.0269 | PASS |
| unet_balanced_rainnet_seed1 | hss_mean@10.0mm | 0.04827785 | 0.0483 | PASS |
| unet_balanced_rainnet_seed2 | overall_mse_norm | 1.717896e-05 | 1.72e-05 | PASS |
| unet_balanced_rainnet_seed2 | overall_mse_mm2 | 0.3367075 | 0.3367 | PASS |
| unet_balanced_rainnet_seed2 | overall_rmse_mm | 0.5802651 | 0.5803 | PASS |
| unet_balanced_rainnet_seed2 | overall_mae_mm | 0.1347584 | 0.1348 | PASS |
| unet_balanced_rainnet_seed2 | csi_mean@0.1mm | 0.3080627 | 0.3081 | PASS |
| unet_balanced_rainnet_seed2 | hss_mean@0.1mm | 0.4097037 | 0.4097 | PASS |
| unet_balanced_rainnet_seed2 | csi_mean@0.5mm | 0.2452235 | 0.2452 | PASS |
| unet_balanced_rainnet_seed2 | hss_mean@0.5mm | 0.3552158 | 0.3552 | PASS |
| unet_balanced_rainnet_seed2 | csi_mean@1.0mm | 0.2077599 | 0.2078 | PASS |
| unet_balanced_rainnet_seed2 | hss_mean@1.0mm | 0.314321 | 0.3143 | PASS |
| unet_balanced_rainnet_seed2 | csi_mean@2.0mm | 0.1597901 | 0.1598 | PASS |
| unet_balanced_rainnet_seed2 | hss_mean@2.0mm | 0.2544421 | 0.2544 | PASS |
| unet_balanced_rainnet_seed2 | csi_mean@5.0mm | 0.08002683 | 0.08 | PASS |
| unet_balanced_rainnet_seed2 | hss_mean@5.0mm | 0.1368901 | 0.1369 | PASS |
| unet_balanced_rainnet_seed2 | csi_mean@10.0mm | 0.02297399 | 0.023 | PASS |
| unet_balanced_rainnet_seed2 | hss_mean@10.0mm | 0.04195031 | 0.042 | PASS |
| simvp_mse_rainnet_seed0 | overall_mse_norm | 1.449869e-05 | 1.45e-05 | PASS |
| simvp_mse_rainnet_seed0 | overall_mse_mm2 | 0.2841743 | 0.2842 | PASS |
| simvp_mse_rainnet_seed0 | overall_rmse_mm | 0.53308 | 0.5331 | PASS |
| simvp_mse_rainnet_seed0 | overall_mae_mm | 0.1104829 | 0.1105 | PASS |
| simvp_mse_rainnet_seed0 | csi_mean@0.1mm | 0.3557458 | 0.3557 | PASS |
| simvp_mse_rainnet_seed0 | hss_mean@0.1mm | 0.4725092 | 0.4725 | PASS |
| simvp_mse_rainnet_seed0 | csi_mean@0.5mm | 0.3123893 | 0.3124 | PASS |
| simvp_mse_rainnet_seed0 | hss_mean@0.5mm | 0.4440428 | 0.444 | PASS |
| simvp_mse_rainnet_seed0 | csi_mean@1.0mm | 0.2430113 | 0.243 | PASS |
| simvp_mse_rainnet_seed0 | hss_mean@1.0mm | 0.3650205 | 0.365 | PASS |
| simvp_mse_rainnet_seed0 | csi_mean@2.0mm | 0.1581369 | 0.1581 | PASS |
| simvp_mse_rainnet_seed0 | hss_mean@2.0mm | 0.2513777 | 0.2514 | PASS |
| simvp_mse_rainnet_seed0 | csi_mean@5.0mm | 0.06480951 | 0.0648 | PASS |
| simvp_mse_rainnet_seed0 | hss_mean@5.0mm | 0.111088 | 0.1111 | PASS |
| simvp_mse_rainnet_seed0 | csi_mean@10.0mm | 0.01630025 | 0.0163 | PASS |
| simvp_mse_rainnet_seed0 | hss_mean@10.0mm | 0.02982838 | 0.0298 | PASS |
| simvp_mse_rainnet_seed1 | overall_mse_norm | 1.445784e-05 | 1.45e-05 | PASS |
| simvp_mse_rainnet_seed1 | overall_mse_mm2 | 0.2833736 | 0.2834 | PASS |
| simvp_mse_rainnet_seed1 | overall_rmse_mm | 0.5323285 | 0.5323 | PASS |
| simvp_mse_rainnet_seed1 | overall_mae_mm | 0.1066508 | 0.1067 | PASS |
| simvp_mse_rainnet_seed1 | csi_mean@0.1mm | 0.3647569 | 0.3648 | PASS |
| simvp_mse_rainnet_seed1 | hss_mean@0.1mm | 0.4827672 | 0.4828 | PASS |
| simvp_mse_rainnet_seed1 | csi_mean@0.5mm | 0.3159976 | 0.316 | PASS |
| simvp_mse_rainnet_seed1 | hss_mean@0.5mm | 0.4485074 | 0.4485 | PASS |
| simvp_mse_rainnet_seed1 | csi_mean@1.0mm | 0.2448255 | 0.2448 | PASS |
| simvp_mse_rainnet_seed1 | hss_mean@1.0mm | 0.367335 | 0.3673 | PASS |
| simvp_mse_rainnet_seed1 | csi_mean@2.0mm | 0.1614421 | 0.1614 | PASS |
| simvp_mse_rainnet_seed1 | hss_mean@2.0mm | 0.2567038 | 0.2567 | PASS |
| simvp_mse_rainnet_seed1 | csi_mean@5.0mm | 0.06277066 | 0.0628 | PASS |
| simvp_mse_rainnet_seed1 | hss_mean@5.0mm | 0.1075588 | 0.1076 | PASS |
| simvp_mse_rainnet_seed1 | csi_mean@10.0mm | 0.01791633 | 0.0179 | PASS |
| simvp_mse_rainnet_seed1 | hss_mean@10.0mm | 0.03287301 | 0.0329 | PASS |
| simvp_mse_rainnet_seed2 | overall_mse_norm | 1.448937e-05 | 1.45e-05 | PASS |
| simvp_mse_rainnet_seed2 | overall_mse_mm2 | 0.2839917 | 0.284 | PASS |
| simvp_mse_rainnet_seed2 | overall_rmse_mm | 0.5329087 | 0.5329 | PASS |
| simvp_mse_rainnet_seed2 | overall_mae_mm | 0.1111901 | 0.1112 | PASS |
| simvp_mse_rainnet_seed2 | csi_mean@0.1mm | 0.3516035 | 0.3516 | PASS |
| simvp_mse_rainnet_seed2 | hss_mean@0.1mm | 0.4681449 | 0.4681 | PASS |
| simvp_mse_rainnet_seed2 | csi_mean@0.5mm | 0.3128649 | 0.3129 | PASS |
| simvp_mse_rainnet_seed2 | hss_mean@0.5mm | 0.4445798 | 0.4446 | PASS |
| simvp_mse_rainnet_seed2 | csi_mean@1.0mm | 0.2416926 | 0.2417 | PASS |
| simvp_mse_rainnet_seed2 | hss_mean@1.0mm | 0.3635863 | 0.3636 | PASS |
| simvp_mse_rainnet_seed2 | csi_mean@2.0mm | 0.1577724 | 0.1578 | PASS |
| simvp_mse_rainnet_seed2 | hss_mean@2.0mm | 0.251039 | 0.251 | PASS |
| simvp_mse_rainnet_seed2 | csi_mean@5.0mm | 0.05988431 | 0.0599 | PASS |
| simvp_mse_rainnet_seed2 | hss_mean@5.0mm | 0.1031914 | 0.1032 | PASS |
| simvp_mse_rainnet_seed2 | csi_mean@10.0mm | 0.01412056 | 0.0141 | PASS |
| simvp_mse_rainnet_seed2 | hss_mean@10.0mm | 0.0261027 | 0.0261 | PASS |
| tau_ddr_rainnet_seed0 | overall_mse_norm | 1.522165e-05 | 1.52e-05 | PASS |
| tau_ddr_rainnet_seed0 | overall_mse_mm2 | 0.2983443 | 0.2983 | PASS |
| tau_ddr_rainnet_seed0 | overall_rmse_mm | 0.5462091 | 0.5462 | PASS |
| tau_ddr_rainnet_seed0 | overall_mae_mm | 0.1341411 | 0.1341 | PASS |
| tau_ddr_rainnet_seed0 | csi_mean@0.1mm | 0.3585462 | 0.3585 | PASS |
| tau_ddr_rainnet_seed0 | hss_mean@0.1mm | 0.476313 | 0.4763 | PASS |
| tau_ddr_rainnet_seed0 | csi_mean@0.5mm | 0.2924341 | 0.2924 | PASS |
| tau_ddr_rainnet_seed0 | hss_mean@0.5mm | 0.4213637 | 0.4214 | PASS |
| tau_ddr_rainnet_seed0 | csi_mean@1.0mm | 0.2246396 | 0.2246 | PASS |
| tau_ddr_rainnet_seed0 | hss_mean@1.0mm | 0.3407631 | 0.3408 | PASS |
| tau_ddr_rainnet_seed0 | csi_mean@2.0mm | 0.150545 | 0.1505 | PASS |
| tau_ddr_rainnet_seed0 | hss_mean@2.0mm | 0.240147 | 0.2401 | PASS |
| tau_ddr_rainnet_seed0 | csi_mean@5.0mm | 0.06523592 | 0.0652 | PASS |
| tau_ddr_rainnet_seed0 | hss_mean@5.0mm | 0.1115953 | 0.1116 | PASS |
| tau_ddr_rainnet_seed0 | csi_mean@10.0mm | 0.01873366 | 0.0187 | PASS |
| tau_ddr_rainnet_seed0 | hss_mean@10.0mm | 0.03420132 | 0.0342 | PASS |
| tau_ddr_rainnet_seed1 | overall_mse_norm | 1.491995e-05 | 1.49e-05 | PASS |
| tau_ddr_rainnet_seed1 | overall_mse_mm2 | 0.292431 | 0.2924 | PASS |
| tau_ddr_rainnet_seed1 | overall_rmse_mm | 0.5407689 | 0.5408 | PASS |
| tau_ddr_rainnet_seed1 | overall_mae_mm | 0.1133112 | 0.1133 | PASS |
| tau_ddr_rainnet_seed1 | csi_mean@0.1mm | 0.3347752 | 0.3348 | PASS |
| tau_ddr_rainnet_seed1 | hss_mean@0.1mm | 0.4471264 | 0.4471 | PASS |
| tau_ddr_rainnet_seed1 | csi_mean@0.5mm | 0.2900668 | 0.2901 | PASS |
| tau_ddr_rainnet_seed1 | hss_mean@0.5mm | 0.4181867 | 0.4182 | PASS |
| tau_ddr_rainnet_seed1 | csi_mean@1.0mm | 0.204029 | 0.204 | PASS |
| tau_ddr_rainnet_seed1 | hss_mean@1.0mm | 0.3131707 | 0.3132 | PASS |
| tau_ddr_rainnet_seed1 | csi_mean@2.0mm | 0.1218989 | 0.1219 | PASS |
| tau_ddr_rainnet_seed1 | hss_mean@2.0mm | 0.1973651 | 0.1974 | PASS |
| tau_ddr_rainnet_seed1 | csi_mean@5.0mm | 0.04383703 | 0.0438 | PASS |
| tau_ddr_rainnet_seed1 | hss_mean@5.0mm | 0.07666705 | 0.0767 | PASS |
| tau_ddr_rainnet_seed1 | csi_mean@10.0mm | 0.01229927 | 0.0123 | PASS |
| tau_ddr_rainnet_seed1 | hss_mean@10.0mm | 0.02282626 | 0.0228 | PASS |
| tau_ddr_rainnet_seed2 | overall_mse_norm | 1.576248e-05 | 1.58e-05 | PASS |
| tau_ddr_rainnet_seed2 | overall_mse_mm2 | 0.3089446 | 0.3089 | PASS |
| tau_ddr_rainnet_seed2 | overall_rmse_mm | 0.5558279 | 0.5558 | PASS |
| tau_ddr_rainnet_seed2 | overall_mae_mm | 0.1335365 | 0.1335 | PASS |
| tau_ddr_rainnet_seed2 | csi_mean@0.1mm | 0.2870103 | 0.287 | PASS |
| tau_ddr_rainnet_seed2 | hss_mean@0.1mm | 0.3857183 | 0.3857 | PASS |
| tau_ddr_rainnet_seed2 | csi_mean@0.5mm | 0.2377938 | 0.2378 | PASS |
| tau_ddr_rainnet_seed2 | hss_mean@0.5mm | 0.3515095 | 0.3515 | PASS |
| tau_ddr_rainnet_seed2 | csi_mean@1.0mm | 0.1751981 | 0.1752 | PASS |
| tau_ddr_rainnet_seed2 | hss_mean@1.0mm | 0.2719665 | 0.272 | PASS |
| tau_ddr_rainnet_seed2 | csi_mean@2.0mm | 0.1129022 | 0.1129 | PASS |
| tau_ddr_rainnet_seed2 | hss_mean@2.0mm | 0.1816738 | 0.1817 | PASS |
| tau_ddr_rainnet_seed2 | csi_mean@5.0mm | 0.04142416 | 0.0414 | PASS |
| tau_ddr_rainnet_seed2 | hss_mean@5.0mm | 0.07099144 | 0.071 | PASS |
| tau_ddr_rainnet_seed2 | csi_mean@10.0mm | 0.00625204 | 0.0063 | PASS |
| tau_ddr_rainnet_seed2 | hss_mean@10.0mm | 0.01188944 | 0.0119 | PASS |

Regression summary: 23 tags PASS, 0 individual metric FAILs.

## 2. Day-block bootstrap CIs (family = seed-mean per-window MSE, mm^2)

| model | seeds | overall MSE mm^2 (per seed) | point | 95% CI |
|---|---|---|---|---|
| Persistence (Eulerian) | 1 | 0.5764 | 0.5764 | [0.5106, 0.6386] |
| Optical flow (Lagrangian) | 1 | 0.5325 | 0.5325 | [0.4731, 0.5875] |
| pysteps LK extrapolation | 1 | 0.5096 | 0.5096 | [0.4524, 0.5643] |
| pysteps S-PROG | 1 | 0.4897 | 0.4897 | [0.4351, 0.5408] |
| ConvLSTM (as released) | 3 | 0.3585, 0.3587, 0.3585 | 0.3586 | [0.3167, 0.3980] |
| ConvLSTM-ED (MSE) | 3 | 0.2894, 0.2886, 0.2889 | 0.2889 | [0.2562, 0.3197] |
| U-Net (MSE) | 3 | 0.2892, 0.2872, 0.2926 | 0.2897 | [0.2575, 0.3204] |
| ConvLSTM-ED (balanced) | 3 | 0.3451, 0.3406, 0.3311 | 0.3389 | [0.3016, 0.3742] |
| U-Net (balanced) | 3 | 0.3199, 0.3501, 0.3367 | 0.3356 | [0.2982, 0.3705] |
| SimVP v1 (MSE) | 3 | 0.2842, 0.2834, 0.2840 | 0.2838 | [0.2521, 0.3139] |
| TAU (SimVP v2 + attention) | 3 | 0.2983, 0.2924, 0.3089 | 0.2999 | [0.2664, 0.3316] |

## 3. Paired per-window tests (A - B, mm^2, day-block bootstrap)

| A | B | mean diff | 95% CI | p (two-sided) |
|---|---|---|---|---|
| SimVP v1 (MSE) | ConvLSTM-ED (MSE) | -0.0051 | [-0.0065, -0.0038] | 0.0005 |
| SimVP v1 (MSE) | U-Net (MSE) | -0.0058 | [-0.0072, -0.0044] | 0.0005 |
| ConvLSTM-ED (MSE) | U-Net (MSE) | -0.0007 | [-0.0021, +0.0006] | 0.2590 |
| ConvLSTM-ED (MSE) | ConvLSTM (as released) | -0.0696 | [-0.0816, -0.0587] | 0.0005 |
| ConvLSTM-ED (MSE) | Optical flow (Lagrangian) | -0.2436 | [-0.2708, -0.2181] | 0.0005 |
| ConvLSTM-ED (MSE) | ConvLSTM-ED (balanced) | -0.0500 | [-0.0566, -0.0440] | 0.0005 |
| U-Net (MSE) | U-Net (balanced) | -0.0459 | [-0.0519, -0.0401] | 0.0005 |
| TAU (SimVP v2 + attention) | SimVP v1 (MSE) | +0.0161 | [+0.0138, +0.0187] | 0.0005 |
| TAU (SimVP v2 + attention) | ConvLSTM-ED (MSE) | +0.0110 | [+0.0092, +0.0130] | 0.0005 |
| pysteps S-PROG | pysteps LK extrapolation | -0.0199 | [-0.0269, -0.0139] | 0.0005 |
| pysteps S-PROG | Optical flow (Lagrangian) | -0.0428 | [-0.0540, -0.0319] | 0.0005 |
| SimVP v1 (MSE) | pysteps S-PROG | -0.2059 | [-0.2291, -0.1832] | 0.0005 |

## 4. FSS (lead-mean; thresholds mm, neighbourhoods px)

| model | FSS@0.5mm nbd1px | FSS@0.5mm nbd3px | FSS@0.5mm nbd5px | FSS@0.5mm nbd9px | FSS@0.5mm nbd17px | FSS@1.0mm nbd1px | FSS@1.0mm nbd3px | FSS@1.0mm nbd5px | FSS@1.0mm nbd9px | FSS@1.0mm nbd17px | FSS@2.0mm nbd1px | FSS@2.0mm nbd3px | FSS@2.0mm nbd5px | FSS@2.0mm nbd9px | FSS@2.0mm nbd17px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (Eulerian) | 0.327 | 0.385 | 0.422 | 0.481 | 0.567 | 0.259 | 0.314 | 0.351 | 0.412 | 0.504 | 0.183 | 0.235 | 0.272 | 0.335 | 0.433 |
| Optical flow (Lagrangian) | 0.364 | 0.438 | 0.483 | 0.549 | 0.637 | 0.299 | 0.372 | 0.418 | 0.487 | 0.581 | 0.223 | 0.292 | 0.338 | 0.410 | 0.511 |
| pysteps LK extrapolation | 0.370 | 0.439 | 0.485 | 0.557 | 0.653 | 0.307 | 0.375 | 0.422 | 0.497 | 0.602 | 0.232 | 0.299 | 0.347 | 0.425 | 0.537 |
| pysteps S-PROG | 0.399 | 0.451 | 0.485 | 0.537 | 0.610 | 0.342 | 0.395 | 0.430 | 0.485 | 0.562 | 0.266 | 0.319 | 0.355 | 0.412 | 0.495 |
| ConvLSTM (as released) | 0.106 | 0.128 | 0.144 | 0.173 | 0.222 | 0.009 | 0.012 | 0.013 | 0.017 | 0.022 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ConvLSTM-ED (MSE) | 0.455 | 0.514 | 0.548 | 0.598 | 0.663 | 0.361 | 0.425 | 0.464 | 0.523 | 0.602 | 0.237 | 0.295 | 0.333 | 0.390 | 0.463 |
| U-Net (MSE) | 0.453 | 0.510 | 0.544 | 0.595 | 0.662 | 0.361 | 0.422 | 0.460 | 0.519 | 0.598 | 0.246 | 0.304 | 0.341 | 0.400 | 0.476 |
| ConvLSTM-ED (balanced) | 0.342 | 0.390 | 0.420 | 0.468 | 0.538 | 0.279 | 0.327 | 0.358 | 0.407 | 0.482 | 0.201 | 0.245 | 0.275 | 0.323 | 0.393 |
| U-Net (balanced) | 0.387 | 0.440 | 0.473 | 0.522 | 0.588 | 0.331 | 0.385 | 0.419 | 0.472 | 0.545 | 0.260 | 0.315 | 0.351 | 0.407 | 0.485 |
| SimVP v1 (MSE) | 0.467 | 0.523 | 0.557 | 0.608 | 0.674 | 0.376 | 0.438 | 0.477 | 0.537 | 0.617 | 0.256 | 0.316 | 0.356 | 0.417 | 0.494 |
| TAU (SimVP v2 + attention) | 0.416 | 0.473 | 0.509 | 0.564 | 0.638 | 0.317 | 0.373 | 0.410 | 0.468 | 0.546 | 0.209 | 0.259 | 0.293 | 0.345 | 0.412 |

## 5. Heavy-rain CSI (5 mm / 10 mm), lead-mean and per lead (seed-summed counts)

| model | CSI@5mm mean | CSI@10mm mean | CSI@5mm +1h..+10h |
|---|---|---|---|
| Persistence (Eulerian) | 0.0526 | 0.0233 | 0.210, 0.094, 0.060, 0.043, 0.033, 0.025, 0.019, 0.016, 0.014, 0.013 |
| Optical flow (Lagrangian) | 0.0735 | 0.0370 | 0.276, 0.142, 0.088, 0.061, 0.045, 0.036, 0.029, 0.023, 0.019, 0.016 |
| pysteps LK extrapolation | 0.0747 | 0.0358 | 0.267, 0.139, 0.090, 0.063, 0.049, 0.039, 0.032, 0.026, 0.023, 0.019 |
| pysteps S-PROG | 0.0890 | 0.0425 | 0.282, 0.166, 0.112, 0.081, 0.063, 0.051, 0.042, 0.035, 0.031, 0.027 |
| ConvLSTM (as released) | 0.0000 | 0.0000 | 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 |
| ConvLSTM-ED (MSE) | 0.0566 | 0.0161 | 0.265, 0.130, 0.070, 0.038, 0.022, 0.015, 0.010, 0.007, 0.005, 0.004 |
| U-Net (MSE) | 0.0589 | 0.0175 | 0.275, 0.139, 0.069, 0.040, 0.026, 0.016, 0.011, 0.007, 0.004, 0.003 |
| ConvLSTM-ED (balanced) | 0.0542 | 0.0122 | 0.256, 0.142, 0.078, 0.039, 0.018, 0.007, 0.002, 0.000, 0.000, 0.000 |
| U-Net (balanced) | 0.0803 | 0.0235 | 0.284, 0.176, 0.120, 0.078, 0.052, 0.035, 0.024, 0.016, 0.011, 0.007 |
| SimVP v1 (MSE) | 0.0625 | 0.0161 | 0.271, 0.149, 0.083, 0.046, 0.027, 0.016, 0.011, 0.009, 0.007, 0.006 |
| TAU (SimVP v2 + attention) | 0.0505 | 0.0126 | 0.246, 0.126, 0.060, 0.028, 0.015, 0.010, 0.007, 0.005, 0.004, 0.003 |

## 6. Train/val temporal overlap (selection-set disclosure)

- validation windows that share at least one frame with a train window: **98.0%**
- mean overlapping train windows per val window: 3.49; max 10
- test files disjoint from train files: True (year-2018 test split is untouched by the carve)

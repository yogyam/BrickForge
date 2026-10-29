| generator | budget | cases | valid_rate | planned_rate | stable_rate | mean_iou | median_iou | mean_coverage | mean_pieces | mean_iou_per_piece_x100 | mean_generate_s | expected_met | expected_total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy_v1 | 50 | 53 | 0.962 | 0.962 | 0.925 | 0.6331 | 0.6982 | 0.6611 | 28.4 | 4.109 | 0.042 | 9 | 13 |
| greedy_v1 | 100 | 53 | 0.962 | 0.962 | 0.925 | 0.7093 | 0.8925 | 0.7316 | 37.3 | 4.07 | 0.061 | 13 | 13 |
| greedy_v1 | 200 | 53 | 0.962 | 0.962 | 0.925 | 0.7185 | 0.913 | 0.7406 | 39.1 | 4.068 | 0.066 | 13 | 13 |

Per case (IoU by budget):

| case | group | greedy_v1@50 | greedy_v1@100 | greedy_v1@200 | note |
|---|---|---|---|---|---|
| proc_bridge | bridge | 0.964 | 0.964 | 0.964 |  |
| proc_cube | simple | 1.000 | 1.000 | 1.000 |  |
| proc_cylinder | budget | 0.476 | 0.892 | 1.000 | partial_candidate |
| proc_flat_sheet_impossible | impossible | 0.050 | 0.050 | 0.050 | partial_candidate |
| proc_l_shape | simple | 0.600 | 1.000 | 1.000 | partial_candidate |
| proc_open_box_invalid | invalid_input | fail | fail | fail | non_watertight_mesh |
| proc_pyramid | overhang | 0.791 | 0.946 | 0.946 | partial_candidate |
| proc_slab | simple | 0.700 | 1.000 | 1.000 | partial_candidate |
| proc_sphere | overhang | 0.670 | 0.975 | 0.975 | partial_candidate |
| proc_t_shape | simple | 1.000 | 1.000 | 1.000 |  |
| proc_thin_pole | impossible | 0.000! | 0.000! | 0.000! | no_valid_candidate |
| proc_tower | simple | 1.000 | 1.000 | 1.000 |  |
| proc_wedge | simple | 1.000 | 1.000 | 1.000 |  |
| thingi_100070 | thingi10k | 0.348 | 0.650 | 1.000 | partial_candidate |
| thingi_100894 | thingi10k | 0.913 | 0.913 | 0.913 |  |
| thingi_105120 | thingi10k | 0.200 | 0.200 | 0.200 | partial_candidate |
| thingi_108578 | thingi10k | 0.057 | 0.057 | 0.057 | partial_candidate |
| thingi_108884 | thingi10k | 0.580 | 0.953 | 0.953 | partial_candidate |
| thingi_110201 | thingi10k | 0.250 | 0.250 | 0.250 | partial_candidate |
| thingi_111878 | thingi10k | 0.109 | 0.109 | 0.109 | partial_candidate |
| thingi_1130077 | thingi10k | 0.693 | 0.698 | 0.698 | partial_candidate |
| thingi_113134 | thingi10k | 0.404 | 0.404 | 0.404 | partial_candidate |
| thingi_130960 | thingi10k | 0.967 | 0.967 | 0.967 |  |
| thingi_1335002 | thingi10k | 0.698 | 0.970 | 0.980 | partial_candidate |
| thingi_1417953 | thingi10k | 1.000 | 1.000 | 1.000 |  |
| thingi_144106 | thingi10k | 0.300 | 0.300 | 0.300 | partial_candidate |
| thingi_1514900 | thingi10k | 0.706 | 0.706 | 0.706 | partial_candidate |
| thingi_200212 | thingi10k | 0.698 | 0.989 | 0.989 | partial_candidate |
| thingi_225935 | thingi10k | 0.699 | 0.907 | 0.907 | partial_candidate |
| thingi_265728 | thingi10k | 0.648 | 0.648 | 0.648 | partial_candidate |
| thingi_39108 | thingi10k | 0.800 | 0.800 | 0.800 |  |
| thingi_39922 | thingi10k | 0.667 | 0.667 | 0.667 | partial_candidate |
| thingi_40447 | thingi10k | 0.702 | 1.000 | 1.000 | partial_candidate |
| thingi_42025 | thingi10k | 0.024 | 0.024 | 0.024 | partial_candidate |
| thingi_42035 | thingi10k | 0.987 | 0.987 | 0.987 |  |
| thingi_48124 | thingi10k | 0.295 | 0.295 | 0.295 | partial_candidate |
| thingi_53749 | thingi10k | 0.458 | 0.458 | 0.458 | partial_candidate |
| thingi_54820 | thingi10k | 0.427 | 0.427 | 0.427 | partial_candidate |
| thingi_56630 | thingi10k | 0.763 | 0.955 | 0.955 | partial_candidate |
| thingi_567026 | thingi10k | 0.200 | 0.200 | 0.200 | partial_candidate |
| thingi_61192 | thingi10k | 1.000 | 1.000 | 1.000 |  |
| thingi_69537 | thingi10k | 0.812 | 0.812 | 0.812 | partial_candidate |
| thingi_70559 | thingi10k | 0.231 | 0.231 | 0.231 | partial_candidate |
| thingi_71307 | thingi10k | 0.237 | 0.237 | 0.237 | partial_candidate |
| thingi_75090 | thingi10k | 0.790 | 0.790 | 0.790 |  |
| thingi_75651 | thingi10k | 0.920 | 0.920 | 0.920 | partial_candidate |
| thingi_80597 | thingi10k | 0.883 | 0.883 | 0.883 | partial_candidate |
| thingi_81589 | thingi10k | 0.972 | 0.972 | 0.972 |  |
| thingi_87521 | thingi10k | 1.000 | 1.000 | 1.000 |  |
| thingi_91350 | thingi10k | 0.628 | 0.993 | 0.993 | partial_candidate |
| thingi_91452 | thingi10k | 0.159 | 0.159 | 0.159 | partial_candidate |
| thingi_93881 | thingi10k | 1.000 | 1.000 | 1.000 |  |
| thingi_966906 | thingi10k | 0.810 | 0.817 | 0.817 | partial_candidate |

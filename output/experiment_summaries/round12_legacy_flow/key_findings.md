result_csv=H:\论文项目\compare\results\md_aligned_main_6run\launch_20260624_100902\final_best_ours_selected_round1_round2.csv
raw_rows=422
selected_points=69
output_rows=422
complete_main_points=56
correctness_all_ok=True

geomean_speedup_baseline_over_ours:
- hash_join: geomean=0.374, mean=0.377, median=0.372, n=56, ours_win=0
- full_skew_join: geomean=1.469, mean=1.474, median=1.456, n=56, ours_win=56
- amjoin_style: geomean=1.462, mean=1.466, median=1.461, n=56, ours_win=56
- rdma_onesize: geomean=1.190, mean=1.193, median=1.197, n=56, ours_win=55
- topology_aware_parallel_join: geomean=0.726, mean=0.729, median=0.725, n=56, ours_win=1
- aqe_join_reselection: geomean=0.849, mean=0.854, median=0.839, n=56, ours_win=3

old_best_available_ours_points=44
current_vs_old_ours_ratio_mean=1.007
current_vs_old_ours_ratio_median=1.009
current_faster_count=18
current_slower_count=26

E7_reuse_mode_mean_time:
- FULL_REUSE: 1.301s
- PARTIAL_REUSE: 3.487s
- REBUILD: 3.332s

E4_diagnostic_points=4
E5_diagnostic_points=3
Adaptive LP experiment outputs archived on 2026-05-29.

Files:
- final_experiment_new.csv: averaged output from the full run.
- final_experiment_new_raw.csv: raw per-instance output from the full run.
- final_experiment_new_raw_checkpoint.csv: checkpointed raw output.
- lp_nan_audit.csv: count of NaN LP columns by size pair.
- plot_dsrse_*_uniform.png and plot_cdsrse_*_uniform.png: generated relative-performance plots.
- full_experiment_run.log: stdout/stderr from the full run.

Important audit note:
- In this archived full run, lp_orig and lp_cons are NaN for every raw row.
- Therefore DLP-P and CDLP-P plot lines are present in the plotting code, but invisible in these plots because all plotted values are NaN.
- The adaptive LP heuristic values are finite, but they should be treated as suspect until the LP solve status is logged inside the adaptive heuristic.

Run settings from run_final_experiments_new.jl at archive time:
- rng_seed = 38072
- run_sdp = false
- num_instances = 10
- dist_types = (:uniform,)
- weight_modes = (:uniform01,)

ENV["GKSwstype"] = "100"
using CSV, DataFrames, Statistics, Plots

input_path = "adaptive_lp_results_corrected_20260529_rerun/final_experiment_new_raw_checkpoint.csv"
large_heuristics_path = "adaptive_lp_results_corrected_20260529_rerun_80_120_heuristics/final_experiment_new_raw.csv"
output_dir = "adaptive_lp_results_corrected_20260529_final_plots"
mkpath(output_dir)

df_raw = CSV.read(input_path, DataFrame)
if isfile(large_heuristics_path)
    df_large = CSV.read(large_heuristics_path, DataFrame)
    df_raw = vcat(df_raw, df_large; cols=:union)
end

if "n_base" in names(df_raw)
    df_raw[df_raw.n_base .>= 80, :dsrse_lp_adaptive] .= NaN
    df_raw[df_raw.n_base .>= 80, :cdsrse_lp_adaptive] .= NaN
end

function add_ratio_columns!(df::DataFrame)
    df.alpha_over_opt = df.alpha_pes ./ max.(df.opt_mc, 1e-9)
    if "alpha_pes_over_pstar" in names(df)
        df.alpha_pes_over_pstar_over_opt = df.alpha_pes_over_pstar ./ max.(df.opt_mc, 1e-9)
    end
    df.lp_cons_over_opt = df.lp_cons ./ max.(df.opt_mc, 1e-9)
    df.lp_orig_over_opt = df.lp_orig ./ max.(df.opt_mc, 1e-9)
    df.dual1_over_opt = df.dual1 ./ max.(df.opt_mc, 1e-9)
    df.dual2_over_opt = df.dual2 ./ max.(df.opt_mc, 1e-9)
    if "mu_pes_bound" in names(df)
        df.mu_pes_bound_over_opt = df.mu_pes_bound ./ max.(df.opt_mc, 1e-9)
    end
    df.cdsrse_w_over_opt = df.cdsrse_weight ./ max.(df.opt_mc, 1e-9)
    df.cdsrse_r_over_opt = df.cdsrse_ratio ./ max.(df.opt_mc, 1e-9)
    df.dsrse_w_over_opt = df.dsrse_weight ./ max.(df.opt_mc, 1e-9)
    df.dsrse_r_over_opt = df.dsrse_ratio ./ max.(df.opt_mc, 1e-9)
    if "cdsrse_lp_adaptive" in names(df)
        df.cdsrse_lp_adaptive_over_opt = df.cdsrse_lp_adaptive ./ max.(df.opt_mc, 1e-9)
    end
    if "dsrse_lp_adaptive" in names(df)
        df.dsrse_lp_adaptive_over_opt = df.dsrse_lp_adaptive ./ max.(df.opt_mc, 1e-9)
    end
    return df
end

add_ratio_columns!(df_raw)
CSV.write(joinpath(output_dir, "final_experiment_new_raw.csv"), df_raw)

group_cols = [:n_base, :m_base, :density, :length, :weight_mode, :dist_type]
ratio_cols = [:alpha_over_opt, :alpha_pes_over_pstar_over_opt, :lp_cons_over_opt,
              :lp_orig_over_opt, :dual1_over_opt, :dual2_over_opt,
              :mu_pes_bound_over_opt, :cdsrse_w_over_opt, :cdsrse_r_over_opt,
              :dsrse_w_over_opt, :dsrse_r_over_opt,
              :cdsrse_lp_adaptive_over_opt, :dsrse_lp_adaptive_over_opt]
num_cols = Symbol.(names(df_raw, Number))
value_cols = setdiff(num_cols, union(group_cols, ratio_cols, [:instance_id]))
df_avg = combine(groupby(df_raw, group_cols), value_cols .=> mean .=> value_cols)
add_ratio_columns!(df_avg)
CSV.write(joinpath(output_dir, "final_experiment_new.csv"), df_avg)

function series_extrema(sub::DataFrame, cols::Vector{Symbol})
    values = Float64[]
    for col in cols
        String(col) in names(sub) || continue
        append!(values, filter(isfinite, collect(skipmissing(sub[!, col]))))
    end
    isempty(values) && return (0.0, 1.0)
    return (minimum(values), maximum(values))
end

function add_series!(plt, xticks, sub::DataFrame, col::Symbol; kwargs...)
    String(col) in names(sub) || return
    y = collect(sub[!, col])
    plot!(plt, xticks, y; kwargs...)
end

for dt in unique(df_avg.dist_type), dens in unique(df_avg.density), len in unique(df_avg.length)
    sub = df_avg[(df_avg.dist_type .== dt) .& (df_avg.density .== dens) .& (df_avg.length .== len), :]
    nrow(sub) == 0 && continue
    sort!(sub, [:n_base, :m_base])
    xlabels = ["n=$(sub.n_base[i]), m=$(sub.m_base[i])" for i in 1:nrow(sub)]
    xticks = collect(1:nrow(sub))
    expected_line = ones(nrow(sub))

    dsrse_cols = [:alpha_over_opt, :lp_orig_over_opt, :dsrse_w_over_opt,
                  :dsrse_r_over_opt, :dsrse_lp_adaptive_over_opt]
    minv, maxv = series_extrema(sub, dsrse_cols)
    minv = min(minv, 1.0)
    maxv = max(maxv, 1.0)
    pad = max(0.05 * (maxv - minv), 0.05)
    p_dsrse = plot(xticks, expected_line; lw=2, marker=:circle, label="expected_stab",
        xlabel="instance", xticks=(xticks, xlabels), xrotation=45,
        ylabel="value / expected_stab", title="DSRSE ($(dens), $(len), dist=$(dt))",
        legend=:outertopright, ylims=(minv - pad, maxv + pad))
    add_series!(p_dsrse, xticks, sub, :alpha_over_opt; lw=2, marker=:square, label="α_pes")
    add_series!(p_dsrse, xticks, sub, :lp_orig_over_opt; lw=2, marker=:utriangle, label="DLP-P")
    add_series!(p_dsrse, xticks, sub, :dsrse_w_over_opt; lw=2, marker=:dtriangle, label="DSRSE weight")
    add_series!(p_dsrse, xticks, sub, :dsrse_r_over_opt; lw=2, marker=:diamond, label="DSRSE ratio")
    add_series!(p_dsrse, xticks, sub, :dsrse_lp_adaptive_over_opt; lw=2, marker=:hexagon, label="DSRSE adaptive LP")
    savefig(p_dsrse, joinpath(output_dir, "plot_dsrse_$(dens)_$(len)_$(dt).png"))

    cdsrse_cols = [:alpha_over_opt, :lp_cons_over_opt, :cdsrse_w_over_opt,
                   :cdsrse_r_over_opt, :cdsrse_lp_adaptive_over_opt]
    minv, maxv = series_extrema(sub, cdsrse_cols)
    minv = min(minv, 1.0)
    maxv = max(maxv, 1.0)
    pad = max(0.05 * (maxv - minv), 0.05)
    p_cdsrse = plot(xticks, expected_line; lw=2, marker=:circle, label="expected_stab",
        xlabel="instance", xticks=(xticks, xlabels), xrotation=45,
        ylabel="value / expected_stab", title="CDSRSE ($(dens), $(len), dist=$(dt))",
        legend=:outertopright, ylims=(minv - pad, maxv + pad))
    add_series!(p_cdsrse, xticks, sub, :alpha_over_opt; lw=2, marker=:square, label="α_pes")
    add_series!(p_cdsrse, xticks, sub, :lp_cons_over_opt; lw=2, marker=:utriangle, label="CDLP-P")
    add_series!(p_cdsrse, xticks, sub, :cdsrse_w_over_opt; lw=2, marker=:dtriangle, label="CDSRSE weight")
    add_series!(p_cdsrse, xticks, sub, :cdsrse_r_over_opt; lw=2, marker=:diamond, label="CDSRSE ratio")
    add_series!(p_cdsrse, xticks, sub, :cdsrse_lp_adaptive_over_opt; lw=2, marker=:hexagon, label="CDSRSE adaptive LP")
    savefig(p_cdsrse, joinpath(output_dir, "plot_cdsrse_$(dens)_$(len)_$(dt).png"))
end

println("Wrote finalized CSVs and plots to $(output_dir). Raw rows: $(nrow(df_raw)), averaged rows: $(nrow(df_avg))")

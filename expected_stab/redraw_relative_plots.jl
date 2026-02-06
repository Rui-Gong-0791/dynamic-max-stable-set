ENV["GKSwstype"] = "100"  # headless GR for Plots
using CSV, DataFrames, Plots

function ensure_ratio_columns!(df::DataFrame)
    denom = max.(df.opt_mc, 1e-9)
    if !("alpha_over_opt" in names(df))
        df.alpha_over_opt = df.alpha_pes ./ denom
    end
    if "alpha_pes_over_pstar" in names(df) && !("alpha_pes_over_pstar_over_opt" in names(df))
        df.alpha_pes_over_pstar_over_opt = df.alpha_pes_over_pstar ./ denom
    end
    if !("lp_cons_over_opt" in names(df))
        df.lp_cons_over_opt = df.lp_cons ./ denom
    end
    if !("lp_orig_over_opt" in names(df))
        df.lp_orig_over_opt = df.lp_orig ./ denom
    end
    if "mu_pes_bound" in names(df) && !("mu_pes_bound_over_opt" in names(df))
        df.mu_pes_bound_over_opt = df.mu_pes_bound ./ denom
    end
    if !("cdsrse_w_over_opt" in names(df))
        df.cdsrse_w_over_opt = df.cdsrse_weight ./ denom
    end
    if !("cdsrse_r_over_opt" in names(df))
        df.cdsrse_r_over_opt = df.cdsrse_ratio ./ denom
    end
    if !("dsrse_w_over_opt" in names(df))
        df.dsrse_w_over_opt = df.dsrse_weight ./ denom
    end
    if !("dsrse_r_over_opt" in names(df))
        df.dsrse_r_over_opt = df.dsrse_ratio ./ denom
    end
    return df
end

function redraw_relative_plots(; csv_path="final_experiment_new.csv", suffix::AbstractString="")
    df = CSV.read(csv_path, DataFrame)
    ensure_ratio_columns!(df)
    out_dir = dirname(abspath(csv_path))
    suffix_str = isempty(suffix) ? "" : suffix

    function series_extrema(sub::DataFrame, cols::Vector{Symbol})
        minv = Inf
        maxv = -Inf
        for col in cols
            if String(col) in names(sub)
                for v in sub[!, col]
                    if isfinite(v)
                        minv = v < minv ? v : minv
                        maxv = v > maxv ? v : maxv
                    end
                end
            end
        end
        if minv == Inf
            return (0.0, 1.0)
        end
        return (minv, maxv)
    end

    for dt in unique(df.dist_type), dens in unique(df.density), len in unique(df.length)
        sub = df[(df.dist_type .== dt) .& (df.density .== dens) .& (df.length .== len), :]
        nrow(sub) == 0 && continue
        sort!(sub, [:n_base, :m_base])
        xlabels = ["n=$(sub.n_base[i]), m=$(sub.m_base[i])" for i in 1:nrow(sub)]
        xticks = collect(1:nrow(sub))
        expected_line = ones(nrow(sub))

        dsrse_cols = [:alpha_over_opt, :lp_orig_over_opt, :dsrse_w_over_opt, :dsrse_r_over_opt]
        if "mu_pes_bound_over_opt" in names(sub)
            push!(dsrse_cols, :mu_pes_bound_over_opt)
        end
        minv, maxv = series_extrema(sub, dsrse_cols)
        minv = min(minv, 1.0)
        maxv = max(maxv, 1.0)
        pad = 0.05 * (maxv - minv)
        pad = pad == 0.0 ? 0.1 * maxv : pad
        ylims = (minv - pad, maxv + pad)

        println("DSRSE plot ", dens, " ", len, " ", dt, " mu_pes max=",
            ("mu_pes_bound_over_opt" in names(sub) ? maximum(sub.mu_pes_bound_over_opt) : "NA"),
            " ylims=", ylims)

        p_dsrse = plot(xticks, expected_line; lw=2, marker=:circle, label="expected_stab",
            xlabel="instance", xticks=(xticks, xlabels), xrotation=45,
            ylabel="value / expected_stab", title="DSRSE ($(dens), $(len), dist=$(dt))",
            legend=:outertopright, ylims=ylims)
        plot!(p_dsrse, xticks, sub.alpha_over_opt; lw=2, marker=:square, label="α_pes")
        plot!(p_dsrse, xticks, sub.lp_orig_over_opt; lw=2, marker=:utriangle, label="DLP-P")
        if "mu_pes_bound_over_opt" in names(sub) &&
           any(x -> isfinite(x), skipmissing(sub.mu_pes_bound_over_opt))
            plot!(p_dsrse, xticks, sub.mu_pes_bound_over_opt; lw=3, marker=:star5,
                label="μ_pes bound", color=:black)
        end
        plot!(p_dsrse, xticks, sub.dsrse_w_over_opt; lw=2, marker=:dtriangle, label="DSRSE weight")
        plot!(p_dsrse, xticks, sub.dsrse_r_over_opt; lw=2, marker=:diamond, label="DSRSE ratio")
        out_path = joinpath(out_dir, "plot_dsrse_$(dens)_$(len)_$(dt)$(suffix_str).png")
        savefig(p_dsrse, out_path)
        println("Saved ", out_path)
        display(p_dsrse)

        cdsrse_cols = [:alpha_over_opt, :lp_cons_over_opt, :cdsrse_w_over_opt, :cdsrse_r_over_opt]
        minv, maxv = series_extrema(sub, cdsrse_cols)
        minv = min(minv, 1.0)
        maxv = max(maxv, 1.0)
        pad = 0.05 * (maxv - minv)
        pad = pad == 0.0 ? 0.1 * maxv : pad
        ylims = (minv - pad, maxv + pad)

        println("CDSRSE plot ", dens, " ", len, " ", dt, " ylims=", ylims)
        p_cdsrse = plot(xticks, expected_line; lw=2, marker=:circle, label="expected_stab",
            xlabel="instance", xticks=(xticks, xlabels), xrotation=45,
            ylabel="value / expected_stab", title="CDSRSE ($(dens), $(len), dist=$(dt))",
            legend=:outertopright, ylims=ylims)
        plot!(p_cdsrse, xticks, sub.alpha_over_opt; lw=2, marker=:square, label="α_pes")
        plot!(p_cdsrse, xticks, sub.lp_cons_over_opt; lw=2, marker=:utriangle, label="CDLP-P")
        plot!(p_cdsrse, xticks, sub.cdsrse_w_over_opt; lw=2, marker=:dtriangle, label="CDSRSE weight")
        plot!(p_cdsrse, xticks, sub.cdsrse_r_over_opt; lw=2, marker=:diamond, label="CDSRSE ratio")
        out_path = joinpath(out_dir, "plot_cdsrse_$(dens)_$(len)_$(dt)$(suffix_str).png")
        savefig(p_cdsrse, out_path)
        println("Saved ", out_path)
        display(p_cdsrse)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    redraw_relative_plots()
end

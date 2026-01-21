ENV["GKSwstype"] = "100"  # headless GR for Plots
using CSV, DataFrames, Plots

function redraw_dsrse_plots(; csv_path="final_experiment_new.csv")
    df = CSV.read(csv_path, DataFrame)
    for dt in unique(df.dist_type), dens in unique(df.density), len in unique(df.length)
        sub = df[(df.dist_type .== dt) .& (df.density .== dens) .& (df.length .== len), :]
        nrow(sub) == 0 && continue
        sort!(sub, [:n_base, :m_base])
        xlabels = ["n=$(sub.n_base[i]), m=$(sub.m_base[i])" for i in 1:nrow(sub)]
        xticks = collect(1:nrow(sub))

        p_dsrse = plot(xticks, sub.opt_mc; lw=2, marker=:circle, label="expected_stab",
            xlabel="instance", xticks=(xticks, xlabels), xrotation=45,
            ylabel="value", title="DSRSE ($(dens), $(len), dist=$(dt))")
        plot!(p_dsrse, xticks, sub.alpha_pes; lw=2, marker=:square, label="α_pes")
        plot!(p_dsrse, xticks, sub.lp_orig; lw=2, marker=:utriangle, label="DLP-P")
        plot!(p_dsrse, xticks, sub.dsrse_weight; lw=2, marker=:dtriangle, label="DSRSE weight")
        plot!(p_dsrse, xticks, sub.dsrse_ratio; lw=2, marker=:diamond, label="DSRSE ratio")
        savefig(p_dsrse, "plot_dsrse_$(dens)_$(len)_$(dt).png")
        display(p_dsrse)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    redraw_dsrse_plots()
end

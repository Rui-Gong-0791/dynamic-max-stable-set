include("run_final_experiments_new.jl")

function run_80_120_heuristics_only(; rng_seed=38072,
    output_dir="adaptive_lp_results_corrected_20260529_rerun_80_120_heuristics",
    num_instances=10,
    num_samples_bounds=1000,
    num_samples_overlap=500,
    num_samples_policy=1000,
    num_samples_opt=1000)

    mkpath(output_dir)
    Random.seed!(rng_seed)
    rng = Random.GLOBAL_RNG
    rows = Dict[]
    n = 80
    m = 120
    wm = :uniform01
    dt = :uniform

    for inst in 1:num_instances
        base_supports = generate_base_supports(n, m; rng=rng)
        base_weights = generate_weights(n; weight_mode=wm, rng=rng)
        keep_idx = sort(sample(rng, 1:n, n ÷ 2; replace=false))
        supports_long, m_long = extend_supports_long(base_supports, m)

        for len in (:short, :long)
            supports_len = len == :short ? base_supports : supports_long
            m_used = len == :short ? m : m_long
            for dens in (:dense, :sparse)
                idx = dens == :dense ? collect(1:n) : keep_idx
                supports_use = supports_len[idx, :]
                weights_use = base_weights[idx]
                jobs = supports_to_jobs(supports_use; dist_type=dt)
                println("Heuristics-only: n=$(n), m=$(m), length=$(len), density=$(dens), rep=$(inst)")
                flush(stdout)

                bounds = conservative_bounds(jobs, weights_use, m_used; num_samples=num_samples_bounds)
                P_del_orig = estimate_overlap_probabilities(jobs, m_used; num_samples=num_samples_overlap, rng=rng)
                P_del_cons = support_overlap_matrix(jobs, m_used)
                cons_w, _ = simulate_policy_conservative(jobs, weights_use, m_used;
                    policy=:weight, num_samples=num_samples_policy, L=bounds.L, P_del=P_del_cons, rng=rng)
                cons_r, _ = simulate_policy_conservative(jobs, weights_use, m_used;
                    policy=:ratio, num_samples=num_samples_policy, L=bounds.L, P_del=P_del_cons, rng=rng)
                orig_w, _ = simulate_policy_original(jobs, weights_use, m_used;
                    policy=:weight, num_samples=num_samples_policy, L=bounds.L, P_del=P_del_orig, rng=rng)
                orig_r, _ = simulate_policy_original(jobs, weights_use, m_used;
                    policy=:ratio, num_samples=num_samples_policy, L=bounds.L, P_del=P_del_orig, rng=rng)
                opt_mc, _ = simulate_optimal_expected(jobs, weights_use, m_used; num_samples=num_samples_opt)
                _, _, P_occ, _ = estimate_position_probabilities(jobs, m_used; num_samples=num_samples_bounds)
                lp_cons, _, lp_cons_status = lp_solution_conservative(P_occ, weights_use)

                row = Dict(
                    :instance_id=>inst,
                    :n_base=>n, :m_base=>m,
                    :n=>length(weights_use), :m=>m_used,
                    :density=>dens, :length=>len,
                    :weight_mode=>wm, :dist_type=>dt,
                    :alpha_pes=>bounds.α_pes, :p_star=>bounds.p_star,
                    :alpha_pes_over_pstar=>bounds.alpha_pes_over_pstar,
                    :dual1=>bounds.dual1, :dual2=>bounds.dual2,
                    :mu_pes_bound=>bounds.mu_pes_bound,
                    :opt_mc=>opt_mc,
                    :lp_cons=>lp_cons, :lp_orig=>NaN, :sdp_orig=>NaN,
                    :lp_cons_status=>lp_cons_status, :lp_orig_status=>"SKIPPED_LARGE",
                    :cdsrse_weight=>cons_w, :cdsrse_ratio=>cons_r,
                    :dsrse_weight=>orig_w, :dsrse_ratio=>orig_r,
                    :cdsrse_lp_adaptive=>NaN,
                    :dsrse_lp_adaptive=>NaN,
                    :cdsrse_lp_adaptive_failed_solves=>0,
                    :cdsrse_lp_adaptive_total_solves=>0,
                    :dsrse_lp_adaptive_failed_solves=>0,
                    :dsrse_lp_adaptive_total_solves=>0
                )
                push!(rows, row)
                df_checkpoint = DataFrame(rows)
                add_ratio_columns!(df_checkpoint)
                add_gap_columns!(df_checkpoint)
                CSV.write(joinpath(output_dir, "final_experiment_new_raw_checkpoint.csv"), df_checkpoint)
            end
        end
    end

    df_raw = DataFrame(rows)
    add_ratio_columns!(df_raw)
    add_gap_columns!(df_raw)
    CSV.write(joinpath(output_dir, "final_experiment_new_raw.csv"), df_raw)
    return df_raw
end

if abspath(PROGRAM_FILE) == @__FILE__
    df = run_80_120_heuristics_only()
    println("Finished 80_120 heuristics-only rows=", nrow(df))
end

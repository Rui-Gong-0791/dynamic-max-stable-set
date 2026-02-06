using CSV, DataFrames, Random

include("run_final_experiments_new.jl")

function ordered_unique(vals)
    seen = Dict{Any, Bool}()
    out = Any[]
    for v in vals
        if !haskey(seen, v)
            push!(out, v)
            seen[v] = true
        end
    end
    return out
end

function ordered_unique_pairs(df::DataFrame, col1::Symbol, col2::Symbol)
    seen = Dict{Tuple{Any, Any}, Bool}()
    out = Tuple{Any, Any}[]
    for (a, b) in zip(df[!, col1], df[!, col2])
        key = (a, b)
        if !haskey(seen, key)
            push!(out, key)
            seen[key] = true
        end
    end
    return out
end

function replay_mu_bounds!(df_raw::DataFrame; rng_seed::Int=38072,
    num_samples_bounds::Int=1000, num_samples_overlap::Int=500,
    num_samples_policy::Int=1000, num_samples_opt::Int=1000)

    size_pairs = ordered_unique_pairs(df_raw, :n_base, :m_base)
    weight_modes = ordered_unique(df_raw.weight_mode)
    dist_types = ordered_unique(df_raw.dist_type)
    densities = ordered_unique(df_raw.density)
    lengths = ordered_unique(df_raw.length)
    num_instances = maximum(df_raw.instance_id)

    Random.seed!(rng_seed)
    rng = Random.GLOBAL_RNG

    results = DataFrame(
        instance_id = Int[],
        n_base = Int[],
        m_base = Int[],
        density = String[],
        length = String[],
        weight_mode = String[],
        dist_type = String[],
        mu_pes_bound = Float64[],
    )

    for (n, m) in size_pairs, wm in weight_modes, inst in 1:num_instances
        base_supports = generate_base_supports(n, m; rng=rng)
        base_weights = generate_weights(n; weight_mode=Symbol(wm), rng=rng)
        n_sparse = max(1, n ÷ 2)
        keep_idx = sort(sample(rng, 1:n, n_sparse; replace=false))

        supports_long, m_long = extend_supports_long(base_supports, m)

        for len in lengths
            supports_len = len == "short" ? base_supports : supports_long
            m_used = len == "short" ? m : m_long
            for dens in densities
                idx = dens == "dense" ? collect(1:n) : keep_idx
                supports_use = supports_len[idx, :]
                weights_use = base_weights[idx]
                for dt in dist_types
                    jobs = supports_to_jobs(supports_use; dist_type=Symbol(dt))

                    bounds = conservative_bounds(jobs, weights_use, m_used;
                        num_samples=num_samples_bounds)
                    estimate_overlap_probabilities(jobs, m_used;
                        num_samples=num_samples_overlap, rng=rng)
                    P_del_cons = support_overlap_matrix(jobs, m_used)
                    simulate_policy_conservative(jobs, weights_use, m_used;
                        policy=:weight, num_samples=num_samples_policy, L=bounds.L, P_del=P_del_cons)
                    simulate_policy_conservative(jobs, weights_use, m_used;
                        policy=:ratio, num_samples=num_samples_policy, L=bounds.L, P_del=P_del_cons)
                    P_del_orig = estimate_overlap_probabilities(jobs, m_used;
                        num_samples=num_samples_overlap, rng=rng)
                    simulate_policy_original(jobs, weights_use, m_used;
                        policy=:weight, num_samples=num_samples_policy, L=bounds.L, P_del=P_del_orig)
                    simulate_policy_original(jobs, weights_use, m_used;
                        policy=:ratio, num_samples=num_samples_policy, L=bounds.L, P_del=P_del_orig)
                    simulate_optimal_expected(jobs, weights_use, m_used;
                        num_samples=num_samples_opt)
                    estimate_position_probabilities(jobs, m_used; num_samples=num_samples_bounds)

                    push!(results, (instance_id=inst, n_base=n, m_base=m, density=dens,
                        length=len, weight_mode=wm, dist_type=dt,
                        mu_pes_bound=bounds.mu_pes_bound))
                end
            end
        end
    end

    return results
end

function merge_new_bounds(; csv_raw="final_experiment_new_raw.csv",
    csv_avg="final_experiment_new.csv", rng_seed::Int=38072,
    num_samples_bounds::Int=1000, num_samples_overlap::Int=500,
    num_samples_policy::Int=1000, num_samples_opt::Int=1000)

    df_raw = CSV.read(csv_raw, DataFrame)
    drop_cols = [
        :mu_pes_bound, :mu_pes_bound_over_opt, :mu_pes_bound_gap,
        :alpha_pes_over_pstar, :alpha_pes_over_pstar_over_opt, :alpha_pes_over_pstar_gap
    ]
    keep = [c for c in names(df_raw) if c ∉ drop_cols]
    select!(df_raw, keep)
    results = replay_mu_bounds!(df_raw; rng_seed=rng_seed,
        num_samples_bounds=num_samples_bounds, num_samples_overlap=num_samples_overlap,
        num_samples_policy=num_samples_policy, num_samples_opt=num_samples_opt)

    join_cols = [:instance_id, :n_base, :m_base, :density, :length, :weight_mode, :dist_type]
    df_raw = leftjoin(df_raw, results, on=join_cols, makeunique=true)
    if :mu_pes_bound_1 in names(df_raw)
        if :mu_pes_bound in names(df_raw)
            df_raw.mu_pes_bound = coalesce.(df_raw.mu_pes_bound_1, df_raw.mu_pes_bound)
            select!(df_raw, Not(:mu_pes_bound_1))
        else
            rename!(df_raw, :mu_pes_bound_1 => :mu_pes_bound)
        end
    end
    if :mu_pes_bound ∉ names(df_raw)
        @warn "mu_pes_bound missing after join; filling with NaN. Check RNG settings or join keys."
        df_raw.mu_pes_bound = fill(NaN, nrow(df_raw))
    end

    df_raw.alpha_pes_over_pstar = map((a, p) -> p > 0 ? a / p : NaN,
        df_raw.alpha_pes, df_raw.p_star)

    add_ratio_columns!(df_raw)
    add_gap_columns!(df_raw)
    CSV.write(csv_raw, df_raw)

    group_cols = [:n_base, :m_base, :density, :length, :weight_mode, :dist_type]
    ratio_cols = [:alpha_over_opt, :alpha_pes_over_pstar_over_opt, :lp_cons_over_opt,
                  :lp_orig_over_opt, :sdp_orig_over_opt, :dual1_over_opt, :dual2_over_opt,
                  :mu_pes_bound_over_opt, :cdsrse_w_over_opt, :cdsrse_r_over_opt,
                  :dsrse_w_over_opt, :dsrse_r_over_opt]
    num_cols = names(df_raw, Number)
    value_cols = setdiff(num_cols, union(group_cols, ratio_cols, [:instance_id]))
    df_avg = combine(groupby(df_raw, group_cols), value_cols .=> mean .=> value_cols)
    add_ratio_columns!(df_avg)
    CSV.write(csv_avg, df_avg)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    merge_new_bounds()
end

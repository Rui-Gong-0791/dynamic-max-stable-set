ENV["GKSwstype"] = "100"  # headless GR for Plots
using Random, Distributions, StatsBase, DataFrames, CSV, Plots, JuMP

# Optional LP solvers (best effort)
const _lp_optimizer = let
    try
        @eval using HiGHS
        () -> HiGHS.Optimizer
    catch
        try
            @eval using GLPK
            () -> GLPK.Optimizer
        catch
            @warn "No LP optimizer found (HiGHS/GLPK). LP bounds will be skipped."
            () -> nothing
        end
    end
end

# Minimal helpers (copied/adapted from discrete_interval.jl to avoid solver deps).
# Bound mapping:
#   - alpha_pes: pessimistic interval bound (deterministic widest support).
#   - dual1, dual2: simple occupancy-based dual bounds from note (using P^o).
#   - opt_mc: Monte Carlo estimate of true expected optimum (exact WIS on each draw).
# LP bound you mentioned (p*-scaled upper bound) is currently commented out/omitted; add if needed.
struct JobDistribution
    start_dist::UnivariateDistribution
    end_dist::UnivariateDistribution
end

function interval_bounds(d::UnivariateDistribution, m::Int; eps=1e-9)
    a = try Distributions.minimum(d) catch; -Inf end
    b = try Distributions.maximum(d) catch;  Inf end
    if !isfinite(a); a = quantile(d, eps); end
    if !isfinite(b); b = quantile(d, 1 - eps); end
    L = max(1, min(m, ceil(Int, a)))
    U = max(1, min(m, floor(Int, b)))
    if U < L; U = L; end
    return (L, U)
end

function compute_A_intervals(jobs::Vector{JobDistribution}, m::Int)
    n = length(jobs)
    A_L = Vector{Int}(undef, n)
    A_R = Vector{Int}(undef, n)
    for i in 1:n
        Ls, _  = interval_bounds(jobs[i].start_dist, m)
        _,  Re = interval_bounds(jobs[i].end_dist,   m)
        A_L[i] = Ls
        A_R[i] = max(Ls, Re)
    end
    return A_L, A_R
end

function sample_job_intervals(jobs::Vector{JobDistribution},k)
    n = length(jobs)
    starts, ends = zeros(n), zeros(n)
    for i in 1:n
        s = round(Int, clamp(rand(jobs[i].start_dist), 1, k))
        e = round(Int, clamp(rand(jobs[i].end_dist), 1, k))
        while e < s
            e = round(Int, clamp(rand(jobs[i].end_dist), 1, k))
        end
        starts[i] = s
        ends[i] = e
    end
    return starts, ends
end

function estimate_position_probabilities(jobs::Vector{JobDistribution}, k::Int; num_samples=1000)
    n = length(jobs)
    P_s = zeros(n, k)
    P_e = zeros(n, k)
    P = zeros(n, k)
    L = zeros(n)
    for i in 1:n
        counts_s = zeros(Int, k)
        counts_e = zeros(Int, k)
        counts = zeros(Int, k)
        for _ in 1:num_samples
            start = round(Int, clamp(rand(jobs[i].start_dist), 1, k))
            e = round(Int, clamp(rand(jobs[i].end_dist), 1, k))
            while e < start
                e = round(Int, clamp(rand(jobs[i].end_dist), 1, k))
            end
            for t in start:e
                counts[t] += 1
            end
            L[i] += (e - start + 1)
            counts_e[e] += 1
            counts_s[start] += 1
        end
        P_s[i, :] = counts_s ./ num_samples
        P_e[i, :] = counts_e ./ num_samples
        P[i,:] = counts ./ num_samples
        L[i] = L[i]./ num_samples
    end
    return P_s, P_e, P, L
end

function weighted_interval_scheduling(starts::Vector{<:Real}, ends::Vector{<:Real}, weights::Vector{<:Real})
    n = length(starts)
    if n == 0; return 0.0 end
    idxs = sortperm(ends)
    s = starts[idxs]; e = ends[idxs]; w = weights[idxs]
    p = zeros(Int, n)
    for j in 1:n
        i = searchsortedlast(e, s[j] - 1e-8)
        p[j] = i
    end
    opt = zeros(Float64, n+1)
    for j in 1:n
        opt[j+1] = max(opt[j], w[j] + opt[p[j]+1])
    end
    return opt[n+1]
end

function generate_support_instance(n::Int, m::Int;
    overlap::Symbol = :moderate,
    weight_mode::Symbol = :uniform,
    dist_type::Symbol = :uniform,
    sigma::Real = 2.0,
    rng = Random.GLOBAL_RNG)

    jobs = JobDistribution[]
    weights = zeros(Float64, n)

    function pick_start_anchor()
        if overlap == :sparse
            return rand(rng, 1:2: max(1, m - 1))
        elseif overlap == :heavy
            return rand(rng, 1:max(1, ceil(Int, m ÷ 3)))
        else
            return rand(rng, 1:m)
        end
    end

    for i in 1:n
        anchor = pick_start_anchor()
        max_start_len = max(1, m - anchor + 1)
        start_len = rand(rng, 1:max_start_len)
        a = clamp(anchor, 1, m)
        b = clamp(a + start_len - 1, 1, m)

       
        
        # pick end anchor c ≥ b (not always equal), then a feasible end length
        c = rand(rng, b:m)
        max_end_len = max(1, m - c + 1)
        end_len = rand(rng, start_len:max(start_len, max_end_len))
        d = clamp(c + end_len - 1, c, m)

        start_dist = dist_type == :uniform ? DiscreteUniform(a, b) :
                     dist_type == :normal  ? truncated(Normal((a + b) / 2, sigma), 1, m) :
                     dist_type == :exp     ? truncated(Exponential(1.0 / max(1, (a + b) / 2)), 1, m) :
                     dist_type == :constant ? Dirac(a) :
                     error("unknown dist_type")

        end_dist   = dist_type == :uniform ? DiscreteUniform(c, d) :
                     dist_type == :normal  ? truncated(Normal((c + d) / 2, sigma), 1, m) :
                     dist_type == :exp     ? truncated(Exponential(1.0 / max(1, (c + d) / 2)), 1, m) :
                     dist_type == :constant ? Dirac(d) :
                     error("unknown dist_type")

        push!(jobs, JobDistribution(start_dist, end_dist))
        weights[i] = weight_mode == :uniform ? 1.0 :
                     weight_mode == :uniform01 ? rand(rng) :
                     weight_mode == :exp1 ? rand(rng, Exponential(1.0)) :
                     rand(rng)
    end
    return jobs, weights
end

function conservative_bounds(jobs::Vector{JobDistribution}, weights::AbstractVector, m::Int; num_samples::Int=1000)
    n = length(jobs)
    P_s, P_e, P_occ, L = estimate_position_probabilities(jobs, m; num_samples=num_samples)
    A_L, A_R = compute_A_intervals(jobs, m)
    α_pes = weighted_interval_scheduling(A_L, A_R, weights)  # pessimistic bound
    p_star = minimum(filter(>(0.0), P_occ))                  # min positive occupancy
    total_len = sum(P_occ, dims=2)
    dual1 = 0.0
    for r in 1:m
        vals = [weights[i]/total_len[i] for i in 1:n if P_occ[i,r]>0 && total_len[i]>0]
        dual1 += isempty(vals) ? 0.0 : maximum(vals)
    end
    total_sq = sum(P_occ .^ 2, dims=2)
    dual2 = 0.0
    for r in 1:m
        vals = [P_occ[i,r]*weights[i]/total_sq[i] for i in 1:n if P_occ[i,r]>0 && total_sq[i]>0]
        dual2 += isempty(vals) ? 0.0 : maximum(vals)
    end
    return (α_pes=α_pes, p_star=p_star, dual1=dual1, dual2=dual2, L=vec(L))
end

# LP for conservative case (CDLP-P): max Σ w_i x_i s.t. Σ_i P_occ[i,k] x_i ≤ 1, x ≥ 0
function lp_bound_conservative(P_occ::AbstractMatrix, weights::AbstractVector)
    optimizer = _lp_optimizer()
    optimizer === nothing && return NaN
    m = size(P_occ, 2); n = size(P_occ, 1)
    model = Model(optimizer)
    @variable(model, x[1:n] >= 0)
    @constraint(model, [k=1:m], sum(P_occ[i,k] * x[i] for i in 1:n) <= 1)
    @objective(model, Max, sum(weights[i]*x[i] for i in 1:n))
    optimize!(model)
    return objective_value(model)
end

# LP for original case (DLP-P) as given in the note.
function lp_bound_original(jobs::Vector{JobDistribution}, P_s::AbstractMatrix, P_e::AbstractMatrix, weights::AbstractVector, m::Int)
    optimizer = _lp_optimizer()
    optimizer === nothing && return NaN
    n = length(jobs)
    T = min(n, m)
    a = Vector{Int}(undef, n); b = Vector{Int}(undef, n)
    c = Vector{Int}(undef, n); d = Vector{Int}(undef, n)
    for i in 1:n
        a[i], b[i] = interval_bounds(jobs[i].start_dist, m)
        c[i], d[i] = interval_bounds(jobs[i].end_dist, m)
    end
    model = Model(optimizer)
    @variable(model, x[1:n, 1:T] >= 0)
    @variable(model, x_ts[1:n, 1:m, 1:T] >= 0)
    @variable(model, x_te[1:n, 1:m, 1:T] >= 0)

    @constraint(model, [i=1:n, t=1:T], x[i,t] == sum(x_ts[i,k,t] for k in 1:m))
    @constraint(model, [i=1:n, t=1:T], x[i,t] == sum(x_te[i,k,t] for k in 1:m))

    for i in 1:n, t in 1:T, k in 1:m
        if k < a[i] || k > b[i]
            @constraint(model, x_ts[i,k,t] == 0)
        end
        if k < c[i] || k > d[i]
            @constraint(model, x_te[i,k,t] == 0)
        end
    end

    @constraint(model, [k=1:m],
        sum(sum(x[i,t] for t in 1:T) for i in 1:n if a[i] <= k <= d[i]) -
        sum(sum(sum(x_ts[i,ℓ,t] for ℓ in (k+1):b[i]) for t in 1:T) for i in 1:n if a[i] <= k <= b[i]) -
        sum(sum(sum(x_te[i,ℓ,t] for ℓ in c[i]:(k-1)) for t in 1:T) for i in 1:n if c[i] <= k <= d[i])
        <= 1)

    @constraint(model, [i=1:n, k=1:m], sum(x_ts[i,k,t] for t in 1:T) <= P_s[i,k])
    @constraint(model, [i=1:n, k=1:m], sum(x_te[i,k,t] for t in 1:T) <= P_e[i,k])

    @objective(model, Max, sum(weights[i]*sum(x[i,t] for t in 1:T) for i in 1:n))
    optimize!(model)
    return objective_value(model)
end

function simulate_policy_original(jobs::Vector{JobDistribution}, weights::AbstractVector, k::Int;
    policy::Symbol = :ratio, num_samples::Int=500, L::Union{Nothing,AbstractVector}=nothing, rng=Random.GLOBAL_RNG)
    n = length(jobs)
    L === nothing && (L = estimate_position_probabilities(jobs, k; num_samples=1000)[4])
    all_weights = zeros(Float64, num_samples)
    for s in 1:num_samples
        remaining = collect(1:n)
        total = 0.0
        while !isempty(remaining)
            scores = zeros(Float64, length(remaining))
            for (idx,i) in pairs(remaining)
                scores[idx] = policy == :ratio ? weights[i]/max(1e-9,L[i]) :
                               policy == :shortest ? -L[i] : weights[i]
            end
            pick_idx = argmax(scores); i_sel = remaining[pick_idx]
            s_i, e_i = sample_job_intervals([jobs[i_sel]], k); s_i=s_i[1]; e_i=e_i[1]
            total += weights[i_sel]
            new_remaining = Int[]
            for j in remaining
                if j == i_sel; continue; end
                s_j, e_j = sample_job_intervals([jobs[j]], k); s_j=s_j[1]; e_j=e_j[1]
                if max(s_i, s_j) > min(e_i, e_j)
                    push!(new_remaining, j)
                end
            end
            remaining = new_remaining
        end
        all_weights[s] = total
    end
    return mean(all_weights), all_weights
end

function simulate_policy_conservative(jobs::Vector{JobDistribution}, weights::AbstractVector, k::Int;
    policy::Symbol = :weight, num_samples::Int=1000, L::Union{Nothing,AbstractVector}=nothing, rng=Random.GLOBAL_RNG)
    n = length(jobs)
    A_L, A_R = compute_A_intervals(jobs, k)
    L === nothing && (L = estimate_position_probabilities(jobs, k; num_samples=1000)[4])
    all_weights = zeros(Float64, num_samples)
    for s in 1:num_samples
        remaining = collect(1:n)
        total = 0.0
        while !isempty(remaining)
            scores = zeros(Float64, length(remaining))
            for (idx,i) in pairs(remaining)
                scores[idx] = policy == :ratio ? weights[i]/max(1e-9,L[i]) : weights[i]
            end
            pick_idx = argmax(scores); i_sel = remaining[pick_idx]
            s_i, e_i = sample_job_intervals([jobs[i_sel]], k); s_i=s_i[1]; e_i=e_i[1]
            total += weights[i_sel]
            new_remaining = Int[]
            for j in remaining
                if j == i_sel; continue; end
                if A_R[j] < s_i || A_L[j] > e_i
                    push!(new_remaining, j)
                end
            end
            remaining = new_remaining
        end
        all_weights[s] = total
    end
    return mean(all_weights), all_weights
end

function simulate_optimal_expected(jobs::Vector{JobDistribution}, weights::AbstractVector, k::Int; num_samples::Int=1000)
    vals = zeros(Float64, num_samples)
    for s in 1:num_samples
        starts, ends = sample_job_intervals(jobs, k)
        vals[s] = weighted_interval_scheduling(starts, ends, weights)
    end
    return mean(vals), vals
end

function run_final_experiments(; rng_seed=38072,
    n_grid = (8, 10, 15, 20),
    m_grid = (10, 15, 20, 25,30),
    overlaps = (:moderate,),
    weight_modes = (:uniform,),
    # dist_types = (:uniform, :normal, :exp),
    dist_types = (:uniform, :normal),
    num_instances=1,
    num_samples_bounds=1000,
    num_samples_policy=1000,
    num_samples_opt=2000)

    Random.seed!(rng_seed)
    rows = Dict[]
    for n in n_grid, m in m_grid, ov in overlaps, wm in weight_modes, dt in dist_types, _ in 1:num_instances
        # if m < n; continue; end    
        jobs, w = generate_support_instance(n, m; overlap=ov, weight_mode=wm, dist_type=dt, sigma=1.0)
        println("Finish generating instance: n=$(n), m=$(m), overlap=$(ov), weight_mode=$(wm), dist_type=$(dt)")
        bounds = conservative_bounds(jobs, w, m; num_samples=num_samples_bounds)
        println("Finish computing bounds for the conservative case.")
        cons_w, _ = simulate_policy_conservative(jobs, w, m; policy=:weight, num_samples=num_samples_policy, L=bounds.L)
        println("Finish simulating conservative greedy policy (weight).")
        cons_r, _ = simulate_policy_conservative(jobs, w, m; policy=:ratio,  num_samples=num_samples_policy, L=bounds.L)
        println("Finish simulating conservative greedy policy (ratio).")
        orig_w, _ = simulate_policy_original(jobs, w, m; policy=:weight, num_samples=num_samples_policy, L=bounds.L)
        println("Finish simulating original greedy policy (weight).")
        orig_r, _ = simulate_policy_original(jobs, w, m; policy=:ratio,  num_samples=num_samples_policy, L=bounds.L)
        println("Finish simulating original greedy policy (ratio).")
        # orig_s, _ = simulate_policy_original(jobs, w, m; policy=:shortest, num_samples=num_samples_policy, L=bounds.L)
        opt_mc, _ = simulate_optimal_expected(jobs, w, m; num_samples=num_samples_opt)
        println("Finish simulating optimal expected stability number.")
        P_s, P_e, P_occ, _ = estimate_position_probabilities(jobs, m; num_samples=num_samples_bounds)
        lp_cons = lp_bound_conservative(P_occ, w)
        lp_orig = lp_bound_original(jobs, P_s, P_e, w, m)
        push!(rows, Dict(
            :n=>n,:m=>m,:overlap=>ov,:weight_mode=>wm,:dist_type=>dt,
            :alpha_pes=>bounds.α_pes,:p_star=>bounds.p_star,
            :dual1=>bounds.dual1,:dual2=>bounds.dual2,
            :opt_mc=>opt_mc,
            :lp_cons=>lp_cons,:lp_orig=>lp_orig,
            :cdsrse_weight=>cons_w,:cdsrse_ratio=>cons_r,
            :dsrse_weight=>orig_w,:dsrse_ratio=>orig_r
        ))
    end
    df = DataFrame(rows)
    df.alpha_over_opt = df.alpha_pes ./ max.(df.opt_mc, 1e-9)
    df.lp_cons_over_opt = df.lp_cons ./ max.(df.opt_mc, 1e-9)
    df.lp_orig_over_opt = df.lp_orig ./ max.(df.opt_mc, 1e-9)
    df.dual1_over_opt = df.dual1 ./ max.(df.opt_mc, 1e-9)
    df.dual2_over_opt = df.dual2 ./ max.(df.opt_mc, 1e-9)
    df.cdsrse_w_over_opt = df.cdsrse_weight ./ max.(df.opt_mc, 1e-9)
    df.cdsrse_r_over_opt = df.cdsrse_ratio ./ max.(df.opt_mc, 1e-9)
    df.dsrse_w_over_opt = df.dsrse_weight ./ max.(df.opt_mc, 1e-9)
    df.dsrse_r_over_opt = df.dsrse_ratio ./ max.(df.opt_mc, 1e-9)

    CSV.write("final_experiment.csv", df)

    # Line plots vs instance size (ordered by n then m) per dist_type
    for dt in unique(df.dist_type)
        sub = df[df.dist_type .== dt, :]
        sort!(sub, [:n, :m])
        xlabels = ["n=$(sub.n[i]), m=$(sub.m[i])" for i in 1:nrow(sub)]
        xticks = collect(1:nrow(sub))

        # DSRSE plot (original)
        p_dsrse = plot(xticks, sub.opt_mc; lw=2, marker=:circle, label="expected_stab",
            xlabel="instance", xticks=(xticks, xlabels), xrotation=45,
            ylabel="value", title="DSRSE vs size (dist=$(dt))")
        plot!(p_dsrse, xticks, sub.alpha_pes; lw=2, marker=:square, label="α_pes")
        plot!(p_dsrse, xticks, sub.lp_orig; lw=2, marker=:utriangle, label="DLP-P")
        plot!(p_dsrse, xticks, sub.dsrse_weight; lw=2, marker=:dtriangle, label="DSRSE weight")
        plot!(p_dsrse, xticks, sub.dsrse_ratio; lw=2, marker=:diamond, label="DSRSE ratio")
        savefig(p_dsrse, "plot_dsrse_$(dt).png")
        display(p_dsrse)

        # CDSRSE plot (conservative)
        p_cdsrse = plot(xticks, sub.opt_mc; lw=2, marker=:circle, label="expected_stab",
            xlabel="instance", xticks=(xticks, xlabels), xrotation=45,
            ylabel="value", title="CDSRSE vs size (dist=$(dt))")
        plot!(p_cdsrse, xticks, sub.alpha_pes; lw=2, marker=:square, label="α_pes")
        plot!(p_cdsrse, xticks, sub.lp_cons; lw=2, marker=:utriangle, label="CDLP-P")
        plot!(p_cdsrse, xticks, sub.cdsrse_weight; lw=2, marker=:dtriangle, label="CDSRSE weight")
        plot!(p_cdsrse, xticks, sub.cdsrse_ratio; lw=2, marker=:diamond, label="CDSRSE ratio")
        savefig(p_cdsrse, "plot_cdsrse_$(dt).png")
        display(p_cdsrse)
    end


    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    df = run_final_experiments()
    println("Finished experiments. Rows: ", nrow(df))
    println(df)
end

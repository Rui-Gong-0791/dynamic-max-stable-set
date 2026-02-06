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

# Optional SDP solvers (best effort)
const _sdp_optimizer = let
    try
        @eval using MosekTools
        () -> MosekTools.Optimizer
    catch
        try
            @eval using SCS
            () -> SCS.Optimizer
        catch
            try
                @eval using COSMO
                () -> COSMO.Optimizer
            catch
                @warn "No SDP optimizer found (MosekTools/SCS/COSMO). SDP bounds will be skipped."
                () -> nothing
            end
        end
    end
end

# Minimal helpers (copied/adapted from discrete_interval.jl to avoid solver deps).
# Bound mapping:
#   - alpha_pes: pessimistic interval bound (deterministic widest support).
#   - dual1, dual2: simple occupancy-based dual bounds from note (using P^o).
#   - opt_mc: Monte Carlo estimate of true expected optimum (exact WIS on each draw).
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

function support_overlap_matrix(jobs::Vector{JobDistribution}, m::Int)
    n = length(jobs)
    A_L, A_R = compute_A_intervals(jobs, m)
    P = zeros(Float64, n, n)
    for i in 1:(n-1)
        for j in (i+1):n
            if max(A_L[i], A_L[j]) <= min(A_R[i], A_R[j])
                P[i, j] = 1.0
                P[j, i] = 1.0
            end
        end
    end
    return P
end

function sample_job_intervals(jobs::Vector{JobDistribution}, k; rng=Random.GLOBAL_RNG)
    n = length(jobs)
    starts, ends = zeros(n), zeros(n)
    for i in 1:n
        s = round(Int, clamp(rand(rng, jobs[i].start_dist), 1, k))
        e = round(Int, clamp(rand(rng, jobs[i].end_dist), 1, k))
        while e < s
            e = round(Int, clamp(rand(rng, jobs[i].end_dist), 1, k))
        end
        starts[i] = s
        ends[i] = e
    end
    return starts, ends
end

function estimate_overlap_probabilities(jobs::Vector{JobDistribution}, k::Int; num_samples=500, rng=Random.GLOBAL_RNG)
    n = length(jobs)
    counts = zeros(Int, n, n)
    for _ in 1:num_samples
        starts, ends = sample_job_intervals(jobs, k; rng=rng)
        for i in 1:(n-1)
            for j in (i+1):n
                if max(starts[i], starts[j]) <= min(ends[i], ends[j])
                    counts[i, j] += 1
                    counts[j, i] += 1
                end
            end
        end
    end
    return counts ./ num_samples
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

function wis_dual_mu(A_L::Vector{Int}, A_R::Vector{Int}, weights::AbstractVector, m::Int)
    optimizer = _lp_optimizer()
    optimizer === nothing && return nothing, NaN
    n = length(weights)
    model = Model(optimizer)
    @variable(model, μ[1:m] >= 0)
    @constraint(model, [i=1:n], sum(μ[r] for r in A_L[i]:A_R[i]) >= weights[i])
    @objective(model, Min, sum(μ))
    optimize!(model)
    return value.(μ), objective_value(model)
end

function mu_pes_bound_nonconservative(mu::AbstractVector, jobs::Vector{JobDistribution},
    P_occ::AbstractMatrix, m::Int)
    n = length(jobs)
    a = Vector{Int}(undef, n); b = Vector{Int}(undef, n)
    c = Vector{Int}(undef, n); d = Vector{Int}(undef, n)
    for i in 1:n
        a[i], b[i] = interval_bounds(jobs[i].start_dist, m)
        c[i], d[i] = interval_bounds(jobs[i].end_dist, m)
    end

    total = 0.0
    for r in 1:m
        sum_s = 0.0
        sum_e = 0.0
        for i in 1:n
            if a[i] <= r <= b[i]
                sum_s += 1 - P_occ[i, r]
            end
            if c[i] <= r <= d[i]
                sum_e += 1 - P_occ[i, r]
            end
        end
        total += mu[r] * (1 + sum_s + sum_e)
    end
    return total
end

function generate_base_supports(n::Int, m::Int; rng=Random.GLOBAL_RNG)
    supports = Array{Int}(undef, n, 4)
    for i in 1:n
        u = rand(rng, 1:m)
        v = rand(rng, 1:m)
        a = min(u, v)
        d = max(u, v)
        if a == d
            b = a
            c = a
        else
            u2 = rand(rng, a:d)
            v2 = rand(rng, a:d)
            b = min(u2, v2)
            c = max(u2, v2)
        end
        supports[i, 1] = a
        supports[i, 2] = b
        supports[i, 3] = c
        supports[i, 4] = d
    end
    return supports
end

function extend_supports_long(supports::Matrix{Int}, m::Int)
    n = size(supports, 1)
    extended = similar(supports)
    for i in 1:n
        a, b, c, d = supports[i, 1], supports[i, 2], supports[i, 3], supports[i, 4]
        gap = c - b
        shift_left = cld(gap, 2)
        shift_right = fld(gap, 2)
        a1 = a - shift_left
        b1 = b - shift_left
        c1 = c + shift_right
        d1 = d + shift_right
        len_start = b - a
        len_end = d - c
        new_a = b1 - 2 * len_start
        new_b = b1
        new_c = c1
        new_d = c1 + 2 * len_end
        extended[i, 1] = new_a
        extended[i, 2] = new_b
        extended[i, 3] = new_c
        extended[i, 4] = new_d
    end
    min_support = minimum(extended)
    max_support = maximum(extended)
    if min_support < 1 || max_support > m
        shift = 1 - min_support
        extended .+= shift
        m_used = max_support - min_support + 1
    else
        m_used = m
    end
    return extended, m_used
end

function generate_weights(n::Int; weight_mode::Symbol=:uniform, rng=Random.GLOBAL_RNG)
    weights = zeros(Float64, n)
    for i in 1:n
        weights[i] = weight_mode == :uniform ? 1.0 :
                     weight_mode == :uniform01 ? rand(rng) :
                     weight_mode == :exp1 ? rand(rng, Exponential(1.0)) :
                     rand(rng)
    end
    return weights
end

function supports_to_jobs(supports::Matrix{Int}; dist_type::Symbol=:uniform)
    n = size(supports, 1)
    jobs = JobDistribution[]
    for i in 1:n
        a, b, c, d = supports[i, 1], supports[i, 2], supports[i, 3], supports[i, 4]
        start_dist = if dist_type == :uniform
            DiscreteUniform(a, b)
        elseif dist_type == :normal
            if a == b
                Dirac(a)
            else
                sigma = max(1e-6, (b - a) / 3)
                truncated(Normal((a + b) / 2, sigma), a, b)
            end
        else
            error("unknown dist_type")
        end

        end_dist = if dist_type == :uniform
            DiscreteUniform(c, d)
        elseif dist_type == :normal
            if c == d
                Dirac(d)
            else
                sigma = max(1e-6, (d - c) / 3)
                truncated(Normal((c + d) / 2, sigma), c, d)
            end
        else
            error("unknown dist_type")
        end

        push!(jobs, JobDistribution(start_dist, end_dist))
    end
    return jobs
end

function conservative_bounds(jobs::Vector{JobDistribution}, weights::AbstractVector, m::Int; num_samples::Int=1000)
    n = length(jobs)
    P_s, P_e, P_occ, L = estimate_position_probabilities(jobs, m; num_samples=num_samples)
    A_L, A_R = compute_A_intervals(jobs, m)
    α_pes = weighted_interval_scheduling(A_L, A_R, weights)  # pessimistic bound
    mu_pes, _ = wis_dual_mu(A_L, A_R, weights, m)
    mu_bound = isnothing(mu_pes) ? NaN : mu_pes_bound_nonconservative(mu_pes, jobs, P_occ, m)
    p_star = minimum(filter(>(0.0), P_occ))                  # min positive occupancy
    alpha_pes_over_pstar = p_star > 0 ? α_pes / p_star : NaN
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
    return (α_pes=α_pes, p_star=p_star, dual1=dual1, dual2=dual2, L=vec(L),
            mu_pes_bound=mu_bound, alpha_pes_over_pstar=alpha_pes_over_pstar)
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

# SDP relaxation for original case (SDP-P).
function sdp_bound_original(jobs::Vector{JobDistribution}, P_s::AbstractMatrix, P_e::AbstractMatrix, weights::AbstractVector, m::Int)
    optimizer = _sdp_optimizer()
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

    NT = n * T
    x_flat = [x[i,t] for t in 1:T for i in 1:n]
    @variable(model, X[1:NT, 1:NT], Symmetric)
    @constraint(model, [1 x_flat'; x_flat X] in PSDCone())
    @constraint(model, X .>= 0)
    @constraint(model, [p=1:NT], X[p,p] == x_flat[p])

    idx(i, t) = (t - 1) * n + i
    for t in 1:T, i in 1:n, j in 1:n
        if i != j
            @constraint(model, X[idx(i,t), idx(j,t)] == 0)
        end
    end
    for i in 1:n, t in 1:T, tau in 1:T
        if t != tau
            @constraint(model, X[idx(i,t), idx(i,tau)] == 0)
        end
    end
    for i in 1:n, t in 2:T
        @constraint(model, x[i,t] == sum(X[idx(i,t), idx(j,t-1)] for j in 1:n if j != i))
    end

    @objective(model, Max, sum(weights[i]*sum(x[i,t] for t in 1:T) for i in 1:n))
    optimize!(model)
    return objective_value(model)
end

function add_ratio_columns!(df::DataFrame)
    df.alpha_over_opt = df.alpha_pes ./ max.(df.opt_mc, 1e-9)
    if :alpha_pes_over_pstar in names(df)
        df.alpha_pes_over_pstar_over_opt = df.alpha_pes_over_pstar ./ max.(df.opt_mc, 1e-9)
    end
    df.lp_cons_over_opt = df.lp_cons ./ max.(df.opt_mc, 1e-9)
    df.lp_orig_over_opt = df.lp_orig ./ max.(df.opt_mc, 1e-9)
    df.sdp_orig_over_opt = df.sdp_orig ./ max.(df.opt_mc, 1e-9)
    df.dual1_over_opt = df.dual1 ./ max.(df.opt_mc, 1e-9)
    df.dual2_over_opt = df.dual2 ./ max.(df.opt_mc, 1e-9)
    if :mu_pes_bound in names(df)
        df.mu_pes_bound_over_opt = df.mu_pes_bound ./ max.(df.opt_mc, 1e-9)
    end
    df.cdsrse_w_over_opt = df.cdsrse_weight ./ max.(df.opt_mc, 1e-9)
    df.cdsrse_r_over_opt = df.cdsrse_ratio ./ max.(df.opt_mc, 1e-9)
    df.dsrse_w_over_opt = df.dsrse_weight ./ max.(df.opt_mc, 1e-9)
    df.dsrse_r_over_opt = df.dsrse_ratio ./ max.(df.opt_mc, 1e-9)
    return df
end

function add_gap_columns!(df::DataFrame)
    df.alpha_gap = abs.(1 .- df.alpha_over_opt)
    if :alpha_pes_over_pstar_over_opt in names(df)
        df.alpha_pes_over_pstar_gap = abs.(1 .- df.alpha_pes_over_pstar_over_opt)
    end
    df.lp_cons_gap = abs.(1 .- df.lp_cons_over_opt)
    df.lp_orig_gap = abs.(1 .- df.lp_orig_over_opt)
    df.sdp_orig_gap = abs.(1 .- df.sdp_orig_over_opt)
    df.dual1_gap = abs.(1 .- df.dual1_over_opt)
    df.dual2_gap = abs.(1 .- df.dual2_over_opt)
    if :mu_pes_bound_over_opt in names(df)
        df.mu_pes_bound_gap = abs.(1 .- df.mu_pes_bound_over_opt)
    end
    df.cdsrse_w_gap = abs.(1 .- df.cdsrse_w_over_opt)
    df.cdsrse_r_gap = abs.(1 .- df.cdsrse_r_over_opt)
    df.dsrse_w_gap = abs.(1 .- df.dsrse_w_over_opt)
    df.dsrse_r_gap = abs.(1 .- df.dsrse_r_over_opt)
    return df
end

function simulate_policy_original(jobs::Vector{JobDistribution}, weights::AbstractVector, k::Int;
    policy::Symbol = :ratio, num_samples::Int=500, L::Union{Nothing,AbstractVector}=nothing,
    P_del::Union{Nothing,AbstractMatrix}=nothing, rng=Random.GLOBAL_RNG)
    n = length(jobs)
    L === nothing && (L = estimate_position_probabilities(jobs, k; num_samples=1000)[4])
    all_weights = zeros(Float64, num_samples)
    for s in 1:num_samples
        remaining = collect(1:n)
        total = 0.0
        while !isempty(remaining)
            scores = zeros(Float64, length(remaining))
            for (idx,i) in pairs(remaining)
                if policy == :ratio
                    scores[idx] = weights[i]/max(1e-9, L[i])
                elseif policy == :shortest
                    scores[idx] = -L[i]
                elseif policy == :weight
                    if P_del === nothing
                        scores[idx] = weights[i]
                    else
                        loss = 0.0
                        for j in remaining
                            j == i && continue
                            loss += weights[j] * P_del[i, j]
                        end
                        scores[idx] = weights[i] - loss
                    end
                else
                    scores[idx] = weights[i]
                end
            end
            pick_idx = argmax(scores); i_sel = remaining[pick_idx]
            s_i, e_i = sample_job_intervals([jobs[i_sel]], k; rng=rng); s_i=s_i[1]; e_i=e_i[1]
            total += weights[i_sel]
            new_remaining = Int[]
            for j in remaining
                if j == i_sel; continue; end
                s_j, e_j = sample_job_intervals([jobs[j]], k; rng=rng); s_j=s_j[1]; e_j=e_j[1]
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
    policy::Symbol = :weight, num_samples::Int=1000, L::Union{Nothing,AbstractVector}=nothing,
    P_del::Union{Nothing,AbstractMatrix}=nothing, rng=Random.GLOBAL_RNG)
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
                if policy == :ratio
                    scores[idx] = weights[i]/max(1e-9, L[i])
                elseif policy == :weight
                    if P_del === nothing
                        scores[idx] = weights[i]
                    else
                        loss = 0.0
                        for j in remaining
                            j == i && continue
                            loss += weights[j] * P_del[i, j]
                        end
                        scores[idx] = weights[i] - loss
                    end
                else
                    scores[idx] = weights[i]
                end
            end
            pick_idx = argmax(scores); i_sel = remaining[pick_idx]
            s_i, e_i = sample_job_intervals([jobs[i_sel]], k; rng=rng); s_i=s_i[1]; e_i=e_i[1]
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

function run_final_experiments_design(; rng_seed=38072,
    n_grid = (8, 10, 15, 20),
    m_grid = (10, 15, 20, 25, 30),
    size_pairs = [(8, 12), (10, 15), (12, 18), (14, 21), (16, 24), (18, 27), (19, 29),
                  (20, 30), (40, 60), (80, 120)],
    weight_modes = (:uniform01,),
    dist_types = (:uniform,),
    densities = (:dense, :sparse),
    lengths = (:short, :long),
    sdp_n_max = 12,
    run_sdp = false,
    num_instances=30,
    num_samples_bounds=1000,
    num_samples_overlap=500,
    num_samples_policy=1000,
    num_samples_opt=1000)

    Random.seed!(rng_seed)
    rows = Dict[]
    rng = Random.GLOBAL_RNG
    if size_pairs === nothing
        size_iter = ((n, m) for n in n_grid for m in m_grid)
    else
        size_iter = size_pairs
    end
    for (n, m) in size_iter, wm in weight_modes, inst in 1:num_instances
        base_supports = generate_base_supports(n, m; rng=rng)
        base_weights = generate_weights(n; weight_mode=wm, rng=rng)
        n_sparse = max(1, n ÷ 2)
        keep_idx = sort(sample(rng, 1:n, n_sparse; replace=false))

        supports_long, m_long = extend_supports_long(base_supports, m)

        for len in lengths
            supports_len = len == :short ? base_supports : supports_long
            m_used = len == :short ? m : m_long
            for dens in densities
                idx = dens == :dense ? collect(1:n) : keep_idx
                supports_use = supports_len[idx, :]
                weights_use = base_weights[idx]
                for dt in dist_types
                    jobs = supports_to_jobs(supports_use; dist_type=dt)
                    println("Instance: n=$(n), m=$(m), length=$(len), density=$(dens), dist=$(dt), rep=$(inst)")
                    bounds = conservative_bounds(jobs, weights_use, m_used; num_samples=num_samples_bounds)
                    P_del_orig = estimate_overlap_probabilities(jobs, m_used; num_samples=num_samples_overlap, rng=rng)
                    P_del_cons = support_overlap_matrix(jobs, m_used)
                    cons_w, _ = simulate_policy_conservative(jobs, weights_use, m_used;
                        policy=:weight, num_samples=num_samples_policy, L=bounds.L, P_del=P_del_cons)
                    cons_r, _ = simulate_policy_conservative(jobs, weights_use, m_used;
                        policy=:ratio,  num_samples=num_samples_policy, L=bounds.L, P_del=P_del_cons)
                    orig_w, _ = simulate_policy_original(jobs, weights_use, m_used;
                        policy=:weight, num_samples=num_samples_policy, L=bounds.L, P_del=P_del_orig)
                    orig_r, _ = simulate_policy_original(jobs, weights_use, m_used;
                        policy=:ratio,  num_samples=num_samples_policy, L=bounds.L, P_del=P_del_orig)
                    opt_mc, _ = simulate_optimal_expected(jobs, weights_use, m_used;
                        num_samples=num_samples_opt)
                    P_s, P_e, P_occ, _ = estimate_position_probabilities(jobs, m_used;
                        num_samples=num_samples_bounds)
                    lp_cons = lp_bound_conservative(P_occ, weights_use)
                    lp_orig = lp_bound_original(jobs, P_s, P_e, weights_use, m_used)
                    sdp_orig = (run_sdp && length(weights_use) <= sdp_n_max) ?
                        sdp_bound_original(jobs, P_s, P_e, weights_use, m_used) : NaN

                    push!(rows, Dict(
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
                        :lp_cons=>lp_cons, :lp_orig=>lp_orig, :sdp_orig=>sdp_orig,
                        :cdsrse_weight=>cons_w, :cdsrse_ratio=>cons_r,
                        :dsrse_weight=>orig_w, :dsrse_ratio=>orig_r
                    ))
                end
            end
        end
    end

    df_raw = DataFrame(rows)
    add_ratio_columns!(df_raw)
    add_gap_columns!(df_raw)
    CSV.write("final_experiment_new_raw.csv", df_raw)

    group_cols = [:n_base, :m_base, :density, :length, :weight_mode, :dist_type]
    ratio_cols = [:alpha_over_opt, :alpha_pes_over_pstar_over_opt, :lp_cons_over_opt,
                  :lp_orig_over_opt, :sdp_orig_over_opt, :dual1_over_opt, :dual2_over_opt,
                  :mu_pes_bound_over_opt, :cdsrse_w_over_opt, :cdsrse_r_over_opt,
                  :dsrse_w_over_opt, :dsrse_r_over_opt]
    num_cols = names(df_raw, Number)
    value_cols = setdiff(num_cols, union(group_cols, ratio_cols, [:instance_id]))
    df_avg = combine(groupby(df_raw, group_cols), value_cols .=> mean .=> value_cols)
    add_ratio_columns!(df_avg)
    CSV.write("final_experiment_new.csv", df_avg)

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

    for dt in unique(df_avg.dist_type), dens in unique(df_avg.density), len in unique(df_avg.length)
        sub = df_avg[(df_avg.dist_type .== dt) .& (df_avg.density .== dens) .& (df_avg.length .== len), :]
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

        p_dsrse = plot(xticks, expected_line; lw=2, marker=:circle, label="expected_stab",
            xlabel="instance", xticks=(xticks, xlabels), xrotation=45,
            ylabel="value / expected_stab", title="DSRSE ($(dens), $(len), dist=$(dt))",
            legend=:outertopright, ylims=ylims)
        plot!(p_dsrse, xticks, sub.alpha_over_opt; lw=2, marker=:square, label="α_pes")
        plot!(p_dsrse, xticks, sub.lp_orig_over_opt; lw=2, marker=:utriangle, label="DLP-P")
        if "mu_pes_bound_over_opt" in names(sub) &&
           any(x -> isfinite(x), skipmissing(sub.mu_pes_bound_over_opt))
            plot!(p_dsrse, xticks, sub.mu_pes_bound_over_opt; lw=2, marker=:star5, label="μ_pes bound")
        end
        plot!(p_dsrse, xticks, sub.dsrse_w_over_opt; lw=2, marker=:dtriangle, label="DSRSE weight")
        plot!(p_dsrse, xticks, sub.dsrse_r_over_opt; lw=2, marker=:diamond, label="DSRSE ratio")
        savefig(p_dsrse, "plot_dsrse_$(dens)_$(len)_$(dt).png")
        display(p_dsrse)

        cdsrse_cols = [:alpha_over_opt, :lp_cons_over_opt, :cdsrse_w_over_opt, :cdsrse_r_over_opt]
        minv, maxv = series_extrema(sub, cdsrse_cols)
        minv = min(minv, 1.0)
        maxv = max(maxv, 1.0)
        pad = 0.05 * (maxv - minv)
        pad = pad == 0.0 ? 0.1 * maxv : pad
        ylims = (minv - pad, maxv + pad)

        p_cdsrse = plot(xticks, expected_line; lw=2, marker=:circle, label="expected_stab",
            xlabel="instance", xticks=(xticks, xlabels), xrotation=45,
            ylabel="value / expected_stab", title="CDSRSE ($(dens), $(len), dist=$(dt))",
            legend=:outertopright, ylims=ylims)
        plot!(p_cdsrse, xticks, sub.alpha_over_opt; lw=2, marker=:square, label="α_pes")
        plot!(p_cdsrse, xticks, sub.lp_cons_over_opt; lw=2, marker=:utriangle, label="CDLP-P")
        plot!(p_cdsrse, xticks, sub.cdsrse_w_over_opt; lw=2, marker=:dtriangle, label="CDSRSE weight")
        plot!(p_cdsrse, xticks, sub.cdsrse_r_over_opt; lw=2, marker=:diamond, label="CDSRSE ratio")
        savefig(p_cdsrse, "plot_cdsrse_$(dens)_$(len)_$(dt).png")
        display(p_cdsrse)
    end

    return df_avg
end

if abspath(PROGRAM_FILE) == @__FILE__
    df_avg = run_final_experiments_design()
    println("Finished experiments (averaged). Rows: ", nrow(df_avg))
    println(df_avg)
end

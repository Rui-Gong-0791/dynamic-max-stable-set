using Random, Distributions, StatsBase
using LinearAlgebra, Random, Statistics

begin
    try
        using Convex
    catch e
        @warn "Convex not available; skipping" exception=e
    end
    try
        using LightGraphs
    catch e
        @warn "LightGraphs not available; skipping" exception=e
    end
    try
        using JuMP
    catch e
        @warn "JuMP not available; skipping" exception=e
    end
end

using DataFrames
using CSV
using Plots          # or: using StatsPlots
using MathOptInterface
const MOI = MathOptInterface

function simulate_job_assignment(k::Int, n::Int, job_distributions, job_weights, num_simulations::Int=1000)
    """
    Simulate job assignment to positions and calculate expected total weight.
    
    Parameters:
    - k: number of positions
    - n: number of jobs
    - job_distributions: vector of length n, each element is a probability vector of length k
    - job_weights: vector of length n with job weights
    - num_simulations: number of simulation runs
    
    Returns:
    - expected_weight: average total weight across simulations
    - all_weights: vector of total weights from all simulations
    """
    
    all_weights = Float64[]
    
    for sim in 1:num_simulations
        # Initialize positions with empty job lists
        position_jobs = [Float64[] for _ in 1:k]
        
        # Assign each job to a position according to its distribution
        for job_idx in 1:n
            # Sample position for this job
            position = sample(1:k, Weights(job_distributions[job_idx]))
            # Add job weight to that position
            push!(position_jobs[position], job_weights[job_idx])
        end
        
        # For each position, take the job with largest weight (if any)
        total_weight = 0.0
        for pos in 1:k
            if !isempty(position_jobs[pos])
                total_weight += maximum(position_jobs[pos])
            end
        end
        
        push!(all_weights, total_weight)
    end
    
    expected_weight = mean(all_weights)
    return expected_weight, all_weights
end

function simulate_greedy(k::Int, n::Int, job_distributions, job_weights, num_simulations::Int=1000)
    """
    Greedy simulation: process jobs in descending weight order, sampling a position
    for each and keeping the heaviest job that lands on a position.
    """
    all_weights = Float64[]

    job_order = sortperm(job_weights; rev = true)  # heaviest first

    for _ in 1:num_simulations
        best_at_pos = fill(0.0, k)

        for job_idx in job_order
            position = sample(1:k, Weights(job_distributions[job_idx]))
            if job_weights[job_idx] > best_at_pos[position]
                best_at_pos[position] = job_weights[job_idx]
            end
        end

        push!(all_weights, sum(best_at_pos))
    end

    expected_weight = mean(all_weights)
    return expected_weight, all_weights
end

# Example usage
function example_simulation(k,n)
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Parameters
    # k = 5  # number of positions
    # n = 30  # number of jobs
    
    # Generate random job distributions (each job has a probability distribution over positions)
    job_distributions = []
    for i in 1:n
        # Generate random probabilities and normalize
        probs = rand(k)
        # probs = ones(k)./k
        probs = probs ./ sum(probs)  # normalize to sum to 1
        push!(job_distributions, probs)
    end
    
    # Generate random job weights
    job_weights = rand(1.0:10.0, n)  # weights between 1 and 10
    
    println("Job Assignment Simulation")
    println("=" ^ 40)
    println("Number of positions (k): $k")
    println("Number of jobs (n): $n")
    println()
    
    println("Job Details:")
    for i in 1:n
        println("Job $i: weight = $(job_weights[i]), distribution = $(round.(job_distributions[i], digits=3))")
    end
    println()
    
    # Run simulation
    expected_weight, all_weights = simulate_job_assignment(k, n, job_distributions, job_weights, 3000)
    
    println("Simulation Results:")
    println("Expected total weight: $(round(expected_weight, digits=3))")
    println("Standard deviation: $(round(std(all_weights), digits=3))")
    println("Min weight observed: $(round(minimum(all_weights), digits=3))")
    println("Max weight observed: $(round(maximum(all_weights), digits=3))")
    
    return expected_weight, all_weights, job_distributions, job_weights
end


# Additional analysis function
function analyze_results(all_weights)
    """
    Provide additional statistical analysis of simulation results
    """
    println("\nDetailed Analysis:")
    println("=" ^ 40)
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90]
    println("Percentiles:")
    for p in percentiles
        val = quantile(all_weights, p/100)
        println("  $(p)th percentile: $(round(val, digits=3))")
    end
    
    # Histogram-like summary
    println("\nDistribution summary:")
    hist_edges = range(minimum(all_weights), maximum(all_weights), length=6)
    for i in 1:(length(hist_edges)-1)
        count = sum((all_weights .>= hist_edges[i]) .& (all_weights .< hist_edges[i+1]))
        if i == length(hist_edges)-1  # Include the maximum in the last bin
            count = sum((all_weights .>= hist_edges[i]) .& (all_weights .<= hist_edges[i+1]))
        end
        println("  [$(round(hist_edges[i], digits=2)), $(round(hist_edges[i+1], digits=2))): $count")
    end
end



function compute_inner_product_matrix(job_distributions)
    """
    Compute the inner product matrix of job probability distributions.
    
    Parameters:
    - job_distributions: vector of length n, each element is a probability vector of length k
    
    Returns:
    - inner_product_matrix: n×n matrix where element (i,j) is the inner product of 
      job i's distribution with job j's distribution
    """
    n = length(job_distributions)
    inner_product_matrix = zeros(n, n)+I(n)
    
    for i in 1:n
        for j in 1:n
            # Compute inner product (dot product) of distributions
            inner_product_matrix[i, j] = dot(job_distributions[i], job_distributions[j])
        end
    end
    # position_prob = [prod(1 .-job_distributions[i]) for i in 1:k]
    return inner_product_matrix
end

struct JobDistribution
    start_dist::UnivariateDistribution
    end_dist::UnivariateDistribution
end

function generate_job_distributions(n,k; start_type=:uniform, end_type=:uniform, 
                                    start_params=Nothing, end_params=Nothing, 
                                    weight_dist=Uniform(1, 10))
    jobs = JobDistribution[]
    for i in 1:n
        # Setup start distribution
        if start_type == :uniform
            s_params = start_params === Nothing ? (1, floor(k/2)) : start_params[i]
            start_dist = DiscreteUniform(s_params...)
        elseif start_type == :normal
            s_params = start_params === Nothing ? (floor(k/4), 2.0) : start_params[i]
            start_dist = truncated(Normal(s_params...), 0, Inf)  # ensure positivity
        elseif start_type == :constant
            s_params = start_params === Nothing ? (1,) : start_params[i]
            start_dist = Dirac(s_params[1])  # constant distribution
        else
            error("Unknown start_type: $start_type")
        end

        # Setup end distribution
        if end_type == :uniform
            e_params = end_params === Nothing ? (ceil(k/2), k) : end_params[i]
            end_dist = DiscreteUniform(e_params...)
        elseif end_type == :normal
            e_params = end_params === Nothing ? (floor(3*k/4), 2.0) : end_params[i]
            end_dist = truncated(Normal(e_params...), 0, Inf)
        elseif end_type == :constant
            e_params = end_params === Nothing ? (k,) : end_params[i]
            println("Using constant end distribution with parameter: ", e_params)
            end_dist = Dirac(e_params[1])  # constant distribution
        else
            error("Unknown end_type: $end_type")
        end

        push!(jobs, JobDistribution(start_dist, end_dist))
    end
    job_weights = rand(1.0:10.0, n)
    return jobs, job_weights
end

function sample_job_intervals(jobs::Vector{JobDistribution},k)
    n = length(jobs)
    starts, ends = zeros(n), zeros(n)
    for i in 1:n
        s = round(Int, clamp(rand(jobs[i].start_dist), 1, k))
        e = round(Int, clamp(rand(jobs[i].end_dist), 1, k))

        # Test the case with constant length 2
        # s = round(Int, clamp(rand(jobs[i].start_dist), 1, k))
        # e = s
        while e < s    # force end after start
            e = round(Int, clamp(rand(jobs[i].end_dist), 1, k))
        end
        starts[i] = s
        ends[i] = e
    end
    return starts, ends
end

function estimate_intersection_probability(job1::JobDistribution, job2::JobDistribution,k; num_samples=10000)
    count = 0
    mins1, _ = interval_bounds(job1.start_dist, k)
    _, maxs1 = interval_bounds(job1.end_dist, k)
    mins2, _ = interval_bounds(job2.start_dist, k)
    _, maxs2 = interval_bounds(job2.end_dist, k)
    count_12 = 0
    count_21 = 0
    for _ in 1:num_samples
        s1 = round(Int, clamp(rand(job1.start_dist), 1, k))
        e1 = round(Int, clamp(rand(job1.end_dist), 1, k))
        while e1 < s1
            e1 = round(Int, clamp(rand(job1.end_dist), 1, k))
        end
        s2 = round(Int, clamp(rand(job2.start_dist), 1, k))
        e2 = round(Int, clamp(rand(job2.end_dist), 1, k))
        while e2 < s2
            e2 = round(Int, clamp(rand(job2.end_dist), 1, k))
        end

        # Test the case where length is constant 4
        # s1= round(Int, clamp(rand(job1.start_dist), 1, k))
        # e1 = s1
        # s2 = round(Int, clamp(rand(job2.start_dist), 1, k))
        # e2 = s2
        if max(s1, s2) <= min(e1, e2)
            count += 1
        end
        if !(s1>=maxs2) && !(e1<=mins2)
            count_12 += 1
        end
        if !(s2>=maxs1) && !(e2<=mins1)
            count_21 += 1
        end
    end
    return count / num_samples, count_12 / num_samples, count_21 / num_samples
end

function compute_intersection_matrix(jobs::Vector{JobDistribution},k; num_samples=10000)
    n = length(jobs)
    P = Matrix{Float64}(I, n, n)
    P_new = Matrix{Float64}(I, n, n)
    for i in 1:n, j in i+1:n
        prob, p_ij, p_ji = estimate_intersection_probability(jobs[i], jobs[j],k, num_samples=num_samples)
        P[i, j] = prob
        P[j, i] = prob
        P_new[i, j] = p_ij
        P_new[j, i] = p_ji
    end
    return P, P_new
end

function estimate_position_probabilities(jobs::Vector{JobDistribution}, k::Int; num_samples=10000)
    n = length(jobs)
    P_s = zeros(n, k)
    P_e = zeros(n, k)
    P = zeros(n, k)
    L = zeros(n)
    for i in 1:n
        counts_s = zeros(Int, k)
        counts_e = zeros(Int, k)
        counts = zeros(Int, k)
        for s in 1:num_samples
            start = round(Int, clamp(rand(jobs[i].start_dist), 1, k))
            e = round(Int, clamp(rand(jobs[i].end_dist), 1, k))
            while e < start
                e = round(Int, clamp(rand(jobs[i].end_dist), 1, k))
            end
            for t in start:e
                if 1 <= t <= k
                    counts[t] += 1
                end
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
    return P_s, P_e,P,L
end

function weighted_interval_scheduling(starts::Vector{<:Real}, ends::Vector{<:Real}, weights::Vector{<:Real})
    n = length(starts)
    if n == 0
        return 0.0
    end

    # Sort jobs by end time
    idxs = sortperm(ends)
    s = starts[idxs]
    e = ends[idxs]
    w = weights[idxs]

    # Compute p[j]: last job before j that does not overlap
    p = zeros(Int, n)
    for j in 1:n
        i = searchsortedlast(e, s[j] - 1e-8) # finds largest i where e[i] < s[j]
        p[j] = i
    end

    # DP: opt[j] = max total weight using jobs 1..j
    opt = zeros(Float64, n+1)
    for j in 1:n
        # opt[j] is max total weight using jobs 1..j
        # opt[p[j]] is the max total weight for compatible jobs before j
        opt[j+1] = max(opt[j], w[j] + opt[p[j]+1])
    end

    return opt[n+1]
end

function simulate_jobs(jobs::Vector{JobDistribution}, weights::Vector{Float64}, k, num_simulations=10000)
    total_weights = Float64[]
    n = length(jobs)
    for sim in 1:num_simulations
        starts, ends = sample_job_intervals(jobs,k)
        # For example, get the maximum total weight of non-overlapping jobs
        # (This is the interval scheduling maximization, greedy by earliest end)
        # idxs = sortperm(ends)
        # last_end, wsum = -Inf, 0.0
        # for i in idxs
        #     if starts[i] > last_end
        #         wsum += weights[i]
        #         last_end = ends[i]
        #     end
        # end

        wsum = weighted_interval_scheduling(starts, ends, weights)
        

        push!(total_weights, wsum)
    end
    return mean(total_weights), total_weights
end


# ---------------------------
# Utilities: A(i) bounds from distributions
# ---------------------------
"""
    interval_bounds(d::UnivariateDistribution, m; eps=1e-9) -> (L,U)

Return integer bounds within 1..m from a distribution's support.
L = ceil(minimum(d)), U = floor(maximum(d)), clipped to [1,m].
If unbounded, use quantiles (eps, 1-eps).
Ensures L ≤ U (collapses to a point if needed).
"""
function interval_bounds(d::UnivariateDistribution, m::Int; eps=1e-9)
    a = try Distributions.minimum(d) catch; -Inf end
    b = try Distributions.maximum(d) catch;  Inf end
    if !isfinite(a); a = quantile(d, eps); end
    if !isfinite(b); b = quantile(d, 1 - eps); end
    L = max(1, min(m, ceil(Int, a)))
    U = max(1, min(m, floor(Int, b)))
    if U < L
        U = L
    end
    return (L, U)
end

"""
    compute_A_intervals(jobs, m) -> (A_L, A_R)

Given `jobs::Vector{JobDistribution}`, return A(i) = [A_L[i], A_R[i]],
where A_L[i] = smallest possible start (from start_dist),
and A_R[i] = largest possible end (from end_dist),
both clipped to 1..m.
"""
function compute_A_intervals(jobs::Vector{JobDistribution}, m::Int)
    n = length(jobs)
    A_L = Vector{Int}(undef, n)
    A_R = Vector{Int}(undef, n)
    for i in 1:n
        Ls, _  = interval_bounds(jobs[i].start_dist, m)
        _,  Re = interval_bounds(jobs[i].end_dist,   m)
        A_L[i] = Ls
        A_R[i] = max(Ls, Re)   # ensure nonempty (collapse if needed)
    end
    return A_L, A_R
end

function compute_B_intervals(jobs::Vector{JobDistribution}, m::Int)
    n = length(jobs)
    A_s = Vector{Int}(undef, n)
    A_e = Vector{Int}(undef, n)
    for i in 1:n
        _, Rs  = interval_bounds(jobs[i].start_dist, m)
        Le,  _ = interval_bounds(jobs[i].end_dist,   m)
        A_s[i] = Rs
        A_e[i] = max(Le, Rs)   # ensure nonempty (collapse if needed)
    end
    return A_s, A_e
end

# ---------------------------
# Model builder
# ---------------------------
"""
    solve_dynamic_star_lp(jobs, m, w, sample_job_intervals;
                          optimizer,
                          binary::Bool=false,
                          add_position_cuts::Bool=false)

Builds and solves:
    max  Σ_i w[i] Σ_t x[i,t]
s.t. Σ_i x[i,t] ≤ 1                        ∀ t
    Σ_t x[i,t] ≤ 1                        ∀ i
    (optional) Σ_{i: k∈B(i)} x[i,t] ≤ 1   ∀ k,t
    Σ_{τ>t} x[j,τ] ≤ 1 - x[i,t]           ∀ i, t, ∀ j s.t. B(i)∩A(j)≠∅
    0 ≤ x[i,t] ≤ 1 (LP) or x[i,t] ∈ {0,1} (MIP)

T = min(n, m).  `sample_job_intervals(jobs, m)` must return two
Int vectors `(B_L, B_R)` of length n giving the realized interval B(i)=[B_L[i], B_R[i]].

Returns a NamedTuple with objective value, x-matrix, and status.
"""
function solve_weight_schedule_conserv(jobs::Vector{JobDistribution},
                               m::Int,
                               w::AbstractVector,
                               A_L, A_R, B_L, B_R;
                               binary::Bool = true,
                               add_position_cuts::Bool = false)

    n = length(jobs)
    length(w) == n || error("w must have length n")
    T = min(n, m)


    length(B_L) == n && length(B_R) == n || error("sample_job_intervals must return two vectors of length n")

    # Precompute intersections for rule: B(i) ∩ A(j) ≠ ∅
    function intersects(i::Int, j::Int)
        # [B_L[i],B_R[i]] ∩ [A_L[j],A_R[j]] ≠ ∅ ?
        return !(B_R[i] < A_L[j] || A_R[j] < B_L[i])
    end
    # adjacency lists: for each i, those j affected
    follows = [Int[] for _ in 1:n]
    for i in 1:n, j in 1:n
        if intersects(i, j)
            push!(follows[i], j)
        end
    end

    # Optional: for each position k, jobs whose realized B(i) contains k
    jobs_at_k = Vector{Vector{Int}}(undef, m)
    for k in 1:m
        jobs_at_k[k] = [i for i in 1:n if (B_L[i] <= k <= B_R[i])]
    end

    # Build model
    model = Model(Gurobi.Optimizer)

    # Variables
    if binary
        @variable(model, x[1:n, 1:T], Bin)
    else
        @variable(model, 0 <= x[1:n, 1:T] <= 1)
    end

    # Time capacity: one job at most per time
    @constraint(model, [t=1:T], sum(x[i,t] for i in 1:n) <= 1)

    # Each job at most once
    @constraint(model, [i=1:n], sum(x[i,t] for t in 1:T) <= 1)

    # Optional position cuts (often redundant with time capacity, but included if requested)
    if add_position_cuts
        @constraint(model, [k=1:m, t=1:T], sum(x[i,t] for i in jobs_at_k[k]) <= 1)
    end

    # Order/precedence-type constraints: if i at t, j with B(i)∩A(j)≠∅ cannot be after t
    # sum_{τ>t} x[j,τ] ≤ 1 - x[i,t]
    for i in 1:n, t in 1:T
        # empty sum when t==T is okay (equals 0 ≤ 1 - x[i,T])
        for j in follows[i]
            if t < T
                @constraint(model, sum(x[j, τ] for τ in (t+1):T) <= 1 - x[i,t])
            else
                @constraint(model, 0 <= 1 - x[i,t])
            end
        end
    end

    # Objective
    @objective(model, Max, sum(w[i] * sum(x[i,t] for t in 1:T) for i in 1:n))

    optimize!(model)
    status = termination_status(model)
    obj     = objective_value(model)
    x_val   = value.(x)
    return (objective = obj, x = x_val, status = status,
            A_L = A_L, A_R = A_R, B_L = B_L, B_R = B_R, T = T)
end

function simulate_jobs_conserv(jobs::Vector{JobDistribution}, weights::Vector{Float64}, k, num_simulations=1000)
    total_weights = Float64[]
    n = length(jobs)
    for sim in 1:num_simulations
        # A(i): earliest start & latest end from distributions
        A_L, A_R = compute_A_intervals(jobs, k)

        # B(i): realized interval from your sampler (assumed valid, 1..m)
        # Expected API: returns (B_L::Vector{Int}, B_R::Vector{Int})
        B_L, B_R = sample_job_intervals(jobs, k)
        # For example, get the maximum total weight of non-overlapping jobs
        # (This is the interval scheduling maximization, greedy by earliest end)
        # idxs = sortperm(ends)
        # last_end, wsum = -Inf, 0.0
        # for i in idxs
        #     if starts[i] > last_end
        #         wsum += weights[i]
        #         last_end = ends[i]
        #     end
        # end

        wsum = solve_weight_schedule_conserv(jobs, k, weights, A_L,A_R,B_L,B_R).objective
        

        push!(total_weights, wsum)
    end
    return mean(total_weights), total_weights
end


"Discrete median index in 1..k: smallest j with CDF(d, j) ≥ 0.5 (clamped)."
function discrete_median_pos(d::UnivariateDistribution, k::Int)
    # handle unbounded supports via quantiles; then clamp to 1..k
    lo = try Distributions.minimum(d) catch; -Inf end
    hi = try Distributions.maximum(d) catch;  Inf end
    if !isfinite(lo); lo = quantile(d, 1e-12); end
    if !isfinite(hi); hi = quantile(d, 1 - 1e-12); end
    # search the first integer j with CDF ≥ 0.5
    j0 = 1
    best = k
    for j in 1:k
        if cdf(d, j) >= 0.5
            best = j
            break
        end
    end
    return max(1, min(k, best))
end

function discrete_survival_median_pos(d::UnivariateDistribution, k::Int)
    best = 1
    for j in 1:k
        if 1 - cdf(d, j-1) >= 0.5   # P(end ≥ j)
            best = j
        else
            break
        end
    end
    return best
end


"""
    median_intervals(jobs, k) -> (s::Vector{Int}, e::Vector{Int})

For each job, pick s_i = discrete median of start_dist, e_i = discrete median of end_dist,
then ensure s_i ≤ e_i by swapping if needed. Returns the start/end vectors.
"""
function median_intervals(jobs::Vector{JobDistribution}, k::Int)
    n = length(jobs)
    s = Vector{Int}(undef, n)
    e = Vector{Int}(undef, n)
    for i in 1:n
        si = discrete_median_pos(jobs[i].start_dist, k)
        ei = discrete_survival_median_pos(jobs[i].end_dist,   k)
        # ensure a valid (non-empty) interval
        if si <= ei
            s[i] = si; e[i] = ei
        else
            s[i] = ei; e[i] = si
        end
    end
    return s, e
end

"""
    stability_number_interval_graph(s, e) -> (alpha::Int, chosen::Vector{Int})

Greedy max independent set for intervals [s[i], e[i]] on the line:
sort by nondecreasing e[i], pick whenever s[i] > last_end.
Returns the size and indices selected.
"""
function stability_number_interval_graph(s::AbstractVector{<:Integer},
                                         e::AbstractVector{<:Integer})
    n = length(s); @assert length(e) == n
    order = sortperm(1:n, by = i -> e[i])
    chosen = Int[]
    last_end = -typemax(Int)  # effectively -∞
    for i in order
        if s[i] > last_end
            push!(chosen, i)
            last_end = e[i]
        end
    end
    return length(chosen), chosen
end

# ===== Convenience wrapper you can call directly =====
"""
    stability_from_job_medians(jobs, k)

Compute s_i and e_i at median positions, build intervals, and return
(alpha, chosen, s, e).
"""
function stability_from_job_medians(jobs::Vector{JobDistribution}, k::Int, weights = ones(length(jobs)))
    s, e = median_intervals(jobs, k)
    α= weighted_interval_scheduling(s, e, weights)
    return α
end

function stability_from_job_opt(jobs::Vector{JobDistribution}, k::Int, weights = ones(length(jobs)))
    s, e = compute_B_intervals(jobs, k)
    α= weighted_interval_scheduling(s, e, weights)
    return α
end




function solve_rui_relaxation(n,k,p, w=ones(n), job_distributions = ones(n,k), pos_prob = ones(k))
    # model = Model(Gurobi.Optimizer)
    model = Model(COPT.ConeOptimizer)
    # set_optimizer_attribute(model, "OutputFlag", 0)

    # @variable(model, X[1:n,1:k])
    @variable(model, M[1:(1+n*k+n),1:(1+n*k+n)], PSD)  # SDP relaxation variable
    # @constraint(model, X .>= 0)  # Ensure non-negativity
    @constraint(model, M .<= 1) 
    # @constraint(model, [i=1:n,j=1:k], M[(i-1)*k+j,1+n*k+n] <= job_distributions[i,j])  # Ensure X[i,j] <= job_distributions[i,j]
    @constraint(model, [i=1:n,j=1:k], M[(i-1)*k+j,(i-1)*k+j] == M[(i-1)*k+j,1+n*k+n])  # Ensure X[i,j] >= 0
    # @constraint(model, [i=1:n,j=1:k], X[i,j] == M[(i-1)*k+j,1+n*k+n])  # Link X to M
    @constraint(model, M[1+n*k+n,1+n*k+n] == 1)  # Ensure M[1,1] is 1 (first element of the SDP matrix)
    # @constraint(model, [i=1:n,j=1:k],sum(M[(i-1)*k+j,l] for l in 1:(n*k)) <= 2*M[(i-1)*k+j,(i-1)*k+j])  # Link weights to M This is not a valid inequality.
    @constraint(model, [i=1:n,j=1:k, l = 1:(n*k)],M[(i-1)*k+j,l] <= M[(i-1)*k+j,(i-1)*k+j])  # Link weights to M
    # @constraint(model, [i=1:n,j=1:k],M[(i-1)*k+j,(i-1)*k+j] == X[i,j])
    @constraint(model,
    [j = 1:k, i = 1:n, g = (i+1):n],
    M[(i - 1)*k + j, (g - 1)*k + j] == 0
    )
    @constraint(model,
    [i = 1:n, j = (i+1):n],
    sum(M[(i - 1)*k + g, (j - 1)*k + l] for g in 1:k, l in 1:k) <= 1 - p[i, j]
    )
    # @constraint(model,
    # [i=1:n, j=1:k, g=1:k; g ≠ j],
    # M[(i-1)*k+j, (i-1)*k+g] == 0
    # )

    @constraint(model, [i=1:n, j = 1:n, t=1:k], M[n*k+i,(j-1)*k+t] == sum(M[(i-1)*k+g, (j-1)*k+t] for g in 1:k))  # Ensure M[n*k+i,(j-1)*n+t] is the sum of X[i,j] for all t
    @constraint(model, [i=1:(n-1), j = (i+1):n], M[n*k+i,(n*k+i)]+M[n*k+j,(n*k+j)] <= 2-p[i,j])  # Ensure M[(i-1)*k+j,(n*k+i)] is the sum of X[i,j] for all t
    @constraint(model, [i=1:n], M[n*k+i,(n*k+i)] == M[n*k+i,n*k+n+1])  # Ensure each job is assigned to exactly one position
    @constraint(model, [i=1:n, j = (i+1):n], M[n*k+i,n*k+j] == sum(M[n*k+i,(j-1)*k+t] for t in 1:k))  # Ensure each job is assigned to exactly one position
    @constraint(model, [i=1:n,j = (i+1:n)], M[n*k+i,n*k+j]  <= 1-p[i,j])  # Ensure each job is assigned to exactly one position


    # @variable(model, y[1:n,1:k])
    # @variable(model, b[1:n,1:(k-1)])
    # @constraint(model, y .<= 1)  
    # @constraint(model, y .>= 0)  
    # @constraint(model, b .<= 1)  
    # @constraint(model, b .>= 0)  
    # @constraint(model, [i=1:n,j=1:k], M[(i-1)*k+j,(i-1)*k+j]<= y[i,j])  # Ensure X[i,j] <= y[i,j]
    # @constraint(model, [i=1:n,j=1:k], M[(i-1)*k+j,(i-1)*k+j] >= -y[i,j])  # Ensure X[i,j] <= y[i,j]
    # @constraint(model, [i=1:n,j=1:(k-1)], y[i,j]-y[i,j+1] <= b[i,j])  # Ensure X[i,j] <= b[i,j]
    # @constraint(model, [i=1:n,j=1:(k-1)], y[i,j]-y[i,j+1] >= -b[i,j])  # Ensure X[i,j] <= b[i,j]
    # @constraint(model, [i=1:n], sum(b[i,:])<=2)  # Ensure X[i,j] <= b[i,j]

    # Add edge constraints
    # @constraint(model, [j=1:k], 1-sum(X[i,j] for i in 1:n) <= prod((1-X[i,j]) for i in 1:n))  # Ensure each position has at most one job assigned

    # @constraint(model, [j=1:k], sum(X[i,j] for i in 1:n) <= 1)
    # @constraint(model, [i=1:n], sum(X[i,j] for j in 1:k) <= 1)

    # @constraint(model, [i=1:n,j=(i+1):n], sum(X[i,g]*(sum(X[j,l] for l in 1:k)-X[j,g]) for g in 1:k)<= 1-p[i,j])  # Ensure x[i,j] is binary

    # @constraint(model,[i = 1:N,j=(i+1):N], x[i]*x[j] <= 1-p[i,j])  # Ensure x[i] is binary

    # Maximize size of independent set
    @objective(model, Max, sum(w[i] * M[n*k+i,n*k+i] for i in 1:n))  # Maximize total weight of jobs assigned to positions

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        solution = value.(M[n*k+1:n*k+n,n*k+1:n*k+n])
        return solution, objective_value(model), value.(M)
    else
        # Return nothing or handle non-optimal cases
        println("Optimization failed with status: ", termination_status(model))
        return nothing, nothing
    end
end

function random_param(n::Integer, k::Integer; rng = Random.GLOBAL_RNG)
    k ≥ 4 || throw(ArgumentError("k must be at least 2"))
    v1 = Vector{NTuple{2,Int}}(undef, n)
    v2 = Vector{NTuple{2,Int}}(undef, n)

    for i in 1:n
        # --- first pair (a, b) with a ≤ b < k -----------------------------
        a = rand(rng, 1:k)         # leave room so b < k
        b = rand(rng, a:k)
        v1[i] = (a, b)

        # --- second pair (c, d) with c > b, c ≤ d ≤ k ---------------------
        c = rand(rng, b:k)         # b+1 … k is non-empty because b < k
        d = rand(rng, c:k)
        v2[i] = (c, d)
    end

    return v1, v2
end


# pick a radius according to a "mode": :random, :max, or a fixed Int

_pick_r(mode, rmax, rng) = mode === :max ? rmax :
                           mode isa Integer ? max(0, min(mode, rmax)) :
                           (rmax == 0 ? 0 : rand(rng, 0:rmax))

function centered_discrete_uniform_pair(μs::Int, μe::Int, k::Int;
    rng = Random.GLOBAL_RNG, start_halfwidth = :random, end_halfwidth = :random)

    (1 ≤ μs ≤ k) || throw(ArgumentError("μs must be in 1..k"))
    (1 ≤ μe ≤ k) || throw(ArgumentError("μe must be in 1..k"))
    μe ≥ μs      || throw(ArgumentError("need μe ≥ μs (nonnegative length)"))

    # boundary-limited symmetric radii (no `_` usage)
    rmax_s = max(0, min(μs - 1, k - μs))
    rmax_e = max(0, min(μe - 1, k - μe))
    gap    = μe - μs  # = ℓ - 1

    # choose end radius first, then fit start into leftover to ensure no overlap
    r_e_max = min(rmax_e, gap)
    r_e = _pick_r(end_halfwidth, r_e_max, rng)

    r_s_max = min(rmax_s, gap - r_e)
    r_s = _pick_r(start_halfwidth, r_s_max, rng)

    a_s, b_s = μs - r_s, μs + r_s
    a_e, b_e = μe - r_e, μe + r_e
    @assert b_s <= a_e  # guaranteed by construction

    return DiscreteUniform(a_s, b_s), DiscreteUniform(a_e, b_e)
end

"""
Generate job distributions from a list of desired *expected lengths*.

For each job:
  1) pick a length ℓ from `lengths` (optionally with probabilities `length_probs`)
  2) pick a start mean μ_s ∈ {1, …, k-ℓ+1}
  3) set end mean μ_e = μ_s + ℓ - 1
  4) build start/end DiscreteUniform distributions centered at μ_s and μ_e
     (exact mean equals the desired value)

Returns: jobs::Vector{JobDistribution}, weights::Vector{Float64},
         meta::NamedTuple (chosen lengths and means)
"""
function generate_job_distributions_from_lengths(n::Int, k::Int, lengths::AbstractVector{<:Integer};
    rng = Random.GLOBAL_RNG,
    length_probs::Union{Nothing,AbstractVector{<:Real}} = nothing,
    start_halfwidth = :random,  # Int, :random, or :max
    end_halfwidth   = :random,  # Int, :random, or :max
    weight_dist = Uniform(1, 10),
 )
    # sanitize lengths
    valid_lengths = filter(ℓ -> 1 <= ℓ <= k, lengths)
    isempty(valid_lengths) && error("No valid lengths (must be between 1 and k).")

    # sampler for ℓ
    pick_len = if isnothing(length_probs)
        () -> rand(rng, valid_lengths)
    else
        length(length_probs) == length(lengths) || error("length_probs must match `lengths`.")
        probs = collect(length_probs)
        sum(probs) > 0 || error("length_probs must have positive sum.")
        probs ./= sum(probs)
        w = Weights(probs)
        () -> lengths[rand(rng, w)]
    end

    jobs = JobDistribution[]
    chosen_lengths = Vector{Int}(undef, n)
    s_means = Vector{Int}(undef, n)
    e_means = Vector{Int}(undef, n)

    for i in 1:n
        ℓ = pick_len()
        ℓ > k && (ℓ = k)  # just in case
        # pick start mean so that end mean stays within 1..k
        μs = rand(rng, 1:(k - ℓ + 1))
        μe = μs + ℓ - 1

        sd, ed = centered_discrete_uniform_pair(μs, μe, k;
    rng=rng, start_halfwidth=start_halfwidth, end_halfwidth=end_halfwidth)

        push!(jobs, JobDistribution(sd, ed))
        chosen_lengths[i] = ℓ
        s_means[i] = μs
        e_means[i] = μe
    end

    # weights
    job_weights = rand(rng, weight_dist, n)

    meta = (lengths = chosen_lengths, s_means = s_means, e_means = e_means)
    return jobs, job_weights, meta
end

function solve_x_it_relaxation(n,k,p, w=ones(n), job_distributions = ones(n,k), job_distributions_s = ones(n,k), job_distributions_e = ones(n,k),pos_prob = ones(k))
    # model = Model(Gurobi.Optimizer)
    model = Model(COPT.ConeOptimizer)
    # set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, M[1:(1+n*k+n),1:(1+n*k+n)], PSD)  # SDP relaxation variable
    @constraint(model, M .>= 0)  # Ensure non-negativity
    @constraint(model, M .<= 1) 
    # @constraint(model, [i=1:n,j=1:k], M[(i-1)*k+j,(i-1)*k+j] == M[(i-1)*k+j,1+n*k+n]) # Last column corresponds to Diag(M)
    @constraint(model, [i=1:n], M[n*k+i,(n*k+i)] == M[n*k+i,n*k+n+1])  # Last column corresponds to Diag(A)
    @constraint(model, M[1+n*k+n,1+n*k+n] == 1)  # Ensure bottom-right element of the SDP matrix is 1


    @constraint(model, [t = 1:k], sum(M[(i-1)*k+t, (i-1)*k+t] for i in 1:n) <= 1)  # Probability of schedule a job at time t is at most 1
    @constraint(model, [i = 1:n], sum(M[(i-1)*k+t, (i-1)*k+t] for t in 1:k) <= 1)  # Probability of schedule job i over all time slots is at most 1
    @constraint(model, [i=1:n,t=1:k, l = 1:(n*k)],M[(i-1)*k+t,l] <= M[(i-1)*k+t,(i-1)*k+t])  # Probability of scheduling job i at time t is at least the probability of scheduling job i at time t and scheduling another job at some time
    @constraint(model,
    [j = 1:k, i = 1:n, g = (i+1):n],
    M[(i - 1)*k + j, (g - 1)*k + j] == 0
    )  # Ensure no two jobs are scheduled at the same time.
    @constraint(model,
    [i=1:n, j=1:k, g=1:k; g ≠ j],
    M[(i-1)*k+j, (i-1)*k+g] == 0
    ) # Ensure no job is scheduled at two different times.
    # @constraint(model, [i=1:n,j=1:k],sum(M[(i-1)*k+j,l] for l in 1:(n*k)) <= 2*M[(i-1)*k+j,(i-1)*k+j])  # Link weights to M This is not a valid inequality.

    @constraint(model, [i=1:n, j = 1:n, t=1:k], M[n*k+i,(j-1)*k+t] == sum(M[(i-1)*k+g, (j-1)*k+t] for g in 1:k))  # D_{i,(j,g)} = \sum_{t=1,t\neq g}^{m}M_{(i,t),(j,g)},~\forall i,~\forall j,g = \sum_{t=1}^{m}M_{(i,t),(j,g)},~\forall i,~\forall j,g
    @constraint(model, [i=1:n, j = (i+1):n], M[n*k+i,n*k+j] == sum(M[n*k+i,(j-1)*k+t] for t in 1:k)) # A_{ij}=\sum_{t=1}^{m}D_{i,(j,t)},~\forall i,j\in [n], i\neq j
    @constraint(model, [i=1:n], M[n*k+i,(n*k+i)] == sum(M[(i-1)*k+t, (i-1)*k+t] for t in 1:k))  # A_{ii}=\sum_{t=1}^{m}M_{(i,t),(i,t)},~\forall i\in[n]
    
    @constraint(model, [i=1:(n-1), j = (i+1):n], M[n*k+i,(n*k+i)]+M[n*k+j,(n*k+j)] <= 2-p[i,j])  #A_ii+A_jj <= 2-p_{ij}
    @constraint(model, [i=1:n,j = (i+1:n)], M[n*k+i,n*k+j]  <= 1-p[i,j])  # A_{ij} <= 1-p_{ij}


    l = vec(maximum(job_distributions, dims = 1))
    # @constraint(model, [t = 1:k], sum(M[n*k+i,n*k+i]*job_distributions[i,t] for i in 1:n) <= 1)  # Ensure each position has at most one job 
    v = vec(maximum(job_distributions_s, dims = 1))
    # @constraint(model, [t = 1:k], sum(M[n*k+i,n*k+i]*job_distributions_s[i,t] for i in 1:n) <= 1)  # Ensure each position has at most one job assigned
    u = vec(maximum(job_distributions_e, dims = 1))
    # @constraint(model, [t = 1:k], sum(M[n*k+i,n*k+i]*job_distributions_e[i,t] for i in 1:n) <= 1)  # Ensure each position has at most one job assigned


    @constraint(model, [t=1:(k-1)], sum(M[(i-1)*k+t, (i-1)*k+t] for i in 1:n) >= sum(M[(i-1)*k+t+1, (i-1)*k+t+1] for i in 1:n)) #probability of scheduling a job at time t is at least the probability of scheduling a job at time t+1
    @constraint(model, [i=1:n, t= 2:k], M[(i-1)*k+t, (i-1)*k+t]<=(1-sum(M[(i-1)*k+tau, (i-1)*k+tau] for tau=1:(t-1))))  # Schedule job i at time t only if it is not scheduled before t
    
    @constraint(model, [i=1:n,t=2:k, tau = 1:(t-1)], M[(i-1)*k+t, (i-1)*k+t] == sum(M[(i-1)*k+t, (j-1)*k+tau] for j=1:n if j != i)) # Schedule job i at time t is equivalent to scheduling job i at time t and scheduling other jobs at any time tau before t
    @constraint(model, [i=1:n, t= 2:k], M[(i-1)*k+t, (i-1)*k+t] <= sum(M[(j-1)*k+t-1, (j-1)*k+t-1]*(1-p[i,j]) for j=1:n if j != i)) # Schedule job i at time t only if another job is scheduled at time t-1 and not delete i.
    
    @constraint(model, sum(M[(i-1)*k+1, (i-1)*k+1]* (1-prod(p[i,j] for j in 1:n if j!=i)) for i in 1:n) == sum(M[(i-1)*k+2, (i-1)*k+2] for i in 1:n))  # Schedule a job at time 2 is equivalent to schedule a job at time 1 and at least one job left at time 2.
    
    # @variable(model, X_s[1:n,1:k]) 
    # @variable(model, X_e[1:n,1:k])
    # @constraint(model, X_s .>= 0)
    # @constraint(model, X_e .>= 0)
    # @constraint(model, [i=1:n], sum(X_s[i,:]) == M[n*k+i,n*k+i])  # Ensure X_s[i,:] sums to M[n*k+i,n*k+i]
    # @constraint(model, [i=1:n], sum(X_e[i,:]) == M[n*k+i,n*k+i])  # Ensure X_e[i,:] sums to M[n*k+i,n*k+i]
    # @constraint(model, [i=1:n,j=1:k], X_s[i,j] <= job_distributions_s[i,j])  # Ensure X_s[i,j] <= job_distributions_s[i,j]
    # @constraint(model, [i=1:n,j=1:k], X_e[i,j] <= job_distributions_e[i,j])  # Ensure X_e[i,j] <= job_distributions_e[i,j]
    # @constraint(model, [j=1:k], sum(X_s[:,j]) <= 1)  # Ensure X_s[i,j] <= X[i]
    # @constraint(model, [j=1:k], sum(X_e[:,j]) <= 1)  # Ensure X_s[i,j] <= X[i]


    # @variable(model, X_s[1:n,1:k]) 
    # @variable(model, X_e[1:n,1:k])
    # @constraint(model, X_s .>= 0)
    # @constraint(model, X_e .>= 0)
    # @constraint(model, [i=1:n], sum(X_s[i,:]) == M[n*k+i,n*k+i])
    # @constraint(model, [i=1:n], sum(X_e[i,:]) == M[n*k+i,n*k+i])
    # @constraint(model, [i=1:n,j=1:k], X_s[i,j] <= job_distributions_s[i,j])  # Ensure X_s[i,j] <= job_distributions_s[i,j]
    # @constraint(model, [i=1:n,j=1:k], X_e[i,j] <= job_distributions_e[i,j])  # Ensure X_e[i,j] <= job_distributions_e[i,j]
    
    # v = vec(maximum(job_distributions_s, dims = 1))
    
    # u = vec(maximum(job_distributions_e, dims = 1))
    # @constraint(model, [j=1:k], sum(X_s[:,j]) <= v[j]) 
    # @constraint(model, [j=1:k], sum(X_e[:,j]) <= u[j]) 
    # u = vec(maximum(p, dims = 1))
    # @constraint(model, [t = 1:k], sum(M[n*k+i,n*k+i]*jj for i in 1:n) <= u[t])  # Ensure each position has at most one job assigned
    # @constraint(model, [t = 1:k], sum(M[(i-1)*k+1,(i-1)*n+1]*job_distributions[i,t] for i in 1:n) <= v[t])



    # @constraint(model, M[(4-1)*k+1,(4-1)*k+1] == 1)  # Ensure no job assigned to position 1 if any job assigned to position 2
    # @constraint(model, M[(2-1)*k+2,(2-1)*k+2] == 0.375)  # Ensure no job assigned to position 1 if any job assigned to position 2
    # @constraint(model, M[(1-1)*k+2,(1-1)*k+2] == 0.625)  # Ensure no job assigned to position 1 if any job assigned to position 2
    # @constraint(model, M[(1-1)*k+3,(1-1)*k+3] == 0.1875)  # Ensure no job assigned to position 1 if any job assigned to position 2
    # @constraint(model, M[(1-1)*k+4,(1-1)*k+4] == 0)  # Ensure no job assigned to position 1 if any job assigned to position 2
    # @constraint(model, M[(3-1)*k+3,(3-1)*k+3] == 0.0625)  # Ensure no job assigned to position 1 if any job assigned to position 2
    # @constraint(model, M[(3-1)*k+4,(3-1)*k+4] == 0.0625)  # Ensure no job assigned to position 1 if any job assigned to position 2

    # @constraint(model, [i = 1:n,j=(i+1):n], sum(M[(i-1)*k+t, (j-1)*k+tau] for t = 1:k, tau= (t+1):k) <= p[i,j])
    # @constraint(model, [i=1:n, t= 2:k], M[(i-1)*k+t, (i-1)*k+t]<= sum(M[(j-1)*k+t-1, (j-1)*k+t-1]*(1-p[i,j]) for j=1:n if j != i))  # Ensure no job assigned to position t if any job assigned to position before t
    # @constraint(model, [t=1:(k-1)], sum(M[(i-1)*k+t, (i-1)*k+t]* prod((1-p[i,j]) for j in 1:n if j != i) for i in 1:n) <= sum(M[(i-1)*k+t+1, (i-1)*k+t+1] for i in 1:n))  # 
    # @constraint(model, [t=1:(k-1)], sum(M[(i-1)*k+t, (i-1)*k+t]* (1-maximum(p[i,:])) for i in 1:n) <= sum(M[(i-1)*k+t+1, (i-1)*k+t+1] for i in 1:n))  # 
    # @variable(model, y[1:n,1:k])
    # @variable(model, b[1:n,1:(k-1)])
    # @constraint(model, y .<= 1)  
    # @constraint(model, y .>= 0)  
    # @constraint(model, b .<= 1)  
    # @constraint(model, b .>= 0)  
    # @constraint(model, [i=1:n,j=1:k], M[(i-1)*k+j,(i-1)*k+j]<= y[i,j])  # Ensure X[i,j] <= y[i,j]
    # @constraint(model, [i=1:n,j=1:k], M[(i-1)*k+j,(i-1)*k+j] >= -y[i,j])  # Ensure X[i,j] <= y[i,j]
    # @constraint(model, [i=1:n,j=1:(k-1)], y[i,j]-y[i,j+1] <= b[i,j])  # Ensure X[i,j] <= b[i,j]
    # @constraint(model, [i=1:n,j=1:(k-1)], y[i,j]-y[i,j+1] >= -b[i,j])  # Ensure X[i,j] <= b[i,j]
    # @constraint(model, [i=1:n], sum(b[i,:])<=2)  # Ensure X[i,j] <= b[i,j]

    # Add edge constraints
    # @constraint(model, [j=1:k], 1-sum(X[i,j] for i in 1:n) <= prod((1-X[i,j]) for i in 1:n))  # Ensure each position has at most one job assigned

    # @constraint(model, [j=1:k], sum(X[i,j] for i in 1:n) <= 1)
    # @constraint(model, [i=1:n], sum(X[i,j] for j in 1:k) <= 1)

    # @constraint(model, [i=1:n,j=(i+1):n], sum(X[i,g]*(sum(X[j,l] for l in 1:k)-X[j,g]) for g in 1:k)<= 1-p[i,j])  # Ensure x[i,j] is binary

    # @constraint(model,[i = 1:N,j=(i+1):N], x[i]*x[j] <= 1-p[i,j])  # Ensure x[i] is binary

    # Maximize size of independent set
    @objective(model, Max, sum(w[i] * M[n*k+i,n*k+i] for i in 1:n))  # Maximize total weight of jobs assigned to positions

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        solution = value.(M[(n*k+1):(n*k+n),(n*k+1):(n*k+n)])
        return solution, objective_value(model), value.(M)
    else
        # Return nothing or handle non-optimal cases
        println("Optimization failed with status: ", termination_status(model))
        return nothing, nothing
    end
end


function solve_x_it_relaxation_new(n,k,p, w=ones(n), job_distributions = ones(n,k), job_distributions_s = ones(n,k), job_distributions_e = ones(n,k), L = ones(n), pos_prob = ones(k))
    # model = Model(Gurobi.Optimizer)
    model = Model(COPT.ConeOptimizer)
    T= min(n, k)  # Number of time slots
    # set_optimizer_attribute(model, "OutputFlag", 0)

    # @variable(model, M[1:(1+n),1:(1+n)], PSD)  # SDP relaxation variable
    @variable(model, M[1:(1+n),1:(1+n)])  
    @constraint(model, M .>= 0)  # Ensure non-negativity
    @constraint(model, M .<= 1) 
    # @constraint(model, [i=1:n,j=1:k], M[(i-1)*k+j,(i-1)*k+j] == M[(i-1)*k+j,1+n*k+n]) # Last column corresponds to Diag(M)
    # @constraint(model, [i=1:n], M[i,i] == M[i,n+1])  # Last column corresponds to Diag(A)
    # @constraint(model, M[1+n,1+n] == 1)  # Ensure bottom-right element of the SDP matrix is 1

    P = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if i == j
                P[i,j] = 0
            else
                P[i,j] = min(p[i,j],p[j,i])
            end
        end
    end
    @constraint(model, [i=1:(n-1), j = (i+1):n], M[i,i]+M[j,j] <= 2-P[i,j])  #A_ii+A_jj <= 2-p_{ij}
    # @constraint(model, [i=1:n,j = (i+1:n)], M[i,j]  <= 1-P[i,j])  # A_{ij} <= 1-p_{ij}


    l = vec(maximum(job_distributions, dims = 1))
    @constraint(model, [m = 1:k], sum(M[i,i]*job_distributions[i,m] for i in 1:n) <= l[m])  # Ensure each position has at most one job 
    v = vec(maximum(job_distributions_s, dims = 1))
    @constraint(model, [m = 1:k], sum(M[i,i]*job_distributions_s[i,m] for i in 1:n) <= v[m])  # Ensure each position has at most one job assigned
    u = vec(maximum(job_distributions_e, dims = 1))
    @constraint(model, [m = 1:k], sum(M[i,i]*job_distributions_e[i,m] for i in 1:n) <= u[m])  # Ensure each position has at most one job assigned

    # @constraint(model, [i=1:n], sum(M[i,i]*L[i] for i in 1:n) <= k)

    # S = n * T
    # @variable(model, Z[1:S+1, 1:S+1], PSD)
    # @constraint(model, Z[S+1, S+1] == 1.0)
    # @constraint(model, Z.>= 0)
    
    # # flat index for selection variables
    # idx_sel(i, t) = (t-1)*n + i
    
    # # Link diagonal and last column of Z to X_t (s_{it})
    # for i in 1:n
    #     @constraint(model, sum(Z[idx_sel(i,t), idx_sel(i,t)] for t in 1:T)     == M[i,i])  # Ensure X[i,j] <= y[i,j]
    #     @constraint(model, [t =1:T], Z[idx_sel(i,t), S+1]   == Z[idx_sel(i,t), idx_sel(i,t)])  # Ensure X[i,j] <= y[i,j]
    # end
    
    # # Zero pattern matching your rules:
    # # (A) same job across different times -> zero
    # for i in 1:n, t in 1:T, τ in (t+1):T
    #     if τ != t
    #         @constraint(model, Z[idx_sel(i,t), idx_sel(i,τ)] == 0.0)
    #     end
    # end
    
    # # (B) different jobs at the same time -> zero
    # for t in 1:T, i in 1:n, j in (i+1):n
    #     if i != j
    #         @constraint(model, Z[idx_sel(i,t), idx_sel(j,t)] == 0.0)
    #     end
    # end
    
    # # Probability of co-selection cap:  P(both i and j) ≤ 1 - p[i,j]
    # @constraint(model, [i=1:n, j=1:n,t=1:T],
    #     sum(Z[idx_sel(i,tau), idx_sel(j,τ)] for tau in 1:t, τ in (t+1):T) ≤ (1 - p[i,j])*sum(Z[idx_sel(i,tau),idx_sel(i,tau)] for tau in 1:t))
    
    # # @constraint(model,[i=1:n, j=1:n, t= 1:T],
    #     # sum(Z[idx_sel(i,tau),idx_sel(i,tau)] for tau in 1:t)*(1-p[i,j])>=sum(Z[idx_sel(j,tau),idx_sel(j,tau)] for tau in (t+1):T))

    # # Select i at time t if co-selected with any job at any prior time
    # @constraint(model, [t=2:T, i=1:n,  τ in 1:(t-1)],
    #     Z[idx_sel(i,t), idx_sel(i,t)] == sum(Z[idx_sel(i,t), idx_sel(j,τ)] for j in 1:n))
    # @constraint(model, [t=1:T], sum(Z[idx_sel(i,t),idx_sel(i,t)] for i in 1:n)<=1)
    # @constraint(model, [i=1:n], sum(Z[idx_sel(i,t),idx_sel(i,t)] for t in 1:T)<=1)

    # @constraint(model, [i=1:n,j=1:n, t=1:T], sum(Z[idx_sel(i,tau),idx_sel(i,tau)] for tau in 1:t)+sum(Z[idx_sel(j,tau),idx_sel(j,tau)] for tau in (t+1):T)<= 2-p[i,j])
    # Maximize size of independent set
    @objective(model, Max, sum(w[i] * M[i,i] for i in 1:n))  # Maximize total weight of jobs assigned to positions

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        solution = value.(M)
        return solution, objective_value(model), value.(M)
    else
        # Return nothing or handle non-optimal cases
        println("Optimization failed with status: ", termination_status(model))
        return nothing, nothing
    end
end

function solve_x_it_relaxation_new_dual(n,k,p, w=ones(n), job_distributions = ones(n,k), job_distributions_s = ones(n,k), job_distributions_e = ones(n,k), L = ones(n), pos_prob = ones(k))
    # model = Model(Gurobi.Optimizer)
    model = Model(COPT.ConeOptimizer)
    # set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, M[1:(1+n),1:(1+n)], PSD)  # SDP relaxation variable
    @variable(model, gamma[1:k])  # SDP relaxation variable
    @variable(model, alpha[1:n,1:n])  # SDP relaxation variable
    @variable(model, beta[1:n,1:n])  # SDP relaxation variable
    # @variable(model, lambda_e[1:k])  # SDP relaxation variable
    # @variable(model, lambda_s[1:k])  # SDP relaxation variable


    @variable(model, u) 
    @variable(model, v)
    @variable(model, z)

    @constraint(model, gamma .>= 0)  # Ensure non-negativity
    @constraint(model, beta .>= 0)  # Ensure non-negativity
    # @constraint(model, lambda_e .>= 0)  # Ensure non-negativity
    # @constraint(model, lambda_s .>= 0)  # Ensure non-negativity
    @constraint(model, alpha .>= 0)  # Ensure non-negativity

    @constraint(model, [i=1:n,j = 1:n], beta[i,j]>= M[i,j])
    @constraint(model, [i=1:n,j = (i+1):n], beta[i,j] == beta[j,i])
    @constraint(model, [i=1:n,j = (i+1):n], alpha[i,j] == alpha[j,i])
    
    @constraint(model, [i=1:n], -w[i] - M[i,i] - 2*M[i,n+1]+sum(alpha[i,j] for j in 1:n if j!=i)+sum(alpha[j,i] for j in 1:n if j!=i)+sum(gamma[t]*job_distributions[i,t] for t in 1:k) == 0)  # Dual constraints for each job

    @constraint(model, sum(alpha[i,j]*(2-p[i,j]) for i=1:n for j=1:n if j!=i) == u)
    @constraint(model, sum(beta[i,j]*(1-p[i,j]) for i=1:n for j=1:n if j!=i) == v)
    @constraint(model, sum(gamma[t] for t=1:k) == z)

    # Maximize size of independent set
    @objective(model, Min, M[n+1,n+1]+u+v+z)  # Maximize total weight of jobs assigned to positions

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        solution = value.(M)
        return solution, objective_value(model), value.(gamma), value.(beta), value.(alpha)
    else
        # Return nothing or handle non-optimal cases
        println("Optimization failed with status: ", termination_status(model))
        return nothing, nothing
    end
end



function solve_x_it_LP(n,k,p, w=ones(n), job_distributions = ones(n,k), job_distributions_s = ones(n,k), job_distributions_e = ones(n,k),pos_prob = ones(k))
    # model = Model(Gurobi.Optimizer)
    model = Model(COPT.ConeOptimizer)
    # set_optimizer_attribute(model, "OutputFlag", 0)

    # @variable(model, X[1:n,1:k])
    @variable(model, X[1:n])  
    @variable(model, X_s[1:n,1:k]) 
    @variable(model, X_e[1:n,1:k])
    @variable(model, X_o[1:n,1:k])
    @constraint(model, X .>= 0)
    @constraint(model, X_s .>= 0)
    @constraint(model, X_e .>= 0)
    @constraint(model, X_o .>= 0)
    @constraint(model, [i=1:n], sum(X_s[i,:]) == X[i])
    @constraint(model, [i=1:n], sum(X_e[i,:]) == X[i])
    @constraint(model, [i=1:n,j=1:k], X_s[i,j] <= job_distributions_s[i,j])  # Ensure X_s[i,j] <= job_distributions_s[i,j]
    @constraint(model, [i=1:n,j=1:k], X_e[i,j] <= job_distributions_e[i,j])  # Ensure X_e[i,j] <= job_distributions_e[i,j]
    @constraint(model, [i=1:n,j=1:k], X_o[i,j] <= job_distributions[i,j])  # Ensure X_e[i,j] <= job_distributions_e[i,j]
    
    
    
    v = vec(maximum(job_distributions_s, dims = 1))
    
    u = vec(maximum(job_distributions_e, dims = 1))
    # l = vec(maximum(job_distributions_e.+job_distributions_s.-job_distributions_e.*job_distributions_s, dims = 1))
    # l = vec(minimum([l';ones(k)'], dims = 1))  # Ensure each position has at least one job assigned
    l = vec(maximum(job_distributions, dims = 1))
    @constraint(model, [j=1:k], sum(X_s[:,j]) <= 1)  # Ensure X_s[i,j] <= X[i]
    @constraint(model, [j=1:k], sum(X_e[:,j]) <= 1)  # Ensure X_s[i,j] <= X[i]
    @constraint(model, [j=1:k], sum(X_o[:,j]) <= 1)  # Ensure X_s[i,j] <= X[i]
    # @constraint(model, [j=1:k], sum(X_s[:,j])+sum(X_e[:,j]) <= l[j])  # Ensure X_s[i,j] <= X[i]

    # P = zeros(n, n)
    # for i in 1:n
    #     for j in 1:n
    #         if i == j
    #             P[i,j] = 1
    #         else
    #             P[i,j] = min(p[i,j],p[j,i])
    #         end
    #     end
    # end
    @constraint(model, [i=1:(n-1), j = (i+1):n], X[i]+X[j] <= 2-p[i,j])  # Ensure M[(i-1)*k+j,(n*k+i)] is the sum of X[i,j] for all t
    # @constraint(model, [i=1:n,j = (i+1:n)], M[n*k+i,n*k+j]  <= 1-p[i,j])  # Ensure each job is assigned to exactly one position
    # @constraint(model, [i=1:n, t = 2:k, ti = 1:(t-1)], sum(M[(i-1)*k+tau,(i-1)*k+tau] for tau in t:k)  <= sum(M[(j-1)*k+ti,(j-1)*k+ti]*(1-p[i,j]) for j in 1:n if j != i) )

    # @constraint(model, [t = 1:k], sum(X[i]*job_distributions[i,t] for i in 1:n) <= sum(X_o[:,t]))  # Ensure each position has at most one job 
    # @constraint(model, [t = 1:k], sum(X[i]*job_distributions_s[i,t] for i in 1:n) <= sum(X_s[:,t]))  # Ensure each position has at most one job assigned
    # @constraint(model, [t = 1:k], sum(X[i]*job_distributions_e[i,t] for i in 1:n) <= sum(X_e[:,t]))  # Ensure each position has at most one job assigned

    # Maximize size of independent set
    @objective(model, Max, sum(w[i] * X[i] for i in 1:n))  # Maximize total weight of jobs assigned to positions

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        solution = value.(X)
        return solution, objective_value(model), value.(X)
    else
        # Return nothing or handle non-optimal cases
        println("Optimization failed with status: ", termination_status(model))
        return nothing, nothing
    end
end

function solve_x_it_LP_2(n,m,p, w=ones(n), job_distributions = ones(n,k), job_distributions_s = ones(n,k), job_distributions_e = ones(n,k),pos_prob = ones(k))
    model = Model(COPT.ConeOptimizer)
    T = 1:min(n, m)
    mindim = min(n, m)
    @variable(model,
    x[s in 0:1, t in T, i in 1:n, k in 1:m] >= 0,
    base_name = "x")
    @variable(model, X[1:n,1:m]>=0)
    # s = 0, start; s = 1, end

    @constraint(model, [i = 1:n, k = 1:m], sum(x[0, t, i, k] for t in T) <= job_distributions_s[i,k])  # Each job is assigned to exactly one position
    @constraint(model, [i = 1:n, k = 1:m], sum(x[1, t, i, k] for t in T) <= job_distributions_e[i,k])  # Each job is assigned to exactly one position

    @constraint(model, [t in T, k = 1:m, j =(k+1):m,], sum(x[0, t, i, k] for i in 1:n) <= 1-sum(x[0, t, i, j] for i in 1:n))  # Each job is assigned to exactly one position
    @constraint(model, [t in T, k = 1:m, j =(k+1):m,], sum(x[1, t, i, k] for i in 1:n) <= 1-sum(x[1, t, i, j] for i in 1:n))  # Each job is assigned to exactly one position

    @constraint(model, [i = 1:n, t in T], X[i,t] == sum(x[0, t, i, k] for k in 1:m))  # Each job is assigned to exactly one position
    @constraint(model, [i = 1:n, t in T], X[i,t] == sum(x[1, t, i, k] for k in 1:m))  # Each job is assigned to exactly one position

    @constraint(model, [i = 1:n, t in T, k = 1:m], x[0,t,i,k]<= sum(x[1,t,i,j] for j in k:m))
    @constraint(model, [i = 1:n, t in T, k = 1:m], x[1,t,i,k]<= sum(x[0,t,i,j] for j in 1:k))

    @constraint(model, [i = 1:n, j = (i+1):n, t in T], X[i,t]<= 1- X[j,t])  # Each job is assigned to exactly one position
    @constraint(model, [t in T], sum(X[i,t+1] for i in 1:n)<= sum(X[i,t] for i in 1:n))  # Each job is assigned to exactly one position

    @constraint(model, [k = 1:m, t in 2:mindim], sum(x[1,t,i,k] for i in 1:n)<= 1-sum(x[1,t-1,i,k] for i in 1:n))
    @constraint(model, [k = 1:m, t in 2:mindim], sum(x[0,t,i,k] for i in 1:n)<= 1-sum(x[0,t-1,i,k] for i in 1:n))



    @constraint(model, [i = 1:n, j = (i+1):n], sum(X[i,t] for t in T)+sum(X[j,t] for t in T) <= 2-p[i,j])  # Each job is assigned to exactly one position
    @constraint(model, [t in mindim-1], sum(X[i,t] for i in 1:n) <=sum(X[i,t+1] for i in 1:n) )  # Each job is assigned to exactly one position

    @objective(model, Max, sum(w[i] * X[i,t] for i in 1:n for t in T))  # Maximize total weight of jobs assigned to positions
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        solution = value.(X)
        return solution, objective_value(model), value.(x)
    else
        # Return nothing or handle non-optimal cases
        println("Optimization failed with status: ", termination_status(model))
        return nothing, nothing
    end
end
function solve_x_it_LP_new(n,k,p, w=ones(n), job_distributions = ones(n,k), job_distributions_s = ones(n,k), job_distributions_e = ones(n,k),pos_prob = ones(k))
    # model = Model(Gurobi.Optimizer)
    model = Model(COPT.ConeOptimizer)
    # set_optimizer_attribute(model, "OutputFlag", 0)

    # @variable(model, X[1:n,1:k])
    @variable(model, X[1:n])  
    @constraint(model, X .>= 0)
    @constraint(model, [j=1:k], sum(X[i]*job_distributions[i,j] for i in 1:n) <= 1)  # Ensure X_s[i,j] <= X[i]
    # @constraint(model, [j=1:k], sum(X_s[:,j])+sum(X_e[:,j]) <= l[j])  # Ensure X_s[i,j] <= X[i]

    P = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if i == j
                P[i,j] = 1
            else
                P[i,j] = min(p[i,j],p[j,i])
            end
        end
    end
    @constraint(model, [i=1:(n-1), j = (i+1):n], X[i]+X[j] <= 2-P[i,j])  # Ensure M[(i-1)*k+j,(n*k+i)] is the sum of X[i,j] for all t
    # @constraint(model, [i=1:n,j = (i+1:n)], M[n*k+i,n*k+j]  <= 1-p[i,j])  # Ensure each job is assigned to exactly one position
    # @constraint(model, [i=1:n, t = 2:k, ti = 1:(t-1)], sum(M[(i-1)*k+tau,(i-1)*k+tau] for tau in t:k)  <= sum(M[(j-1)*k+ti,(j-1)*k+ti]*(1-p[i,j]) for j in 1:n if j != i) )

    # @constraint(model, [t = 1:k], sum(X[i]*job_distributions[i,t] for i in 1:n) <= sum(X_o[:,t]))  # Ensure each position has at most one job 
    # @constraint(model, [t = 1:k], sum(X[i]*job_distributions_s[i,t] for i in 1:n) <= sum(X_s[:,t]))  # Ensure each position has at most one job assigned
    # @constraint(model, [t = 1:k], sum(X[i]*job_distributions_e[i,t] for i in 1:n) <= sum(X_e[:,t]))  # Ensure each position has at most one job assigned

    # Maximize size of independent set
    @objective(model, Max, sum(w[i] * X[i] for i in 1:n))  # Maximize total weight of jobs assigned to positions

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        solution = value.(X)
        return solution, objective_value(model), value.(X)
    else
        # Return nothing or handle non-optimal cases
        println("Optimization failed with status: ", termination_status(model))
        return nothing, nothing
    end
end


# support-aware greedy: delete j if its support conflicts with scheduled [s_i:e_i]
function simulate_greedy_policy_support(
    jobs::Vector{JobDistribution},
    weights::AbstractVector{<:Real},
    P2::AbstractMatrix{<:Real},
    J::AbstractMatrix{<:Real},   # n×k occupancy probabilities (from estimate_position_probabilities)
    k::Int; num_sims::Int=1000)
    n = length(jobs)
    # Boolean support: job i can possibly occupy slot t iff J[i,t] > 0
    support_mask = J .> 0.0

    all_weights = Vector{Float64}(undef, num_sims)

    for s in 1:num_sims
        remaining = collect(1:n)
        total = 0.0

        while !isempty(remaining)
            # pick argmax score_i = w_i - sum_{j∈remaining\{i}} P2[i,j]*w_j
            best_i = remaining[1]; best_score = -Inf
            for i in remaining
                score = weights[i]
                @inbounds for j in remaining
                    if j != i
                        score -= P2[i,j] * weights[j]
                    end
                end
                if score > best_score
                    best_score = score; best_i = i
                end
            end

            # realize only the chosen job's interval
            s_i = clamp(round(Int, rand(jobs[best_i].start_dist)), 1, k)
            e_i = clamp(round(Int, rand(jobs[best_i].end_dist)),   1, k)
            while e_i < s_i
                e_i = clamp(round(Int, rand(jobs[best_i].end_dist)), 1, k)
            end

            total += weights[best_i]

            # delete j if its support intersects any occupied slot of i
            occ = s_i:e_i
            new_remaining = Int[]
            for j in remaining
                if j == best_i; continue; end
                # conflict if ∃t ∈ [s_i:e_i] with support_mask[j,t] = true
                conflicted = any(t -> (1 <= t <= k) && support_mask[j,t], occ)
                if !conflicted
                    push!(new_remaining, j)
                end
            end
            remaining = new_remaining
        end

        all_weights[s] = total
    end

    return mean(all_weights), all_weights
end

# ---------------------------
# Experiment scaffolding
# ---------------------------

"Generate start/end supports with controllable overlap and distribution family."
function generate_support_instance(n::Int,
                                   m::Int;
                                   overlap::Symbol = :moderate,
                                   weight_mode::Symbol = :uniform,
                                   dist_type::Symbol = :uniform,
                                   sigma::Real = 2.0,
                                   rng = Random.GLOBAL_RNG)
    jobs = JobDistribution[]
    weights = zeros(Float64, n)

    # crude knobs for overlap: how wide supports are and how concentrated starts are
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

        max_end_len = max(1, m - b + 1)
        end_len = rand(rng, start_len:max(start_len, max_end_len))
        c = clamp(b, 1, m)
        d = clamp(c + end_len - 1, c, m)

        start_dist = if dist_type == :uniform
            DiscreteUniform(a, b)
        elseif dist_type == :normal
            truncated(Normal((a + b) / 2, sigma), 1, m)
        elseif dist_type == :exp
            truncated(Exponential(1.0 / max(1, (a + b) / 2)), 1, m)
        elseif dist_type == :constant
            Dirac(a)
        else
            error("unknown dist_type: $dist_type")
        end

        end_dist = if dist_type == :uniform
            DiscreteUniform(c, d)
        elseif dist_type == :normal
            truncated(Normal((c + d) / 2, sigma), 1, m)
        elseif dist_type == :exp
            truncated(Exponential(1.0 / max(1, (c + d) / 2)), 1, m)
        elseif dist_type == :constant
            Dirac(d)
        else
            error("unknown dist_type: $dist_type")
        end

        push!(jobs, JobDistribution(start_dist, end_dist))

        weights[i] = weight_mode == :uniform ? 1.0 :
                     weight_mode == :uniform01 ? rand(rng) :
                     weight_mode == :exp1 ? rand(rng, Exponential(1.0)) :
                     rand(rng)  # default
    end
    meta = (overlap = overlap, weight_mode = weight_mode, dist_type = dist_type)
    return jobs, weights, meta
end

# Backward-compatible alias
generate_uniform_support_instance(args...; kwargs...) = generate_support_instance(args...; kwargs...)

"Compute pessimist bound and simple dual bounds for conservative model."
function conservative_bounds(jobs::Vector{JobDistribution},
                             weights::AbstractVector,
                             m::Int;
                             num_samples::Int = 5000)
    n = length(jobs)
    P_s, P_e, P_occ, L = estimate_position_probabilities(jobs, m; num_samples=num_samples)
    A_L, A_R = compute_A_intervals(jobs, m)

    α_pes = weighted_interval_scheduling(A_L, A_R, weights)

    p_star = minimum(filter(>(0.0), P_occ))
    upper_pstar = isfinite(p_star) && p_star > 0 ? α_pes / p_star : Inf

    # Dual 1: μ_r = max_i { w_i / Σ_k P^o_{ik} : P^o_{ir} > 0 }
    total_len = sum(P_occ, dims=2)
    dual1 = 0.0
    for r in 1:m
        vals = Float64[]
        for i in 1:n
            if P_occ[i, r] > 0 && total_len[i] > 0
                push!(vals, weights[i] / total_len[i])
            end
        end
        dual1 += isempty(vals) ? 0.0 : maximum(vals)
    end

    # Dual 2: μ_r = max_i { P^o_{ir} * w_i / Σ_k (P^o_{ik})^2 : P^o_{ir} > 0 }
    total_sq = sum(P_occ .^ 2, dims=2)
    dual2 = 0.0
    for r in 1:m
        vals = Float64[]
        for i in 1:n
            if P_occ[i, r] > 0 && total_sq[i] > 0
                push!(vals, P_occ[i, r] * weights[i] / total_sq[i])
            end
        end
        dual2 += isempty(vals) ? 0.0 : maximum(vals)
    end

    return (α_pes = α_pes,
            p_star = p_star,
            upper_pstar = upper_pstar,
            dual1 = dual1,
            dual2 = dual2,
            P_occ = P_occ,
            L = vec(L))
end

"Monte Carlo estimate of conservative OPT via per-sample MIP."
function estimate_conservative_opt(jobs::Vector{JobDistribution},
                                   weights::Vector{Float64},
                                   m::Int;
                                   num_samples::Int = 200)
    avg, allw = simulate_jobs_conserv(jobs, weights, m, num_samples)
    return avg, allw
end

"Simulate original dynamic process with different policies."
function simulate_policy_original(jobs::Vector{JobDistribution},
                                  weights::AbstractVector,
                                  k::Int;
                                  policy::Symbol = :ratio,
                                  num_samples::Int = 500,
                                  L::Union{Nothing,AbstractVector} = nothing,
                                  rng = Random.GLOBAL_RNG)
    n = length(jobs)
    L === nothing && (L = estimate_position_probabilities(jobs, k; num_samples=2000)[4])

    all_weights = zeros(Float64, num_samples)
    for sim in 1:num_samples
        remaining = collect(1:n)
        total = 0.0
        while !isempty(remaining)
            scores = zeros(Float64, length(remaining))
            for (idx, i) in pairs(remaining)
                if policy == :ratio
                    scores[idx] = weights[i] / max(1e-9, L[i])
                elseif policy == :weight
                    scores[idx] = weights[i]
                elseif policy == :shortest
                    scores[idx] = -L[i]
                elseif policy == :random
                    scores[idx] = rand(rng)
                else
                    scores[idx] = weights[i]
                end
            end
            pick_idx = argmax(scores)
            i_sel = remaining[pick_idx]

            # realize interval
            s_i, e_i = sample_job_intervals([jobs[i_sel]], k)
            s_i = s_i[1]; e_i = e_i[1]
            total += weights[i_sel]

            # delete conflicting remaining jobs using realized interval
            new_remaining = Int[]
            for j in remaining
                if j == i_sel; continue; end
                s_j, e_j = sample_job_intervals([jobs[j]], k)
                s_j = s_j[1]; e_j = e_j[1]
                if max(s_i, s_j) > min(e_i, e_j)
                    push!(new_remaining, j)
                end
            end
            remaining = new_remaining
        end
        all_weights[sim] = total
    end
    return mean(all_weights), all_weights
end

"Simulate conservative greedy policies: delete any job whose support intersects a realized interval."
function simulate_policy_conservative(jobs::Vector{JobDistribution},
                                      weights::AbstractVector,
                                      k::Int;
                                      policy::Symbol = :weight,
                                      num_samples::Int = 500,
                                      L::Union{Nothing,AbstractVector} = nothing,
                                      rng = Random.GLOBAL_RNG)
    n = length(jobs)
    A_L, A_R = compute_A_intervals(jobs, k)
    L === nothing && (L = estimate_position_probabilities(jobs, k; num_samples=2000)[4])
    all_weights = zeros(Float64, num_samples)

    for sim in 1:num_samples
        remaining = collect(1:n)
        total = 0.0
        while !isempty(remaining)
            scores = zeros(Float64, length(remaining))
            for (idx, i) in pairs(remaining)
                if policy == :ratio
                    scores[idx] = weights[i] / max(1e-9, L[i])
                elseif policy == :weight
                    scores[idx] = weights[i]
                else
                    scores[idx] = weights[i]
                end
            end
            pick_idx = argmax(scores)
            i_sel = remaining[pick_idx]

            # realize interval
            s_i, e_i = sample_job_intervals([jobs[i_sel]], k)
            s_i = s_i[1]; e_i = e_i[1]
            total += weights[i_sel]

            new_remaining = Int[]
            for j in remaining
                if j == i_sel; continue; end
                # conservative delete if support intersects realized [s_i,e_i]
                if A_R[j] < s_i || A_L[j] > e_i
                    push!(new_remaining, j)
                end
            end
            remaining = new_remaining
        end
        all_weights[sim] = total
    end
    return mean(all_weights), all_weights
end

"Run a small suite of conservative-bound metrics on one instance."
function run_conservative_instance(n::Int, m::Int;
                                   overlap::Symbol = :moderate,
                                   weight_mode::Symbol = :uniform,
                                   dist_type::Symbol = :uniform,
                                   sigma::Real = 2.0,
                                   num_samples_bounds::Int = 2000,
                                   num_samples_opt::Int = 200)
    jobs, weights, meta = generate_support_instance(n, m;
        overlap=overlap, weight_mode=weight_mode, dist_type=dist_type, sigma=sigma)
    bounds = conservative_bounds(jobs, weights, m; num_samples=num_samples_bounds)
    est_opt, _ = estimate_conservative_opt(jobs, weights, m; num_samples=num_samples_opt)

    return Dict(
        "n" => n,
        "m" => m,
        "overlap" => overlap,
        "weight_mode" => weight_mode,
        "dist_type" => dist_type,
        "alpha_pes" => bounds.α_pes,
        "upper_pstar" => bounds.upper_pstar,
        "dual1" => bounds.dual1,
        "dual2" => bounds.dual2,
        "p_star" => bounds.p_star,
        "est_opt" => est_opt,
        "alpha_pes_over_opt" => bounds.α_pes / max(1e-9, est_opt),
        "upper_over_opt" => bounds.upper_pstar / max(1e-9, est_opt),
        "dual1_over_opt" => bounds.dual1 / max(1e-9, est_opt),
        "dual2_over_opt" => bounds.dual2 / max(1e-9, est_opt)
    )
end

# Batch driver: generate many instances, compute bounds, return DataFrame and optional CSV.
function run_conservative_batch(num_instances::Int;
                                n::Int = 10,
                                m::Int = 15,
                                overlaps = (:sparse, :moderate, :heavy),
                                weight_modes = (:uniform, :uniform01, :exp1),
                                dist_types = (:uniform, :normal, :constant),
                                num_samples_bounds::Int = 2000,
                                num_samples_opt::Int = 200,
                                rng = Random.GLOBAL_RNG,
                                save_csv::Union{Nothing,String} = nothing)
    rows = Dict[]
    idx = 1
    for ov in overlaps, wm in weight_modes, dt in dist_types, _ in 1:num_instances
        push!(rows, run_conservative_instance(n, m;
                                              overlap=ov,
                                              weight_mode=wm,
                                              dist_type=dt,
                                              num_samples_bounds=num_samples_bounds,
                                              num_samples_opt=num_samples_opt))
    end
    df = DataFrame(rows)
    if save_csv !== nothing
        CSV.write(save_csv, df)
    end
    return df
end

"Summaries of UB/OPT ratios over a DataFrame produced by run_conservative_batch."
function summarize_bound_ratios(df::DataFrame)
    cols = [:alpha_pes_over_opt, :upper_over_opt, :dual1_over_opt, :dual2_over_opt]
    stats = Dict{Symbol,Dict{String,Float64}}()
    for c in cols
        vals = skipmissing(df[!, c])
        stats[c] = Dict("mean" => mean(vals), "max" => maximum(vals), "min" => minimum(vals))
    end
    return stats
end

"Quick bar plot of mean UB/OPT ratios."
function plot_bound_means(df::DataFrame)
    cols = [:alpha_pes_over_opt, :upper_over_opt, :dual1_over_opt, :dual2_over_opt]
    means = [mean(skipmissing(df[!, c])) for c in cols]
    labels = ["α_pes/opt", "p*-ub/opt", "dual1/opt", "dual2/opt"]
    bar(labels, means; legend=false, ylabel="mean(UB / est_opt)", xticks=(1:length(labels), labels))
end

# ---------------------------
# Full experiment runner (conservative vs original policies)
# ---------------------------
"Run experiments over grids of (n,m, overlap, weight_mode, dist_type); return DataFrame and optional CSV."
function run_full_batch(num_instances::Int;
                        n_grid = (10, 15),
                        m_grid = (10, 20),
                        overlaps = (:sparse, :moderate, :heavy),
                        weight_modes = (:uniform, :uniform01, :exp1),
                        dist_types = (:uniform, :normal, :constant),
                        num_samples_bounds::Int = 2000,
                        num_samples_opt::Int = 200,
                        num_samples_policy::Int = 500,
                        rng = Random.GLOBAL_RNG,
                        save_csv::Union{Nothing,String} = nothing)

    rows = Dict[]
    for n in n_grid, m in m_grid, ov in overlaps, wm in weight_modes, dt in dist_types, _ in 1:num_instances
        jobs, weights, meta = generate_support_instance(n, m; overlap=ov, weight_mode=wm, dist_type=dt, rng=rng)
        bounds = conservative_bounds(jobs, weights, m; num_samples=num_samples_bounds)
        optc, _ = estimate_conservative_opt(jobs, weights, m; num_samples=num_samples_opt)

        L = estimate_position_probabilities(jobs, m; num_samples=2000)[4]
        pol_ratio, _ = simulate_policy_original(jobs, weights, m; policy=:ratio, num_samples=num_samples_policy, L=L)
        pol_weight, _ = simulate_policy_original(jobs, weights, m; policy=:weight, num_samples=num_samples_policy, L=L)
        pol_short, _ = simulate_policy_original(jobs, weights, m; policy=:shortest, num_samples=num_samples_policy, L=L)

        push!(rows, Dict(
            :n => n, :m => m, :overlap => ov, :weight_mode => wm, :dist_type => dt,
            :alpha_pes => bounds.α_pes, :upper_pstar => bounds.upper_pstar,
            :dual1 => bounds.dual1, :dual2 => bounds.dual2, :p_star => bounds.p_star,
            :opt_conserv => optc,
            :alpha_pes_over_opt => bounds.α_pes / max(1e-9, optc),
            :upper_over_opt => bounds.upper_pstar / max(1e-9, optc),
            :dual1_over_opt => bounds.dual1 / max(1e-9, optc),
            :dual2_over_opt => bounds.dual2 / max(1e-9, optc),
            :pol_ratio => pol_ratio,
            :pol_weight => pol_weight,
            :pol_short => pol_short,
            :opt_over_ratio => optc / max(1e-9, pol_ratio),
            :opt_over_weight => optc / max(1e-9, pol_weight),
            :opt_over_short => optc / max(1e-9, pol_short)
        ))
    end
    df = DataFrame(rows)
    if save_csv !== nothing
        CSV.write(save_csv, df)
    end
    return df
end

"""
solve_ximt_LP builds and solves the LP with x_{im}^{t,o} variables.

Arguments
---------
n::Int            # number of jobs (i=1..n)
m::Int            # number of positions (m=1..m)
T::Int            # number of stages/times (t=1..T)
p::AbstractMatrix # n×n pairwise probabilities used for coin-deletion constraint (optional; can pass ones(n,n))

Keyword args
------------
w::AbstractVector                  = ones(n)         # job weights for objective sum_i w[i]*x_i
job_distributions_o::AbstractMatrix = ones(n, m)     # P^o_{im}: prob job i occupies position m at time 1
job_distributions_s::AbstractMatrix = ones(n, m)     # P^s_{im}: prob job i starts at position m at time 1
exp_length_selected1::AbstractVector = ones(n)       # E[length of i | selected at time 1]
ending_independent::Bool = false    # if true and consecutive support, enforce x_{i,g_i}^{t,o} == x_i^t
use_general_lower_bounds::Bool = false  # if true, add x_{im}^{t,o} >= P^s_{im} * x_i^t for all m in support
proportionality_mode::Symbol = :inequality
#  :inequality => enforce P^s_{im} * x_{iℓ}^{t,s} ≥ x_{im}^{t,s} * P^s_{iℓ} only for m<ℓ (avoids over-tightening)
#  :equality    => enforce both directions for all (m,ℓ), which forces exact proportionality over available starts
quiet::Bool = true                  # silence solver logs

Returns
-------
NamedTuple with solution arrays (X, X_t, Xs, Xo), objective value, and termination status.
"""
function solve_ximt_LP(n,
                       m,
                       T,
                       p;
                       w = ones(n),
                       job_distributions_o = ones(n, m),
                       job_distributions_s = ones(n, m),
                       job_distributions_e = ones(n, m),
                       exp_length_selected1 = ones(n),
                       ending_independent::Bool = true,
                       use_general_lower_bounds::Bool = true,
                       proportionality_mode::Symbol = :inequality,
                       quiet::Bool = false)

    # @assert size(job_distributions_o) == (n, m)
    # @assert size(job_distributions_s) == (n, m)
    # @assert length(w) == n
    # # @assert length(exp_length_selected1) == n
    # @assert size(p) == (n, n)

    # Infer each job's start-support "g_i" (largest m with positive start-prob).
    # We also check if the support is consecutive from 1..g_i.
    tol = 1e-12
    a = fill(0, n)                 # first start index in support (0 if empty)
    b = fill(0, n)                 # last  start index in support (0 if empty)

    c = fill(0, n)                 # first start index in support (0 if empty)
    d = fill(0, n)           
        # last  start index in support (0 if empty)
    start_support_is_consecutive = falses(n)
    in_start_support = falses(n, m)      # boolean mask for fast membership checks

    end_support_is_consecutive = falses(n)
    in_end_support = falses(n, m)      # boolean mask for fast membership checks

    for i in 1:n
        Si = Int[]
        for mm in 1:m
            if job_distributions_s[i, mm] > tol
                push!(Si, mm)
                in_start_support[i, mm] = true
            end
        end
        if !isempty(Si)
            a[i] = minimum(Si)
            b[i] = maximum(Si)
            start_support_is_consecutive[i] = (b[i] - a[i] + 1 == length(Si))
        else
            a[i] = 0
            b[i] = 0
            start_support_is_consecutive[i] = false
        end
    end
    for i in 1:n
        Si = Int[]
        for mm in 1:m
            if job_distributions_e[i, mm] > tol
                push!(Si, mm)
                in_end_support[i, mm] = true
            end
        end
        if !isempty(Si)
            c[i] = minimum(Si)
            d[i] = maximum(Si)
            end_support_is_consecutive[i] = (d[i] - c[i] + 1 == length(Si))
        else
            c[i] = 0
            d[i] = 0
            end_support_is_consecutive[i] = false
        end
    end
    println("\n")
    println(a)
    println("\n")
    println(b)
    model = Model(COPT.ConeOptimizer)
    # if quiet
    #     set_optimizer_attribute(model, "Logging", 0)  # COPT quiet flag
    # end

    # Variables
    @variable(model, 0 ≤ X[1:n] ≤ 1)                    # x_i
    @variable(model, 0 ≤ X_t[1:n, 1:T]<=1)                 # x_i^t
    @variable(model, 0 ≤ Xs[1:n, 1:m, 1:T]<=1)             # x_{im}^{t,s}
    @variable(model, 0 ≤ Xe[1:n, 1:m, 1:T]<=1)  
    @variable(model, 0 ≤ Xo[1:n, 1:m, 1:T]<=1)             # x_{im}^{t,o}

    # Linking: sum_t x_i^t = x_i
    @constraint(model, [i=1:n], sum(X_t[i, t] for t in 1:T) == X[i])

    # Linking: sum_m x_{im}^{t,s} = x_i^t
    @constraint(model, [i=1:n, t=1:T], sum(Xs[i, mm, t] for mm in 1:m) == X_t[i, t])
    @constraint(model, [i=1:n, t=1:T], sum(Xe[i, mm, t] for mm in 1:m) == X_t[i, t])

    # println(g)
    # println(support_is_consecutive)
    # Zero-start outside support (keeps things tidy)
    # for i in 1:n, t in 1:T, mm in g[i]+1:m
    #     @constraint(model, Xs[i, mm, t] == 0.0)
    # end
    for i in 1:n, t in 1:T, mm in 1:m
        if !in_start_support[i, mm]
            @constraint(model, Xs[i, mm, t] == 0.0)
            # @constraint(model, Xo[i, mm, t] == 0.0)
        end
        if !in_end_support[i, mm]
            @constraint(model, Xe[i, mm, t] == 0.0)
            # @constraint(model, Xo[i, mm, t] == 0.0)
        end
    end
    for i in 1:n, t in 1:T, mm in 1:m
        @constraint(model, Xo[i, mm, t] <= job_distributions_o[i,mm])
    end 

    for i in 1:n, t in 1:T
        if a[i] > 0  # has nonempty support
            # x_{i,m}^{t,o} - x_{i,m+1}^{t,o} = x_{i,m+1}^{t,s}  for m = a_i..b_i-1
            for mm in a[i]:(b[i]-1)
                @constraint(model, Xo[i, mm+1, t] - Xo[i, mm, t] == Xs[i, mm+1, t])
                # @constraint(model, Xo[i, mm+1, t] - Xo[i, mm, t] >= 0)
            end
        end
        if c[i] > 0  # has nonempty support
            # x_{i,m}^{t,o} - x_{i,m+1}^{t,o} = x_{i,m+1}^{t,s}  for m = a_i..b_i-1
            for mm in c[i]:(d[i]-1)
                @constraint(model, Xo[i, mm, t] - Xo[i, mm+1, t] == Xe[i, mm, t])
                # @constraint(model, Xo[i, mm+1, t] - Xo[i, mm, t] >= 0)
            end
        end
    end
    for i in 1:n, t in 1:T
        if b[i] > 0
            if ending_independent && start_support_is_consecutive[i]
                # when ending time independent and start support is consecutive:
                @constraint(model, Xo[i, b[i], t] == X_t[i, t])
            else
                # general lower bound
                @constraint(model, Xo[i, b[i], t] >= job_distributions_s[i, b[i]] * X_t[i, t])
                if use_general_lower_bounds
                    # optionally add for all supported m
                    for mm in a[i]:b[i]
                        if in_start_support[i, mm]
                            @constraint(model, Xo[i, mm, t] >= job_distributions_s[i, mm] * X_t[i, t])
                        end
                    end
                end
            end
        end
        if c[i] > 0 
            if ending_independent && end_support_is_consecutive[i]
                # when ending time independent and start support is consecutive:
                @constraint(model, Xo[i, c[i], t] == X_t[i, t])
                @constraint(model,[k in b[i]:c[i]], Xo[i, k, t] == X_t[i, t])

            else
                # general lower bound
                @constraint(model, Xo[i, b[i], t] >= job_distributions_s[i, b[i]] * X_t[i, t])
                if use_general_lower_bounds
                    # optionally add for all supported m
                    for mm in a[i]:b[i]
                        if in_start_support[i, mm]
                            @constraint(model, Xo[i, mm, t] >= job_distributions_s[i, mm] * X_t[i, t])
                        end
                    end
                end
            end
        end
    end

    # Proportionality across available starts:
    # Default (inequality): only one direction and only for m<ℓ to avoid forcing equality.
    # If proportionality_mode == :equality, add both directions for all (m,ℓ), which forces exact ratios.
    if proportionality_mode == :inequality
        for i in 1:n, t in 1:T
            for ll in 1:b[i], mm in 1:(ll-1)
                @constraint(model, job_distributions_s[i, mm] * Xs[i, ll, t] >= Xs[i, mm, t] * job_distributions_s[i, ll])
            end
        end
        for i in 1:n, t in 1:T
            for ll in c[i]:(m-1), mm in (ll+1):m
                @constraint(model, job_distributions_e[i, mm] * Xe[i, ll, t] >= Xe[i, mm, t] * job_distributions_e[i, ll])
            end
        end
    elseif proportionality_mode == :equality
        for i in 1:n, t in 1:T
            for mm in 1:m, ll in 1:m
                if mm != ll
                    @constraint(model, job_distributions_s[i, mm] * Xs[i, ll, t] ≥ Xs[i, mm, t] * job_distributions_s[i, ll])
                end
            end
        end
    else
        error("proportionality_mode must be :inequality or :equality")
    end

    # Capacity: at most one job occupies position m at time t
    v = maximum(job_distributions_o, dims = 1)
    @constraint(model, [mm=1:m, t=1:T], sum(Xo[i, mm, t] for i in 1:n) ≤ v[mm])

    # Position m can be occupied at most once in the whole horizon (sum over t)
    @constraint(model, [mm=1:m], sum(sum(Xo[i, mm, t] for i in 1:n) for t in 1:T) ≤ 1)

    # For each job i, a given position m is occupied at most once across time
    @constraint(model, [i=1:n, mm=1:m], sum(Xo[i, mm, t] for t in 1:T) ≤ X[i])

    # Expected length upper bound per job
    # @constraint(model, [i=1:n], sum(sum(Xo[i, mm, t] for mm in 1:m) for t in 1:T) ≤ exp_length_selected1[i] * X[i])

    # t=1 occupancy identity: sum_i x_{im}^{1,o} = sum_i P^o_{im} * x_i^1
    @constraint(model, [mm=1:m], sum(Xo[i, mm, 1] for i in 1:n) ==
                                 sum(job_distributions_o[i, mm] * X_t[i, 1] for i in 1:n))

    @constraint(model, [mm=1:m, i =1:n], Xs[i, mm, 1] ==
                                 job_distributions_s[i, mm] * X_t[i, 1])
    @constraint(model, [mm=1:m, i =1:n], Xe[i, mm, 1] ==
                                 job_distributions_e[i, mm] * X_t[i, 1])
    @constraint(model, [mm=1:m, i =1:n], Xo[i, mm, 1] ==
                                 job_distributions_o[i, mm] * X_t[i, 1])

    @constraint(model, sum(X_t[i,1] for i in 1:n) == 1)

    @constraint(model, [mm=1:m], sum(job_distributions_o[i,mm]*X_t[i,1] for i in 1:n) <= v[mm])
    @constraint(model, [i=1:n-1, j=i+1:n], X[i] + X[j] ≤ 2 - p[i, j])
    
    @constraint(model, [t=2:T], sum(X_t[i,t] for i in 1:n)<=sum(X_t[i,t-1] for i in 1:n))

    # Objective: maximize sum_i w_i x_i
    @objective(model, Max, sum(w[i] * X[i] for i in 1:n))

    optimize!(model)

    status = termination_status(model)
    obj = objective_value(model)

    if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.ALMOST_OPTIMAL
        X_val  = value.(X)
        X_tval = value.(X_t)
        Xs_val = value.(Xs)
        Xe_val = value.(Xe)
        Xo_val = value.(Xo)
        return (X = X_val,
                X_t = X_tval,
                Xs = Xs_val,
                Xe = Xe_val,
                Xo = Xo_val,
                objective = obj,
                status = status)
    else
        @warn "Optimization did not reach OPTIMAL: $status"
        return (X = nothing, X_t = nothing, Xs = nothing, Xo = nothing,
                objective = nothing, status = status)
    end
end

using JuMP
# Pick one solver that supports SDP (install first):
# using MosekTools
# using COSMO
# using SCS
# using Hypatia

function solve_ximt_SDP(n,
                        m,
                        T,
                        p;
                        w = ones(n),
                        job_distributions_o = ones(n, m),  # P^o_{im}
                        job_distributions_s = ones(n, m),  # P^s_{im}
                        job_distributions_e = ones(n, m),  # P^e_{im}
                        exp_length_selected1 = ones(n),
                        ending_independent::Bool = true,
                        use_general_lower_bounds::Bool = true,
                        proportionality_mode::Symbol = :inequality,
                        quiet::Bool = false)

    # @assert optimizer !== nothing "Pass a PSD-capable optimizer, e.g. optimizer=MosekTools.Optimizer (or COSMO, SCS, Hypatia)."
    @assert size(job_distributions_o) == (n, m)
    @assert size(job_distributions_s) == (n, m)
    @assert size(job_distributions_e) == (n, m)
    @assert length(w) == n
    @assert length(exp_length_selected1) == n
    @assert size(p) == (n, n)

    # --- Detect supports (any contiguous block [a..b] allowed) ---
    tol = 1e-12
    a = fill(0, n); b = fill(0, n)               # start support [a[i]..b[i]]
    c = fill(0, n); d = fill(0, n)               # end support [c[i]..d[i]]
    start_support_is_consecutive = falses(n)
    end_support_is_consecutive   = falses(n)
    in_start_support = falses(n, m)
    in_end_support   = falses(n, m)

    for i in 1:n
        Si = Int[]
        for mm in 1:m
            if job_distributions_s[i, mm] > tol
                push!(Si, mm)
                in_start_support[i, mm] = true
            end
        end
        if !isempty(Si)
            a[i] = minimum(Si); b[i] = maximum(Si)
            start_support_is_consecutive[i] = (b[i] - a[i] + 1 == length(Si))
        end
    end

    for i in 1:n
        Ei = Int[]
        for mm in 1:m
            if job_distributions_e[i, mm] > tol
                push!(Ei, mm)
                in_end_support[i, mm] = true
            end
        end
        if !isempty(Ei)
            c[i] = minimum(Ei); d[i] = maximum(Ei)
            end_support_is_consecutive[i] = (d[i] - c[i] + 1 == length(Ei))
        end
    end

    # --- Model ---
    model = Model(COPT.ConeOptimizer)

    # Variables (same as your LP)
    @variable(model, 0 ≤ X[1:n] ≤ 1)                    # x_i
    @variable(model, 0 ≤ X_t[1:n, 1:T] ≤ 1)             # x_i^t
    @variable(model, 0 ≤ Xs[1:n, 1:m, 1:T] ≤ 1)         # x_{im}^{t,s}
    @variable(model, 0 ≤ Xe[1:n, 1:m, 1:T] ≤ 1)         # x_{im}^{t,e}
    @variable(model, 0 ≤ Xo[1:n, 1:m, 1:T] ≤ 1)         # x_{im}^{t,o}

    # Linking: sum_t x_i^t = x_i
    @constraint(model, [i=1:n], sum(X_t[i, t] for t in 1:T) == X[i])

    # Linking: sum_m starts/ends = x_i^t
    @constraint(model, [i=1:n, t=1:T], sum(Xs[i, mm, t] for mm in 1:m) == X_t[i, t])
    @constraint(model, [i=1:n, t=1:T], sum(Xe[i, mm, t] for mm in 1:m) == X_t[i, t])

    # Zero outside supports
    for i in 1:n, t in 1:T, mm in 1:m
        if !in_start_support[i, mm]; @constraint(model, Xs[i, mm, t] == 0.0); end
        if !in_end_support[i, mm];   @constraint(model, Xe[i, mm, t] == 0.0); end
    end

    # Optional per-(i,m,t) upper bound from P^o (you had this)
    for i in 1:n, t in 1:T, mm in 1:m
        @constraint(model, Xo[i, mm, t] ≤ job_distributions_o[i, mm])
    end

    # Start-chain and end-chain
    for i in 1:n, t in 1:T
        if a[i] > 0
            for mm in a[i]:(b[i]-1)
                @constraint(model, Xo[i, mm+1, t] - Xo[i, mm, t] == Xs[i, mm+1, t])
            end
        end
        if c[i] > 0
            for mm in c[i]:(d[i]-1)
                @constraint(model, Xo[i, mm, t] - Xo[i, mm+1, t] == Xe[i, mm, t])
            end
        end
    end

    # Tail constraints (fixed a small bug to use 'e' on the end side)
    for i in 1:n, t in 1:T
        if b[i] > 0
            if ending_independent && start_support_is_consecutive[i]
                @constraint(model, Xo[i, b[i], t] == X_t[i, t])
            else
                @constraint(model, Xo[i, b[i], t] ≥ job_distributions_s[i, b[i]] * X_t[i, t])
                if use_general_lower_bounds
                    for mm in a[i]:b[i]
                        if in_start_support[i, mm]
                            @constraint(model, Xo[i, mm, t] ≥ job_distributions_s[i, mm] * X_t[i, t])
                        end
                    end
                end
            end
        end
        if d[i] > 0
            if ending_independent && end_support_is_consecutive[i]
                @constraint(model, Xo[i, c[i], t] == X_t[i, t])
            else
                @constraint(model, Xo[i, c[i], t] ≥ job_distributions_e[i, c[i]] * X_t[i, t])
                if use_general_lower_bounds
                    for mm in c[i]:d[i]
                        if in_end_support[i, mm]
                            @constraint(model, Xo[i, mm, t] ≥ job_distributions_e[i, mm] * X_t[i, t])
                        end
                    end
                end
            end
        end
        if c[i] > 0 && b[i] > 0
            if ending_independent && start_support_is_consecutive[i] && end_support_is_consecutive[i]
                for j in b[i]:c[i]
                    @constraint(model, Xo[i, j, t] == X_t[i, t])
                end
            end
        end 
    end

    # Proportionality (restrict to supported indices)
    if proportionality_mode == :inequality
        for i in 1:n, t in 1:T
            if a[i] > 0
                for ll in a[i]:b[i], mm in a[i]:(ll-1)
                    if in_start_support[i,mm] && in_start_support[i,ll]
                        @constraint(model, job_distributions_s[i, mm] * Xs[i, ll, t] >= Xs[i, mm, t] * job_distributions_s[i, ll])
                        # @constraint(model, Xs[i, ll, t] >= X_t[i,t] * job_distributions_s[i, ll])
                    end
                end
            end
            if c[i] > 0
                for ll in c[i]:(d[i]-1), mm in (ll+1):d[i]
                    if in_end_support[i,mm] && in_end_support[i,ll]
                        @constraint(model, job_distributions_e[i, mm] * Xe[i, ll, t] >= Xe[i, mm, t] * job_distributions_e[i, ll])
                        # @constraint(model, Xe[i, ll, t] >= X_t[i,t] * job_distributions_e[i, ll])
                    end
                end
                # @constraint(model, Xe[i, d[i], t] >= X_t[i,t] * job_distributions_e[i, d[i]])
            end
        end
    elseif proportionality_mode == :equality
        for i in 1:n, t in 1:T
            if a[i] > 0
                for mm in a[i]:b[i], ll in a[i]:b[i]
                    if mm != ll && in_start_support[i,mm] && in_start_support[i,ll]
                        @constraint(model, job_distributions_s[i, mm] * Xs[i, ll, t] >= Xs[i, mm, t] * job_distributions_s[i, ll])
                    end
                end
            end
            if c[i] > 0
                for mm in c[i]:d[i], ll in c[i]:d[i]
                    if mm != ll && in_end_support[i,mm] && in_end_support[i,ll]
                        @constraint(model, job_distributions_e[i, mm] * Xe[i, ll, t] ≥ Xe[i, mm, t] * job_distributions_e[i, ll])
                    end
                end
            end
        end
    else
        error("proportionality_mode must be :inequality or :equality")
    end



    # Expected length upper bound per job


    # t=1 identities
    
    @constraint(model, [mm=1:m, i=1:n], Xs[i, mm, 1] == job_distributions_s[i, mm] * X_t[i, 1])
    @constraint(model, [mm=1:m, i=1:n], Xe[i, mm, 1] == job_distributions_e[i, mm] * X_t[i, 1])
    
    
    @constraint(model, [i=1:n, mm = 1:m], sum(Xs[i,mm,t] for t in 1:T)<=job_distributions_s[i,mm])
    @constraint(model, [i=1:n, mm = 1:m], sum(Xe[i,mm,t] for t in 1:T)<=job_distributions_e[i,mm])
    


        # Capacity / occupancy constraints
    @constraint(model, [i=1:n],
        sum(sum(Xo[i, mm, t] for mm in 1:m) for t in 1:T) ≤ exp_length_selected1[i] * X[i])
    v = vec(maximum(job_distributions_o, dims = 1))  # 1×m -> Vector{Float64}(m)
    @constraint(model, [mm=1:m, t=1:T], sum(Xo[i, mm, t] for i in 1:n) ≤ v[mm])
    @constraint(model, [mm=1:m], sum(job_distributions_o[i, mm] * X_t[i, 1] for i in 1:n) ≤ v[mm])
    @constraint(model, [mm=1:m], sum(sum(Xo[i, mm, t] for i in 1:n) for t in 1:T) ≤ 1)
    @constraint(model, [i=1:n, mm=1:m], sum(Xo[i, mm, t] for t in 1:T) ≤ X[i])
    @constraint(model, [mm=1:m], sum(Xo[i, mm, 1] for i in 1:n) ==
                                     sum(job_distributions_o[i, mm] * X_t[i, 1] for i in 1:n))
    @constraint(model, [mm=1:m, i=1:n], Xo[i, mm, 1] == job_distributions_o[i, mm] * X_t[i, 1])
    @constraint(model, [i=1:n, mm = 1:m], sum(Xo[i,mm,t] for t in 1:T)<=job_distributions_o[i,mm])
    
    @constraint(model, [t = 2:T, mm = 1:m], sum(sum(Xo[i,mm, tau] for i in 1:n) for tau in t:T) <= 1- sum(sum(Xo[i,mm,tau] for i in 1:n) for tau in 1:(t-1)))
    @constraint(model, [i=1:n-1, j=i+1:n], X[i] + X[j] ≤ 2 - p[i, j])


    

    @constraint(model, sum(X_t[i, 1] for i in 1:n) == 1)
    @constraint(model, [t=2:T], sum(X_t[i, t] for i in 1:n) ≤ sum(X_t[i, t-1] for i in 1:n))

    # At most one job selected per time
    @constraint(model, [t=1:T], sum(X_t[i,t] for i in 1:n) <= 1)



    @constraint(model, [t = 2:T], sum(X_t[i,t] for i in 1:n) <= sum(X_t[i,t-1] for i in 1:n))


    
    # @constraint(model,[t = 2:T,i=1:n, tau = 1:(t-1)], X_t[i,t] <= sum(X_t[j,tau]*(1-p[i,j]) for j in 1:n))
    @constraint(model, [t = 2:T, i = 1:n], sum(X_t[i,tau] for tau in t:T)<= 1- sum(X_t[i,tau] for tau in 1:t-1))


    #     for k in 1:m
    #     expr = @expression(model,
    #         sum(
    #             # + sum_{i: a_i ≤ k ≤ d_i} x_i^t
    #             sum((a[i] > 0 && c[i] > 0 && a[i] <= k <= d[i]) ? X_t[i, t] : 0.0 for i in 1:n)
    #             # - sum_{i: a_i ≤ k ≤ b_i} sum_{ℓ=k+1}^{b_i} x^{ts}_{iℓt}
    #           - sum((a[i] > 0 && a[i] <= k <= b[i]) ?
    #                    sum(Xs[i, ℓ, t] for ℓ in max(k+1, a[i]):b[i]) : 0.0 for i in 1:n)
    #             # - sum_{i: c_i ≤ k ≤ d_i} sum_{ℓ=c_i}^{k-1} x^{te}_{iℓt}
    #           - sum((c[i] > 0 && c[i] <= k <= d[i]) ?
    #                    sum(Xe[i, ℓ, t] for ℓ in c[i]:min(k-1, d[i])) : 0.0 for i in 1:n)
    #         for t in 1:T)
    #     )
    #     @constraint(model, expr <= 1.0)
    # end

    # @constraint(model, [mm = 1:m], sum(job_distributions_o[i,mm]*X[i] for i in 1:n)<=2)
    # ---------------------------
    # SDP lifting over Xo: Y PSD
    # ---------------------------
    # Q = n * m * T
    # @variable(model, Y[1:Q+1, 1:Q+1], PSD)
    # @constraint(model, Y[Q+1, Q+1] == 1.0)

    # # flat index: q = ((t-1)*m + (mm-1)) * n + i
    # idx(i, mm, t) = (t-1)*m*n + (mm-1)*n + i

    # # Link diagonal and last column to Xo
    # for i in 1:n, mm in 1:m, t in 1:T
    #     q = idx(i, mm, t)
    #     @constraint(model, Y[q, q]     == Xo[i, mm, t])
    #     @constraint(model, Y[q, Q+1]   == Xo[i, mm, t])   # last column equals x^o
    #     # (Y is symmetric PSD, so Y[Q+1, q] equals Y[q, Q+1] implicitly)
    # end


    # # 1) Cross-time orthogonality: zero all entries with different times
    # for t in 1:T, τ in (t+1):T
    #         for i in 1:n, j in 1:n, k in 1:m, ℓ in 1:m
    #             @constraint(model, Y[idx(i,k,t), idx(j,k,τ)] == 0.0)
    #             @constraint(model, Y[idx(i,k,t), idx(i,ℓ,τ)] == 0.0)
    #         end
    # end

    # # 2) Within a time block, zero cross-job pairs (any positions k, ℓ)
    # for t in 1:T
    #     for i in 1:n, j in (i+1):n
    #             for k in 1:m, ℓ in 1:m
    #                 @constraint(model, Y[idx(i,k,t), idx(j,ℓ,t)] == 0.0)
    #             end
    #     end
    # end

    S = n * T
    @variable(model, Z[1:S+1, 1:S+1], PSD)
    @constraint(model, Z[S+1, S+1] == 1.0)
    
    # flat index for selection variables
    idx_sel(i, t) = (t-1)*n + i
    
    # Link diagonal and last column of Z to X_t (s_{it})
    for i in 1:n, t in 1:T
        q = idx_sel(i, t)
        @constraint(model, Z[q, q]     == X_t[i, t])
        @constraint(model, Z[q, S+1]   == X_t[i, t])
    end
    
    # Zero pattern matching your rules:
    # (A) same job across different times -> zero
    for i in 1:n, t in 1:T, τ in 1:T
        if τ != t
            @constraint(model, Z[idx_sel(i,t), idx_sel(i,τ)] == 0.0)
        end
    end
    
    # (B) different jobs at the same time -> zero
    for t in 1:T, i in 1:n, j in 1:n
        if i != j
            @constraint(model, Z[idx_sel(i,t), idx_sel(j,t)] == 0.0)
        end
    end
    
    # -----------------------------
    # Co-selection cap:  P(both i and j) ≤ 1 - p[i,j]
    # -----------------------------
    @constraint(model, [i=1:n-1, j=i+1:n],
        sum(Z[idx_sel(i,t), idx_sel(j,τ)] for t in 1:T, τ in 1:T) ≤ 1 - p[i,j])

    @constraint(model, [t=2:T, i=1:n,  τ in 1:(t-1)],
        X_t[i,t] == sum(Z[idx_sel(i,t), idx_sel(j,τ)] for j in 1:n))
    # Objective (same)
    @objective(model, Max, sum(w[i] * X[i] for i in 1:n))

    optimize!(model)

    status = termination_status(model)
    obj = objective_value(model)

    if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        return (
            X  = value.(X),
            X_t = value.(X_t),
            Xs = value.(Xs),
            Xe = value.(Xe),
            Xo = value.(Xo),
            # Y  = value.(Y),
            objective = obj,
            status = status,
        )
    else
        @warn "Optimization did not reach OPTIMAL: $status"
        return (X=nothing, X_t=nothing, Xs=nothing, Xe=nothing, Xo=nothing, Y=nothing,
                objective=nothing, status=status)
    end
end


"""
Solve the original LP:
  max ∑_{i,t} x_i^t
  s.t. x_i^t = ∑_{k∈S_i} x^{ts}_{ikt} = ∑_{k∈E_i} x^{te}_{ikt}
       support zeros for x^{ts}, x^{te}
       capacity_k:  ∑_t [∑_{i:a_i≤k≤d_i} x_i^t
                         - ∑_{i:a_i≤k≤b_i} ∑_{ℓ=k+1}^{b_i} x^{ts}_{iℓt}
                         - ∑_{i:c_i≤k≤d_i} ∑_{ℓ=c_i}^{k-1} x^{te}_{iℓt}] ≤ 1
       ∑_t x^{ts}_{ikt} ≤ P_s[i,k],  ∑_t x^{te}_{ikt} ≤ P_e[i,k]
       0 ≤ variables ≤ 1
Inputs:
  - job_distributions_s = P^s (n×m), job_distributions_e = P^e (n×m)
  - n jobs, m positions, T time periods
Optional:
  - quiet: suppress solver logs (COPT)
Returns named tuple with values and objective.
"""
function solve_original_LP(n::Int, m::Int, T::Int, p;
                            w = ones(n),
                           job_distributions_s::AbstractMatrix,
                           job_distributions_e::AbstractMatrix,
                           job_distributions_o::AbstractMatrix,
                           epsilon = 1e-3,
                           quiet::Bool=false)

    P_s = job_distributions_s
    P_e = job_distributions_e
    @assert size(P_s) == (n, m)
    @assert size(P_e) == (n, m)

    # Infer supports S_i=[a[i],b[i]], E_i=[c[i],d[i]] from P_s, P_e
    tol = 1e-12
    a = fill(0, n); b = fill(0, n)
    c = fill(0, n); d = fill(0, n)
    in_start_support = falses(n, m)
    in_end_support   = falses(n, m)

    for i in 1:n
        S = findall(mm -> P_s[i, mm] > tol, 1:m)
        if !isempty(S)
            a[i] = first(S); b[i] = last(S)
            @inbounds @simd for mm in a[i]:b[i]
                if P_s[i, mm] > tol
                    in_start_support[i, mm] = true
                end
            end
        end
    end
    for i in 1:n
        E = findall(mm -> P_e[i, mm] > tol, 1:m)
        if !isempty(E)
            c[i] = first(E); d[i] = last(E)
            @inbounds @simd for mm in c[i]:d[i]
                if P_e[i, mm] > tol
                    in_end_support[i, mm] = true
                end
            end
        end
    end
    println("Start supports [a..b]: ", [(a[i], b[i]) for i in 1:n])
    println("End   supports [c..d]: ", [(c[i], d[i]) for i in 1:n])
    println("In start support matrix:\n", in_start_support)
    println("In end   support matrix:\n", in_end_support)
    model = Model(COPT.ConeOptimizer)
    # model = Model(Gurobi.Optimizer)
    if quiet
        set_optimizer_attribute(model, "Logging", 0)
    end

    # Variables
    @variable(model, 0 <= X_t[1:n, 1:T] <= 1)             # x_i^t
    @variable(model, 0 <= Xs[1:n, 1:m, 1:T] <= 1)         # x_{ik}^{ts}
    @variable(model, 0 <= Xe[1:n, 1:m, 1:T] <= 1)         # x_{ik}^{te}
    @variable(model, y1[1:n, 1:T] >= 0)
    @variable(model, y2[1:n, 1:T] >= 0)

    # Zero outside support
    for i in 1:n, t in 1:T, k in 1:m
        if !in_start_support[i, k]
            @constraint(model, Xs[i, k, t] == 0.0)
        end
        if !in_end_support[i, k]
            @constraint(model, Xe[i, k, t] == 0.0)
        end
    end

    # Linking equalities: x_i^t = sum_k x^{ts}_{ikt} = sum_k x^{te}_{ikt}
    for i in 1:n, t in 1:T
        if a[i] > 0
            @constraint(model, sum(Xs[i, k, t] for k in a[i]:b[i]) == X_t[i, t])
        else
            @constraint(model, X_t[i, t] == 0.0)
        end
        if c[i] > 0
            @constraint(model, sum(Xe[i, k, t] for k in c[i]:d[i]) == X_t[i, t])
        else
            @constraint(model, X_t[i, t] == 0.0)
        end
    end

    # # Supply constraints: sum_t x^{ts}_{ik} ≤ P^s_{ik}, sum_t x^{te}_{ik} ≤ P^e_{ik}
    # for i in 1:n, k in 1:m
    #     if in_start_support[i, k]
    #         @constraint(model, sum(Xs[i, k, t] for t in 1:T) <= P_s[i, k])
    #     end
    #     if in_end_support[i, k]
    #         @constraint(model, sum(Xe[i, k, t] for t in 1:T) <= P_e[i, k])
    #     end
    # end

    for i in 1:n, k in 1:m, t in 1:T
        if in_start_support[i, k]
            @constraint(model, Xs[i, k, t] == P_s[i, k]*X_t[i,t])
        end
        if in_end_support[i, k]
            @constraint(model, Xe[i, k, t] == P_e[i, k]*X_t[i,t])
        end
    end

    # Capacity per position k (across all t)
    for k in 1:m
        expr = @expression(model,
            sum(
                # + sum_{i: a_i ≤ k ≤ d_i} x_i^t
                sum((a[i] > 0 && c[i] > 0 && a[i] <= k <= d[i]) ? X_t[i, t] : 0.0 for i in 1:n)
                # - sum_{i: a_i ≤ k ≤ b_i} sum_{ℓ=k+1}^{b_i} x^{ts}_{iℓt}
              - sum((a[i] > 0 && a[i] <= k <= b[i]) ?
                       sum(Xs[i, ℓ, t] for ℓ in max(k+1, a[i]):b[i]) : 0.0 for i in 1:n)
                # - sum_{i: c_i ≤ k ≤ d_i} sum_{ℓ=c_i}^{k-1} x^{te}_{iℓt}
              - sum((c[i] > 0 && c[i] <= k <= d[i]) ?
                       sum(Xe[i, ℓ, t] for ℓ in c[i]:min(k-1, d[i])) : 0.0 for i in 1:n)
            for t in 1:T)
        )
        @constraint(model, expr <= 1.0)
    end
    
    for k in 1:m
        expr = @expression(model,
            sum(
                # + sum_{i: a_i ≤ k ≤ d_i} x_i^t
                sum((a[i] > 0 && c[i] > 0 && b[i] <= k <= c[i]) ? X_t[i, t] : 0.0 for i in 1:n)
                # - sum_{i: a_i ≤ k ≤ b_i} sum_{ℓ=k+1}^{b_i} x^{ts}_{iℓt}
              + sum((a[i] > 0 && a[i] <= k <= b[i]-1) ?
                       sum(Xs[i, ℓ, t] for ℓ in a[i]:k) : 0.0 for i in 1:n)
                # - sum_{i: c_i ≤ k ≤ d_i} sum_{ℓ=c_i}^{k-1} x^{te}_{iℓt}
              + sum((c[i] > 0 && c[i]+1 <= k <= d[i]) ?
                       sum(Xe[i, ℓ, t] for ℓ in k:d[i]) : 0.0 for i in 1:n)
            for t in 1:T)
        )
        @constraint(model, expr <= 1.0)
    end



    # for i in 1:n, t in 1:T
    #         if a[i] > 0
    #             for ll in a[i]:b[i], mm in a[i]:(ll-1)
    #                 if in_start_support[i,mm] && in_start_support[i,ll]
    #                     @constraint(model, job_distributions_s[i, mm] * Xs[i, ll, t] >= Xs[i, mm, t] * job_distributions_s[i, ll])
    #                     # @constraint(model, Xs[i, ll, t] >= X_t[i,t] * job_distributions_s[i, ll])
    #                 end
    #             end
    #         end
    #         if c[i] > 0
    #             for ll in c[i]:(d[i]-1), mm in (ll+1):d[i]
    #                 if in_end_support[i,mm] && in_end_support[i,ll]
    #                     @constraint(model, job_distributions_e[i, mm] * Xe[i, ll, t] >= Xe[i, mm, t] * job_distributions_e[i, ll])
    #                     # @constraint(model, Xe[i, ll, t] >= X_t[i,t] * job_distributions_e[i, ll])
    #                 end
    #             end
    #             # @constraint(model, Xe[i, d[i], t] >= X_t[i,t] * job_distributions_e[i, d[i]])
    #         end
    # end
        # for i in 1:n, t in 1:T
        #     if a[i] > 0
        #         for k in a[i]:(b[i]-1)
        #                 @constraint(model, job_distributions_s[i, k] * Xs[i, k+1, t] >= Xs[i, k, t] * job_distributions_s[i, k+1])
        #                 # @constraint(model, Xs[i, ll, t] >= X_t[i,t] * job_distributions_s[i, ll])
                    
        #         end
        #     end
        #     if c[i] > 0
        #         for ll in c[i]:(d[i]-1)
        #                 @constraint(model, job_distributions_e[i, ll+1] * Xe[i, ll, t] >= Xe[i, ll+1, t] * job_distributions_e[i, ll])
        #                 # @constraint(model, Xe[i, ll, t] >= X_t[i,t] * job_distributions_e[i, ll])
        #         end
        #         # @constraint(model, Xe[i, d[i], t] >= X_t[i,t] * job_distributions_e[i, d[i]])
        #     end
        # end

    
    # for i in 1:n, t in 1:T
    #      if a[i] > 0
    #                     @constraint(model,[k = a[i]:(b[i]-1)], y1[i,t] >= job_distributions_s[i, k] * Xs[i, k+1, t] - Xs[i, k, t] * job_distributions_s[i, k+1])
    #                     @constraint(model, y1[i,t]==sum(job_distributions_s[i, k] * Xs[i, k+1, t] for k in a[i]:(b[i]-1)) - sum(job_distributions_s[i, k+1] * Xs[i, k, t] for k in a[i]:(b[i]-1)))
    #                     # @constraint(model, Xs[i, ll, t] >= X_t[i,t] * job_distributions_s[i, ll])
                    
    #     end
    #     if d[i] > 0
    #                     @constraint(model,[k = c[i]:(d[i]-1)], y2[i,t] >= job_distributions_e[i, k+1] * Xe[i, k, t] - Xe[i, k+1, t] * job_distributions_e[i, k])
    #                     @constraint(model, y2[i,t]==sum(job_distributions_e[i, k+1] * Xe[i, k, t] for k in c[i]:(d[i]-1)) - sum(job_distributions_e[i, k] * Xe[i, k+1, t] for k in c[i]:(d[i]-1)))
    #                     # @constraint(model, Xe[i, ll, t] >= X_t[i,t] * job_distributions_e[i, ll])
    #     end
    # end



    

    @variable(model, 0 ≤ X[1:n] ≤ 1)                    # x_i
    @variable(model, 0 ≤ Xo[1:n, 1:m, 1:T] ≤ 1)         # x_{im}^{t,o}

    # Linking: sum_t x_i^t = x_i
    @constraint(model, [i=1:n], sum(X_t[i, t] for t in 1:T) == X[i])

    # Linking: sum_m starts/ends = x_i^t
    @constraint(model, [i=1:n, t=1:T], sum(Xs[i, mm, t] for mm in 1:m) == X_t[i, t])
    @constraint(model, [i=1:n, t=1:T], sum(Xe[i, mm, t] for mm in 1:m) == X_t[i, t])

    # Zero outside supports
    for i in 1:n, t in 1:T, mm in 1:m
        if !in_start_support[i, mm]; @constraint(model, Xs[i, mm, t] == 0.0); end
        if !in_end_support[i, mm];   @constraint(model, Xe[i, mm, t] == 0.0); end
    end

    # Optional per-(i,m,t) upper bound from P^o (you had this)
    for i in 1:n, t in 1:T, mm in 1:m
        @constraint(model, Xo[i, mm, t] ≤ job_distributions_o[i, mm])
    end

    # Start-chain and end-chain
    for i in 1:n, t in 1:T
        if a[i] > 0
            for mm in a[i]:(b[i]-1)
                @constraint(model, Xo[i, mm+1, t] - Xo[i, mm, t] == Xs[i, mm+1, t])
            end
        end
        if c[i] > 0
            for mm in c[i]:(d[i]-1)
                @constraint(model, Xo[i, mm, t] - Xo[i, mm+1, t] == Xe[i, mm, t])
            end
        end
    end

    # Tail constraints (fixed a small bug to use 'e' on the end side)
    for i in 1:n, t in 1:T
        if b[i] > 0
                @constraint(model, Xo[i, b[i], t] == X_t[i, t])
        end
        if d[i] > 0
                @constraint(model, Xo[i, c[i], t] == X_t[i, t])
        end
        if c[i] > 0 && b[i] > 0
                for j in b[i]:c[i]
                    @constraint(model, Xo[i, j, t] == X_t[i, t])
                end
        end 
    end

    # # Proportionality (restrict to supported indices)
    #     for i in 1:n, t in 1:T
    #         if a[i] > 0
    #             for ll in a[i]:b[i], mm in a[i]:(ll-1)
    #                 if in_start_support[i,mm] && in_start_support[i,ll]
    #                     @constraint(model, job_distributions_s[i, mm] * Xs[i, ll, t] >= Xs[i, mm, t] * job_distributions_s[i, ll])
    #                     # @constraint(model, Xs[i, ll, t] >= X_t[i,t] * job_distributions_s[i, ll])
    #                 end
    #             end
    #         end
    #         if c[i] > 0
    #             for ll in c[i]:(d[i]-1), mm in (ll+1):d[i]
    #                 if in_end_support[i,mm] && in_end_support[i,ll]
    #                     @constraint(model, job_distributions_e[i, mm] * Xe[i, ll, t] >= Xe[i, mm, t] * job_distributions_e[i, ll])
    #                     # @constraint(model, Xe[i, ll, t] >= X_t[i,t] * job_distributions_e[i, ll])
    #                 end
    #             end
    #             # @constraint(model, Xe[i, d[i], t] >= X_t[i,t] * job_distributions_e[i, d[i]])
    #         end
    #     end


    # Expected length upper bound per job


    # t=1 identities
    
    # @constraint(model, [mm=1:m, i=1:n], Xs[i, mm, 1] == job_distributions_s[i, mm] * X_t[i, 1])
    # @constraint(model, [mm=1:m, i=1:n], Xe[i, mm, 1] == job_distributions_e[i, mm] * X_t[i, 1])
    
    
    # @constraint(model, [i=1:n, mm = 1:m], sum(Xs[i,mm,t] for t in 1:T)<=job_distributions_s[i,mm])
    # @constraint(model, [i=1:n, mm = 1:m], sum(Xe[i,mm,t] for t in 1:T)<=job_distributions_e[i,mm])
    


        # Capacity / occupancy constraints
    # @constraint(model, [i=1:n],
    #     sum(sum(Xo[i, mm, t] for mm in 1:m) for t in 1:T) ≤ exp_length_selected1[i] * X[i])
    # v = vec(maximum(job_distributions_o, dims = 1))  # 1×m -> Vector{Float64}(m)
    # @constraint(model, [mm=1:m, t=1:T], sum(Xo[i, mm, t] for i in 1:n) ≤ v[mm])
    # @constraint(model, [mm=1:m], sum(job_distributions_o[i, mm] * X_t[i, 1] for i in 1:n) ≤ v[mm])
    # @constraint(model, [mm=1:m], sum(sum(Xo[i, mm, t] for i in 1:n) for t in 1:T) ≤ 1)
    # @constraint(model, [i=1:n, mm=1:m], sum(Xo[i, mm, t] for t in 1:T) ≤ X[i])
    # @constraint(model, [mm=1:m], sum(Xo[i, mm, 1] for i in 1:n) ==
    #                                  sum(job_distributions_o[i, mm] * X_t[i, 1] for i in 1:n))
    # @constraint(model, [mm=1:m, i=1:n], Xo[i, mm, 1] == job_distributions_o[i, mm] * X_t[i, 1])
    # @constraint(model, [i=1:n, mm = 1:m], sum(Xo[i,mm,t] for t in 1:T)<=job_distributions_o[i,mm])
    
    @constraint(model, [t = 2:T, mm = 1:m], sum(sum(Xo[i,mm, tau] for i in 1:n) for tau in t:T) <= 1- sum(sum(Xo[i,mm,tau] for i in 1:n) for tau in 1:(t-1)))
    # @constraint(model, [i=1:n-1, j=i+1:n], X[i] + X[j] ≤ 2 - p[i, j])


    

    @constraint(model, sum(X_t[i, 1] for i in 1:n) == 1)
    @constraint(model, [t=2:T], sum(X_t[i, t] for i in 1:n) ≤ sum(X_t[i, t-1] for i in 1:n))

    # At most one job selected per time
    @constraint(model, [t=1:T], sum(X_t[i,t] for i in 1:n) <= 1)



    @constraint(model, [t = 2:T], sum(X_t[i,t] for i in 1:n) <= sum(X_t[i,t-1] for i in 1:n))


    
    # @constraint(model,[t = 2:T,i=1:n, tau = 1:(t-1)], X_t[i,t] <= sum(X_t[j,tau]*(1-p[i,j]) for j in 1:n))
    @constraint(model, [t = 2:T, i = 1:n], sum(X_t[i,tau] for tau in t:T)<= 1- sum(X_t[i,tau] for tau in 1:t-1))

    # @constraint(model, [i=1:n, j=1:n, t = 1:T],
    #     sum(X_t[j,tau] for tau in (t+1:T))+sum(X_t[i,tau] for tau in 1:t) ≤ p[i,j])

    # S = n * T
    # @variable(model, Z[1:S+1, 1:S+1], PSD)
    # @constraint(model, Z[S+1, S+1] == 1.0)
    # @constraint(model, Z.>= 0)
    
    # # flat index for selection variables
    # idx_sel(i, t) = (t-1)*n + i
    
    # # Link diagonal and last column of Z to X_t (s_{it})
    # for i in 1:n, t in 1:T
    #     q = idx_sel(i, t)
    #     @constraint(model, Z[q, q]     == X_t[i, t])
    #     @constraint(model, Z[q, S+1]   == X_t[i, t])
    # end
    
    # # Zero pattern matching your rules:
    # # (A) same job across different times -> zero
    # for i in 1:n, t in 1:T, τ in (t+1):T
    #     if τ != t
    #         @constraint(model, Z[idx_sel(i,t), idx_sel(i,τ)] == 0.0)
    #     end
    # end
    
    # # (B) different jobs at the same time -> zero
    # for t in 1:T, i in 1:n, j in (i+1):n
    #     if i != j
    #         @constraint(model, Z[idx_sel(i,t), idx_sel(j,t)] == 0.0)
    #     end
    # end
    
    # # Probability of co-selection cap:  P(both i and j) ≤ 1 - p[i,j]
    # @constraint(model, [i=1:n, j=1:n],
    #     sum(Z[idx_sel(i,t), idx_sel(j,τ)] for t in 1:T, τ in (t+1):T) ≤ 1 - p[i,j])
    
    
    # # Select i at time t if co-selected with any job at any prior time
    # @constraint(model, [t=2:T, i=1:n,  τ in 1:(t-1)],
    #     X_t[i,t] == sum(Z[idx_sel(i,t), idx_sel(j,τ)] for j in 1:n))

    # Objective: maximize sum_{i,t} x_i^t
    # @objective(model, Max, sum(X_t[i, t]*w[i] for i in 1:n, t in 1:T)+epsilon*sum(y1[i,t]+y2[i,t] for i in 1:n, t in 1:T))
    @objective(model, Max, sum(X_t[i, t]*w[i] for i in 1:n, t in 1:T))

    optimize!(model)
    status = termination_status(model)
    obj    = objective_value(model)

    X_t_val = value.(X_t)
    Xs_val  = value.(Xs)
    Xe_val  = value.(Xe)
    println("X_t = \n", X_t_val)
    println("objective = ", obj)
    println("y = ", value.(y1))
    println("y = ", value.(y2))
    println("X_s = \n", Xs_val)
    return obj,
            X_t_val,
            Xs_val,
            Xe_val
end
using JuMP
import COPT

"""
Solve the dual LP:

min   sum_r π_r
    + sum_i sum_{k∈S_i} P^s_{ik} μ^s_{ik}
    + sum_i sum_{k∈E_i} P^e_{ik} μ^e_{ik}
    + sum_{i,t} u_{it}
    + sum_{i,t} sum_{k∈S_i} v^s_{ikt}
    + sum_{i,t} sum_{k∈E_i} v^e_{ikt}

s.t.  α_{it} + β_{it} + u_{it} + sum_{r∈K_i} π_r ≥ 1                 (∀i,t)
      -α_{it} - sum_{r=a_i}^{k-1} π_r + μ^s_{ik} + v^s_{ikt} ≥ 0     (∀i,t, k∈S_i)
      -β_{it} - sum_{r=k+1}^{d_i} π_r + μ^e_{ik} + v^e_{ikt} ≥ 0     (∀i,t, k∈E_i)

      π, μ^s, μ^e, u, v^s, v^e ≥ 0;   α, β free.

Inputs:
  - n, m, T
  - P_s (n×m) = P^s, P_e (n×m) = P^e
  - quiet::Bool (optional)
Returns: named tuple of variable values and objective.
"""

function solve_dual_original_LP(n::Int, m::Int, T::Int;
    w = ones(n),
                                P_s::AbstractMatrix,  # = P^s (n×m)
                                P_e::AbstractMatrix,  # = P^e (n×m)
                                quiet::Bool=false)

    @assert size(P_s) == (n, m)
    @assert size(P_e) == (n, m)
    tol = 1e-12

    # --- infer supports S_i=[a_i,b_i], E_i=[c_i,d_i] from P_s, P_e ---
    a = fill(0, n); b = fill(0, n)
    c = fill(0, n); d = fill(0, n)
    in_start_support = falses(n, m)
    in_end_support   = falses(n, m)

    for i in 1:n
        S = findall(mm -> P_s[i, mm] > tol, 1:m)
        if !isempty(S)
            a[i] = first(S); b[i] = last(S)
            for mm in a[i]:b[i]
                in_start_support[i, mm] = P_s[i, mm] > tol
            end
        end
    end
    for i in 1:n
        E = findall(mm -> P_e[i, mm] > tol, 1:m)
        if !isempty(E)
            c[i] = first(E); d[i] = last(E)
            for mm in c[i]:d[i]
                in_end_support[i, mm] = P_e[i, mm] > tol
            end
        end
    end

    model = Model(COPT.ConeOptimizer)
    if quiet
        set_optimizer_attribute(model, "Logging", 0)
    end

    # ---------------- variables ----------------
    @variable(model, pi[1:m] >= 0)                 # π_r

    @variable(model, mu_s[1:n, 1:m] >= 0)          # μ^s_{ik}
    @variable(model, mu_e[1:n, 1:m] >= 0)          # μ^e_{ik}

    @variable(model, u[1:n, 1:T] >= 0)             # u_{it}
    @variable(model, v_s[1:n, 1:m, 1:T] >= 0)      # v^s_{ikt}
    @variable(model, v_e[1:n, 1:m, 1:T] >= 0)      # v^e_{ikt}

    @variable(model, alpha[1:n, 1:T])              # α_{it} free
    @variable(model, beta_[1:n, 1:T])              # β_{it} free

    # zero variables outside support (tidy)
    for i in 1:n, k in 1:m
        if !in_start_support[i, k]
            @constraint(model, mu_s[i, k] == 0)
            for t in 1:T
                @constraint(model, v_s[i, k, t] == 0)
            end
        end
        if !in_end_support[i, k]
            @constraint(model, mu_e[i, k] == 0)
            for t in 1:T
                @constraint(model, v_e[i, k, t] == 0)
            end
        end
    end

    # ---------------- helper expressions (as AffExpr) ----------------
    # sum_{r ∈ K_i} π_r, where K_i = [a_i..d_i] if both supports exist
    sum_pi_K = Vector{AffExpr}(undef, n)
    for i in 1:n
        if a[i] > 0 && d[i] > 0
            sum_pi_K[i] = @expression(model, sum(pi[r] for r in a[i]:d[i]))
        else
            sum_pi_K[i] = @expression(model, 0.0)
        end
    end

    # left_pi[i,k] = sum_{r=a_i}^{k-1} π_r; right_pi[i,k] = sum_{r=k+1}^{d_i} π_r
    left_pi  = Matrix{AffExpr}(undef, n, m)
    right_pi = Matrix{AffExpr}(undef, n, m)
    for i in 1:n, k in 1:m
        if a[i] > 0 && k-1 >= a[i]
            left_pi[i, k] = @expression(model, sum(pi[r] for r in a[i]:(k-1)))
        else
            left_pi[i, k] = @expression(model, 0.0)
        end
        if d[i] > 0 && k+1 <= d[i]
            right_pi[i, k] = @expression(model, sum(pi[r] for r in (k+1):d[i]))
        else
            right_pi[i, k] = @expression(model, 0.0)
        end
    end

    # ---------------- constraints ----------------
    # α_{it} + β_{it} + u_{it} + ∑_{r∈K_i} π_r ≥ 1
    for i in 1:n, t in 1:T
        @constraint(model, alpha[i, t] + beta_[i, t] + u[i, t] + sum_pi_K[i] >= w[i])
    end

    # -α_{it} - ∑_{r=a_i}^{k-1} π_r + μ^s_{ik} + v^s_{ikt} ≥ 0  (k ∈ S_i)
    for i in 1:n, k in 1:m
        if in_start_support[i, k]
            for t in 1:T
                @constraint(model, -alpha[i, t] - left_pi[i, k] + mu_s[i, k] + v_s[i, k, t] >= 0)
            end
        end
    end

    # -β_{it} - ∑_{r=k+1}^{d_i} π_r + μ^e_{ik} + v^e_{ikt} ≥ 0  (k ∈ E_i)
    for i in 1:n, k in 1:m
        if in_end_support[i, k]
            for t in 1:T
                @constraint(model, -beta_[i, t] - right_pi[i, k] + mu_e[i, k] + v_e[i, k, t] >= 0)
            end
        end
    end

    # ---------------- objective ----------------
    @objective(model, Min,
          sum(pi)                                    # ∑_r π_r
        + sum(P_s[i, k] * mu_s[i, k] for i in 1:n, k in 1:m if in_start_support[i, k])
        + sum(P_e[i, k] * mu_e[i, k] for i in 1:n, k in 1:m if in_end_support[i, k])
        + sum(u)                                     # ∑_{i,t} u_{it}
        + sum(v_s[i, k, t] for i in 1:n, k in 1:m, t in 1:T if in_start_support[i, k])
        + sum(v_e[i, k, t] for i in 1:n, k in 1:m, t in 1:T if in_end_support[i, k])
    )

    optimize!(model)

    return (
        status     = termination_status(model),
        objective  = objective_value(model),
        pi   = value.(pi),
        mu_s = value.(mu_s),
        mu_e = value.(mu_e),
        u    = value.(u),
        v_s  = value.(v_s),
        v_e  = value.(v_e),
        alpha = value.(alpha),
        beta  = value.(beta_),
        a = a, b = b, c = c, d = d,
        in_start_support = in_start_support,
        in_end_support   = in_end_support,
    )
end


function solve_dual_original_LP_wo_ab(n::Int, m::Int, T::Int;
    w = ones(n),
                                P_s::AbstractMatrix,  # = P^s (n×m)
                                P_e::AbstractMatrix,  # = P^e (n×m)
                                quiet::Bool=false)

    @assert size(P_s) == (n, m)
    @assert size(P_e) == (n, m)
    tol = 1e-12

    # --- infer supports S_i=[a_i,b_i], E_i=[c_i,d_i] from P_s, P_e ---
    a = fill(0, n); b = fill(0, n)
    c = fill(0, n); d = fill(0, n)
    in_start_support = falses(n, m)
    in_end_support   = falses(n, m)

    for i in 1:n
        S = findall(mm -> P_s[i, mm] > tol, 1:m)
        if !isempty(S)
            a[i] = first(S); b[i] = last(S)
            for mm in a[i]:b[i]
                in_start_support[i, mm] = P_s[i, mm] > tol
            end
        end
    end
    for i in 1:n
        E = findall(mm -> P_e[i, mm] > tol, 1:m)
        if !isempty(E)
            c[i] = first(E); d[i] = last(E)
            for mm in c[i]:d[i]
                in_end_support[i, mm] = P_e[i, mm] > tol
            end
        end
    end

    model = Model(COPT.ConeOptimizer)
    if quiet
        set_optimizer_attribute(model, "Logging", 0)
    end

    # ---------------- variables ----------------
    @variable(model, pi[1:m] >= 0)                 # π_r

    @variable(model, mu_s[1:n, 1:m] >= 0)          # μ^s_{ik}
    @variable(model, mu_e[1:n, 1:m] >= 0)          # μ^e_{ik}

    @variable(model, u[1:n, 1:T] >= 0)             # u_{it}
    @variable(model, v_s[1:n, 1:m, 1:T] >= 0)      # v^s_{ikt}
    @variable(model, v_e[1:n, 1:m, 1:T] >= 0)      # v^e_{ikt}

    # @variable(model, alpha[1:n, 1:T])              # α_{it} free
    # @variable(model, beta_[1:n, 1:T])              # β_{it} free

    # zero variables outside support (tidy)
    for i in 1:n, k in 1:m
        if !in_start_support[i, k]
            @constraint(model, mu_s[i, k] == 0)
            for t in 1:T
                @constraint(model, v_s[i, k, t] == 0)
            end
        end
        if !in_end_support[i, k]
            @constraint(model, mu_e[i, k] == 0)
            for t in 1:T
                @constraint(model, v_e[i, k, t] == 0)
            end
        end
    end

    # ---------------- helper expressions (as AffExpr) ----------------
    # sum_{r ∈ K_i} π_r, where K_i = [a_i..d_i] if both supports exist
    sum_pi_K = Vector{AffExpr}(undef, n)
    for i in 1:n
        if a[i] > 0 && d[i] > 0
            sum_pi_K[i] = @expression(model, sum(pi[r] for r in a[i]:d[i]))
        else
            sum_pi_K[i] = @expression(model, 0.0)
        end
    end

    # left_pi[i,k] = sum_{r=a_i}^{k-1} π_r; right_pi[i,k] = sum_{r=k+1}^{d_i} π_r
    left_pi  = Matrix{AffExpr}(undef, n, m)
    right_pi = Matrix{AffExpr}(undef, n, m)
    for i in 1:n, k in 1:m
        if a[i] > 0 && k-1 >= a[i]
            left_pi[i, k] = @expression(model, sum(pi[r] for r in a[i]:(k-1)))
        else
            left_pi[i, k] = @expression(model, 0.0)
        end
        if d[i] > 0 && k+1 <= d[i]
            right_pi[i, k] = @expression(model, sum(pi[r] for r in (k+1):d[i]))
        else
            right_pi[i, k] = @expression(model, 0.0)
        end
    end

    # ---------------- constraints ----------------
    # α_{it} + β_{it} + u_{it} + ∑_{r∈K_i} π_r ≥ 1
    for i in 1:n, t in 1:T
        @constraint(model, u[i, t] + sum_pi_K[i] >= w[i])
    end

    # -α_{it} - ∑_{r=a_i}^{k-1} π_r + μ^s_{ik} + v^s_{ikt} ≥ 0  (k ∈ S_i)
    for i in 1:n, k in 1:m
        if in_start_support[i, k]
            for t in 1:T
                @constraint(model,  - left_pi[i, k] + mu_s[i, k] + v_s[i, k, t] >= 0)
            end
        end
    end

    # -β_{it} - ∑_{r=k+1}^{d_i} π_r + μ^e_{ik} + v^e_{ikt} ≥ 0  (k ∈ E_i)
    for i in 1:n, k in 1:m
        if in_end_support[i, k]
            for t in 1:T
                @constraint(model, - right_pi[i, k] + mu_e[i, k] + v_e[i, k, t] >= 0)
            end
        end
    end

    # ---------------- objective ----------------
    @objective(model, Min,
          sum(pi)                                    # ∑_r π_r
        + sum(P_s[i, k] * mu_s[i, k] for i in 1:n, k in 1:m if in_start_support[i, k])
        + sum(P_e[i, k] * mu_e[i, k] for i in 1:n, k in 1:m if in_end_support[i, k])
        + sum(u)                                     # ∑_{i,t} u_{it}
        + sum(v_s[i, k, t] for i in 1:n, k in 1:m, t in 1:T if in_start_support[i, k])
        + sum(v_e[i, k, t] for i in 1:n, k in 1:m, t in 1:T if in_end_support[i, k])
    )

    optimize!(model)

    return (
        status     = termination_status(model),
        objective  = objective_value(model),
        pi   = value.(pi),
        mu_s = value.(mu_s),
        mu_e = value.(mu_e),
        u    = value.(u),
        v_s  = value.(v_s),
        v_e  = value.(v_e),
        a = a, b = b, c = c, d = d,
        in_start_support = in_start_support,
        in_end_support   = in_end_support,
    )
end

# n = 20
# k = 24

# Run the example
# expected_weight, all_weights, job_distributions, job_weights = example_simulation(k,n)
# P = compute_inner_product_matrix(job_distributions)
# # Run additional analysis
# analyze_results(all_weights)
# job_distributions = hcat(job_distributions...)'

# println("Job Distributions: ", job_distributions)
# # n=2
# # k=3
# # job_weights = [1, 1]
# # job_distributions = [0.5 0.5 0.0; 0 0.5 0.5]
# # P = [ 0 0.25; 0.25 0 ]


# display("Rui's Relaxation Solution:")
# solution, obj,M = solve_rui_relaxation(n, k, P, job_weights,job_distributions)
# println("Solution: ", solution)
# println("Objective Value: ", obj)


# # n = 50
# k = 100
# n = 5
# k = 8
# start_params = [(2,2),(2,4),(3,4),(4,5),(7,7)]        # per-job uniform start
# end_params   = [(2,3),(5,6),(5,6),(6,7),(8,8)]      # per-job uniform end
# # start_params = [(1,1+1e-10),(2,4),(3,4),(4,5),(7,7+1e-10)]        # per-job uniform start
# # end_params   = [(2,3),(5,6),(5,6),(6,7),(8-1e-10,8)]      # per-job uniform end
# # # start_params = [(i,2) for i in 1:n]        # per-job uniform start
# # end_params   = [(i+2,2) for i in 1:n]      # per-job uniform end
# # start_params = [1,3,4,6,6,7,7,7,8,5]
# # end_params   = [2,3,6,6,7,7,7,8,8,6]
# seed = 42
# Random.seed!(seed)  # For reproducibility
# start_params, end_params = random_param(n, k)

# # Step 1: Define job distributions
# # start_params = [(1,1+1e-10),(1,2),(3,3+1e-10)]
# # end_params = [(1,1+1e-10),(2,3),(3,3+1e-10)]

# # jobs, weights = generate_job_distributions(n,k, start_type=:normal, end_type=:normal,
#                                                 # start_params=start_params, end_params=end_params)
# jobs, weights = generate_job_distributions(n,k, start_type=:uniform, end_type=:uniform,
#                                                 start_params=start_params, end_params=end_params)                                                
# # jobs, weights = generate_job_distributions(n,k, start_type=:constant, end_type=:constant,
#                                                 # start_params=start_params, end_params=end_params)



# # n = 4
# # k = 4
# # # weights = [1.0,2.0,1.0]
# # jobs = JobDistribution[
# #     JobDistribution(Dirac(1), Dirac(1)),
# #     JobDistribution(DiscreteNonParametric([1,2],[0.5,0.5]), DiscreteNonParametric([2,3],[0.5,0.5])),
# #     JobDistribution(Dirac(3), Dirac(3)),
# #     # JobDistribution(Dirac(3), Dirac(4)),
# #     JobDistribution(DiscreteNonParametric([2,3,4],[0.5,0.25,0.25]), Dirac(4))
# # ]
# # weights = [1.0, 2.0, 1.0, 2.0]

# # Step 2: Compute intersection probabilities
# P,P_new = compute_intersection_matrix(jobs,num_samples = 1000)
# P = P-I(n)  # Adjust to ensure diagonal is zero
# P_new = P_new-I(n)  # Adjust to ensure diagonal is zero
# J_s,J_e, J,L = estimate_position_probabilities(jobs, k,num_samples = 1000)
# mean_weight, all_weights = simulate_jobs(jobs, weights, 1000)
# solution_it, obj_it,M_it = solve_x_it_relaxation_new(n, k, P_new, weights,J,J_s,J_e,L)
# solution_it_dual, obj_it_dual,gamma, beta, lambda_e,lambda_s,lambda_o = solve_x_it_relaxation_new_dual(n, k, P, weights,J,J_s,J_e)

# LP_solution, LP_obj,_ = solve_x_it_LP(n, k, P, weights,J,J_s,J_e)
# solution, obj,M = solve_rui_relaxation(n, k, P, weights,J)

# # Step 3: Simulate scheduling and get average max weight
# function run_test(sim,n,k)
    
#     obj_it_avg = 0
#     obj_avg = 0
#     obj_LP_avg = 0
#     mean_weight_avg = 0
#     for i in 1:sim
#         Random.seed!(42+i)  # For reproducibility
        
#         start_params, end_params = random_param(n, k)
#         jobs, weights = generate_job_distributions(n,k, start_type=:uniform, end_type=:uniform,
#                                                 start_params=start_params, end_params=end_params)  
#         P = compute_intersection_matrix(jobs,num_samples = 100000)
#         P = P-I(n)  # Adjust to ensure diagonal is zero
#         J_s,J_e, J = estimate_position_probabilities(jobs, k,num_samples = 100000)
#         mean_weight, all_weights = simulate_jobs(jobs, weights, 100000)
#         solution_it, obj_it,M_it = solve_x_it_relaxation(n, k, P, weights,J,J_s,J_e)
#         solution, obj,M = solve_rui_relaxation(n, k, P, weights,J)
#         LP_solution, LP_obj,_ = solve_x_it_LP(n, k, P, weights,J,J_s,J_e)

#         mean_weight_avg += mean_weight
#         obj_it_avg += obj_it
#         obj_avg += obj
#         obj_LP_avg += LP_obj
#     end
#     println("Rui's Relaxation Solution:")
#     # println("Solution: ", solution)
#     # println("X_it Solution: ", solution_it)
#     println("\nMean weight from simulation: ", mean_weight_avg/sim)
#     println("Objective Value: ", obj_avg/sim)
#     println("X_it Objective Value: ", obj_it_avg/sim)
#     println("LP Objective Value: ", obj_LP_avg/sim)
# end

# # run_test(100,8,10)
# println("\nProblem Descriptions: ")
# println("Number of jobs (n): ", n)
# println("Number of positions (k): ", k)
# println("Start Parameters: ", start_params)
# println("End Parameters: ", end_params)
# println("Job Distributions: ", jobs)
# println("Random Seed: ", seed)
# println("Rui's Relaxation Solution:")
# println("Solution: ", solution)
# println("X_it Solution: ", solution_it)
# println("\nMean weight from simulation: ", mean_weight)
# println("Objective Value: ", obj)
# println("X_it Objective Value: ", obj_it)
# println("LP Objective Value: ", LP_obj)




"""
Run one (n,k) instance and return a NamedTuple of metrics.
You can tweak num_samples, start/end types, etc.
"""
function run_instance(n::Int, k::Int; seed=42, num_samples=1000)
    Random.seed!(seed)

    # start_params, end_params = random_param(n, k)
    # jobs, weights = generate_job_distributions(n, k;
    #     start_type=:uniform, end_type=:uniform,
    #     start_params=start_params, end_params=end_params)
    lengths = rand(1:Int(floor(k/2)), n)  # Random lengths for each job
    jobs, weights, _ = generate_job_distributions_from_lengths(
        n, k, lengths;
        start_halfwidth = :random,      # or an Int (e.g., 1) or :max
        end_halfwidth   = :random,
    )
    weights = ones(n)  # Equal weights for simplicity
    println("Job lengths: ", lengths)
    P, P2 = compute_intersection_matrix(jobs, k; num_samples=num_samples)
    println("Computed intersection matrix P2")
    P2 = P2 .- I(n)
    P = P .- I(n)
    # weights = ones(n)
    J_s, J_e, J, L = estimate_position_probabilities(jobs, k; num_samples=num_samples)
    println("Computed J")

    mean_weight, _ = simulate_jobs(jobs, weights, k, num_samples)
    mean_weight_conserv,_ = simulate_jobs_conserv(jobs,weights,k)
    # mean_weight_conserv = 0
    println("Simulated jobs")
    # choose whichever greedy you keep:
    greedy_mean, _ = simulate_greedy_policy_support(jobs, weights, P2, J, k; num_sims=num_samples)
    # greedy_mean, _ = simulate_greedy(k,n,jobs, weights,num_samples)
    println("Finished greedy_conserv simulation")
    _, obj_it, _ = solve_x_it_relaxation_new(n, k, P2, weights, J, J_s, J_e, L)
    median_stab = stability_from_job_medians(jobs, k, weights)
    opt_stab = stability_from_job_opt(jobs, k, weights)
    # _, obj_SDP, _ = solve_x_it_relaxation(n, k, P, weights, J, J_s, J_e, L)
    _, LP_obj, _  = solve_x_it_LP(n, k, P, weights, J, J_s, J_e)
   _,_,_,_,_,obj_new_LP,_ = solve_ximt_LP(
           n, k, min(n,k), P;
           w = weights,                                   # Vector length n
           job_distributions_o = J, # your P^o_{im} matrix, size n×m
           job_distributions_s = J_s, # your P^s_{im} matrix, size n×m
           job_distributions_e = J_e, # your P^e_{im} matrix, size n×m
           exp_length_selected1 = L               # Vector length n
       )
    #    _,_,_,_,_,obj_new_SDP,_ = solve_ximt_SDP(
    #     n, k, min(n,k), P;
    #     w = weights,                                   # Vector length n
    #     job_distributions_o = J, # your P^o_{im} matrix, size n×m
    #     job_distributions_s = J_s, # your P^s_{im} matrix, size n×m
    #     job_distributions_e = J_e, # your P^e_{im} matrix, size n×m
    #     exp_length_selected1 = L               # Vector length n
    # )
    obj_new_SDP = 0
    
    obj_org_LP,_,_,_ = solve_original_LP(n, k, min(n,k),P2;
    w = weights,
        job_distributions_s = J_s,
        job_distributions_e = J_e,
        job_distributions_o = J, # your P^o_{im} matrix, size n×m
        quiet=true)
        println(obj_org_LP)
    dual_output = solve_dual_original_LP(n, k, min(n,k);
    w = weights,
        P_s = J_s,
        P_e = J_e,
        quiet=true)
    dual_output_wo_ab = solve_dual_original_LP_wo_ab(n, k, min(n,k);
    w = weights,
        P_s = J_s,
        P_e = J_e,
        quiet=true)
    
    # display(dual_output.alpha)
    # display(dual_output.beta)
    # display(dual_output.pi)
    # display(dual_output.mu_s)
    # display(dual_output.mu_e)
    # display(dual_output.u)
    # display(dual_output.v_s)
    # display(dual_output.v_e)
    # display(J_s)
    # display(J_e)
    # display(weights)
    println("\nDual gap (with α,β): ", obj_org_LP - dual_output.objective)
    println("\nα,β gap: ", dual_output.objective - dual_output_wo_ab.objective)

    return (n=n, k=k, mean_weight=mean_weight, mean_weight_conserv = mean_weight_conserv, opt_stab = opt_stab, two_median_stab = 2*median_stab, obj_it=obj_it,obj_SDP=0, LP_obj=LP_obj, obj_new_LP=obj_new_LP, obj_new_SDP=obj_new_SDP, obj_org_LP = obj_org_LP, greedy=greedy_mean, dual_gap = obj_org_LP - dual_output.objective, ab_gap = dual_output.objective - dual_output_wo_ab.objective)
end


"""
    generate_start_graph(n; p=nothing, rng=Random.GLOBAL_RNG,
                         job0=:uniform, job0_dist=nothing, sparse=true)

Builds the start-time probability matrix P^s (jobs × positions).

- Jobs are 0..n  -> rows 1..n+1 (row 1 corresponds to job 0).
- Positions are 1..2n -> columns 1..2n.
- Job 0: by default uniform over positions 1..n. Options:
    * job0 = :uniform        -> each of 1..n gets 1/n
    * job0 = :custom         -> pass `job0_dist` (length n), will be normalized
    * job0 = j::Int          -> deterministic at position j (1 ≤ j ≤ n)
- Jobs i = 1..n:
    P^s[i, i]   = p_i
    P^s[i, n+i] = 1 - p_i
  If `p` is not provided, p_i ~ Uniform(0,1).

Returns:
    job_distributions_s::AbstractMatrix{Float64}  # (n+1) × (2n)
    p_used::Vector{Float64}                       # length n, p_i actually used
    meta::NamedTuple                              # handy indices/info
"""
function generate_start_graph(n::Int;
    p::Union{Nothing,AbstractVector{<:Real}} = nothing,
    rng = Random.GLOBAL_RNG,
    job0 = :uniform,                 # :uniform | :custom | Int (1..n)
    job0_dist::Union{Nothing,AbstractVector{<:Real}} = nothing,
    sparse::Bool = true,
    )
    K = 2*n

    # p_i vector
    p_used = if p === nothing
        rand(rng, n)
    else
        length(p) == n || error("p must have length n")
        pv = collect(p)
        all(0 .<= pv .<= 1) || error("p entries must be in [0,1]")
        pv
    end

    # Job 0 distribution over positions 1..n
    job0_probs = if job0 === :uniform
        fill(1.0/n, n)
    elseif job0 === :custom
        job0_dist === nothing && error("Provide job0_dist when job0=:custom")
        length(job0_dist) == n || error("job0_dist must have length n")
        s = sum(job0_dist)
        s > 0 || error("job0_dist must have positive sum")
        collect(job0_dist) ./ s
    elseif job0 isa Integer
        1 <= job0 <= n || error("job0 as Int must be in 1..n")
        v = zeros(n); v[Int(job0)] = 1.0; v
    else
        error("job0 must be :uniform, :custom, or an Int in 1..n")
    end

    if sparse
        I = Int[]; J = Int[]; V = Float64[]

        # row for job 0 is 1
        for kpos in 1:n
            push!(I, 1); push!(J, kpos); push!(V, job0_probs[kpos])
        end

        # jobs 1..n: row = i+1
        for i in 1:n
            Pi = p_used[i]
            r = i + 1
            # position i with prob p_i
            push!(I, r); push!(J, i);     push!(V, Pi)
            # position n+i with prob 1-p_i
            push!(I, r); push!(J, n+i);   push!(V, 1 - Pi)
        end

        job_distributions_s = sparse(I, J, V, n+1, K)
    else
        job_distributions_s = zeros(n+1, K)
        job_distributions_s[1, 1:n] .= job0_probs
        for i in 1:n
            job_distributions_s[i+1, i]   = p_used[i]
            job_distributions_s[i+1, n+i] = 1 - p_used[i]
        end
    end

    meta = (
        K = K,
        job_row = Dict(0 => 1, (i => i+1 for i in 1:n)...),
        left_positions  = 1:n,
        right_positions = (n+1):(2n),
    )

    return job_distributions_s, p_used, meta
end


using Random, Statistics
# Optional (for plotting). Comment out if you don't want plots:
# ] add Plots
using Plots

# ===== Data container =====
struct StarInstance
    p::Vector{Float64}   # length n, p[i] = P(job i goes LEFT to i)
    w::Vector{Float64}   # length n, weights for jobs 1..n
    w0::Float64          # weight for job 0 (the long interval [1,n])
end

"""
    generate_star_instance(n; p=nothing, w=nothing, w0=nothing,
                           rng=Random.GLOBAL_RNG,
                           p_sampler=() -> rand(),       # draws in (0,1)
                           w_sampler=() -> 1 + 9*rand(), # e.g., Uniform(1,10)
                           w0_sampler=() -> 1 + 9*rand())

Build a star instance:
- Job 0 is the interval [1,n] with weight `w0`.
- Each job i=1..n is a point-interval at i (LEFT) w.p. p[i], or at n+i (RIGHT) w.p. 1-p[i], with weight w[i].
If `p`, `w`, or `w0` are not provided, they’re sampled from the given samplers.
"""
function generate_star_instance(n::Int;
    p::Union{Nothing,AbstractVector{<:Real}}=nothing,
    w::Union{Nothing,AbstractVector{<:Real}}=nothing,
    w0::Union{Nothing,Real}=nothing,
    rng = Random.GLOBAL_RNG,
    p_sampler = () -> rand(rng),            # p_i ~ Uniform(0,1)
    w_sampler = () -> 1 + 9*rand(rng),      # w_i ~ Uniform(1,10)
    w0_sampler = () -> 1 + 9*rand(rng)      # w0  ~ Uniform(1,10)
)
    p_vec = p === nothing ? [p_sampler() for _ in 1:n] : collect(p)
    w_vec = w === nothing ? [w_sampler() for _ in 1:n] : collect(w)
    w0v   = w0 === nothing ? w0_sampler() : float(w0)
    length(p_vec) == n || error("p must have length n")
    length(w_vec) == n || error("w must have length n")
    all(0 .<= p_vec .<= 1) || error("p entries must be in [0,1]")
    return StarInstance(float.(p_vec), float.(w_vec), w0v)
end

# ===== One-shot solver for a single realization =====
# Realization logic:
# - RIGHT jobs never conflict with anything -> always included
# - LEFT jobs mutually non-overlapping, but all conflict with job 0
#   So left-side contribution is max( sum of LEFT job weights, w0 )
#   Total optimum = sum(RIGHT weights) + max(w0, sum(LEFT weights))
@inline function star_realization_opt(left::AbstractVector{Bool}, w::AbstractVector, w0::Real)
    @inbounds begin
        s_left = 0.0
        s_right = 0.0
        @assert length(left) == length(w)
        for i in eachindex(left)
            if left[i]
                s_left += w[i]
            else
                s_right += w[i]
            end
        end
        return s_right + max(w0, s_left)
    end
end

"""
    simulate_expected_star(inst::StarInstance; N=100_000, rng=Random.GLOBAL_RNG)

Monte-Carlo estimate of E[optimal stability number] for the star model.
Returns (mean, stderr).
"""
function simulate_expected_star(inst::StarInstance; N::Int=100_000, rng=Random.GLOBAL_RNG)
    n = length(inst.p)
    vals = Vector{Float64}(undef, N)
    left = Vector{Bool}(undef, n)
    @inbounds for s in 1:N
        for i in 1:n
            left[i] = rand(rng) < inst.p[i]
        end
        vals[s] = star_realization_opt(left, inst.w, inst.w0)
    end
    μ = mean(vals)
    se = std(vals) / sqrt(N)
    return μ, se
end

function LP_star_graph(inst::StarInstance)
    n = length(inst.p)

    model = Model(COPT.ConeOptimizer)

    @variable(model, 0 ≤ X_t[1:(n+1), 1:(n+1)]<=1) 
    @constraint(model, [i=1:(n+1)], sum(X_t[i,t] for t in 1:(n+1)) <= 1)  # each job at most once
    @constraint(model, [t=1:(n+1)], sum(X_t[i,t] for i in 1:(n+1)) <= 1)  # each job at most once
    # Constraints
    @constraint(model, sum(X_t[i,1] for i in 1:(n+1)) == 1)  # exactly one job at time 1
    @constraint(model, [t = 2:(n+1)], sum(X_t[i,t] for i in 1:(n+1)) <= sum(X_t[j,t-1] for j in 1:n))
    @constraint(model, [i = 1:n, t=2:(n+1)], X_t[i,t] <= sum(X_t[j,t-1] for j in 1:(n+1) if j!=i)+X_t[n+1,t-1]*(1-inst.p[i]))  # at most one job can be selected at time 1
    @constraint(model, [t=2:(n+1)],X_t[n+1,t] <= sum(X_t[j,t-1]*(1-inst.p[j]) for j in 1:n))  # at most one job can be selected at time 1

    # Objective
    @objective(model, Max, sum(inst.w[i] * sum(X_t[i,t] for t in 1:(n+1)) for i in 1:n)+inst.w0*sum(X_t[n+1,t] for t in 1:(n+1)))

    optimize!(model)

    status = termination_status(model)
    obj = objective_value(model)

    if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        return (
            X_t= value.(X_t),
            objective = obj,
            status = status,
        )
    else
        @warn "Optimization did not reach OPTIMAL: $status"
        return (X=nothing, Xs=nothing, Xe=nothing, Xo=nothing,
                objective=nothing, status=status)
    end
end
# ===== Plotting helper (compare your LP relaxation to MC) =====
"""
    plot_lp_vs_mc(lp_value::Real, inst::StarInstance; N=100_000, label="instance")

Simulates MC expectation and draws a bar plot:
 - bar 1: LP value (your relaxation)
 - bar 2: MC mean with error bar (±1 s.e.)
Returns the `Plots.Plot` object and (mc_mean, mc_se).
"""
function plot_lp_vs_mc(lp_value::Real, inst::StarInstance; N::Int=100_000, label::AbstractString="instance", rng=Random.GLOBAL_RNG)
    mc_mean, mc_se = simulate_expected_star(inst; N=N, rng=rng)
    # simple two-bar chart with error bar on MC
    cats = ["LP", "MC"]
    X,lp_value,_ = LP_star_graph(inst)
    display(X)
    y = [lp_value, mc_mean]
    plt = bar(cats, y, legend=false, title="LP vs MC — $(label)",
              ylabel="value", xlabel="method", bar_width=0.6)
    scatter!([2], [mc_mean], yerror=[mc_se], ms=0, lc=:black)
    annotate!(2, mc_mean, text("±$(round(mc_se, digits=3))", 8))
    # also show numeric difference in the title
    Δ = lp_value - mc_mean
    plot!(title="LP vs MC — $(label)  (Δ = LP − MC = $(round(Δ,digits=3)))")
    return plt, (mc_mean, mc_se), X
end

# (disabled) old ad-hoc runs
# df_cons = run_conservative_batch(20; n=10, m=15, dist_types=(:uniform,:normal,:constant), save_csv="cons.csv")
# df_full = run_full_batch(10; n_grid=(8,12), m_grid=(10,20), save_csv="full.csv")
# p = plot_bound_means(df_cons); savefig(p, "ub_means.png")


# # # # --- choose sizes to sweep ---
# # n_list = [5, 8, 10, 12, 18, 25, 30, 37, 43, 47, 52, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# # k_list = [8, 10, 12, 18, 25, 30, 37, 43, 47, 52, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# n_list = [5, 8, 10,12,18,25,30]
# k_list = [8, 10,12,18,25,30]
# # n_list = [5]
# # k_list = [8]
# seeds   = 1:4                     # or e.g. 1:5 to average over randomness
# num_samples = 1000


# # # run and collect
# rows = DataFrame(n=Int[], k=Int[], seed=Int[], mean_weight=Float64[], mean_weight_conserv=Float64[], opt_stab = Float64[], two_median_stab= Float64[], obj_it=Float64[], obj_SDP=Float64[], LP_obj=Float64[], obj_new_LP = Float64[], obj_new_SDP = Float64[], obj_org_LP = Float64[], greedy = Float64[])
# gap_rows = DataFrame(n=Int[], k=Int[], seed=Int[], dual_gap=Float64[], ab_gap=Float64[])
# for n in n_list, k in k_list, s in seeds
#     if n <= k
#         # try 
#             println(k)
#             r = run_instance(n, k; seed=s, num_samples=num_samples)
#             push!(rows, (n=r.n, k=r.k, seed=s, mean_weight=r.mean_weight, mean_weight_conserv = r.mean_weight_conserv, opt_stab = r.opt_stab, two_median_stab = r.two_median_stab, obj_it=r.obj_it, obj_SDP=r.obj_SDP, LP_obj=r.LP_obj, obj_new_LP = r.obj_new_LP, obj_new_SDP = r.obj_new_SDP, obj_org_LP = r.obj_org_LP, greedy = r.greedy))
#             push!(gap_rows, (n=r.n, k=r.k, seed=s, dual_gap=r.dual_gap, ab_gap=r.ab_gap))
#         # catch err
#             # @warn "Instance failed" n k s err
#         # end
#     end
# end


# # if you used multiple seeds, average them here
# results = combine(groupby(rows, [:n, :k])) do df
#     (; mean_weight = mean(df.mean_weight),
#         mean_weight_conserv = mean(df.mean_weight_conserv),
#         opt_stab = mean(df.opt_stab),
#         two_median_stab = mean(df.two_median_stab),
#        obj_it      = mean(df.obj_it),
#        obj_SDP      = mean(df.obj_SDP),
#        LP_obj      = mean(df.LP_obj),
#        obj_new_LP  = mean(df.obj_new_LP),
#        obj_new_SDP  = mean(df.obj_new_SDP),
#          obj_org_LP  = mean(df.obj_org_LP),
#        greedy      = mean(df.greedy))
# end

# gaps = combine(groupby(gap_rows, [:n, :k])) do df
#     (; dual_gap = mean(df.dual_gap),
#        ab_gap   = mean(df.ab_gap))
# end

# # # long form for plotting
# # long = stack(results, [:mean_weight, :obj_it, :LP_obj], variable_name=:metric, value_name=:value)
# # long.label = string.("n=", long.n, ", k=", long.k)

# # # --- Plot: grouped bars per (n,k) comparing the three metrics ---
# # # With Plots.jl:
# # group_order = unique(long.label)
# # metric_order = ["mean_weight", "obj_it", "LP_obj"]

# # plotdata = [long[(long.label .== lbl) .& (long.metric .== metric), :value][1]
# #             for lbl in group_order, metric in metric_order]

# # bar(group_order, plotdata;
# #     group = metric_order,
# #     legend = :topleft,
# #     xlabel = "Instance (n,k)",
# #     ylabel = "Value",
# #     title  = "Mean weight vs. relaxation vs. LP",
# #     lw=0.5, framestyle=:box, size=(900,420), dpi=150,
# # )

# x = 1:nrow(results)
# labels = ["$(results.n[i]), $(results.k[i])" for i in 1:nrow(results)]

# step = max(1, round(Int, length(x) / 10))  # ~10 ticks total
# sel  = 1:step:length(x)

# plot(x, results.mean_weight; label="mean_weight", marker=:auto,
#      xticks=(x[sel], labels[sel]), xrotation=30, xlabel="(n,k)",
#      legend=:topleft, framestyle=:box)
# # plot!(x, results.LP_obj;  label="LP_obj",  marker=:auto)
# plot!(x, results.mean_weight_conserv;  label="mean_weight_conserv",  marker=:auto)
# plot!(x, results.opt_stab;  label="opt_stab",  marker=:auto)
# plot!(x, results.two_median_stab;  label="two_median_stab",  marker=:auto)
# plot!(x, results.greedy;  label="greedy",  marker=:auto)
# plot!(x, results.obj_it;  label="obj_it",  marker=:auto)
# plot!(x, results.obj_SDP;  label="obj_SDP",  marker=:auto)
# plot!(x, results.LP_obj;  label="LP_obj",  marker=:auto)
# plot!(x, results.obj_new_LP;  label="obj_new_LP",  marker=:auto)
# plot!(x, results.obj_new_SDP;  label="obj_new_SDP",  marker=:auto)
# plot!(x, results.obj_org_LP;  label="obj_org_LP",  marker=:auto)
# savefig("comparison_new_LP.pdf")

# println(gap_rows)







# n = 10
# k = n
# P2 = 1/n * (ones(n,n)-I(n))
# # P2 = zeros(n,n)
# J = 1/n * ones(n,n)
# J_s = J
# J_e = J
# L = ones(n)
# weights = ones(n)
# _, obj_it, _ = solve_x_it_relaxation_new(n, k, P2, weights, J, J_s, J_e, L)
# X,X_t,X_s,X_o, obj,status = solve_ximt_LP(
#            n, k, min(n,k), P2;
#            w = weights,                                   # Vector length n
#            job_distributions_o = J, # your P^o_{im} matrix, size n×m
#            job_distributions_s = J_s, # your P^s_{im} matrix, size n×m
#            job_distributions_e = J_e, # your P^e_{im} matrix, size n×m
#            exp_length_selected1 = L               # Vector length n
#        )
# Random.seed!(42)
# lengths = rand(1:Int(floor(k/2)), n)  # Random lengths for each job

#     jobs, weights, _ = generate_job_distributions_from_lengths(
#         n, k, lengths;
#         start_halfwidth = :random,      # or an Int (e.g., 1) or :max
#         end_halfwidth   = :random,
#     )
#     println("Job lengths: ", lengths)
#     P, P2 = compute_intersection_matrix(jobs, k; num_samples=num_samples)
#     println("Computed intersection matrix P2")
#     P2 = P2 .- I(n)
#     P = P .- I(n)
#     weights = ones(n)
#     J_s, J_e, J, L = estimate_position_probabilities(jobs, k; num_samples=num_samples)
#     println("Computed J")

#     mean_weight, _ = simulate_jobs(jobs, weights, k, num_samples)
#     println("Simulated jobs")
#     # choose whichever greedy you keep:
#     greedy_mean, _ = simulate_greedy_policy_support(jobs, weights, P, J, k; num_sims=num_samples)
#     println("Finished greedy simulation")
#     _, obj_it, _ = solve_x_it_relaxation_new(n, k, P, weights, J, J_s, J_e, L)
#     _, LP_obj, _  = solve_x_it_LP(n, k, P, weights, J, J_s, J_e)
#  X,X_t,X_s,X_o, obj,status = solve_ximt_LP(
#            n, k, min(n,k), P;
#            w = weights,                                   # Vector length n
#            job_distributions_o = J, # your P^o_{im} matrix, size n×m
#            job_distributions_s = J_s, # your P^s_{im} matrix, size n×m
#            exp_length_selected1 = L               # Vector length n
#        )

# ===== Example =====
# rng = MersenneTwister(42)
# n = 50
# inst = generate_star_instance(n; rng=rng)              # random p_i, w_i, w0
# mc_mean, mc_se = simulate_expected_star(inst; N=200_000, rng=rng)
# println(("MC mean", mc_mean, "MC se", mc_se))
# plt, (μ, se),X = plot_lp_vs_mc(lp_value, inst; N=100_000, label="n=$(n)", rng=rng)
# display(plt)


# # ---------- Experiment driver ----------
# """
#     run_star_experiment(ns; trials=20, N_mc=50_000, rng_seed=42, lp_callback, gen_kwargs...)

# For each n in `ns` do `trials` random instances:
#   1) generate instance via `generate_star_instance(n; gen_kwargs...)`
#   2) compute LP relaxation value via `lp_callback(inst)::Real`
#   3) estimate MC mean & s.e. via `simulate_expected_star`

# Returns a NamedTuple with raw per-instance results and per-n summaries.
# """
# function run_star_experiment(ns::AbstractVector{<:Integer};
#     trials::Int = 20,
#     N_mc::Int = 50_000,
#     rng_seed::Integer = 42,
#     lp_callback::Function,
#     gen_kwargs...
# )
#     rng = MersenneTwister(rng_seed)
#     # per-instance records
#     n_rec   = []
#     trial_i = []
#     lp_val  = []
#     mc_mean = []
#     mc_se   = []
#     gaps    = []   # LP − MC

#     for n in ns
#         for r in 1:trials
#             inst = generate_star_instance(n; rng=rng, gen_kwargs...)
#             _,lp,_   = lp_callback(inst)
#             μ, se = simulate_expected_star(inst; N=N_mc, rng=rng)
#             push!(n_rec, n); push!(trial_i, r)
#             push!(lp_val, lp); push!(mc_mean, μ); push!(mc_se, se); push!(gaps, lp - μ)
#         end
#     end

#     # summarize by n
#     n_to_inds = Dict(n => findall(==(n), n_rec) for n in ns)
#     summary = Dict{Int,NamedTuple}()
#     for n in ns
#         I = n_to_inds[n]
#         lpv  = lp_val[I]; mcv = mc_mean[I]; gapv = gaps[I]
#         summary[n] = (
#             n = n,
#             lp_mean = mean(lpv), lp_sd = std(lpv),
#             mc_mean = mean(mcv), mc_sd = std(mcv),
#             gap_mean = mean(gapv), gap_sd = std(gapv),
#             rel_gap_mean = mean((lpv .- mcv) ./ max.(mcv, 1e-12)),
#             rel_gap_sd   = std((lpv .- mcv) ./ max.(mcv, 1e-12)),
#         )
#     end

#     return (
#         per_instance = (
#             n = n_rec, trial = trial_i, lp = lp_val, mc = mc_mean, mc_se = mc_se, gap = gaps
#         ),
#         per_n = summary
#     )
# end

# # ---------- Plots ----------
# "Plot mean (LP − MC) vs n with ±1 SD ribbon, plus LP/MC means vs n."
# function plot_star_experiment(results; title_tag::AbstractString="")
#     ns = sort(collect(keys(results.per_n)))
#     gap_mean = [results.per_n[n].gap_mean for n in ns]
#     gap_sd   = [results.per_n[n].gap_sd   for n in ns]
#     lp_mean  = [results.per_n[n].lp_mean  for n in ns]
#     mc_mean  = [results.per_n[n].mc_mean  for n in ns]

#     p1 = plot(ns, gap_mean; ribbon=gap_sd, fillalpha=0.2, lw=2, marker=:circle,
#               xlabel="n", ylabel="LP − MC", title="Gap vs n $(title_tag)")
#     p2 = plot(ns, lp_mean; lw=2, marker=:circle, label="LP")
#     plot!(p2, ns, mc_mean; lw=2, marker=:square, label="MC",
#           xlabel="n", ylabel="value", title="LP & MC vs n $(title_tag)")
#     return p1, p2
# end

# "Optional: scatter of per-instance gaps vs n (jittered)."
# function scatter_gaps(results)
#     n = results.per_instance.n
#     g = results.per_instance.gap
#     # simple jitter
#     x = Float64.(n) .+ (rand(length(n)) .- 0.5) * 0.6
#     scatter(x, g; ms=3, alpha=0.6, xlabel="n", ylabel="LP − MC", title="Per-instance gaps")
# end

# # ---------- Example usage ----------
# # Define how to compute your LP value for an instance:
# # (Replace this stub with your actual LP relaxation solver)
# dummy_lp = inst -> begin
#     # quick, optimistic upper bound: E[w_right] + max(w0, E[w_left])
#     #   E[w_right] = sum((1-p_i)*w_i)
#     #   E[w_left]  = sum(p_i*w_i)
#     sum((1 .- inst.p) .* inst.w) + max(inst.w0, sum(inst.p .* inst.w))
# end

# ns   = [10, 20, 40, 80, 160]
# res  = run_star_experiment(ns; trials=20, N_mc=30_000, rng_seed=123, lp_callback=LP_star_graph)
# pGap, pCurves = plot_star_experiment(res; title_tag="(trials=20)")
# display(pGap); display(pCurves)
# scatter_gaps(res) |> display

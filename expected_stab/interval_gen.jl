using Statistics,Random
using Convex, LightGraphs, LinearAlgebra, Random, Statistics
using JuMP, COPT, Ipopt, Gurobi, SCS, Mosek, MosekTools

function generate_interval()
    # Step 1: Randomly select b and c in [0, 100]
    # b = rand() * 100
    c = rand() * 100
    b = (c > 0) ? rand() * c : 0.0
    
    # Step 2: Generate a ≤ b uniformly and d ≥ c uniformly
    a = (b > 0) ? rand() * b : 0.0
    d = (c < 100) ? c + rand() * (100 - c) : 100.0
    
    # Step 3: Starting time uniformly in [a, b], ending time in [c, d]
    s = a + (b - a) * rand()
    e = c + (d - c) * rand()
    
    # Ensure start < end
    start, ending = minmax(s, e)
    return start, ending, [a,b,c,d]
end

function gen_interval_abcd(times)
    a = times[1]
    b = times[2]
    c = times[3]
    d = times[4]
    
    s = a + (b - a) * rand()
    e = c + (d - c) * rand()
    
    # Ensure start < end
    start, ending = minmax(s, e)
    return start, ending
end

function generate_intervals(N)
    starts = zeros(N)
    ends = zeros(N)
    distrs = zeros(N, 4)  # Store the [a, b, c, d] for each interval
    for i in 1:N
        starts[i], ends[i], distrs[i,:] = generate_interval()
    end
    return starts, ends, distrs
end

function generate_intervals_abcd(N,distrs)
    starts = zeros(N)
    ends = zeros(N)
    for i in 1:N
        starts[i], ends[i]= gen_interval_abcd(distrs[i,:])
    end
    return starts, ends
end

function intervals_overlap(s1, e1, s2, e2)
    return (s1<= e2 && s2<=e1)
end

function generate_interval_graph(starts, ends)
    N = length(starts)
    A = zeros(Bool, N, N)
    for i in 1:N
        for j in (i+1):N
            A[i,j] = A[j,i] = intervals_overlap(starts[i], ends[i], starts[j], ends[j])
        end
    end
    return A
end

function prob_uniform_le_uniform(u1, u2, v1, v2)
    # Ensure valid intervals
    if u1 >= u2 || v1 >= v2
        error("Invalid interval bounds: u1 < u2 and v1 < v2 must hold.")
    end

    # Lengths of the intervals
    len_u = u2 - u1
    len_v = v2 - v1

    # The integral of the joint PDF f(x, y) = 1 / (len_u * len_v) over the region x <= y
    # within the support [u1, u2] x [v1, v2].
    # The formula used here is a simplified way to compute this integral based on
    # the geometry of the region of integration.
    numerator = max(0.0, v2 - u1)^2 - max(0.0, v1 - u1)^2 - max(0.0, v2 - u2)^2 + max(0.0, v1 - u2)^2
    denominator = 2 * len_u * len_v

    # The probability
    prob = numerator / denominator

    # Probability should be between 0 and 1
    return clamp(prob, 0.0, 1.0)
end

function prob_interval_intersect(I1, I2)
    p_s1_le_e2 = prob_uniform_le_uniform(I1[1], I1[2], I2[3], I2[4])
    p_s2_le_e1 = prob_uniform_le_uniform(I2[1], I2[2], I1[3], I1[4])

    prob_intersection = p_s1_le_e2 * p_s2_le_e1
    return prob_intersection
end

function compute_P(distrs)
    N = size(distrs, 1)
    P = zeros(N, N)
    
    for i in 1:N
        for j in (i+1):N
            prob = prob_interval_intersect(distrs[i,:], distrs[j,:])
            P[i,j] = prob
            P[j,i] = prob  # Symmetric matrix
        end
    end
    
    return P
end

function exact_stability_simulation(n::Int, Prob_P::Matrix, distrs, w = ones(n), num_sims::Int=1000)
    stability_numbers = Int[]

    for sim in 1:num_sims
        starts, ends = generate_intervals_abcd(n,distrs)
        A = generate_interval_graph(starts, ends)

        # Create IP model
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)

        # Binary variables for vertex selection
        @variable(model, x[1:n], Bin)

        # Add edge constraints
        for i in 1:n
            for j in i+1:n
                if A[i, j] == 1
                    @constraint(model, x[i] + x[j] <= 1)
                end
            end
        end

        # Maximize size of independent set
        @objective(model, Max, w'*x)

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            push!(stability_numbers, Int(objective_value(model)))
        end
    end

    return Dict(
        "values" => stability_numbers,
        "mean" => mean(stability_numbers),
        "std" => std(stability_numbers)
    )
end

function spanning_tree_gen(n::Int)
    # Initialize empty tree edges
    tree_edges = []
    
    # Set of visited nodes
    in_tree = Set{Int}()
    
    # Arbitrary root to start the tree
    root = rand(1:n)
    push!(in_tree, root)
    
    # Loop over unvisited vertices
    while length(in_tree) < n
        # Choose a random start vertex not in the tree
        start = rand(setdiff(1:n, collect(in_tree)))
        path = [start]
        visited = Dict(start => 1)
        
        # Perform a loop-erased random walk
        current = start
        while !(current in in_tree)
            neighbor = rand(setdiff(1:n, current))  # Avoid self-loop
            if haskey(visited, neighbor)
                # Loop detected, erase the loop
                idx = visited[neighbor]
                path = path[1:idx]
                visited = Dict(k => i for (i, k) in enumerate(path))
            else
                push!(path, neighbor)
                visited[neighbor] = length(path)
            end
            current = neighbor
        end
        
        # Add path to the tree
        for i in 1:length(path)-1
            push!(tree_edges, (path[i], path[i+1]))
            push!(in_tree, path[i])
        end
        push!(in_tree, path[end])
    end
    
    adj_matrix = zeros(Int, n, n)
    for (u, v) in tree_edges
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1  # undirected
    end
    return adj_matrix
end

function spanning_tree_sim(n::Int, num_sims::Int=1000)
    stability_numbers = Int[]

    for sim in 1:num_sims
        A = spanning_tree_gen(n)

        # Create IP model
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)

        # Binary variables for vertex selection
        @variable(model, x[1:n], Bin)

        # Add edge constraints
        for i in 1:n
            for j in i+1:n
                if A[i, j] == 1
                    @constraint(model, x[i] + x[j] <= 1)
                end
            end
        end

        # Maximize size of independent set
        @objective(model, Max, w'*x)

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            push!(stability_numbers, Int(objective_value(model)))
        end
    end

    return Dict(
        "values" => stability_numbers,
        "mean" => mean(stability_numbers),
        "std" => std(stability_numbers)
    )
end

function solve_sdp(N, p, w = ones(N))
    model = Model(COPT.ConeOptimizer)
    # model = Model(SCS.Optimizer)
    println("Model created")
    @variable(model, X[1:(N+1), 1:(N+1)], PSD)
    @variable(model, u[1:N])
    # @variable(model, beta[1:N, 1:N])
    # @variable(model, V[1:N, 1:N])
    # @constraint(model, beta.>= 0)
    # @constraint(model, V.>= 0)
    # @constraint(model, [i=1:N,j=1:N], X[i,j] == beta[i,j] - V[i,j]+p[i,j])
    @constraint(model, [i=1:N], X[i,i] == -2*X[N+1,i] - w[i])
    # @constraint(model, [i=1:N], X[i,i] == -X[N+1,i])
    @constraint(model, [i=1:N], -X[N+1,i] >= w[i])
    @constraint(model, X[1:N,1:N] .>= 0)

    # @constraint(model, [i=1:N,j=(i+1):N], p[i,j]*min(w[i],w[j]) <= X[i,j] <= p[i,j]*sqrt(w[i]*w[j]))
    # @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= p[i,j]*sqrt(w[i]*w[j]))
    # @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= p[i,j])
    # @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= p[i,j]*w[i])
    # @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= p[i,j]*w[j])
    @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= -p[i,j]*X[N+1,i])
    @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= -p[i,j]*X[N+1,j])

    @constraint(model, [i=1:N], u[i] == -X[N+1,i]-sum(X[i,j] for j in 1:N)+X[i,i])
    @constraint(model, X[N+1,N+1] == sum(u[i] for i in 1:N)+sum(X[i,j] for i in 1:N for j in (i+1):N))
    println("Constraints set up")
    # obj = t + 2*sum(Q[i,j] * (1 - p[i,j]) for i in 1:N for j in i+1:N)

    # @objective(model, Min, X[N+1,N+1] + 2*sum(X[i,j] * (1 - p[i,j]) for i in 1:N for j in (i+1):N))
    @objective(model, Min, X[N+1,N+1])
        # Solve the model
        optimize!(model)
 
        # Extract results
        if termination_status(model) == MOI.OPTIMAL
            solution = value.(X)
            return solution, objective_value(model)
        else
            error("Optimization did not converge to an optimal solution.")
        end
end

# Example usage
n = 50
Random.seed!(1234)  # Set a random seed for reproducibility
# _,_,distrs = generate_intervals(n)
# P = compute_P(distrs)
P = ones(n,n)
P = P - I(n)
P = P*(2/n)
w = ones(n)
sdp_optsoln, sdp_optval = solve_sdp(n, P,w)
# simulated_expectation = simulate_expected_stability(n1, n2, p; num_samples=1000)
# stability_numbers = exact_stability_simulation(n, P, distrs, w,1000)
stability_numbers = spanning_tree_sim(n, 1000)

# remain = collect(1:n)
# deleteat!(remain, 1)
# deleteat!(remain, 1)
# l = length(remain)
# P_2 = P[remain, remain]
# sdp_optsoln2, sdp_optval2 = solve_sdp(l, P_2,w[remain])
# stability_numbers2 = exact_stability_simulation(l, P_2, distrs[remain,:], w[remain],2000)

println("SDP Optimal Value: ", sdp_optval)
println("Simulated Stability Number: $(stability_numbers["mean"]) ± $(stability_numbers["std"])")
# println("SDP Optimal Value: ", sdp_optval2)
# println("Simulated Stability Number: $(stability_numbers2["mean"]) ± $(stability_numbers2["std"])")

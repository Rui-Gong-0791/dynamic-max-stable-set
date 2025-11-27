using Convex, LightGraphs, LinearAlgebra, Random, Statistics
using JuMP, COPT, Ipopt, Gurobi, SCS, Mosek, MosekTools

function generate_p_matrix_bip(n1, n2)
    N = n1 + n2
    p = zeros(N, N)
    for i in 1:N
        for j in i+1:N
            if (i <= n1 && j > n1) || (i > n1 && j <= n1)
                p_val = rand()
                p[i,j] = p_val
                p[j,i] = p_val
            else
                p[i,j] = 0.0
                p[j,i] = 0.0
            end
        end
    end
    return p
end

function generate_graph(p, n1, n2)
    N = n1 + n2
    g = SimpleGraph(N)
    for i in 1:N
        for j in i+1:N
            if (i <= n1 && j > n1) || (i > n1 && j <= n1)
                if rand() < p[i,j]
                    add_edge!(g, i, j)
                end
            end
        end
    end
    return g
end

function simulate_expected_stability(n1, n2, p; num_samples=1000)
    N = n1 + n2
    total = 0.0
    for _ in 1:num_samples
        g = generate_graph(p, n1, n2)
        matching_size = length(maximum_matching(g))  # Use LightGraphs' function
        stability_number = N - matching_size  # König's theorem for bipartite graphs
        total += stability_number
    end
    return total / num_samples
end

function exact_stability_simulation(n::Int, Prob_P::Matrix, w = ones(n), num_sims::Int=1000)
    stability_numbers = Int[]

    for sim in 1:num_sims
        # Generate Random graph as adjacency matrix
        # Initialize an n x n matrix of zeros
        A = zeros(Int, n, n)
        Random.seed!(1234 + sim)

        # Fill the upper triangle with random edges
        for i in 1:n-1
            for j in i+1:n
                A[i, j] = rand() < Prob_P[i, j] ? 1 : 0
            end
        end

        # Symmetrize to make the graph undirected
        A += A'

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

function exact_stability_simulation_rui(n::Int, Prob_P::Matrix, w = ones(n), num_sims::Int=1000)
    stability_numbers = Int[]

    for sim in 1:num_sims
        # Generate Random graph as adjacency matrix
        # Initialize an n x n matrix of zeros
        A = zeros(Int, n, n)
        Random.seed!(1234 + sim)

        # Fill the upper triangle with random edges
        for i in 1:n-1
            for j in i+1:n
                A[i, j] = rand() < Prob_P[i, j] ? 1 : 0
            end
        end

        # Symmetrize to make the graph undirected
        A += A'

        # Create IP model
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)

        # Binary variables for vertex selection
        @variable(model, x[1:n])
        @constraint(model, x .>= 0)  # Ensure non-negativity
        @constraint(model, x .<= 1)
        # Add edge constraints
        for i in 1:n
            for j in i+1:n
                if A[i, j] == 1
                    @constraint(model, 1-x[i] + 1-x[j] >= 1)
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


function solve_kevin_sdp(n1,n2,p,w=ones(n1+n2))
    N = n1 + n2
    model = Model(COPT.ConeOptimizer)
    # set_optimizer_attribute(model, "Scaling", 2)
    set_optimizer_attribute(model, "RelGap", 1e-9)           # Relative optimality gap
    set_optimizer_attribute(model, "AbsGap", 1e-9)           # Absolute optimality gap
    set_optimizer_attribute(model, "FeasTol", 1e-9)          # Primal feasibility tolerance
    set_optimizer_attribute(model, "IntTol", 1e-9)           # Integer feasibility tolerance
    set_optimizer_attribute(model, "TimeLimit", 600.0)       # Optional: allow more time
    # model = Model(SCS.Optimizer)
    println("Model created")
    @variable(model, X[1:(N+1), 1:(N+1)], PSD)
    @constraint(model, X[1:N,1:N] .>= 0)
    @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= 1-p[i,j])
    @constraint(model, [i=1:N,j=(i+1):N], X[i,i]+X[j,j] <= 2-p[i,j])
    @constraint(model, [i=1:N], X[i,i] == X[N+1,i])
    @constraint(model, X[N+1,N+1] == 1)
    @objective(model, Max, sum(X[i,i]*w[i] for i in 1:N))
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

function solve_rui_LP(n1,n2,p,w=ones(n1+n2))
    N = n1 + n2
    model = Model(Gurobi.Optimizer)
    # set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, x[1:N])
    @constraint(model, x .>= 0)  # Ensure non-negativity
    @constraint(model, x .<= 1)  
    # Add edge constraints
    # @constraint(model, [i=1:N,j=(i+1):N], 1-x[i] + 1 - x[j] == 1 - p[i,j])
    for i in 1:N
        for j in (i+1):N
                @constraint(model, 1- x[i] + 1- x[j] >= p[i,j])
                # @constraint(model, 1- x[i] + 1- x[j] <= 2*p[i,j])
        end
    end

    # @constraint(model,[i = 1:N,j=(i+1):N], x[i]*x[j] <= 1-p[i,j])  # Ensure x[i] is binary

    # Maximize size of independent set
    @objective(model, Max, w'*x)

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        solution = value.(x)
        return solution, objective_value(model)
    else
        # Return nothing or handle non-optimal cases
        println("Optimization failed with status: ", termination_status(model))
        return nothing, nothing
    end
end

function solve_sdp(n1, n2, p, w = ones(n1+n2))
    N = n1 + n2
    model = Model(COPT.ConeOptimizer)
    # set_optimizer_attribute(model, "Scaling", 2)
    set_optimizer_attribute(model, "RelGap", 1e-9)           # Relative optimality gap
    set_optimizer_attribute(model, "AbsGap", 1e-9)           # Absolute optimality gap
    set_optimizer_attribute(model, "FeasTol", 1e-9)          # Primal feasibility tolerance
    set_optimizer_attribute(model, "IntTol", 1e-9)           # Integer feasibility tolerance
    set_optimizer_attribute(model, "TimeLimit", 600.0)       # Optional: allow more time
    # model = Model(SCS.Optimizer)
    println("Model created")
    @variable(model, X[1:(N+1), 1:(N+1)], PSD)
    @variable(model, u[1:N])
    # @variable(model, beta[1:N, 1:N])
    # @variable(model, V[1:N, 1:N])
    # @constraint(model, beta.>= 0)
    # @constraint(model, V.>= 0)
    # @constraint(model, [i=1:N,j=1:N], X[i,j] == beta[i,j] - V[i,j]+p[i,j])
    # @constraint(model, [i=1:N], X[i,i] == -2*X[N+1,i] - w[i])
    @constraint(model, [i=1:N], X[i,i] == -X[N+1,i])
    # @constraint(model, [i=1:N], X[i,i] == -X[N+1,i])
    @constraint(model, [i=1:N], -X[N+1,i] >= w[i])
    @constraint(model, X[1:N,1:N] .>= 0)
    @constraint(model, u[1:N] .>= 0)

    # @constraint(model, [i=1:N,j=(i+1):N], p[i,j]*min(w[i],w[j]) <= X[i,j] <= p[i,j]*sqrt(w[i]*w[j]))
    # @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= p[i,j]*sqrt(w[i]*w[j]))
    # @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= p[i,j])
    # @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= p[i,j]*w[i])
    # @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= p[i,j]*w[j])
    @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= -X[N+1,i])
    @constraint(model, [i=1:N,j=(i+1):N], X[i,j] <= -X[N+1,j])

    @constraint(model, [i=1:N], u[i] == -X[N+1,i]-sum(X[i,j] for j in 1:N)+X[i,i])
    # @constraint(model, X[N+1,N+1] == sum(u[i] for i in 1:N)+sum(X[i,j] for i in 1:N for j in (i+1):N))
    @constraint(model, X[N+1,N+1] == sum(X[i,i] for i in 1:N)-sum(X[i,j] for i in 1:N for j in (i+1):N))
    # @constraint(model, X[N+1,N+1] == (sum(-X[N+1,i] for i in 1:N)+sum(u[i] for i in 1:N))/2)
    println("Constraints set up")
    # obj = t + 2*sum(Q[i,j] * (1 - p[i,j]) for i in 1:N for j in i+1:N)

    # @objective(model, Min, X[N+1,N+1] + 2*sum(beta[i,j] * (1 - p[i,j]) for i in 1:N for j in (i+1):N))
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

function solve_sdp_primal(n1,n2,p,w=ones(n1+n2))
    N = n1 + n2
    model = Model(COPT.ConeOptimizer)
    # set_optimizer_attribute(model, "Scaling", 2)
    # set_optimizer_attribute(model, "RelGap", 1e-9)           # Relative optimality gap
    # set_optimizer_attribute(model, "AbsGap", 1e-9)           # Absolute optimality gap
    # set_optimizer_attribute(model, "FeasTol", 1e-9)          # Primal feasibility tolerance
    # set_optimizer_attribute(model, "IntTol", 1e-9)           # Integer feasibility tolerance
    # set_optimizer_attribute(model, "TimeLimit", 600.0)       # Optional: allow more time
    # model = Model(SCS.Optimizer)
    println("Model created")
    @variable(model, X[1:(N+1), 1:(N+1)], PSD)
    @variable(model, k[1:N])
    @variable(model, Y[i=1:N, j=1:N; i < j])
    @variable(model, Z[i=1:N, j=1:N; i < j])
    
    @constraint(model, k.>= 0)
    @constraint(model, Y.>= 0)
    @constraint(model, Z.>= 0)
    # @constraint(model, X.>= 0)


    @constraint(model, [i=1:N], 2*X[N+1,i] == k[i]+2*X[i,i]+X[N+1,N+1]-1+sum(Y[i,j]*p[i,j] for j in (i+1):N) + sum(Z[j,i]*p[j,i] for j in 1:(i-1)))
    @constraint(model, [i=1:N,j=(i+1):N], 2*X[i,j] <= X[N+1,N+1]-1+Y[i,j]+Z[i,j])

    println("Constraints set up")

    @objective(model, Max, sum((k[i]+X[i,i])*w[i] for i in 1:N))
        # Solve the model
        optimize!(model)
 
        # Extract results
        if termination_status(model) == MOI.OPTIMAL
            solution = [value.(X), value.(k), value.(Y), value.(Z)]
            # solution_k = value.(k)
            return solution, objective_value(model)
        else
            error("Optimization did not converge to an optimal solution.")
        end
end

# Example usage
n1 = 30
n2 = 30
w = ones(n1+n2)
# w = rand(1:10, n1+n2)
# w[1:Int(floor(n1/2))] .= 2
# w[(n1+1):(n1+Int(floor(n2/2)))] .= 9
P = 8*generate_p_matrix_bip(n1, n2)./(n1+n2)
# # P = generate_p_matrix_bip(n1, n2)
# P[1:Int(floor(n1/2)), (n1+1):(n1+n2)] .= 0
# P[(n1+1):(n1+n2), 1:Int(floor(n1/2))] .= 0
# P = zeros(n1+n2, n1+n2)
# p = 0.3
# P[1:Int(floor(n1/5)), (n1+1):(n1+n2)] .= p
# P[1:n1, (n1+1):(n1+n2)] .= p
# P = (P+P')
# P = P./(n1+n2)

sdp_optsoln, sdp_optval = solve_sdp(n1, n2, P,w)
sdp_optsoln_primal, sdp_optval_primal = solve_sdp_primal(n1, n2, P,w)
LP_optsoln, LP_optval = solve_rui_LP(n1, n2, P,w)
# simulated_expectation = simulate_expected_stability(n1, n2, p; num_samples=1000)
stability_numbers = exact_stability_simulation(n1+n2, P,w, 1000)
stability_numbers2 = exact_stability_simulation_rui(n1+n2, P,w, 1000)
println("SDP Optimal Value: ", sdp_optval)
println("SDP Optimal Primal Value: ", sdp_optval_primal)
println("LP Optimal Primal Value: ", LP_optval)
println("Simulated Stability Number: $(stability_numbers["mean"]) ± $(stability_numbers["std"])")
println("Simulated Stability Number: $(stability_numbers2["mean"]) ± $(stability_numbers2["std"])")


function run_trial(n1, n2)
    w = ones(n1+n2)
    # w[1:Int(floor(n1/2))] .= 2
    # w[(n1+1):(n1+Int(floor(n2/2)))] .= 9
    P = generate_p_matrix_bip(n1, n2)./(n1+n2)
    P = P.*8
    # P = generate_p_matrix_bip(n1, n2)
    P[1:Int(floor(n1/2)), (n1+1):(n1+n2)] .= 0
    P[(n1+1):(n1+n2), 1:Int(floor(n1/2))] .= 0

    # P = zeros(n1+n2, n1+n2)
    # p = 1
    # P[1:n1, (n1+1):(n1+n2)] .= p
    # P = (P+P')
    # P = P./(n1+n2)

    sdp_optsoln, sdp_optval = solve_sdp(n1, n2, P,w)
    stability_numbers = exact_stability_simulation(n1+n2, P,w, 1000)
    return sdp_optval, stability_numbers
end

function run_experiment(trials::Int)
    count_diff = 0

    for _ in 1:trials
        n1 = rand(0:70)
        n2 = rand(0:70)
        
        sdp_optval, stability_number = run_trial(n1, n2)
        
        if abs(sdp_optval - stability_number["mean"]) > 0.001
            count_diff += 1
        end
    end

    return count_diff
end

# diff_count = run_experiment(1000)
# println("Number of trials with different results: ", diff_count)


# # Prepare storage
# n_values = Int[]
# means = Float64[]
# stds = Float64[]
# sdp_vals = Float64[]

# # Loop over total size from 2 to 150
# for n in 2:50
#     n1 = div(n, 2)
#     n2 = n - n1
#     sdp_val, stab = run_trial(n1, n2)

#     push!(n_values, n)
#     push!(sdp_vals, sdp_val)
#     push!(means, stab["mean"])
#     push!(stds, stab["std"])
# end

# # Plot
# plot(n_values, means;
#     ribbon = stds,
#     label = "Stability Number Mean ± Std",
#     xlabel = "n1 + n2",
#     ylabel = "Value",
#     legend = :topleft,
#     lw = 0.1)

# plot!(n_values, sdp_vals;
#     label = "SDP Optimal Value",
#     lw = 0.1,
#     ls = :dash,
#     color = :red)

# # Save to Desktop (macOS or Linux)
# savefig(joinpath(homedir(), "Desktop", "new_obj_sdp_vs_stability_constant_random_p_8n_1w.pdf"))
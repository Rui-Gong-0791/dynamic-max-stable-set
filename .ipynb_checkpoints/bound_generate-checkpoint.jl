using Statistics
using Random, Graphs
using Combinatorics, LinearAlgebra
using JuMP, COPT,Ipopt,Gurobi, SCS, Mosek, MosekTools

function find_S_star(P, t, i)
    n = size(P, 1)
    N = 1:n  # Full set of indices
    
    min_value = Inf
    S_star = nothing
    
    # Generate all subsets of size n-t+1
    for S in combinations(N, n-t+1)
        S = collect(S)  # Convert the subset to a vector for indexing
        
        # Compute the first term: prod_{j ∈ S \ i} p_{ij}
        S_without_i = setdiff(S, i)
        prod_term = prod(P[i, S_without_i])
        
        # Compute the second term: sum_{j ∈ S \ i} p_{ij} * prod_{k ∈ S \ i \ j} (1 - p_{ik})
        sum_term = 0.0
        for j in S_without_i
            other_indices = setdiff(S_without_i, j)
            prod_other = prod(1 .- P[i, other_indices])
            sum_term += P[i, j] * prod_other
        end
        
        # Compute the total value
        value = prod_term + sum_term
        
        # Update the maximum and best subset
        if value <min_value
            min_value = value
            S_star = S
        end
    end
    
    return S_star, min_value
end



function generate_symmetric_matrix(n)
    M = rand(n, n)  # Generate a random n x n matrix with entries ~ U(0, 1)
    return (M + M') / 2  # Symmetrize the matrix
end


function compute_expression(P,t)
    n = size(P, 1)
    results = zeros(n)  # Store the result for each row i
    
    for i in 1:n
        _,results[i] = find_S_star(P,t,i)
    end
    
    return results
end

function product_of_t_largest(P, row_index, t)
    # Extract the row
    row = P[row_index, :]
    
    # Sort the row in descending order and take the k largest entries
    largest_t = sort(row, rev=true)[1:t]
    
    # Compute the product of the k largest entries
    return prod(largest_t)
end



function fill_symmetric_matrix(entries, n)
    if length(entries) != div(n * (n - 1), 2)
        error("The number of entries must be n(n-1)/2.")
    end
    
    # Initialize an n x n matrix with zeros
    M = zeros(n, n)
    
    # Fill the upper triangular part (excluding diagonal)
    idx = 1
    for i in 1:n-1
        for j in i+1:n
            M[i, j] = entries[idx]
            idx += 1
        end
    end
    
    # Copy the upper triangular part to the lower triangular part
    M += M'
    
    return M
end


function solve_lp(P)
    n = size(P, 1) 
    # function compute_lhs(S, i, P)
    #     S_minus_i = setdiff(S, [i])
    #     prod_term = prod(1 - P[i, j] for j in S_minus_i)
    #     sum_term = sum(
    #         P[i, j] * prod(1 - P[i, k] for k in setdiff(S_minus_i, [j]))
    #         for j in S_minus_i
    #     )
    #     return prod_term + sum_term
    # end
    
    # # Compute the minimum LHS for each i and each t
    # results = Dict()  # Store results as (i, t) => min_lhs
    # for t in 1:n
    #     subset_size = n - t
    #     for i in 1:n
    #         # Generate all subsets of size at most max_subset_size that include i
    #         # Generate all subsets of size at most max_subset_size
    #         all_subsets = collect(combinations(1:n, subset_size))
            
    #         # Filter subsets to include only those that contain i
    #         subsets = filter(S -> i in S, all_subsets)
            
    #         # Compute the minimum LHS for this t and i
    #         if !isempty(subsets)
    #             min_lhs = minimum(compute_lhs(S, i, P) for S in subsets)
    #         else
    #             min_lhs = Inf  # No valid subsets for this (unlikely in practice)
    #         end
            
    #         # Store the result
    #         results[(i, t)] = min_lhs
    #     end
    # end

    n = size(P, 1)  # Number of rows/columns in P
    #model = Model(Mosek.Optimizer)  # Create model with HiGHS solver
    model = Model(COPT.ConeOptimizer)  # Create model with HiGHS solver

    # Variables: x[i, t] for i ∈ 1:n, t ∈ 1:n
    @variable(model, x[1:n, 1:n] >= 0)
    @variable(model, X[1:n, 1:n], PSD)
    # @variable(model, X[1:n, 1:n], Symmetric)
    # @constraint(model, X in PSDCone())

    # Precompute I_i^t for all i, t
    I = Dict{Tuple{Int, Int}, Vector{Int}}()  # I_i^t mapping
    J = Dict{Tuple{Int, Int}, Vector{Int}}()  # I_i^t mapping
    for i in 1:n
        for t in 1:n
            if t > 1
                # Extract off-diagonal entries for row i
                row_off_diag = collect(1:n)
                deleteat!(row_off_diag, i)  # Remove diagonal index
                sorted_indices = sort(row_off_diag, by=j -> P[i, j])
                I[(i, t)] = sorted_indices[1:(t-1)]  # Take top t-1 indices
                J[(i, t)] = sorted_indices[1:end-(t-1)]  # Take last t-1 indices
            else
                I[(i, t)] = []
                J[(i, t)] = []
            end
        end
    end

    # Constraints
    for i in 1:n
        # γ_i = prod((1-P[i, j]) for j in 1:n if j != i)
        @constraint(model, X[i, i] == sum(x[i, t] for t in 1:n))
        for t in 1:n
            # if t==1
            #     @constraint(model, sum(x[i,t:end]) >= prod((1 - P[i, j]) for j in 1:n if j != i))
            # else
            #     function safe_product(iter)
            #         prod = 1.0
            #         for v in iter
            #             prod *= v
            #         end
            #         return prod
            #     end
                
            #     # Add constraints based on the given expression
            #     @constraint(model, sum(x[i, t:end]) >= sum(
            #         sum(
            #             prod(
            #                 safe_product((1 - P[j, k]) for k in T if j != k) *
            #                 safe_product((1 - safe_product(1 - P[j, k] for k in T))) *
            #                 safe_product((1 - P[k, j]) for k in (setdiff(1:n, union(S,T))))
            #                 for j in T
            #             ) *
            #             safe_product((1 - P[i, j]) for j in S if j != i)
            #             for T in combinations(setdiff(1:n,S), t - 1)  # All subsets T \ S of size t - 1
            #         )
            #         for s in 1:(n - (t - 1)), S in combinations(1:n, s) if i in S  # All subsets S containing i and size <= n - (t - 1)
            #     ))
            # end
            
            for j in i:n
                @constraint(model, X[i,j] <= 1 - P[i, j])
                # @constraint(model, X[i,j] >= X[i,i]+X[j,j]-1)
            end
            if t==1
                @constraint(model, sum(x[:, t]) <= 1)
            else
                @constraint(model, sum(x[:, t]) <= 1)
            end 
            # if t < n
            #     @constraint(model, sum(x[:,t])>=sum(x[:,t+1]))
            # end

            # @constraint(model, sum(x[t, :])<= 1)
            if t > 1
                # Compute ∏_{j ∈ I_i^t} (1 - p_ij)
                prod_term = prod(1 .- P[i, I[(i, t)]])
                # prod_term_J = prod(P[i, J[(i, t)]])
                # Add constraint: x[i, t] ≤ (1 - sum(x[i, 1:t-1])) * prod_term
                @constraint(model, x[i, t] <=  (1 - sum(x[i, 1:t-1]))*prod_term)
                #@constraint(model, x[i, t] >=  1-prod_term_J)
                #@constraint(model, x[i, t] <=  prod(1 .- x[i, 1:t-1])*prod_term)
                #@constraint(model, x[i, t] <=  prod_term)
                # @constraint(model, x[i, t] <=  1 - sum(x[i, 1:t-1]))
            else
                # Add constraint: x[i, 1] ≤ 1
                @constraint(model, x[i, 1] <= 1)
            end
            # @constraint(model, x[i, t] >= results[i,t])
        end
    end

    q = 1 .- P
    # Add constraints for all i, j, k
    for i in 1:n
        for j in 1:n
            for k in 1:n
                if i != j && i != k && j != k  # Ensure distinct indices
                    α_i = 1 - P[i, j] * P[i, k] + q[i, j] * q[i, k] * q[j, k]
                    β_i = max(
                       ((1 + P[j, k]) / α_i - 1) / (1 - q[i, j] * (P[j, k] + q[j, k] * q[i, k])),
                       ((1+P[j, k])/α_i-1)/(1-q[i,k]*(P[j,k]+q[j,k]*q[i,j])),
                        ((1 + q[j, k] * (P[i, k] + q[i, k] * q[i, j])) / α_i - 1) / (1 - q[i, k]),
                        ((1 + q[j, k] * (P[i, j] + q[i, j] * q[i, k])) / α_i - 1) / (1 - q[i, j])
                    )
                    @constraint(
                        model,
                        β_i * X[i,i] + (X[j,j] + X[k,k]) / α_i <= 1 + β_i
                    )
                end
            end
        end
    end

    # for i in 1:n
    #     for j in i+1:n
    #         @constraint(model, sum(x[i, :]) * sum(x[j, :]) <= 1 - P[i, j])
    #     end
    # end
    
    # for i in 1:n
    #     # Compute the term: ∏_{j ∈ N \ i} p_ij
    #     prod_term = prod((1-P[i, j]) for j in 1:n if j != i)
        
    #     # # Compute the term: ∑_{j ∈ N \ i} p_ij * ∏_{k ∈ N \ i \ j} (1 - p_ik)
    #     # sum_term = 0.0
    #     # for j in 1:n
    #     #     if j != i
    #     #         prod_sub_term = prod(1 - P[i, k] for k in 1:n if k != i && k != j)
    #     #         sum_term += P[i, j] * prod_sub_term
    #     #     end
    #     # end
        
    #     # Add the constraint
    #     for t in 1:n
    #         @constraint(model, x[i, t] >= (1 - sum(x[i, 1:t-1]))*prod_term)
    #     end
    # end

    # for i in 1:n
    #     for j in i+1:n
    #         @constraint(model, sum(x[i, :]) + sum(x[j, :]) <= 2 - P[i, j])
    #     end
    # end

    # Objective: Minimize sum of x[i, t] over all i, t
    @objective(model, Max, tr(X))

    # Solve the model
    optimize!(model)

    # Extract results
    if termination_status(model) == MOI.OPTIMAL
        solution = value.(x)
        return solution, objective_value(model)
    else
        error("Optimization did not converge to an optimal solution.")
    end
end

function solve_easy_lp(P)
    n = size(P, 1)  # Number of rows/columns in P
    model = Model(Gurobi.Optimizer)  # Create model with HiGHS solver

    @variable(model, x[1:n] >= 0)


    for i in 1:n
        for j in i+1:n
            @constraint(model, x[i]+ x[j] <= 2 - P[i, j])
        end
    end

    # Objective: Minimize sum of x[i, t] over all i, t
    @objective(model, Max, sum(x))

    # Solve the model
    optimize!(model)

    # Extract results
    if termination_status(model) == MOI.OPTIMAL
        solution = value.(x)
        return solution, objective_value(model)
    else
        error("Optimization did not converge to an optimal solution.")
    end
end

function solve_kevin_lp(n, p)
    model = Model(Gurobi.Optimizer)  # Using Gurobi as the solver

    # Decision variables: x[i] for i ∈ [1, n]
    @variable(model, x[1:n] >= 0)

    # Constraint: x lies in the probability simplex
    @constraint(model, sum(x) == 1)

    # Compute q_l = (1-p)^(binomial(l, 2)) for each l
    q = [(1 - p)^binomial(ℓ, 2) for ℓ in 1:n]

    # Add constraints: ∑_{i=ℓ}^n [(n-ℓ choose i-ℓ) / (n choose i)] * x[i] ≤ q[ℓ] for all ℓ ∈ [1, n]
    for ℓ in 1:n
        coeffs = [binomial(BigInt(n-ℓ), BigInt(i-ℓ)) / binomial(BigInt(n), BigInt(i)) for i in ℓ:n]
        @constraint(model, sum(coeffs[j-ℓ+1] * x[j] for j in ℓ:n) <= q[ℓ])
    end

    # Objective: Maximize ∑_{i=1}^n i * x[i]
    @objective(model, Max, sum(i * x[i] for i in 1:n))

    # Solve the model
    optimize!(model)

    # Check solver status
    if termination_status(model) == MOI.OPTIMAL
        solution = value.(x)
        return solution, objective_value(model)
    else
        error("Optimization did not converge to an optimal solution.")
    end
end

function find_stability_num(n, p)
    log_n = log(n)  # Compute ln(n)
    
    for k in 1:n
        # Compute the binomial coefficients and probability term
        binom_n_k = binomial(BigInt(n), BigInt(k))
        prob_term = (1 - p)^(binomial(k, 2))
        
        # Check the inequality
        if binom_n_k * prob_term <= log_n
            return k  # Return the minimum k
        end
    end
    
    error("No valid k found for the given n and p.")
end

function exact_stability_simulation(n::Int, p::Float64; num_sims::Int=100)
    stability_numbers = Int[]
    
    for _ in 1:num_sims
        # Generate Erdős-Rényi graph as adjacency matrix
        # Initialize an n x n matrix of zeros
        P = zeros(Int, n, n)
    
        # Fill the upper triangle with random edges
        for i in 1:n-1
            for j in i+1:n
                P[i, j] = rand() < p ? 1 : 0
            end
         end
    
        # Symmetrize to make the graph undirected
        P += P'
        
        # Create IP model
        model = Model(COPT.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)
        
        # Binary variables for vertex selection
        @variable(model, x[1:n], Bin)
        
        # Add edge constraints
        for i in 1:n
            for j in i+1:n
                if P[i,j] == 1
                    @constraint(model, x[i] + x[j] <= 1)
                end
            end
        end
        
        # Maximize size of independent set
        @objective(model, Max, sum(x))
        
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
function exact_stability_simulation(n::Int, Prob_P::Matrix; num_sims::Int=100)
    stability_numbers = Int[]
    
    for sim in 1:num_sims
        # Generate Erdős-Rényi graph as adjacency matrix
        # Initialize an n x n matrix of zeros
        P = zeros(Int, n, n)
        Random.seed!(1234 + sim)
    
        # Fill the upper triangle with random edges
        for i in 1:n-1
            for j in i+1:n
                P[i, j] = rand() < Prob_P[i,j] ? 1 : 0
            end
         end
    
        # Symmetrize to make the graph undirected
        P += P'
        
        # Create IP model
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)
        
        # Binary variables for vertex selection
        @variable(model, x[1:n], Bin)
        
        # Add edge constraints
        for i in 1:n
            for j in i+1:n
                if P[i,j] == 1
                    @constraint(model, x[i] + x[j] <= 1)
                end
            end
        end
        
        # Maximize size of independent set
        @objective(model, Max, sum(x))
        
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

function compute_v(n, p)
    q = 1 - p
    v = zeros(n + 1)  # Array to store v_{k,p} for k = 0 to n
    v[1] = 0          # v_{0,p} = 0
    v[2] = 1          # v_{1,p} = 1

    for k in 2:n
        sum_term = 0
        for i in 0:k-1
            bin_coeff = binomial(BigInt(k-1), BigInt(i))
            sum_term += bin_coeff * (p^i) * (q^(k-1-i)) * v[k-1-i+1]
        end
        v[k+1] = 1 + sum_term
    end

    return v
end

function solve_submod_LP(n, P, GNP::Bool=false)
    
    
    
    # Define the optimization model
    model = Model(Gurobi.Optimizer)  # Use the GLPK solver
    @variable(model, z[1:n] >= 0)  # Non-negativity constraints
    
    # Add constraints for all subsets S
    if GNP == true
        # Compute all v_{|S|,p} values
        p=P[1,1]
        v = compute_v(n, p)
        for k in 1:n  # Size of subset S
            subsets = combinations(1:n, k)
            for S in subsets
                @constraint(model, sum(z[i] for i in S) <= v[k+1])
            end
        end
    end


    q = 1 .- P
    # Add constraints for all i, j, k
    for i in 1:n
        for j in 1:n
            for k in 1:n
                if i != j && i != k && j != k  # Ensure distinct indices
                    α_i = 1 - P[i, j] * P[i, k] + q[i, j] * q[i, k] * q[j, k]
                    β_i = max(
                       ((1 + P[j, k]) / α_i - 1) / (1 - q[i, j] * (P[j, k] + q[j, k] * q[i, k])),
                       ((1+P[j, k])/α_i-1)/(1-q[i,k]*(P[j,k]+q[j,k]*q[i,j])),
                        ((1 + q[j, k] * (P[i, k] + q[i, k] * q[i, j])) / α_i - 1) / (1 - q[i, k]),
                        ((1 + q[j, k] * (P[i, j] + q[i, j] * q[i, k])) / α_i - 1) / (1 - q[i, j])
                    )
                    @constraint(
                        model,
                        β_i * z[i] + (z[j] + z[k]) / α_i <= 1 + β_i
                    )
                end
            end
        end
    end


    # Objective function: maximize sum of z_i
    @objective(model, Max, sum(z))
    
    # Solve the problem
    optimize!(model)
    
    # Get results
    
    return value.(z), objective_value(model)
end

# Example usage 
# Generate a random n x n matrix with entries ~ U(0, 1)
#p=vcat(fill(0.1, 100), fill(0.2, 100), fill(0.3, 100), fill(0.4, 100), fill(0.5, 35))
#P=fill_symmetric_matrix(p,30)
# n = 300
# p=0.4
# Random.seed!(1234)
# P = generate_symmetric_matrix(n)
# P = ones(n,n).*p

# solution, obj_value = solve_lp(P)
# solution2, obj_value2 = solve_kevin_lp(n,p)

# println("Solution:\n")
# display(solution)
# display(sum(solution, dims=2))
# println("Objective Value: ", obj_value)
# println("Kevin LP Objective Value: ", obj_value2)
# println("Expected Stability Number Iteration: $(compute_v(n, p)[end])")

# t=50
# result = compute_expression(P,1)
# bound=(1-(t-1)*result[1])*product_of_t_largest(ones(n,n)-P,1,t-1)
# println(bound)
# println(mean(result))





# Example usage 
# Generate a random n x n matrix with entries ~ U(0, 1)
#p=vcat(fill(0.1, 100), fill(0.2, 100), fill(0.3, 100), fill(0.4, 100), fill(0.5, 35))

n = 200
#p=0.4
Random.seed!(1234)
probs = rand(n*(n-1)÷2)
P=fill_symmetric_matrix(probs,n)
display(P)
solution, obj_value = solve_lp(P)
solution2, obj_value2 = solve_submod_LP(n, P,false)
#solution2, obj_value2 = solve_kevin_lp(n,p)
# Parameters
simulations = 1000  # Number of simulations for statistical significance

# Run simulation
stability_numbers = exact_stability_simulation(n, P; num_sims=simulations)

# z_values, obj_value3 = solve_submod_LP(n, p)
# println("Optimal z values: $z_values")

println("Solution:\n")
display(solution)
display(sum(solution, dims=2))
display(sum(solution, dims=1))
println("Objective Value: ", obj_value)
println("Submodular LP Objective Value: ", obj_value2)
println("Simulated Stability Number: $(stability_numbers["mean"]) ± $(stability_numbers["std"])")
display(P)
display(sum(P,dims=2))

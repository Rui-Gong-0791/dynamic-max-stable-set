using Statistics
using Random, Graphs
using Combinatorics, LinearAlgebra
using JuMP, COPT, Ipopt, Gurobi, SCS, Mosek, MosekTools
using Suppressor, StatsBase


function find_S_star(P, t, i)
    n = size(P, 1)
    N = 1:n  # Full set of indices

    min_value = Inf
    S_star = nothing

    # Generate all subsets of size n-t+1
    for S in combinations(N, n - t + 1)
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
        if value < min_value
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


function compute_expression(P, t)
    n = size(P, 1)
    results = zeros(n)  # Store the result for each row i

    for i in 1:n
        _, results[i] = find_S_star(P, t, i)
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

# Function to compute the term for one particular combination
function term_for_combo(combo::Vector{Int}, i::Int, t::Int, q)
    # Construct the full sequence: j1 = 1, then combo, then jt = i
    indices = [1; combo; i]
    prod_term = 1 / 1
    # For stages k = 2, ..., t-1 (corresponding to indices[2]...indices[t-1]),
    # multiply by factor = ((1 - q^(k-1)) / (1 - q^k))^( indices[k] - indices[k-1] - 1 )
    for k in 2:t
        gap = indices[k] - indices[k-1]
        prod_term *= (1 - q^(k - 1))^gap
    end
    # The last factor from the final gap (from j_{t-1} to j_t=i) is typically
    # given by q^(i - indices[t-1] - 1)*(1 - q)^(t-1)
    # (or in your formula the constant (1-q^{t-1})/(1-q) is outside the summation).
    # For this example we assume that the overall constant will be applied later.
    return prod_term
end

# Function to compute x[i,t] via enumeration of combinations
function gnp_prob_sum(i::Int, t::Int, q)
    if i < t || t < 2
        return 0
    end
    # There are binom(i-2, t-2) combinations for the internal indices
    s = 0
    for combo in combinations(2:i-1, t - 2)
        s += term_for_combo(combo, i, t, q)
    end
    # Multiply by the remaining constant factors:
    expo = Int(t * (t - 1) / 2)
    const_factor = q^(expo) / prod(1 - q^k for k in 1:t-1)
    # println(typeof(const_factor))
    # println(typeof(s))
    return const_factor * s
end

# function gnp_prob_sum(i::Int, t::Int, p::Float64)
#     # Define a recursive helper function.
#     # Here, j₁ is fixed as 1.
#     # rec(j_prev, k) computes the partial sum starting from index j_prev at stage k,
#     # where k runs from 2 up to t.
#     function rec(j_prev::Int, k::Int)
#         if k == t
#             # Base case: force j_t = i.
#             # Multiply by p^(i - j_prev - 1) and include the factor for stage t: (1-p)^(t-1)
#             return (1-(1-p)^(t-1))^(i - j_prev - 1) * (1 - p)^(t - 1)
#         elseif k == t - 1
#             # For the second-to-last stage, sum over j from j_prev+1 to i-1
#             s = 0.0
#             for j in (j_prev+1):(i-1)  # j_{t-1} can range up to i-1 so that j_t = i is valid
#                 # Use the modified factor for this stage:
#                 factor = (1 - (1 - p)^(t - 2))^(j - j_prev - 1)
#                 # Also include the (1-p) factor for stage k
#                 s += factor * (1 - p)^(k - 1) * rec(j, k + 1)
#             end
#             return s
#         else
#             # For stages k = 2, …, t-2 use the original factor.
#             s = 0.0
#             for j in (j_prev+1):(i-(t-k))  # ensure enough room for later indices
#                 s += (1-(1-p)^(k-1))^(j - j_prev - 1) * (1 - p)^(k - 1) * rec(j, k + 1)
#             end
#             return s
#         end
#     end
#     return rec(1, 2)
# end

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
    model = Model(Gurobi.Optimizer)  # Create model with HiGHS solver

    # Variables: x[i, t] for i ∈ 1:n, t ∈ 1:n
    @variable(model, x[1:n, 1:n] >= 0)
    # @variable(model, X[1:n, 1:n], Symmetric)
    # @constraint(model, X in PSDCone())

    # Precompute I_i^t for all i, t
    I = Dict{Tuple{Int,Int},Vector{Int}}()  # I_i^t mapping
    J = Dict{Tuple{Int,Int},Vector{Int}}()  # I_i^t mapping
    for i in 1:n
        for t in 1:n
            if t > 1
                # Extract off-diagonal entries for row i
                row_off_diag = collect(1:n)
                deleteat!(row_off_diag, i)  # Remove diagonal index
                sorted_indices = sort(row_off_diag, by=j -> P[i, j])
                I[(i, t)] = sorted_indices[1:(t-1)]  # Take top t-1 indices
                J[(i, t)] = sorted_indices[end:-1:end-(t-1)+1]  # Take last t-1 indices
            else
                I[(i, t)] = []
                J[(i, t)] = []
            end
        end
    end

    # Sum of x[i, t] over all i is at most 1 for all t
    for t in 1:n
        @constraint(model, sum(x[:, t]) <= 1)
        # p=P[2]
        # if t == 1
        #     @constraint(model, sum(x[:,t])==1)
        # end
        # println(1-(1-(1-p)^(t-1))^(n-t+1))
        # @constraint(model, sum(x[:, t]) >= 1 - (1-(1-p)^(t-1))^(n-t+1))

        # if t >= 2
        #     max_rhs = -Inf
        #     # Iterate over all combinations H ⊆ N of size t-1.
        #     for H in combinations(1:n, t - 1)
        #         # Compute the inner product for each i in N \ H:
        #         prod_over_i = 1.0
        #         for i in setdiff(1:n, H)
        #             prod_over_j = 1.0
        #             for j in H
        #                 prod_over_j *= 1 - P[i, j]
        #             end
        #             prod_over_i *= (1 - prod_over_j)
        #         end
        #         candidate = 1 - prod_over_i
        #         max_rhs = max(max_rhs, candidate)
        #     end
        #     # Add the constraint to the model:
        #     # max_prob = (1-P[2])^binomial(t-1,2)*binomial(n,t-1)
        #     #max_prob = 1
        #     # sum(x[:, t-1])*binomial(n,t-1)
        #     @constraint(model, sum(x[1:n, t]) <= max_rhs)
        #     #@constraint(model, sum(x[1:n, (t-1)])*(1-P[2])<= sum(x[1:n, t])) #//TODO: check if this is correct
        # end
    end

    # @constraint(model, sum(x[:,2:n]) == n-2)

    # Sum of x[i, t] over all t is at most 1 for all i
    # Sum of x[i,t] from t to n is bounded
    for i in 1:n
        # γ_i = prod((1-P[i, j]) for j in 1:n if j != i)
        @constraint(model, sum(x[i, :]) <= 1)

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

            # for j in i:n
            #     @constraint(model, X[i,j] <= 1 - P[i, j])
            #     # @constraint(model, X[i,j] >= X[i,i]+X[j,j]-1)
            # end
            # if t < n
            #     @constraint(model, sum(x[:,t])>=sum(x[:,t+1]))
            # end

            # @constraint(model, sum(x[t, :])<= 1)
            if t > 1
                # Compute ∏_{j ∈ I_i^t} (1 - p_ij)
                prod_term = prod(1 .- P[i, I[(i, t)]])
                prod_term_J = prod(1 .- P[i, J[(i, t)]])
                # prod_term_J = prod(P[i, J[(i, t)]])
                # Add constraint: x[i, t] ≤ (1 - sum(x[i, 1:t-1])) * prod_term


                # @constraint(model, x[i, t] <= (1 - sum(x[i, 1:t-1])) * prod_term)

                #@constraint(model, sum(x[i, t:n]) <= (1 - sum(x[i, 1:t-1])) * prod_term)

                @constraint(model, sum(x[i, t:n]) <= (1 - sum(x[i, 1:t-1])) * prod_term)
                # @constraint(model, sum(x[i, t:n]) <= (1 - sum(x[i, 1:t-1])) * prod_term_J)

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

    # Function to compute the coefficient terms
    # function var_bound(i, t, q)
    #     binomial_coeff = binomial(i - 2, t - 2)
    #     q_factor = q^(t * (t - 1) / 2)
    #     product_term = prod(1 / (1 - q^k) for k in 1:(t-1))
    #     return binomial_coeff * q_factor * product_term
    # end

    # q = 1 - P[2]

    # # Add constraints
    # for i in 2:n, t in 2:i  # Ensure valid (i,t) pairs
    #     coeff = var_bound(i, t, q)
    #     println((1 - q)^(i - 1) * coeff)
    #     println((1 - q^(t-1))^(i - 1) * coeff)
    #     @constraint(model, (1 - q)^(i - 1) * coeff <= x[i, t])
    #     @constraint(model, x[i, t] <= (1 - q^(t - 1))^(i - 1) * coeff)
    # end

    # --------------------
    # Exactly model G(n,p) variables by having vertex 1 chosen in the first iteration 
    # -------------------- 
    # p = P[2]
    # for i in 2:n
    #     for t in 2:n
    #         if t <= i
    #             @constraint(model, x[i, t] == gnp_prob_sum(i, t, 1-p))
    #         else 
    #             @constraint(model, x[i, t] == 0)
    #         end
    #     end
    # end
    # @constraint(model, x[1, 1] == 1)

    # q = 1 .- P
    # # Add constraints for all triangles i, j, k
    # for i in 1:n
    #     for j in 1:n
    #         for k in 1:n
    #             if i != j && i != k && j != k  # Ensure distinct indices
    #                 α_i = 1 - P[i, j] * P[i, k] + q[i, j] * q[i, k] * q[j, k]
    #                 β_i = max(
    #                     ((1 + P[j, k]) / α_i - 1) / (1 - q[i, j] * (P[j, k] + q[j, k] * q[i, k])),
    #                     ((1 + P[j, k]) / α_i - 1) / (1 - q[i, k] * (P[j, k] + q[j, k] * q[i, j])),
    #                     ((1 + q[j, k] * (P[i, k] + q[i, k] * q[i, j])) / α_i - 1) / (1 - q[i, k]),
    #                     ((1 + q[j, k] * (P[i, j] + q[i, j] * q[i, k])) / α_i - 1) / (1 - q[i, j])
    #                 )
    #                 @constraint(
    #                     model,
    #                     β_i * sum(x[i,:]) + (sum(x[j,:]) + sum(x[k,:])) / α_i <= 1 + β_i
    #                 )
    #             end
    #         end
    #     end
    # end

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

    # Objective: Maximize sum of x[i, t] over all i, t
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
function solve_sdp(P)
    function interaction_prod(P::AbstractMatrix, i::Int, j::Int, t::Int)
        N = size(P, 1)

        # Basic checks
        @assert 1 ≤ i ≤ N && 1 ≤ j ≤ N "Indices i,j must be within matrix dimensions."
        @assert i != j "No feasible set if i == j but i must be in H and j must be excluded."
        @assert 1 ≤ t ≤ N - 1 "t must be at most N-1 to exclude j while including i."

        # We already know i ∈ H, j ∉ H
        # So we need t-1 more elements from {1,…,N} \ {i,j}.
        candidates = [k for k in 1:N if k != i && k != j]

        if t - 1 > length(candidates)
            error("Not enough elements to form a set of size t=$t with i in H and j out.")
        end

        # Pick the t-1 indices that have the smallest p_{j,k} (largest (1 - p_{j,k}))
        chosen_indices = partialsortperm(candidates, t - 1, by=k -> P[j, k], rev=false)

        # Compute the product: (1 - p_{j,i}) times the product of the chosen t-1
        product_value = (1 - P[j, i]) *
                        prod((1 - P[j, candidates[idx]]) for idx in chosen_indices)

        return product_value
    end
    function max_product_corrected(P::Matrix{Float64}, t::Int, i::Int; ϵ=1e-10)
        n = size(P, 1)
        @assert 1 ≤ t ≤ n "t must be between 1 and $n"
        @assert 1 ≤ i ≤ n "i must be between 1 and $n"

        # Handle zeros by replacing with ϵ
        P_modified = copy(P)
        P_modified[P_modified.==0] .= ϵ
        logP = log.(P_modified)

        # Precompute row sums R_j = sum(logP[j, :])
        R = vec(sum(logP, dims=2))

        # Generate all subsets of size t-1 from remaining elements (excluding i)
        candidates = setdiff(1:n, i)
        max_total = -Inf
        best_H = []

        # Iterate over all possible combinations of t-1 elements
        for subset in combinations(candidates, t - 1)
            H = vcat(i, subset)
            # Compute sum(R_j for j in H)
            sum_R = sum(R[H])
            # Compute sum(logP[j, k] for j, k in H)
            sum_inner = sum(logP[j, k] for j in H, k in H)
            notH = setdiff(1:n, H)
            sum_inner2 = sum(logP[j, k] for j in notH, k in notH)
            total = sum_R - sum_inner - sum_inner2

            if total > max_total
                max_total = total
                best_H = H
            end
        end

        max_prod = exp(max_total)
        return max_prod
    end
    function max_probability(P::Matrix{Float64}, i::Int, remain::Int)
        if remain == 0
            return 1
        end
        n = size(P, 1)  # Number of nodes
        N = collect(1:n)  # Node set

        max_val = 0.0  # Store the maximum probability

        # Generate all subsets H of size t that contain i
        for H in combinations(setdiff(N, [i]), remain)  # Select t-1 other nodes
            # H = [i; H]  # Ensure i is in H
            H_set = Set(H)
            # complement_H = setdiff(N, H_set)  # Nodes not in H

            # Compute the double product
            # prod_val = prod(P[j, k] for k in complement_H for j in H_set)
            prod_val = prod(P[i, k] for k in H_set)
            # prod_val = prod_val * prod(1 - P[j, k] for k in H_set for j in H_set)

            # Update max_val
            max_val = max(max_val, prod_val)
        end

        return max_val
    end
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

    # map the variable to the corresponding index in the matrix
    function vec2mat(i, t)
        return (t - 1) * n + i
    end

    function mat2vec(j)
        if j % n == 0
            return [n, j ÷ n]
        else
            return [j % n, (j ÷ n) + 1]
        end
    end

    Q = 1 .- P

    n = size(P, 1) # Number of rows/columns in P
    m = n^2  # Each column/row corresponds to a variable x[i, t]
    #model = Model(Mosek.Optimizer)  # Create model with HiGHS solver
    model = Model(COPT.ConeOptimizer)  # Create model with HiGHS solver

    # Variables: x[i, t] for i ∈ 1:n, t ∈ 1:n
    @variable(model, X[1:(m+1), 1:(m+1)], PSD)
    @constraint(model, X .>= 0)
    @constraint(model, X[m+1, m+1] == 1)
    @constraint(model, [i = 1:m, j = 1:m; i ≠ j], X[i, i] >= X[i, j])
    for i in 1:m
        @constraint(model, X[i, i] == X[m+1, i])

        #@constraint(model, 2*X[i, i] <= sum(X[i, 1:m]))
    end
    for i in 1:n
        for j in 1:m
            # @constraint(model, X[j,j]* prod(Q[mat2vec(j)[1], l] for l in 1:n if l != Q[mat2vec(j)[1]]) <= X[vec2mat(i,n),j])
            @constraint(model, prod(Q[mat2vec(j)[1], l] for l in 1:n if l != mat2vec(j)[1]) >= X[vec2mat(i, n), j])
            @constraint(model, X[i, j] == X[j, i])
        end
    end

    #=
        For the entries corresponding to each variable, we have the following mapping:
        for x_i^t, it is the entry X[(t-1)n+i, (t-1)n+i]; for the interaction entry x_ij^tτ, it is the entry X[(t-1)n+i, (τ-1)n+j].
    =#

    # @variable(model, X[1:n, 1:n], Symmetric)
    # @constraint(model, X in PSDCone())

    # Precompute I_i^t for all i, t
    I = Dict{Tuple{Int,Int},Vector{Int}}()  # I_i^t mapping
    J = Dict{Tuple{Int,Int},Vector{Int}}()  # I_i^t mapping
    for i in 1:n
        for t in 1:n
            if t > 1
                # Extract off-diagonal entries for row i
                row_off_diag = collect(1:n)
                deleteat!(row_off_diag, i)  # Remove diagonal index
                sorted_indices = sort(row_off_diag, by=j -> P[i, j])
                I[(i, t)] = sorted_indices[1:(t-1)]  # Take top t-1 indices
                J[(i, t)] = sorted_indices[end-t+2:end]  # Take last t-1 indices
            else
                I[(i, t)] = []
                J[(i, t)] = []
            end
        end
    end
    avg = (sum(P) - sum(diag(P))) / (n * (n - 1))
    for j in 1:m
        i = mat2vec(j)[1]
        t = mat2vec(j)[2]
        if t > 1
            for τ in 1:(t-1)
                @constraint(model, X[j, j] == sum(X[j, vec2mat(l, τ)] for l in 1:n if l != i))
            end
            if t <= n - 1
                # println(n-avg*t-t)
                # @constraint(model, X[j,j] <= sum(X[j, vec2mat(l, t+1)] for l in 1:n if l!=i)+prod(P[i, J[(i, max(floor(Int64,n-avg*t-t),0)+1)]]))
                # println(prod(P[i, J[(i, max(floor(Int64,n-avg*t-t),0)+1)]]))
                @constraint(model, X[j, j] <= sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i) + max_probability(P, i, n - t - floor(Int64, avg * t)))
                println("prob bound: $(max_probability(P,i,n-t-floor(Int64,avg*t)))")
            end
        elseif t == 1
            @constraint(model, (1 - prod(P[i, l] for l in 1:n if l != i)) * X[j, j] == sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i))
        end
    end
    # --------------------
    # Exactly model G(n,p) variables by having vertex 1 chosen in the first iteration 
    # --------------------
    # for i in 2:n
    #     for t in 2:n
    #         if t <= i
    #             @constraint(model, x[i, t] == gnp_prob_sum(i, t, p))
    #         else 
    #             @constraint(model, x[i, t] == 0)
    #         end
    #     end
    # end
    # @constraint(model, x[1, 1] == 1)

    # add constraints $\sum_{i=1}^n x_i^t \leq 1$ for all $t$
    for t in 1:n
        @constraint(model, sum(diag(X)[vec2mat(1, t):vec2mat(n, t)]) <= 1)
        # Consider a lower bound on x[:,t] for each t, when the graph is G(n,p)
        #@constraint(model, sum(x[:, t]) >= 1 - (1-(1-P[2])^(t-1))^(n-t+1))
        if t >= 2
            max_rhs = -Inf
            # Iterate over all combinations H ⊆ N of size t-1.
            for H in combinations(1:n, t - 1)
                # Compute the inner product for each i in N \ H:
                prod_over_i = 1.0
                for i in setdiff(1:n, H)
                    prod_over_j = 1.0
                    for j in H
                        prod_over_j *= 1 - P[i, j]
                    end
                    prod_over_i *= (1 - prod_over_j)
                end
                candidate = 1 - prod_over_i
                max_rhs = max(max_rhs, candidate)
            end
            # Add the constraint to the model:
            # max_prob = (1-P[2])^binomial(t-1,2)*binomial(n,t-1)
            #max_prob = 1
            # sum(x[:, t-1])*binomial(n,t-1)
            @constraint(model, sum(diag(X)[vec2mat(1, t):vec2mat(n, t)]) <= max_rhs)
            #@constraint(model, sum(x[1:n, (t-1)])*(1-P[2])<= sum(x[1:n, t])) #//TODO: check if this is correct
        end
    end

    for k in 1:m
        for l in (k+1):m
            i = mat2vec(k)[1]
            t = mat2vec(k)[2]
            j = mat2vec(l)[1]
            τ = mat2vec(l)[2]
            if i != j && t < τ
                max_rhs = -Inf
                for H in combinations(setdiff(collect(1:n), [i, j]), τ - t - 1)
                    # Compute the inner product for each i in N \ H:
                    candidate = 1.0
                    for v in H
                        candidate *= (1 - P[i, v]) * (1 - P[j, v])
                    end
                    max_rhs = max(max_rhs, candidate)
                end
                @constraint(model, X[k, l] <= max_rhs * X[k, k])
            end
        end
    end
    # Constraints
    for i in 1:n
        # γ_i = prod((1-P[i, j]) for j in 1:n if j != i)

        # add constraints $\sum_{t=1}^n x_i^t \leq 1$ for all $i$
        @constraint(model, sum(diag(X)[i:n:m]) <= 1)
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

            # for j in i:n
            #     @constraint(model, X[i, j] <= 1 - P[i, j])
            #     # @constraint(model, X[i,j] >= X[i,i]+X[j,j]-1)
            # end
            # if t < n
            #     @constraint(model, sum(x[:,t])>=sum(x[:,t+1]))
            # end

            # @constraint(model, sum(x[t, :])<= 1)
            if t > 1
                # Compute ∏_{j ∈ I_i^t} (1 - p_ij)
                prod_term = prod(1 .- P[i, I[(i, t)]])
                prod_term_J = prod(1 .- P[i, J[(i, t)]])
                # Add constraint: x[i, t] ≤ (1 - sum(x[i, 1:t-1])) * prod_term
                @constraint(model, sum(diag(X)[vec2mat(i, t):n:vec2mat(i, n)]) <= (1 - sum(diag(X)[vec2mat(i, 1):n:vec2mat(i, t - 1)])) * prod_term)
                #(1 - sum(x[i, 1:t-1])) * prod_term)
                # @constraint(model, sum(x[i, t:n]) >= (1 - sum(x[i, 1:t-1])) * prod_term_J)

                #@constraint(model, x[i, t] >=  1-prod_term_J)
                #@constraint(model, x[i, t] <=  prod(1 .- x[i, 1:t-1])*prod_term)
                #@constraint(model, x[i, t] <=  prod_term)
                # @constraint(model, x[i, t] <=  1 - sum(x[i, 1:t-1]))
            else
                # Add constraint: x[i, 1] ≤ 1
                @constraint(model, X[vec2mat(i, 1), vec2mat(i, 1)] <= 1)
            end
            # @constraint(model, x[i, t] >= results[i,t])
        end
    end

    for i in 1:n
        for j in (i+1):n
            @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in t+1:n) <= sum(interaction_prod(P, i, j, t) * sum(diag(X)[i:n:vec2mat(i, t)]) for t in 2:n-1) + X[i, i])

            #TODO: check if this is correct
            @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in (t+1):n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in (t+1):n) <= 1 - P[i, j])

            @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in t+1:n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in t+1:n) >= prod(Q[i, l] for l in 1:n if l != i) * prod(Q[j, l] for l in 1:n if l != j && l != i))


        end
    end

    for i in 1:m
        for j in 1:m
            row = mat2vec(i)
            col = mat2vec(j)
            if row[1] == col[1] && row[2] != col[2]
                @constraint(model, X[i, j] == 0)
            elseif row[1] != col[1] && row[2] == col[2]
                @constraint(model, X[i, j] == 0)
            end
        end
    end

    q = 1 .- P
    # Add constraints for all triangles i, j, k
    for i in 1:n
        for j in 1:n
            for k in 1:n
                if i != j && i != k && j != k  # Ensure distinct indices
                    α_i = 1 - P[i, j] * P[i, k] + q[i, j] * q[i, k] * q[j, k]
                    β_i = max(
                        ((1 + P[j, k]) / α_i - 1) / (1 - q[i, j] * (P[j, k] + q[j, k] * q[i, k])),
                        ((1 + P[j, k]) / α_i - 1) / (1 - q[i, k] * (P[j, k] + q[j, k] * q[i, j])),
                        ((1 + q[j, k] * (P[i, k] + q[i, k] * q[i, j])) / α_i - 1) / (1 - q[i, k]),
                        ((1 + q[j, k] * (P[i, j] + q[i, j] * q[i, k])) / α_i - 1) / (1 - q[i, j])
                    )
                    @constraint(
                        model,
                        β_i * X[i, i] + (X[j, j] + X[k, k]) / α_i <= 1 + β_i
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
    @objective(model, Max, tr(X) - 1)

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



function solve_sdp_temp(P)
    function interaction_prod(P::AbstractMatrix, i::Int, j::Int, t::Int)
        N = size(P, 1)

        # Basic checks
        @assert 1 ≤ i ≤ N && 1 ≤ j ≤ N "Indices i,j must be within matrix dimensions."
        @assert i != j "No feasible set if i == j but i must be in H and j must be excluded."
        @assert 1 ≤ t ≤ N - 1 "t must be at most N-1 to exclude j while including i."

        # We already know i ∈ H, j ∉ H
        # So we need t-1 more elements from {1,…,N} \ {i,j}.
        candidates = [k for k in 1:N if k != i && k != j]

        if t - 1 > length(candidates)
            error("Not enough elements to form a set of size t=$t with i in H and j out.")
        end

        # Pick the t-1 indices that have the smallest p_{j,k} (largest (1 - p_{j,k}))
        chosen_indices = partialsortperm(candidates, t - 1, by=k -> P[j, k], rev=false)

        # Compute the product: (1 - p_{j,i}) times the product of the chosen t-1
        product_value = (1 - P[j, i]) *
                        prod((1 - P[j, candidates[idx]]) for idx in chosen_indices)

        return product_value
    end
    function max_product_corrected(P::Matrix{Float64}, t::Int, i::Int; ϵ=1e-10)
        n = size(P, 1)
        @assert 1 ≤ t ≤ n "t must be between 1 and $n"
        @assert 1 ≤ i ≤ n "i must be between 1 and $n"

        # Handle zeros by replacing with ϵ
        P_modified = copy(P)
        P_modified[P_modified.==0] .= ϵ
        logP = log.(P_modified)

        # Precompute row sums R_j = sum(logP[j, :])
        R = vec(sum(logP, dims=2))

        # Generate all subsets of size t-1 from remaining elements (excluding i)
        candidates = setdiff(1:n, i)
        max_total = -Inf
        best_H = []

        # Iterate over all possible combinations of t-1 elements
        for subset in combinations(candidates, t - 1)
            H = vcat(i, subset)
            # Compute sum(R_j for j in H)
            sum_R = sum(R[H])
            # Compute sum(logP[j, k] for j, k in H)
            sum_inner = sum(logP[j, k] for j in H, k in H)
            notH = setdiff(1:n, H)
            sum_inner2 = sum(logP[j, k] for j in notH, k in notH)
            total = sum_R - sum_inner - sum_inner2

            if total > max_total
                max_total = total
                best_H = H
            end
        end

        max_prod = exp(max_total)
        return max_prod
    end
    function max_probability(P::Matrix{Float64}, i::Int, remain::Int)
        if remain == 0
            return 1
        end
        n = size(P, 1)  # Number of nodes
        N = collect(1:n)  # Node set

        max_val = 0.0  # Store the maximum probability

        # Generate all subsets H of size t that contain i
        for H in combinations(setdiff(N, [i]), remain)  # Select t-1 other nodes
            # H = [i; H]  # Ensure i is in H
            H_set = Set(H)
            # complement_H = setdiff(N, H_set)  # Nodes not in H

            # Compute the double product
            # prod_val = prod(P[j, k] for k in complement_H for j in H_set)
            prod_val = prod(P[i, k] for k in H_set)
            # prod_val = prod_val * prod(1 - P[j, k] for k in H_set for j in H_set)

            # Update max_val
            max_val = max(max_val, prod_val)
        end

        return max_val
    end

    # map the variable to the corresponding index in the matrix
    function vec2mat(i, t)
        return (t - 1) * n + i
    end

    function mat2vec(j)
        if j % n == 0
            return [n, j ÷ n]
        else
            return [j % n, (j ÷ n) + 1]
        end
    end

    Q = 1 .- P

    n = size(P, 1) # Number of rows/columns in P
    m = n^2  # Each column/row corresponds to a variable x[i, t]
    #model = Model(Mosek.Optimizer)  # Create model with HiGHS solver
    model = Model(COPT.ConeOptimizer)  # Create model with HiGHS solver

    # Variables: x[i, t] for i ∈ 1:n, t ∈ 1:n
    @variable(model, X[1:(m+1), 1:(m+1)], PSD)
    @constraint(model, X .>= 0)
    @constraint(model, X[m+1, m+1] == 1)
    @constraint(model, [i = 1:m, j = 1:m; i ≠ j], X[i, i] >= X[i, j])
    for i in 1:m
        @constraint(model, X[i, i] == X[m+1, i])

        #@constraint(model, 2*X[i, i] <= sum(X[i, 1:m]))
    end
    for i in 1:n
        for j in 1:m
            # @constraint(model, X[j,j]* prod(Q[mat2vec(j)[1], l] for l in 1:n if l != Q[mat2vec(j)[1]]) <= X[vec2mat(i,n),j])
            @constraint(model, prod(Q[mat2vec(j)[1], l] for l in 1:n if l != mat2vec(j)[1]) >= X[vec2mat(i, n), j])
            @constraint(model, X[i, j] == X[j, i])
            if mat2vec(i)[2] - mat2vec(j)[2] > 1
                @constraint(model, X[i, j] == 0)
            end
        end
    end

    #=
        For the entries corresponding to each variable, we have the following mapping:
        for x_i^t, it is the entry X[(t-1)n+i, (t-1)n+i]; for the interaction entry x_ij^tτ, it is the entry X[(t-1)n+i, (τ-1)n+j].
    =#

    # @variable(model, X[1:n, 1:n], Symmetric)
    # @constraint(model, X in PSDCone())

    # Precompute I_i^t for all i, t
    I = Dict{Tuple{Int,Int},Vector{Int}}()  # I_i^t mapping
    J = Dict{Tuple{Int,Int},Vector{Int}}()  # I_i^t mapping
    for i in 1:n
        for t in 1:n
            if t > 1
                # Extract off-diagonal entries for row i
                row_off_diag = collect(1:n)
                deleteat!(row_off_diag, i)  # Remove diagonal index
                sorted_indices = sort(row_off_diag, by=j -> P[i, j])
                I[(i, t)] = sorted_indices[1:(t-1)]  # Take top t-1 indices
                J[(i, t)] = sorted_indices[end-t+2:end]  # Take last t-1 indices
            else
                I[(i, t)] = []
                J[(i, t)] = []
            end
        end
    end
    avg = (sum(P) - sum(diag(P))) / (n * (n - 1))
    for j in 1:m
        i = mat2vec(j)[1]
        t = mat2vec(j)[2]
        if t > 1
            for τ in 1:(t-1)
                @constraint(model, X[j, j] == sum(X[j, vec2mat(l, τ)] for l in 1:n if l != i))
            end
            if t <= n - 1
                # println(n-avg*t-t)
                # @constraint(model, X[j,j] <= sum(X[j, vec2mat(l, t+1)] for l in 1:n if l!=i)+prod(P[i, J[(i, max(floor(Int64,n-avg*t-t),0)+1)]]))
                # println(prod(P[i, J[(i, max(floor(Int64,n-avg*t-t),0)+1)]]))
                @constraint(model, X[j, j] <= sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i) + max_probability(P, i, n - t - floor(Int64, avg * t)))
                println("prob bound: $(max_probability(P,i,n-t-floor(Int64,avg*t)))")
            end
        elseif t == 1
            @constraint(model, (1 - prod(P[i, l] for l in 1:n if l != i)) * X[j, j] == sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i))
        end
    end

    # add constraints $\sum_{i=1}^n x_i^t \leq 1$ for all $t$
    for t in 1:n
        @constraint(model, sum(diag(X)[vec2mat(1, t):vec2mat(n, t)]) <= 1)
        # Consider a lower bound on x[:,t] for each t, when the graph is G(n,p)
        #@constraint(model, sum(x[:, t]) >= 1 - (1-(1-P[2])^(t-1))^(n-t+1))
        if t >= 2
            max_rhs = -Inf
            # Iterate over all combinations H ⊆ N of size t-1.
            for H in combinations(1:n, t - 1)
                # Compute the inner product for each i in N \ H:
                prod_over_i = 1.0
                for i in setdiff(1:n, H)
                    prod_over_j = 1.0
                    for j in H
                        prod_over_j *= 1 - P[i, j]
                    end
                    prod_over_i *= (1 - prod_over_j)
                end
                candidate = 1 - prod_over_i
                max_rhs = max(max_rhs, candidate)
            end
            # Add the constraint to the model:
            # max_prob = (1-P[2])^binomial(t-1,2)*binomial(n,t-1)
            #max_prob = 1
            # sum(x[:, t-1])*binomial(n,t-1)
            @constraint(model, sum(diag(X)[vec2mat(1, t):vec2mat(n, t)]) <= max_rhs)
            #@constraint(model, sum(x[1:n, (t-1)])*(1-P[2])<= sum(x[1:n, t])) #//TODO: check if this is correct
        end
    end

    for k in 1:m
        for l in (k+1):m
            i = mat2vec(k)[1]
            t = mat2vec(k)[2]
            j = mat2vec(l)[1]
            τ = mat2vec(l)[2]
            if i != j && t < τ
                max_rhs = -Inf
                for H in combinations(setdiff(collect(1:n), [i, j]), τ - t - 1)
                    # Compute the inner product for each i in N \ H:
                    candidate = 1.0
                    for v in H
                        candidate *= (1 - P[i, v]) * (1 - P[j, v])
                    end
                    max_rhs = max(max_rhs, candidate)
                end
                @constraint(model, X[k, l] <= max_rhs * X[k, k])
            end
        end
    end
    # Constraints
    for i in 1:n
        # γ_i = prod((1-P[i, j]) for j in 1:n if j != i)

        # add constraints $\sum_{t=1}^n x_i^t \leq 1$ for all $i$
        @constraint(model, sum(diag(X)[i:n:m]) <= 1)
        for t in 1:n
            if t > 1
                # Compute ∏_{j ∈ I_i^t} (1 - p_ij)
                prod_term = prod(1 .- P[i, I[(i, t)]])
                prod_term_J = prod(1 .- P[i, J[(i, t)]])
                # Add constraint: x[i, t] ≤ (1 - sum(x[i, 1:t-1])) * prod_term
                @constraint(model, sum(diag(X)[vec2mat(i, t):n:vec2mat(i, n)]) <= (1 - sum(diag(X)[vec2mat(i, 1):n:vec2mat(i, t - 1)])) * prod_term)
                #(1 - sum(x[i, 1:t-1])) * prod_term)
                # @constraint(model, sum(x[i, t:n]) >= (1 - sum(x[i, 1:t-1])) * prod_term_J)

                #@constraint(model, x[i, t] >=  1-prod_term_J)
                #@constraint(model, x[i, t] <=  prod(1 .- x[i, 1:t-1])*prod_term)
                #@constraint(model, x[i, t] <=  prod_term)
                # @constraint(model, x[i, t] <=  1 - sum(x[i, 1:t-1]))
            else
                # Add constraint: x[i, 1] ≤ 1
                @constraint(model, X[vec2mat(i, 1), vec2mat(i, 1)] <= 1)
            end
            # @constraint(model, x[i, t] >= results[i,t])
        end
    end

    for i in 1:n
        for j in (i+1):n
            @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in t+1:n) <= sum(interaction_prod(P, i, j, t) * sum(diag(X)[i:n:vec2mat(i, t)]) for t in 2:n-1) + X[i, i])

            #TODO: check if this is correct
            @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in (t+1):n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in (t+1):n) <= 1 - P[i, j])

            @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in t+1:n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in t+1:n) >= prod(Q[i, l] for l in 1:n if l != i) * prod(Q[j, l] for l in 1:n if l != j && l != i))


        end
    end

    for i in 1:m
        for j in 1:m
            row = mat2vec(i)
            col = mat2vec(j)
            if row[1] == col[1] && row[2] != col[2]
                @constraint(model, X[i, j] == 0)
            elseif row[1] != col[1] && row[2] == col[2]
                @constraint(model, X[i, j] == 0)
            end
        end
    end

    q = 1 .- P
    # Add constraints for all triangles i, j, k
    for i in 1:n
        for j in 1:n
            for k in 1:n
                if i != j && i != k && j != k  # Ensure distinct indices
                    α_i = 1 - P[i, j] * P[i, k] + q[i, j] * q[i, k] * q[j, k]
                    β_i = max(
                        ((1 + P[j, k]) / α_i - 1) / (1 - q[i, j] * (P[j, k] + q[j, k] * q[i, k])),
                        ((1 + P[j, k]) / α_i - 1) / (1 - q[i, k] * (P[j, k] + q[j, k] * q[i, j])),
                        ((1 + q[j, k] * (P[i, k] + q[i, k] * q[i, j])) / α_i - 1) / (1 - q[i, k]),
                        ((1 + q[j, k] * (P[i, j] + q[i, j] * q[i, k])) / α_i - 1) / (1 - q[i, j])
                    )
                    @constraint(
                        model,
                        β_i * X[i, i] + (X[j, j] + X[k, k]) / α_i <= 1 + β_i
                    )
                end
            end
        end
    end

    # Objective: Minimize sum of x[i, t] over all i, t
    @objective(model, Max, tr(X) - 1)

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

function solve_easy_lp(P)
    n = size(P, 1)  # Number of rows/columns in P
    model = Model(Gurobi.Optimizer)  # Create model with HiGHS solver

    @variable(model, x[1:n] >= 0)


    for i in 1:n
        for j in i+1:n
            @constraint(model, x[i] + x[j] <= 2 - P[i, j])
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
        coeffs = [binomial(BigInt(n - ℓ), BigInt(i - ℓ)) / binomial(BigInt(n), BigInt(i)) for i in ℓ:n]
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
                if P[i, j] == 1
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
            bin_coeff = binomial(BigInt(k - 1), BigInt(i))
            sum_term += bin_coeff * (p^i) * (q^(k - 1 - i)) * v[k-1-i+1]
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
        p = P[1, 1]
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
                        ((1 + P[j, k]) / α_i - 1) / (1 - q[i, k] * (P[j, k] + q[j, k] * q[i, j])),
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

function sample_process(input, P::Matrix{Float64}, x::Vector{Float64})
    n = length(x)
    if n == 0 || length(input) == 0
        return ([], zeros(0, 0), NaN)
    end
    # Sample initial index using probability vector x
    r = rand()
    total = 0.0
    k = 0
    for i in 1:(n+1)
        total += x[i]
        # println(r)
        if r < total
            k = i
            break
        end
    end

    if k == n + 1
        return (input, P, NaN)
    else
        # Get the k-th row probabilities
        row_probs = P[k, input]

        # Determine which columns to keep (Bernoulli trial failed)
        keep_columns = [rand() ≥ p for p in row_probs]
        remaining = findall(keep_columns)
        
        remaining = input[remaining]
        filter!(x -> x ≠ k, remaining)
        # Create principal submatrix
        ret_P = ones(n,n)
        ret_P[remaining, remaining] = P[remaining, remaining]
        return (remaining, ret_P, k)
    end
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

n = 50
# p = 0.1
# P = (ones(n, n) - I) .* p
Random.seed!(1234)
probs = rand(n * (n - 1) ÷ 2).* (0.6-0.4) .+ 0.4
probs = rand(n * (n - 1) ÷ 2)
P = fill_symmetric_matrix(probs, n)
min_p = minimum(P+10*I)
max_p = maximum(P)
opt_ratio = compute_v(n, max_p)[end] / compute_v(n, min_p)[end]


# P= [0 0 0 0.1; 0 0 0 0.2; 0 0 0 0.3; 0.1 0.2 0.3 0]

# display(P)
#=
X = [x_1^1, x_1^2,..., x_1^n; 
     x_2^1, x_2^2,..., x_2^n; 
                  ...; 
     x_n^1, x_n^2,..., x_n^n]
=#
X, obj_value = solve_lp(P)
function alg(X, P, n)
    selected = zeros(n)
    remain = collect(1:n)
    for i in 1:n
        gone = vec(setdiff(collect(1:(n+1)), remain))
        x = X[:, i]
        x = vcat(x, 1 - sum(x))
        x[gone] .= 0
        if sum(x) != 0
            x = x ./ sum(x)
            remain, _, choice = sample_process(remain, P, x)
        else
            choice = NaN
        end  
        
        if isnan(choice)
            break
        else
            selected[choice] = 1
        end
    end
    return selected
end

function alg2(P, n)
    selected = zeros(n)
    remain = collect(1:n)
    for i in 1:n
        if length(remain) == 0
            break
        end
        # gone = vec(setdiff(collect(1:(n+1)), remain))
        P_remain = P[remain,remain]
        X,_ = solve_lp(P_remain)
        x = zeros(n)
        x[remain] = X[:, 1]
        println(x)
        x = vcat(x, 0)
        if sum(x) != 0
            x = x ./ sum(x)
            remain, _, choice = sample_process(remain, P, x)
        else
            choice = NaN
        end  
        println("\n $choice \n")
        println("\n $remain \n")
        if isnan(choice)
            break
        else
            selected[choice] = 1
        end
    end
    return selected
end


function alg3(P, n)
    selected = zeros(n)
    remain = collect(1:n)
    for i in 1:n
        if length(remain) == 0
            break
        end
        # gone = vec(setdiff(collect(1:(n+1)), remain))
        P_remain = P[remain,remain]
        X,_ = solve_lp(P_remain)
        x = zeros(n)
        x[remain] .= 1/length(remain)
        x = vcat(x, 1 - sum(x))
        if sum(x) != 0
            x = x ./ sum(x)
            remain, _, choice = sample_process(remain, P, x)
        else
            choice = NaN
        end  
        println("\n $choice \n")
        println("\n $remain \n")
        if isnan(choice)
            break
        else
            selected[choice] = 1
        end
    end
    return selected
end


function sim_alg(num, X, P, n)
    avg = 0 
        for i in 1:num
            selected = alg(X, P, n)
            avg += sum(selected)
        end
    return avg/num
end
function sim_alg2(num, P, n)
    avg = 0 
        for i in 1:num
            selected = alg2(P, n)
            avg += sum(selected)
        end
    return avg/num
end
function sim_alg3(num, P, n)
    avg = 0 
        for i in 1:num
            selected = alg3(P, n)
            avg += sum(selected)
        end
    return avg/num
end
# selected = alg(X, P, n)
# alg_avg = sim_alg(3000, X, P, n)
# println("alg2")
# selected = alg2(P, n)
# display(selected)
alg_avg2 = sim_alg2(1000, P, n)
alg_avg3 = sim_alg3(1000, P, n)

# display(selected)
# solution4, obj_value4 = solve_sdp(P)
# solution2, obj_value2 = solve_submod_LP(n, P, false)
# solution3, obj_value3 = solve_kevin_lp(n, p)
# Parameters

println("Simulation")
stability_numbers = 0
# Run simulation
simulations = 1000  # Number of simulations for statistical significance
@suppress begin
    global stability_numbers = exact_stability_simulation(n, P; num_sims=simulations)
end


# # z_values, obj_value3 = solve_submod_LP(n, p)
# # println("Optimal z values: $z_values")

println("LP Solution:\n")
solution, obj_value = solve_lp(P)
display(solution)
# alg_avg = sim_alg(1000, solution, P, n)
display(sum(solution, dims=2))
display(sum(solution, dims=1))

# println("SDP Solution:\n")
# display(solution4)

# v = diag(solution4)
# display(v)
# pop!(v)
# display(sum.(eachcol(reshape(v, n, :))))
# display(sum(diag(solution4), dims=1))
println("LP Objective Value: ", obj_value)

# println("SDP Objective Value: ", obj_value4)
# println("Submodular LP Objective Value: ", obj_value2)
# println("Kevin LP Objective Value: ", obj_value3)
println("Optimal value for G(n,p): ", compute_v(n, p)[end])
println("Optimal ratio: ", opt_ratio)
# println("Selected vertices: ", selected)
# println("Alg Value: ", alg_avg)
println("Alg 2 Avg: ", alg_avg2)
println("Alg Uniform choose Avg: ", alg_avg3)
# println("Alg Ratio: ", alg_avg/compute_v(n, min_p)[end])
println("Alg2 Ratio: ", alg_avg2/compute_v(n, min_p)[end])
println("LP Ratio: ", obj_value/compute_v(n, min_p)[end])
println("Simulated Stability Number: $(stability_numbers["mean"]) ± $(stability_numbers["std"])")
# display(P)
# display(sum(P,dims=2))

# n=3

# # x = zeros(Rational{Int},n,n)
x = zeros(n,n)
for i in 2:n
    for t in 2:n
        if t <= i
            x[i, t] = gnp_prob_sum(i, t, 1-p)
        else 
            x[i, t] = 0
        end
    end
end
# x[1,1]=1
# println("Exact formulation for G(n,p): ", sum(x))
# # println("Extreme Points: " )
# display(x)
# X = solution4
# function vec2mat(i, t, n = 8)
#     return (t - 1) * n + i
# end
for i in 1:n
    for j in (i+1):n
        println(sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in (t+1):n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in (t+1):n))
        #//TODO: check if this is correct
    end
end
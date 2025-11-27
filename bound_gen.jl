### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 05cd9dab-2719-4daf-b088-61c7fae40119


# ╔═╡ a3b980d5-e45c-44ab-9d4f-47b760f042fa
compute_v(n, min_p)[end]

# ╔═╡ 0dca4d73-bdbb-4615-b01b-69f50907f7ea
obj

# ╔═╡ dedb9435-3f7f-488e-b8f1-fa640e3bcb43
large_LP_avg = sim_alg_itersolve_large_LP(1000, P, n)

# ╔═╡ 60640e36-881a-40f2-ac6f-7ef1cc5bece5
large_LP_avg

# ╔═╡ 21557e7f-8e10-4a88-8efe-3a71007fda1f
x[1:12]

# ╔═╡ c62d2ba4-93e1-4030-b48a-9d3da63bdaf6
begin
    x_LP,obj_LP=solve_lp(P)
    x_LP
    obj_LP
end

# ╔═╡ 1bc70a94-6438-40fe-988a-41857d129f26
x[1:12]

# ╔═╡ 9ec02064-a9d9-40a2-8172-b60c4691ab31
begin
    println("LP Solution:\n")
    solution, obj_value = solve_lp(P)
    display(solution)
    # alg_avg = sim_alg(1000, solution, P, n)
    display(sum(solution, dims=2))
    display(sum(solution, dims=1))
    println("\nLP Objective Value: ", obj_value)

end

# ╔═╡ ff8c5b26-e86a-4070-930e-df13bb8b961f
begin
    alg_avg3 = sim_alg3(100, P, n)
    println("Alg Uniform choose Avg: ", alg_avg3)
end

# ╔═╡ 93f3bb92-dd46-469e-8ba9-4ddd7bc98167
diag(X)[1:12]

# ╔═╡ 3b90f02b-fabc-4f04-ae46-7a9f84fcef87
begin
    println("Alg Solve n by n SDP Once Avg: ", avg_n_by_n)
    println("Alg Solve 3n PSD Avg: ", alg_avg_3n_PSD)
    println("Alg Solve 3n PSD Avg Once: ", alg_avg_3n_PSD_once)
end

# ╔═╡ 350f8e68-fa5b-480b-ac2c-0b9931e0b982
begin
    # # println("Simulation")
    # stability_numbers = 0
    # Run simulation
    simulations = 500  # Number of simulations for statistical significance
    @suppress begin
        global stability_numbers = exact_stability_simulation(n, P; num_sims=simulations)
    end
    println("Simulated Stability Number: $(stability_numbers["mean"]) ± $(stability_numbers["std"])")
end

# ╔═╡ 11301ab8-b686-4121-8c30-25947223891f
md"""
println(\"Simulated Stability Number: $(stability_numbers[\"mean\"]) ± $(stability_numbers[\"std\"])\")"""

# ╔═╡ 718644eb-208e-4d33-babd-6b36906c9757
begin
    alg_avg_3n_PSD = sim_alg_3n_PSD(100, P, n)
    
    X,_ = solve_SDP_3n_PSD(P,n)
    
    alg_avg_3n_PSD = sim_alg_3n_PSD(100, P, n)
    
    X,_ = solve_SDP_3n_PSD(P,n)
    alg_avg_3n_PSD_once = sim_alg_3n_PSD_once(100, P, n,X)
    
    
    # X,_ = solve_sdp(P)
    # X = reshape(diag(X)[1:end-1],n,n)
    # avg5 = sim_alg_sdp_once(100, P, n, X)
    
    X,_ = solve_SDP_n_by_n(P,n)
    avg_n_by_n = sim_alg_n_by_n_sdp(100, X, P, n)
    alg_avg4 = sim_alg4(100, P, n)
    
    println("Alg Solve SDP Iterativly Avg: ", alg_avg4)
    # println("Alg Solve SDP Once Avg: ", avg5)
    println("Alg Solve n by n SDP Once Avg: ", avg_n_by_n)
    println("Alg Solve 3n PSD Avg: ", alg_avg_3n_PSD)
    println("Alg Solve 3n PSD Avg Once: ", alg_avg_3n_PSD_once)
end

# ╔═╡ ced4e571-c6cf-45ae-8413-afbeaca04db1
begin
    function solve_large_LP(P, n=size(P,1), w=ones(n))
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
        
        if n == 1
            return [1], w[1]
        end
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "Method", 2)
        @variable(model, x[1:n^2]>=0)
        # @variable(model, z[1:n^2,1:n]>=0)
    
        m = n^2
        @variable(model, z[i=1:m, j=1:m; i <= j])
        @constraint(model, [i=1:n,t=1:n], z[vec2mat(i,t),vec2mat(i,t)] == x[vec2mat(i,t)])
        @constraint(model, [i=1:n], x[i] == sum(z[vec2mat(i,1),vec2mat(j,2)] for j in 1:n if j!=i)+prod(P[i,j] for j in 1:n if j!=i)*x[i])
    
        @constraint(model, [i=1:n, t=2:n, tau = 1:(t-1)], x[vec2mat(i,t)] == sum(z[vec2mat(j,tau),vec2mat(i,t)] for j in 1:n if j!=i))
        @constraint(model, [i=1:n, t=1:(n-1), tau = (t+1):n], x[vec2mat(i,t)] >= sum(z[vec2mat(i,t),vec2mat(j,tau)] for j in 1:n if j!=i))
        @constraint(model, sum(x[1:n]) == 1)
        @constraint(model, [t=2:n], sum(x[(t-1)*n.+1:n]) <= 1)
        @constraint(model, [i=1:n], sum(x[vec2mat(i,t)] for t in 1:n) <= 1)
    
        @constraint(model,[i=1:n,t=1:(n-1), j = 1:n], z[vec2mat(i,t),vec2mat(j,t+1)]<=x[vec2mat(i,t)]*(1-P[i,j]))
        sorted_products = Matrix{Vector{Float64}}(undef, n, n)
        for i in 1:n
            for j in 1:n
                # Collect all k ≠ i and k ≠ j
                ks = [k for k in 1:n if k ∉ (i, j)]
                # Compute products (1 - P[j,k])(1 - P[i,k]) for each valid k
                products = [(1 - P[j,k]) * (1 - P[i,k]) for k in ks]
                # Sort the products (ascending order; use `rev=true` for descending)
                sort!(products,rev=true)
                # Store sorted products for (i,j)
                sorted_products[i,j] = products
            end
        end
        @constraint(model, [i=1:n,j=1:n,t=1:(n-2), tau = (t+1):n], z[vec2mat(i,t),vec2mat(j,tau)]<=x[vec2mat(i,t)] *prod(sorted_products[i,j][1:(tau-t-1)])*(1-P[i,j]))
    
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
    
        # Sum of x[i, t] over all t is at most 1 for all i
        # Sum of x[i,t] from t to n is bounded
        for i in 1:n
            for t in 1:n
                if t > 1
                    # Compute ∏_{j ∈ I_i^t} (1 - p_ij)
                    prod_term = prod(1 .- P[i, I[(i, t)]])
                    prod_term_J = prod(1 .- P[i, J[(i, t)]])
                    @constraint(model, sum(x[(τ-1)*n+i] for τ in t:n) <= (1 - sum(x[(τ-1)+i] for τ in 1:(t-1))) * prod_term)
                end
            end
        end
    
        for t in 2:n
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
            
                # \sum_{i=1}^{n}x_i^t \leq \max_{H\subseteq N,|H|=t-1}\left[1-\left(\prod_{j\notin H}\left(1-\prod_{k\in H}(1-p_{jk})\right)\right)\right]\prod_{j\neq k\in H}(1-p_{jk}),\quad \forall t\geq 2\\
                # Probability a vertex is chosen at $t$ is bounded by the max probability a set of vertices is chosen during t-1 and there is at least one vertex remain
                @constraint(model, sum(x[(t-1)*n+i] for i in 1:n) <= max_rhs)
    
                products = [prod(P[i, [1:i-1; i+1:end]]) for i in 1:n]
                @constraint(model, sum(x[(t-1)*n+i] for i in 1:n) <= (1-minimum(products))*sum(x[(t-2)*n+i] for i in 1:n))
                if t == 2
                    @constraint(model, sum(x[(t-1)*n+i] for i in 1:n) <= sum(x[(t-2)*n+i] for i in 1:n))
                end
        end
        @objective(model, Max, dot(x,repeat(w, n)))
    
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
    
    
    function alg_itersolve_large_LP(P, n = size(P,1), w = ones(n))
        selected = zeros(n)
        remain = collect(1:n)
        for i in 1:n
            if length(remain) == 0
                break
            end
            # gone = vec(setdiff(collect(1:(n+1)), remain))
            P_remain = P[remain,remain]
            X,_ = solve_large_LP(P_remain, length(remain), w[remain])
            println(X)
            x = zeros(n)
            x[remain] = X[1:length(remain)]
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
    
    
    function sim_alg_itersolve_large_LP(num, P, n=size(P,1), w = ones(n))
        avg = 0 
            for i in 1:num
                selected = alg_itersolve_large_LP(P, n, w)
                avg += sum(selected)
            end
        return avg/num
    end
end

# ╔═╡ b98a3f21-fb4d-40e8-9f27-fc70eb367bd1
begin
    x, obj =solve_large_LP(P)
    
    println(P)
    println(obj)
    println(x)
    println(compute_v(n, min_p)[end])
end

# ╔═╡ a73b3637-28c4-47e5-b861-f744b2bc7567
begin
    using Statistics
    using Random, Graphs
    using Combinatorics, LinearAlgebra
    using JuMP, COPT, Ipopt, Gurobi, SCS, Mosek, MosekTools
    using Suppressor, StatsBase, SparseArrays
    
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
    
    function solve_lp(P, bound = Inf, max_p = 1, min_p = 0)
        n = size(P, 1)  # Number of rows/columns in P
        bound = min(bound, n)  # Ensure bound is at most n
        # avg_nb = 2*sum(tril(P,-1))/n/(n-1)
        # exp_values = log(1-avg_nb-1/n,1/n)
        #model = Model(Mosek.Optimizer)  # Create model with HiGHS solver
        model = Model(Gurobi.Optimizer)  # Create model with HiGHS solver
    
        # Variables: x[i, t] for i ∈ 1:n, t ∈ 1:n
        @variable(model, x[1:n, 1:n] >= 0)
        # if isnan(avg_nb) != true
        #     @constraint(model, sum(x) >= max(avg_nb,0))
        # end
        @constraint(model, sum(x) <= bound)
        @constraint(model, sum(x) <= compute_v(n, min_p)[end])
        @constraint(model, sum(x) >= compute_v(n, max_p)[end])
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
            # if t <= ceil(exp_values)
            #     # @constraint(model, sum(x[:, t]) == 1)
            #     @constraint(model, sum(x[:, t]) <= 1) 
            # else
                # @constraint(model, sum(x[:, t]) <= 1)
            # end
            if t >= 2
                @constraint(model, sum(x[:, t]) <= sum(x[:, t-1]))
            else
                @constraint(model, sum(x[:, t]) == 1)
            end
            if t == 2
                products = [prod(P[i, [1:i-1; i+1:end]]) for i in 1:n]
                @constraint(model, sum(x[:, t]) <= (1-minimum(products))*sum(x[:, t-1]))
            end
            
        end
    
        # Sum of x[i, t] over all t is at most 1 for all i
        # Sum of x[i,t] from t to n is bounded
        for i in 1:n
            # γ_i = prod((1-P[i, j]) for j in 1:n if j != i)
            @constraint(model, sum(x[i, :]) <= 1)
    
            for t in 1:n
                if t > 1
                    # Compute ∏_{j ∈ I_i^t} (1 - p_ij)
                    prod_term = prod(1 .- P[i, I[(i, t)]])
                    prod_term_J = prod(1 .- P[i, J[(i, t)]])
                    @constraint(model, sum(x[i, t:n]) <= (1 - sum(x[i, 1:t-1])) * prod_term)
                else
                    # Add constraint: x[i, 1] ≤ 1
                    # @constraint(model, x[i, 1] == 1/n)
                end
            end
        end
    
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
    function solve_sdp(P, w = ones(size(P,1)))
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
                H_set = Set(H)
    
                # Compute the double product
                prod_val = prod(P[i, k] for k in H_set)
    
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
        model = Model(COPT.ConeOptimizer)  # Create model with HiGHS solver
        # model = Model(Mosek.Optimizer)
        if n == 1
            return reshape([1],1,1), 1
        end 
        # Variables: x[i, t] for i ∈ 1:n, t ∈ 1:n
        @variable(model, X[1:(m+1), 1:(m+1)], PSD)
        @constraint(model, X .>= 0)
        @constraint(model, X[m+1, m+1] == 1)
        @constraint(model, [i = 1:m, j = 1:m; i ≠ j], X[i, i] >= X[i, j])
        for i in 1:m
            @constraint(model, X[i, i] == X[m+1, i])
        end
        for i in 1:n
            for j in 1:m
                @constraint(model, prod(Q[mat2vec(j)[1], l] for l in 1:n if l != mat2vec(j)[1]) >= X[vec2mat(i, n), j])
                @constraint(model, X[i, j] == X[j, i])
            end
        end
    
        #=
            For the entries corresponding to each variable, we have the following mapping:
            for x_i^t, it is the entry X[(t-1)n+i, (t-1)n+i]; for the interaction entry x_ij^tτ, it is the entry X[(t-1)n+i, (τ-1)n+j].
        =#
    
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
                for τ in t+1:n
                    @constraint(model, X[j, j] >= sum(X[j, vec2mat(l, τ)] for l in 1:n if l != i))
                end
                # if t <= n - 1
                #     @constraint(model, X[j, j] <= sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i) + max_probability(P, i, n - t - floor(Int64, avg * t)))
                #     # println("prob bound: $(max_probability(P,i,n-t-floor(Int64,avg*t)))")
                # end
            elseif t == 1
                @constraint(model, (1 - prod(P[i, l] for l in 1:n if l != i)) * X[j, j] == sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i))
            end
        end
        # add constraints $\sum_{i=1}^n x_i^t \leq 1$ for all $t$
        @constraint(model, sum(diag(X)[vec2mat(1, 1):vec2mat(n, 1)]) == 1)
        for t in 1:n
            @constraint(model, sum(diag(X)[vec2mat(1, t):vec2mat(n, t)]) <= 1)
            # Consider a lower bound on x[:,t] for each t, when the graph is G(n,p)
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
            # add constraints $\sum_{t=1}^n x_i^t \leq 1$ for all $i$
            @constraint(model, sum(diag(X)[i:n:m]) <= 1)
            for t in 1:n
                if t > 1
                    # Compute ∏_{j ∈ I_i^t} (1 - p_ij)
                    prod_term = prod(1 .- P[i, I[(i, t)]])
                    prod_term_J = prod(1 .- P[i, J[(i, t)]])
                    @constraint(model, sum(diag(X)[vec2mat(i, t):n:vec2mat(i, n)]) <= (1 - sum(diag(X)[vec2mat(i, 1):n:vec2mat(i, t - 1)])) * prod_term)
                else
                    # Add constraint: x[i, 1] ≤ 1
                    @constraint(model, X[vec2mat(i, 1), vec2mat(i, 1)] <= 1)
                end
            end
        end
    
        for i in 1:n
            for j in (i+1):n
                @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in t+1:n) <= sum(interaction_prod(P, i, j, t) * sum(diag(X)[i:n:vec2mat(i, t)]) for t in 2:n-1) + X[i, i])
    
                #TODO: check if this is correct
                @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in (t+1):n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in (t+1):n) <= 1 - P[i, j])
    
                # @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in t+1:n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in t+1:n) >= prod(Q[i, l] for l in 1:n if l != i) * prod(Q[j, l] for l in 1:n if l != j && l != i))
    
    
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
        @objective(model, Max, dot(diag(X)[1:end-1],repeat(w, n)))
    
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
                H_set = Set(H)
    
                # Compute the double product
                prod_val = prod(P[i, k] for k in H_set)
    
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
        model = Model(COPT.ConeOptimizer)  # Create model with HiGHS solver
    
        # Variables: x[i, t] for i ∈ 1:n, t ∈ 1:n
        @variable(model, X[1:(m+1), 1:(m+1)], PSD)
        @constraint(model, X .>= 0)
        @constraint(model, X[m+1, m+1] == 1)
        @constraint(model, [i = 1:m, j = 1:m; i ≠ j], X[i, i] >= X[i, j])
        for i in 1:m
            @constraint(model, X[i, i] == X[m+1, i])
        end
        for i in 1:n
            for j in 1:m
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
                    @constraint(model, X[j, j] <= sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i) + max_probability(P, i, n - t - floor(Int64, avg * t)))
                    println("prob bound: $(max_probability(P,i,n-t-floor(Int64,avg*t)))")
                end
            elseif t == 1
                @constraint(model, (1 - prod(P[i, l] for l in 1:n if l != i)) * X[j, j] == sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i))
            end
        end
    
        # add constraints $\sum_{i=1}^n x_i^t \leq 1$ for all $t$
        @constraint(model, sum(diag(X)[vec2mat(1, 1):vec2mat(n, 1)]) == 1)
        for t in 1:n
            @constraint(model, sum(diag(X)[vec2mat(1, t):vec2mat(n, t)]) <= 1)
            # Consider a lower bound on x[:,t] for each t, when the graph is G(n,p)
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
            end
        end
    
        # for k in 1:m
        #     for l in (k+1):m
        #         i = mat2vec(k)[1]
        #         t = mat2vec(k)[2]
        #         j = mat2vec(l)[1]
        #         τ = mat2vec(l)[2]
        #         if i != j && t < τ
        #             max_rhs = -Inf
        #             for H in combinations(setdiff(collect(1:n), [i, j]), τ - t - 1)
        #                 # Compute the inner product for each i in N \ H:
        #                 candidate = 1.0
        #                 for v in H
        #                     candidate *= (1 - P[i, v]) * (1 - P[j, v])
        #                 end
        #                 max_rhs = max(max_rhs, candidate)
        #             end
        #             @constraint(model, X[k, l] <= max_rhs * X[k, k])
        #         end
        #     end
        # end
        # Constraints
        for i in 1:n
    
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
                else
                    # Add constraint: x[i, 1] ≤ 1
                    @constraint(model, X[vec2mat(i, 1), vec2mat(i, 1)] <= 1)
                end
            end
        end
    
        # for i in 1:n
        #     for j in (i+1):n
        #         @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in t+1:n) <= sum(interaction_prod(P, i, j, t) * sum(diag(X)[i:n:vec2mat(i, t)]) for t in 2:n-1) + X[i, i])
    
        #         #TODO: check if this is correct
        #         @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in (t+1):n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in (t+1):n) <= 1 - P[i, j])
    
        #         @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in t+1:n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in t+1:n) >= prod(Q[i, l] for l in 1:n if l != i) * prod(Q[j, l] for l in 1:n if l != j && l != i))
        #     end
        # end
    
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
    
    
    
    function solve_sdp_simple(P)
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
    
        n = size(P, 1)  # Number of rows/columns in P
        i0 = n^2+1
        rows = vcat([fill(i, n) for i in 1:(n^2-n)]...)
        cols = [] 
        for i in 1:(n^2)
            t = floor(Int, (i-1)/n) + 1
            if t < n 
                cols = vcat(cols, t*n .+ collect(1:n))
            end
        end
        # println(cols)
        model = Model(COPT.ConeOptimizer)
        @variable(model, x[1:n^2])
        @variable(model, X[1:n^2*(n-1)])  # off-diagonal entries
        # X_plus = Symmetric(sparse(rows, cols, X, i0, i0) + sparse(collect(1:n^2), fill(i0, n^2), x, i0, i0)) + Diagonal([x; 1])
        X_plus = Symmetric(sparse(rows, cols, X, n^2, n^2)) + sparse(Diagonal(x))
        @constraint(model, X_plus in PSDCone())
        @objective(model, Max, sum(x))
        @constraint(model, X .>= 0)
    
        # Precompute I_i^t for all i, t
        # I = Dict{Tuple{Int,Int},Vector{Int}}()  # I_i^t mapping
        # J = Dict{Tuple{Int,Int},Vector{Int}}()  # I_i^t mapping
        # for i in 1:n
        #     for t in 1:n
        #         if t > 1
        #             # Extract off-diagonal entries for row i
        #             row_off_diag = collect(1:n)
        #             deleteat!(row_off_diag, i)  # Remove diagonal index
        #             sorted_indices = sort(row_off_diag, by=j -> P[i, j])
        #             I[(i, t)] = sorted_indices[1:(t-1)]  # Take top t-1 indices
        #             J[(i, t)] = sorted_indices[end-t+2:end]  # Take last t-1 indices
        #         else
        #             I[(i, t)] = []
        #             J[(i, t)] = []
        #         end
        #     end
        # end
    
        for i in 1:n^2
            for j in i:n^2
                v1 = mat2vec(i)
                v2 = mat2vec(j)
                if v1[2] == v2[2] && v1[1] != v1[1]
                    @constraint(model, X_plus[i, j] == 0)
                end
            end 
        end
    
    
    
        for i in 1:n
            @constraint(model,x[vec2mat(i, 1)] == sum(X[((i-1)*n+1):i*n]) + prod(P[i,j] for j in 1:n if j != i))
            @constraint(model,sum(x[((i-1)*n+1):(i*n)]) <= 1)
            if i > 1
                @constraint(model,sum(x[(i-1)*n+1:i*n]) <= 1)
                @constraint(model,sum(x[(i-1)*n+1:i*n]) <= sum(x[(i-2)*n+1:(i-1)*n]))
            else
                @constraint(model,sum(x[(i-1)*n+1:i*n]) == 1)
            end
            for t in 2:n 
                # prod_term = prod(1 .- P[i, I[(i, t)]])
                j = vec2mat(i, t)
                # @constraint(model, sum(x[vec2mat(i, t):n:vec2mat(i, n)]) <= (1 - sum(x[vec2mat(i, 1):n:vec2mat(i, t - 1)])) * prod_term)
                @constraint(model, x[j] == sum(X[(t-2)*n.+collect(i:n:(i*n))]))
            end 
        end 
        optimize!(model)
    
        X_val = Symmetric(value.(X_plus))
        obj_val = objective_value(model)
    
      return (X=X_val, value=obj_val)
    
    end
    
    function solve_sdp_simple2(P)
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
    
        n = size(P, 1)  # Number of rows/columns in P
        i0 = 2*n+1
        rows = vcat([fill(i, n) for i in 1:(n)]...)
        cols = [] 
        for i in 1:n
            t = floor(Int, (i-1)/n) + 1
            if t < n 
                cols = vcat(cols, t*n .+ collect(1:n))
            end
        end
        # println(cols)
        model = Model(COPT.ConeOptimizer)
        @variable(model, x[1:2*n])
        @variable(model, X[1:n^2])  # off-diagonal entries
        X_plus = Symmetric(sparse(rows, cols, X, i0, i0) + sparse(collect(1:2*n), fill(i0, 2*n), x, i0, i0)) + Diagonal([x; 1])
        @constraint(model, X_plus in PSDCone())
        @objective(model, Max, sum(x))
        @constraint(model, X .>= 0)
    
        # Precompute I_i^t for all i, t
        # I = Dict{Tuple{Int,Int},Vector{Int}}()  # I_i^t mapping
        # J = Dict{Tuple{Int,Int},Vector{Int}}()  # I_i^t mapping
        # for i in 1:n
        #     for t in 1:n
        #         if t > 1
        #             # Extract off-diagonal entries for row i
        #             row_off_diag = collect(1:n)
        #             deleteat!(row_off_diag, i)  # Remove diagonal index
        #             sorted_indices = sort(row_off_diag, by=j -> P[i, j])
        #             I[(i, t)] = sorted_indices[1:(t-1)]  # Take top t-1 indices
        #             J[(i, t)] = sorted_indices[end-t+2:end]  # Take last t-1 indices
        #         else
        #             I[(i, t)] = []
        #             J[(i, t)] = []
        #         end
        #     end
        # end
    
        for i in 1:2*n
            for j in i:2*n
                v1 = mat2vec(i)
                v2 = mat2vec(j)
                if v1[2] == v2[2] && v1[1] != v1[1]
                    @constraint(model, X_plus[i, j] == 0)
                end
            end 
        end
    
    
    
        for i in 1:n
            @constraint(model,x[vec2mat(i, 1)] == sum(X[(i-1)*n+1:i*n]) + prod(P[i,j] for j in 1:n if j != i))
            # @constraint(model,sum(x[((i-1)*n+1):(i*n)]) <= 1)
            if i > 1
                @constraint(model,sum(x[i:n:2*n]) <= 1)
            else
                @constraint(model,sum(x[i:n:2*n]) == 1)
            end
            for t in 2:2
                # prod_term = prod(1 .- P[i, I[(i, t)]])
                j = vec2mat(i, t)
                # @constraint(model, sum(x[vec2mat(i, t):n:vec2mat(i, n)]) <= (1 - sum(x[vec2mat(i, 1):n:vec2mat(i, t - 1)])) * prod_term)
                @constraint(model, x[j] == sum(X[(t-2)*n.+collect(i:n:(i*n))]))
            end 
        end 
        optimize!(model)
    
        X_val = Symmetric(value.(X_plus))
        obj_val = objective_value(model)
    
      return (X=X_val, value=obj_val)
    
    end
    
    
    function solve_SDP_n_by_n(P,n)
        Q = 1 .- P
    
        n = size(P, 1) # Number of rows/columns in P
        model = Model(COPT.ConeOptimizer)  # Create model with HiGHS solver
        # Variables: x[i, t] for i ∈ 1:n, t ∈ 1:n
        @variable(model, X[1:(n+1), 1:(n+1)], PSD)
        @constraint(model, X .>= 0)
        @constraint(model, X[n+1, n+1] == 1)
        for i in 1:n
            @constraint(model, X[i, i] == X[n+1, i])
        end
        for i in 1:n
            for j in i+1:n
                @constraint(model, X[i, j] <= Q[i,j])
                @constraint(model, X[i, j] == X[j, i])
            end
        end
    
    
        @objective(model, Max, tr(X) - 1)
        optimize!(model)
        # Extract results
        if termination_status(model) == MOI.OPTIMAL
            solution = value.(X)
            return solution, objective_value(model)
        else
            error("Optimization did not converge to an optimal solution.")
        end
    end
    
    
    function solve_SDP_3n_PSD(P,n)
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
                H_set = Set(H)
    
                # Compute the double product
                prod_val = prod(P[i, k] for k in H_set)
    
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
        # model = Model(optimizer_with_attributes(COPT.ConeOptimizer,
        # "AbsGap" => 1e-7,    # Absolute tolerance (default: 1e-5)
        # "RelGap" => 1e-7,    # Relative tolerance (default: 1e-5)
        # "FeasTol" => 1e-7,
        # "Scaling" => 1
        # ))  # Create model with HiGHS solver
        model = Model(Mosek.Optimizer)
        @variable(model, X[1:(m+1), 1:(m+1)])
        if n == 1
            return reshape([1],1,1), 1
        end 
        # Variables: x[i, t] for i ∈ 1:n, t ∈ 1:n
        submatrices = []
        for i in 1:n-2
            # push!(submatrices, vcat((i-1)*n+1:(i+2)*n, m+1))
            push!(submatrices, vcat((i-1)*n+1:(i+2)*n))
        end
        for idx in submatrices
            @constraint(model, X[idx, idx] in PSDCone())
        end
    
        @constraint(model, X .>= 0)
        @constraint(model, X[m+1, m+1] == 1)
        @constraint(model, [i = 1:m, j = 1:m; i ≠ j], X[i, i] >= X[i, j])
        for i in 1:m
            @constraint(model, X[i, i] == X[m+1, i])
            for j in 1:m 
                @constraint(model, X[i,j] == X[j,i])
            end
        end
    
        for i in 1:n
            for j in 1:m
                # X_{(i,n),(j,t)}\leq \prod_{k\neq j}1-p_{jk},\quad\forall i,j,t
                @constraint(model, prod(Q[mat2vec(j)[1], l] for l in 1:n if l != mat2vec(j)[1]) >= X[vec2mat(i, n), j])
                # @constraint(model, X[i, j] == X[j, i])
    
                # X_{(i,t),(i,\tau)} = 0,\quad\forall t\neq \tau
                # X_{(i,t),(j,t)} = 0,\quad\forall i\neq j
                row = mat2vec(i)
                col = mat2vec(j)
                if row[1] == col[1] && row[2] != col[2]
                    @constraint(model, X[i, j] == 0)
                elseif row[1] != col[1] && row[2] == col[2]
                    @constraint(model, X[i, j] == 0)
                end
            end
        end
    
        #=
            For the entries corresponding to each variable, we have the following mapping:
            for x_i^t, it is the entry X[(t-1)n+i, (t-1)n+i]; for the interaction entry x_ij^tτ, it is the entry X[(t-1)n+i, (τ-1)n+j].
        =#
    
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
            @constraint(model, sum(diag(X)[i:n:m]) <= 1)
            for t in 1:n
                if t > 1
                    # Compute ∏_{j ∈ I_i^t} (1 - p_ij)
                    prod_term = prod(1 .- P[i, I[(i, t)]])
                    prod_term_J = prod(1 .- P[i, J[(i, t)]])
    
                    # \sum_{\tau=t}^{n}x_i^\tau\leq \max_{H\subseteq N\setminus\{i\},|H|=t-1}\prod_{j\in H}(1-p_{ij})\left(1-\sum_{\tau=1}^{t-1}x_i^\tau\right),\quad \forall i,t\\
                    @constraint(model, sum(diag(X)[vec2mat(i, t):n:vec2mat(i, n)]) <= (1 - sum(diag(X)[vec2mat(i, 1):n:vec2mat(i, t - 1)])) * prod_term)
                else
                    # Add constraint: x[i, 1] ≤ 1
                    @constraint(model, X[vec2mat(i, 1), vec2mat(i, 1)] <= 1)
                end
            end
        end
        avg = (sum(P) - sum(diag(P))) / (n * (n - 1))
        for k in 1:m
            i = mat2vec(k)[1]
            t = mat2vec(k)[2]
            if t > 1
                # x_{i}^{t} = \sum_{j\neq i}X_{(i,t),(j,\tau)},\quad\forall \tau<t\\
                for τ in 1:(t-1)
                    @constraint(model, X[k, k] == sum(X[k, vec2mat(l, τ)] for l in 1:n if l != i))
                end
                # x_{i}^{t} \geq \sum_{j\neq i}X_{(i,t),(j,\tau)},\quad\forall \tau>t\\
                for τ in t+1:n
                    @constraint(model, X[k, k] >= sum(X[k, vec2mat(l, τ)] for l in 1:n if l != i))
                end
                # if t <= n - 1
                #     @constraint(model, X[j, j] <= sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i) + max_probability(P, i, n - t - floor(Int64, avg * t)))
                #     # println("prob bound: $(max_probability(P,i,n-t-floor(Int64,avg*t)))")
                # end
            elseif t == 1
                # x_i^1 = \sum_{j\neq i}X_{(i,1),(j,2)} + \left(\prod_{j\neq i}p_{ij}\right)x_i^1,\quad \forall i
                @constraint(model, (1 - prod(P[i, l] for l in 1:n if l != i)) * X[k, k] == sum(X[k, vec2mat(l, t + 1)] for l in 1:n if l != i))
            end
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
                    # X_{(i,t),(j,\tau)}\leq x_i^t~\max_{H\subseteq H\setminus\{i,j\},|H|=\tau-t-1}\prod_{k\in H}(1-p_{ik})(1-p_{jk}),\quad\forall i,j,t<\tau\\
                    @constraint(model, X[k, l] <= max_rhs * X[k, k])
                end
            end
        end
    
        # add constraints $\sum_{i=1}^n x_i^t \leq 1$ for all $t$
        for t in 1:n
            if t == 1
                # \sum_{i=1}^{n}x_i^1 = 1
                @constraint(model, sum(diag(X)[vec2mat(1, 1):vec2mat(n, 1)]) == 1)
            # @constraint(model, sum(diag(X)[vec2mat(1, t):vec2mat(n, t)]) <= 1)
            # Consider a lower bound on x[:,t] for each t, when the graph is G(n,p)
            else
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
    
                # \sum_{i=1}^{n}x_i^t \leq \max_{H\subseteq N,|H|=t-1}\left[1-\left(\prod_{j\notin H}\left(1-\prod_{k\in H}(1-p_{jk})\right)\right)\right]\prod_{j\neq k\in H}(1-p_{jk}),\quad \forall t\geq 2\\
                # Probability a vertex is chosen at $t$ is bounded by the max probability a set of vertices is chosen during t-1 and there is at least one vertex remain
                @constraint(model, sum(diag(X)[vec2mat(1, t):vec2mat(n, t)]) <= max_rhs)
            end
        end
    
    
        for i in 1:n
            for j in (i+1):n
                # Probability that $i$ is chosen before $j$ is chosen.
                @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in t+1:n) <= sum(interaction_prod(P, i, j, t) * sum(diag(X)[i:n:vec2mat(i, t)]) for t in 2:n-1) + X[i, i])
    
                #TODO: check if this is correct
                # Probability both i and j are chosen = probability i chosen before j + probability j chosen before i
                # @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in (t+1):n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in (t+1):n) <= (1 - P[i, j])*sum(X[i:n:n^2,m+1]))
                # @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in (t+1):n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in (t+1):n) <= (1 - P[i, j])*sum(X[j:n:n^2,m+1]))
                # @constraint(model, sum(X[vec2mat(i, t), vec2mat(j, τ)] for t in 1:n-1 for τ in t+1:n) + sum(X[vec2mat(j, t), vec2mat(i, τ)] for t in 1:n-1 for τ in t+1:n) >= prod(Q[i, l] for l in 1:n if l != i) * prod(Q[j, l] for l in 1:n if l != j && l != i))
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
            error("Optimization problem was not solved.")
        end
    end
    
    # Only enforce the PSD and interaction constraints on the first two stages
    function solve_sdp_lp_mix(P, bound = Inf, max_p = 1, min_p = 0)
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
        function max_probability(P::Matrix{Float64}, i::Int, remain::Int)
            if remain == 0
                return 1
            end
            n = size(P, 1)  # Number of nodes
            N = collect(1:n)  # Node set
    
            max_val = 0.0  # Store the maximum probability
    
            # Generate all subsets H of size t that contain i
            for H in combinations(setdiff(N, [i]), remain)  # Select t-1 other nodes
                H_set = Set(H)
    
                # Compute the double product
                prod_val = prod(P[i, k] for k in H_set)
    
                # Update max_val
                max_val = max(max_val, prod_val)
            end
    
            return max_val
        end
        n = size(P, 1)  # Number of rows/columns in P
        bound = min(bound, n)  # Ensure bound is at most n
        # avg_nb = 2*sum(tril(P,-1))/n/(n-1)
        # exp_values = log(1-avg_nb-1/n,1/n)
        #model = Model(Mosek.Optimizer)  # Create model with HiGHS solver
        # model = Model(Gurobi.Optimizer)  # Create model with HiGHS solver
        model = Model(COPT.ConeOptimizer)  # Create model with HiGHS solver
    
        # Variables: x[i, t] for i ∈ 1:n, t ∈ 1:n
        
        # Variables: x[i, t] for i ∈ 1:n, t ∈ 1:n
        @variable(model, x[1:n, 1:n] >= 0)
        
        if n>= 2
            γ = ceil(Int, sqrt(n))
            m = γ*n
            @variable(model, X[1:m, 1:m], PSD)
            @constraint(model, [i = 1:m, j = 1:m; i ≠ j], X[i, i] >= X[i, j])
            @constraint(model, X .>= 0)
            @constraint(model, diag(X) == vcat(x[:, 1:γ]...))
            for i in 1:m
                for j in 1:m
                    row = mat2vec(i)
                    col = mat2vec(j)
                    if row[1] == col[1] && row[2] != col[2]
                        @constraint(model, X[i, j] == 0)
                    elseif row[1] != col[1] && row[2] == col[2]
                        @constraint(model, X[i, j] == 0)
                    else 
                        # @constraint(model, X[i,j] >= prod(P[row, l] for l in 1:n if l != row && l!=col)*x[row,1])
                    end
                end
            end
            @constraint(model, 
            sum(x[:,2]) == sum((1-prod(P[i, l] for l in 1:n if l != i))*x[i,1] for i in 1:n))
            for j in 1:m
                i = mat2vec(j)[1]
                t = mat2vec(j)[2]
                if t > 1
                    if t<=γ-1
                    @constraint(model, X[j, j] == sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i))
                    end
                    for τ in 1:(t-1)
                        @constraint(model, X[j, j] == sum(X[j, vec2mat(l, τ)] for l in 1:n if l != i))
                    end
                    if t <= n - 1
                        # @constraint(model, X[j, j] <= sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i) + max_probability(P, i, n - t - floor(Int64, avg * t)))
                        # println("prob bound: $(max_probability(P,i,n-t-floor(Int64,avg*t)))")
                    end
                elseif t == 1
                    @constraint(model, (1 - prod(P[i, l] for l in 1:n if l != i)) * X[j, j] == sum(X[j, vec2mat(l, t + 1)] for l in 1:n if l != i))
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
        end
        # if isnan(avg_nb) != true
        #     @constraint(model, sum(x) >= max(avg_nb,0))
        # end
        @constraint(model, sum(x) <= bound)
        @constraint(model, sum(x) <= compute_v(n, min_p)[end])
        @constraint(model, sum(x) >= compute_v(n, max_p)[end])
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
            # if t <= ceil(exp_values)
            #     # @constraint(model, sum(x[:, t]) == 1)
            #     @constraint(model, sum(x[:, t]) <= 1) 
            # else
                # @constraint(model, sum(x[:, t]) <= 1)
            # end
            if t >= 2
                @constraint(model, sum(x[:, t]) <= sum(x[:, t-1]))
            else
                @constraint(model, sum(x[:, t]) == 1)
            end
            if t == 2
                products = [prod(P[i, [1:i-1; i+1:end]]) for i in 1:n]
                @constraint(model, sum(x[:, t]) <= (1-minimum(products))*sum(x[:, t-1]))
            end
            
        end
    
        # Sum of x[i, t] over all t is at most 1 for all i
        # Sum of x[i,t] from t to n is bounded
        for i in 1:n
            # γ_i = prod((1-P[i, j]) for j in 1:n if j != i)
            @constraint(model, sum(x[i, :]) <= 1)
    
            for t in 1:n
                if t > 1
                    # Compute ∏_{j ∈ I_i^t} (1 - p_ij)
                    prod_term = prod(1 .- P[i, I[(i, t)]])
                    prod_term_J = prod(1 .- P[i, J[(i, t)]])
                    @constraint(model, sum(x[i, t:n]) <= (1 - sum(x[i, 1:t-1])) * prod_term)
                else
                    # Add constraint: x[i, 1] ≤ 1
                    # @constraint(model, x[i, 1] == 1/n)
                end
            end
        end
    
        # Objective: Maximize sum of x[i, t] over all i, t
        @objective(model, Max, sum(x))
    
        # Solve the model
        optimize!(model)
    
        # Extract results
        if termination_status(model) == MOI.OPTIMAL
            solution = value.(x)
            if n >= 2
                return solution, value.(X), objective_value(model)
            else
                return solution, [], objective_value(model)
            end
            return solution, value.(X), objective_value(model)
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
    function alg_reweight_LP(X, P, n)
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
    
    function alg_itersolve_LP(P, n)
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
            # X,_ = solve_lp(P_remain)
            x = zeros(n)
            x[remain] .= 1/length(remain)
            x = vcat(x, 1 - sum(x))
            if sum(x) != 0
                x = x ./ sum(x)
                println("\n $x \n")
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
    
    function alg4(P, n)
        selected = zeros(n)
        remain = collect(1:n)
        for i in 1:n
            if length(remain) == 0
                break
            end
            # gone = vec(setdiff(collect(1:(n+1)), remain))
            P_remain = P[remain,remain]
            X,_ = solve_sdp(P_remain)
            x = zeros(n)
            println(typeof(diag(X)))
            x[remain] = diag(X)[1:length(remain)]
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
    
    function alg_solve_SDP_once(P,n, X = zeros(n,n))
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
    
    function alg_solve_n_by_n_SDP(P,n, X = zeros(n,n))
        selected = zeros(n)
        remain = collect(1:n)
        for i in 1:n
            gone = vec(setdiff(collect(1:(n+1)), remain))
            # X,_ = solve_SDP_n_by_n(P, n)
            x = diag(X)[1:end-1]
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
    function alg_solve_SDP_3n_PSD(P, n)
        selected = zeros(n)
        remain = collect(1:n)
        for i in 1:n
            if length(remain) == 0
                break
            end
            # gone = vec(setdiff(collect(1:(n+1)), remain))
            P_remain = P[remain,remain]
            X,_ = solve_SDP_3n_PSD(P_remain,n)
            x = zeros(n)
            println(typeof(diag(X)))
            x[remain] = diag(X)[1:length(remain)]
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
    
    function alg_solve_SDP_3n_PSD_once(P, n, X)
        selected = zeros(n)
        remain = collect(1:n)
        y = diag(X)
        for i in 1:n
            if length(remain) == 0
                break
            end
            # gone = vec(setdiff(collect(1:(n+1)), remain))
            P_remain = P[remain,remain]
            x = zeros(n)
            x[remain] = y[1:length(remain)]
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
    
    function sim_reweight_LP(num, X, P, n)
        avg = 0 
            for i in 1:num
                selected = alg_reweight_LP(X, P, n)
                avg += sum(selected)
            end
        return avg/num
    end
    function sim_itersolve_LP(num, P, n)
        avg = 0 
            for i in 1:num
                selected = alg_itersolve_LP(P, n)
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
    
    function sim_alg4(num, P, n)
        avg = 0 
            for i in 1:num
                selected = alg4(P, n)
                avg += sum(selected)
            end
        return avg/num
    end
    
    function sim_alg_3n_PSD(num, P, n)
        avg = 0 
            for i in 1:num
                selected = alg_solve_SDP_3n_PSD(P, n)
                avg += sum(selected)
            end
        return avg/num
    end
    
    function sim_alg_3n_PSD_once(num, P, n, X)
        avg = 0 
            for i in 1:num
                
                selected = alg_solve_SDP_3n_PSD_once(P, n,X)
                avg += sum(selected)
            end
        return avg/num
    end
    
    function sim_alg_sdp_once(num, P, n, X)
        avg = 0 
            for i in 1:num
                selected = alg_solve_SDP_once(P, n, X)
                avg += sum(selected)
            end
        return avg/num
    end
    
    function sim_alg_n_by_n_sdp(num, X, P, n)
        avg = 0 
            for i in 1:num
                selected = alg_solve_n_by_n_SDP(P, n, X)
                avg += sum(selected)
            end
        return avg/num
    end
    
    function x_i_t_gnp(n,p)
        x = zeros(n^2)
        cur = 1/n
        x[1:n] .= cur 
        for i in 2:n 
            cur = sum(binomial(n-i,j-1)*(1-(1-p)^(i-1))^(n-i+1-j)*(1-p)^((i-1)*j) / j for j in 1:(n-i+1))
            x[(i-1)*n+1:i*n] .= cur 
        end
        return x
    end
    

end

# ╔═╡ 3e4e1f50-85ca-4468-bd3f-41d794cd7932
begin
    n = 12
    p = 0.2
    P = (ones(n, n) - I) .* p
    # Random.seed!(1999)
    # probs = rand(n * (n - 1) ÷ 2).* (0.8-0.2) .+ 0.2
    # probs = rand(n * (n - 1) ÷ 2)
    # avg_p = mean(probs)
    # P = fill_symmetric_matrix(probs, n)
    min_p = minimum(P+10*I)
    max_p = maximum(P)
    opt_ratio = compute_v(n, max_p)[end] / compute_v(n, min_p)[end]
    println("Optimal ratio: ", opt_ratio)
    println("min GNP:", compute_v(n, min_p)[end])
    println("max GNP:", compute_v(n, max_p)[end])
    # println("Alg Ratio: ", alg_avg/compute_v(n, min_p)[end])
    # println("Alg2 Ratio: ", alg_avg2/compute_v(n, min_p)[end])
    # println("LP Ratio: ", obj_value/compute_v(n, min_p)[end])

end

# ╔═╡ 20ac478f-3566-4c1d-be38-38d09b11724a
begin
    X,obj2 = solve_sdp(P)
    
    println(diag(X)[1:end-1])
    println(obj2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
COPT = "227a2e2d-e949-4d8e-a1da-7384fe6f0b9f"
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
Gurobi = "2e9cd046-0924-5485-92f1-d5272153d98b"
Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Mosek = "6405355b-0ac2-5fba-af84-adbd65488c0e"
MosekTools = "1ec41992-ff65-5c91-ac43-2df89e9693a4"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SCS = "c946c3f1-0d1f-5ce8-9dea-7daa1f7e2d13"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Suppressor = "fd094767-a336-5f1f-9728-57cf17d0bbfb"

[compat]
COPT = "~1.1.20"
Combinatorics = "~1.0.2"
Graphs = "~1.11.2"
Gurobi = "~1.6.0"
Ipopt = "~1.7.1"
JuMP = "~1.23.6"
Mosek = "~10.2.0"
MosekTools = "~0.15.4"
SCS = "~2.0.2"
Statistics = "~1.11.1"
StatsBase = "~0.34.4"
Suppressor = "~0.2.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "038be5c1a916aa7711ea0336f424af8611711c1f"

[[deps.ASL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6252039f98492252f9e47c312c8ffda0e3b9e78d"
uuid = "ae81ac8f-d209-56e5-92de-9978fef736f9"
version = "0.1.3+0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+4"

[[deps.COPT]]
deps = ["LazyArtifacts", "Libdl", "LinearAlgebra", "MathOptInterface", "Pkg", "Random", "SparseArrays"]
git-tree-sha1 = "0c87eba5590802cdd11f349c6e3fbd230aee4d1c"
uuid = "227a2e2d-e949-4d8e-a1da-7384fe6f0b9f"
version = "1.1.20"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "ebd18c326fa6cee1efb7da9a3b45cf69da2ed4d9"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.11.2"

[[deps.Gurobi]]
deps = ["Gurobi_jll", "Libdl", "MathOptInterface"]
git-tree-sha1 = "73e12786165a1c11217e63e15f26d0803feae74c"
uuid = "2e9cd046-0924-5485-92f1-d5272153d98b"
version = "1.6.0"

[[deps.Gurobi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "91f655524b6600deeda43054793535db81569b11"
uuid = "c018c7e6-a5b0-4aea-8f80-9c1ef9991411"
version = "12.0.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "50aedf345a709ab75872f80a2779568dc0bb461b"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.11.2+3"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Ipopt]]
deps = ["Ipopt_jll", "LinearAlgebra", "MathOptInterface", "OpenBLAS32_jll", "PrecompileTools"]
git-tree-sha1 = "33f1ef97ebd99b6741ea71dfe2ca3ce68943d11f"
uuid = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
version = "1.7.1"

[[deps.Ipopt_jll]]
deps = ["ASL_jll", "Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "MUMPS_seq_jll", "SPRAL_jll", "libblastrampoline_jll"]
git-tree-sha1 = "4f55ad688c698a4f77d892a1cb673f7e8a30f178"
uuid = "9cc047cb-c261-5740-88fc-0cf96f7bdcc7"
version = "300.1400.1700+0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "02b6e65736debc1f47b40b0f7d5dfa0217ee1f09"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.23.6"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "1c20a46719c0dc4ec4e7021ca38f53e1ec9268d9"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.2+1"

[[deps.MUMPS_seq_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "METIS_jll", "libblastrampoline_jll"]
git-tree-sha1 = "0eab12f94948ca67908aec14b9f2ebefd17463fe"
uuid = "d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"
version = "500.700.301+0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "e065ca5234f53fd6f920efaee4940627ad991fb4"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.34.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.Mosek]]
deps = ["Libdl", "Pkg", "Printf", "SparseArrays"]
git-tree-sha1 = "3b3f443422b385733fcc52da0a8de8607cd85482"
uuid = "6405355b-0ac2-5fba-af84-adbd65488c0e"
version = "10.2.0"

[[deps.MosekTools]]
deps = ["MathOptInterface", "Mosek", "Printf"]
git-tree-sha1 = "944b53ab2dab8de7aa82d650536f9177a74ca723"
uuid = "1ec41992-ff65-5c91-ac43-2df89e9693a4"
version = "0.15.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "a2710df6b0931f987530f59427441b21245d8f5e"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "030ea22804ef91648f29b7ad3fc15fa49d0e6e71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dd806c813429ff09878ea3eeb317818f3ca02871"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.28+3"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SCS]]
deps = ["MathOptInterface", "Requires", "SCS_jll", "SparseArrays"]
git-tree-sha1 = "aa3fcff53da363b4ba4b54d4ac4c9186ab00d703"
uuid = "c946c3f1-0d1f-5ce8-9dea-7daa1f7e2d13"
version = "2.0.2"

    [deps.SCS.extensions]
    SCSSCS_GPU_jllExt = ["SCS_GPU_jll"]
    SCSSCS_MKL_jllExt = ["SCS_MKL_jll"]

    [deps.SCS.weakdeps]
    SCS_GPU_jll = "af6e375f-46ec-5fa0-b791-491b0dfa44a4"
    SCS_MKL_jll = "3f2553a9-4106-52be-b7dd-865123654657"

[[deps.SCS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "OpenBLAS32_jll"]
git-tree-sha1 = "902cc4e42ecca21bbd74babf899b2a5b12add323"
uuid = "f4f2fc5b-1d94-523c-97ea-2ab488bedf4b"
version = "3.2.7+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SPRAL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "Libdl", "METIS_jll", "libblastrampoline_jll"]
git-tree-sha1 = "11f3da4b25efacd1cec8e263421f2a9003a5e8e0"
uuid = "319450e9-13b8-58e8-aa9f-8fd1420848ab"
version = "2024.5.8+0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.Suppressor]]
deps = ["Logging"]
git-tree-sha1 = "6dbb5b635c5437c68c28c2ac9e39b87138f37c0a"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.8"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═05cd9dab-2719-4daf-b088-61c7fae40119
# ╠═a73b3637-28c4-47e5-b861-f744b2bc7567
# ╠═ced4e571-c6cf-45ae-8413-afbeaca04db1
# ╠═3e4e1f50-85ca-4468-bd3f-41d794cd7932
# ╠═a3b980d5-e45c-44ab-9d4f-47b760f042fa
# ╠═b98a3f21-fb4d-40e8-9f27-fc70eb367bd1
# ╠═0dca4d73-bdbb-4615-b01b-69f50907f7ea
# ╠═dedb9435-3f7f-488e-b8f1-fa640e3bcb43
# ╠═60640e36-881a-40f2-ac6f-7ef1cc5bece5
# ╠═21557e7f-8e10-4a88-8efe-3a71007fda1f
# ╠═c62d2ba4-93e1-4030-b48a-9d3da63bdaf6
# ╠═20ac478f-3566-4c1d-be38-38d09b11724a
# ╠═93f3bb92-dd46-469e-8ba9-4ddd7bc98167
# ╠═1bc70a94-6438-40fe-988a-41857d129f26
# ╠═9ec02064-a9d9-40a2-8172-b60c4691ab31
# ╠═ff8c5b26-e86a-4070-930e-df13bb8b961f
# ╠═718644eb-208e-4d33-babd-6b36906c9757
# ╠═3b90f02b-fabc-4f04-ae46-7a9f84fcef87
# ╠═350f8e68-fa5b-480b-ac2c-0b9931e0b982
# ╟─11301ab8-b686-4121-8c30-25947223891f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

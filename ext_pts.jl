using Polyhedra
using CDDLib
using LinearAlgebra


function scale_vector_rhs(v, rhs)
    # Find maximum absolute value in the vector
    abs_values = abs.(v)
    max_val = maximum(abs_values)

    # If max_val ≤ 1 or all zeros, no scaling needed
    if max_val <= 1 || max_val == 0
        return (v, rhs)
    end

    # Compute scaling factor (reciprocal of max_val as a rational)
    scaling_factor = 1 // max_val

    # Scale vector and RHS
    v_scaled = scaling_factor .* v
    rhs_scaled = scaling_factor * rhs

    return (v_scaled, rhs_scaled)
end

function main()
    # ========================
    # Parameters (local to function)
    # ========================
    vertices = [1, 2, 3]
    T = 4
    
    P = [0 1/3 1//7; 1//3 0 1//5; 1//7 1//5 0]
    
    # p = 3//10
    # P = (ones(4,4) - I).*p
    # ========================
    # Helper Functions
    # ========================
    function generate_subsets(S)
        subsets = Vector{Vector{Int}}()
        for mask in 0:(1<<length(S))-1
            subset = [S[i+1] for i in 0:length(S)-1 if (mask & (1<<i)) != 0]
            push!(subsets, subset)
        end
        return subsets
    end

    # ========================
    # State Generation (local)
    # ========================
    states = Dict{Int, Vector{Vector{Int}}}()
    states[1] = [copy(vertices)]

    for t in 2:T
        new_states = Set{Vector{Int}}()
        for prev_S in states[t-1]
            push!(new_states, Int[])  # Do nothing
            for i in prev_S
                remaining = filter(x -> x ≠ i, prev_S)
                for S_new in generate_subsets(remaining)
                    push!(new_states, S_new)
                end
            end
        end
        states[t] = collect(new_states)
    end

    # ========================
    # Variable Indexing (local)
    # ========================
    var_index = Dict{Tuple{Vector{Int}, Int, String}, Int}()
    var_count = 0  # Local variable
    IT_index = Dict{Tuple{Int, Int}, Vector{Int}}()
    for t in 1:T, i in copy(vertices)
        IT_index[(t, i)] = Int[]
    end

    for t in 1:T
        for S in states[t]
            sorted_S = sort(S)
            var_count += 1
            var_index[(sorted_S, t, "nothing")] = var_count
            push!(get!(IT_index, (t, 0), Int[]), var_count)
            if t < T && !isempty(S)
                for i in S
                    var_count += 1
                    var_index[(sorted_S, t, "pick$i")] = var_count
                    push!(get!(IT_index, (t, i), Int[]), var_count)
                end
            end
        end
    end

    total_vars = var_count
    println("Total variables: ", total_vars)
    println("states: ", states)
    # ========================
    # Constraint Construction (local)
    # ========================
    A_eq = zeros(0, total_vars)
    b_eq = zeros(0)
    A_ineq = zeros(0, total_vars)
    b_ineq = zeros(0)

   # Flow conservation constraints
   for t in 1:T
    for S in states[t]
        row_eq = zeros(total_vars)
        sorted_S = sort(S)
        
        # Inflow from previous states (t > 1)
        if t > 1
            for prev_S in states[t-1]
                prev_sorted = sort(prev_S)
                # Do nothing transitions
                if isempty(S)
                    var = get(var_index, (prev_sorted, t-1, "nothing"), 0)
                    var > 0 && (row_eq[var] = -1)
                end
                # if isempty(prev_S) && isempty(S)
                #     continue
                # end
                # Pick transitions
                for i in prev_S

                    var = get(var_index, (prev_sorted, t-1, "pick$i"), 0)
                    if var > 0
                        candidate = filter(x -> x ≠ i, prev_S)
                        if issubset(S, candidate)
                            discard = setdiff(candidate, S)
                            present = intersect(candidate, S)

                            prob = -1
                            for j in discard
                                prob *= P[i,j]
                            end
                            for j in present
                                prob *= 1 - P[i,j]
                            end

                            row_eq[var] = prob
                        else
                            # Invalid transition: probability = 0
                            row_eq[var] = 0
                        end
                    end
                end
            end
            push!(b_eq, 0)
        else
            # # Initial state constraint: sum of all actions = 1
            #     # Get all actions for the initial state (do, pick1, pick2, pick3)
            #     actions = ["nothing"]
            #     append!(actions, ["pick$i" for i in S])
            #     for action in actions
            #         var = get(var_index, (sorted_S, t, action), 0)
            #         if var > 0
            #             row_eq[var] = 1
            #         end
            #     end
                push!(b_eq, 1)  # RHS = 1
        end

        # Outflow to next states (t < T)
        if t < T
            # Do nothing outflow
            # var = get(var_index, (sorted_S, t, "nothing"), 0)
            # var > 0 && (row_eq[var] = -1)
            # println("var nothing: ", var)
            # Pick outflows
            if !isempty(S)
                for i in S
                    var = get(var_index, (sorted_S, t, "pick$i"), 0)
                    var > 0 && (row_eq[var] = 1)
                    println("var $i: ", var)
                end
            else
                # Do nothing outflow
                var = get(var_index, (sorted_S, t, "nothing"), 0)
                var > 0 && (row_eq[var] = 1)
            end
        else
            # Do nothing outflow
            var = get(var_index, (sorted_S, t, "nothing"), 0)
            var > 0 && (row_eq[var] = 1)
        end

        A_eq = [A_eq; row_eq']
    end
end

# Non-negativity constraints
for i in 1:total_vars
    row_ineq = zeros(total_vars)
    row_ineq[i] = -1
    A_ineq = [A_ineq; row_ineq']
    push!(b_ineq, 0)
end
    # ========================
    # Combine constraints into H-representation
    # ========================
    m_ineq = size(A_ineq, 1)  # Number of inequalities
    m_eq = size(A_eq, 1)       # Number of equalities

    # Stack inequalities and equalities
    A = [A_ineq; A_eq]
    # A = [zeros(m_ineq,total_vars); A_eq]
    # A = [zeros(m_ineq,total_vars); zeros(m_eq,total_vars)]
    b = [b_ineq; b_eq]

    A = rationalize.(A,tol=1e-6)
    b = rationalize.(b,tol=1e-6)
    # Mark equality rows (indices start after inequalities)
    linset = BitSet((m_ineq + 1) : (m_ineq+m_eq))
    # linset = BitSet(1:-1)

    # Create H-representation
    hrep = Polyhedra.hrep(A, b, linset)
    display(hrep)
    # Build the polyhedron
    lib = CDDLib.Library(:exact)
    poly = Polyhedra.polyhedron(hrep, lib)
    open("output.txt", "w") do io
        redirect_stdout(io) do
    # Compute vertices
    println("\nComputing extreme points...")
    verts = collect(Polyhedra.points(poly))

    it_verts = Vector{Vector{Rational{BigInt}}}(undef, 0)
    if isempty(verts)
        println("No extreme points found - check constraints!")
    else
        println("Found ", length(verts), " extreme points:")
        v_it = []
        for (i, v) in enumerate(verts)
            # println("Vertex $i: ", round.(v, digits=3))
            v_it = [sum(v[IT_index[(t, j)]]) for t in 1:T, j in pushfirst!(copy(vertices),0)]
            push!(it_verts, (vec(transpose(v_it))))
            # [it_verts; Rational{Int64}.(vec(transpose(v_it)))']
            println(typeof((vec(transpose(v_it)))))
            println("Vertex $i: ", v_it)

            # for t in 1:(T-1)
            #     temp = copy(v_it)
            #     temp[(t+1):end,:] .= 0
            #     push!(it_verts, (vec(transpose(temp))))
            #     # [it_verts; Rational{Int64}.(vec(transpose(v_it)))']
            #     println(typeof((vec(transpose(temp)))))
            #     println("Vertex $i: ", temp)
            # end
        end
        # push!(it_verts, zeros(length(vec(transpose(v_it)))))
    end

    # Build polyhedron using CDDLib's exact backend for rational arithmetic
    it_reps = Polyhedra.vrep(it_verts)
    
    poly_it = polyhedron(it_reps, lib)
    removevredundancy!(poly_it)
    # Obtain the H-representation (includes facets)
    
    # removehredundancy!(poly_it)
    h_it = MixedMatHRep(poly_it)
    # Extract the halfspaces defining the facets
    facets = halfspaces(poly_it)
    

               # Display each facet as an inequality a⋅x ≤ b
            for i in eachindex(facets)
                facet = get(poly_it, i)
                facet_vertices = incidentpoints(poly_it, i)
                # To get the actual coordinates, convert indices to points from the vertex representation:
                # facet_vertices = collect(points(vrep(poly_it)))[facet_vertex_indices]
                println("\n$(facet.a)x ≤ $(facet.β)")
                (a, RHS) = scale_vector_rhs(facet.a, facet.β)
                println("$(a)x ≤ $RHS\n")
                println("Facet $i is defined by vertices: ", facet_vertices)
            end
            # For example, if you have a DataFrame `df`:
            # println(df)
        end
    end

    # verts_it = collect(Polyhedra.points(poly_it))
    # if isempty(verts_it)
    #     println("No extreme points found - check constraints!")
    # else
    #     println("Found ", length(verts_it), " extreme points:")
    #     for (i, v) in enumerate(verts_it)
    #         # println("Vertex $i: ", round.(v, digits=3))
    #         println("Vertex $i: ", v)
    #     end
    # end

    sorted_index = sort(collect(var_index); by=p -> p.second[1], rev=false)
    display(sorted_index)
    display(IT_index)
    display(hrep)




end

# Run the entire pipeline
main()


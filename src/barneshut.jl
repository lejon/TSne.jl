using NearestNeighbors
using NearestNeighborDescent

# Use maxthreadid() on Julia 1.9+ (interactive thread pool can exceed nthreads())
_max_thread_id() = @static if VERSION >= v"1.9"
    Threads.maxthreadid()
else
    Threads.nthreads() + 16
end

function find_knn_byrow(D::AbstractMatrix{T}, k::Int; progress::Bool=false) where T<:Number
    n = size(D, 1)
    neighbors = [Vector{Int}(undef, k) for _ in 1:n]
    dists = [Vector{Float64}(undef, k) for _ in 1:n]
    progress && (pb = Progress(n; desc="Finding nearest neighbors"))
    Threads.@threads for i in 1:n
        row = [Float64(D[i, j]) for j in 1:n]
        row[i] = Float64(Inf)
        perm = partialsortperm(row, 1:k; initialized=false)
        for idx in 1:k
            dist = row[perm[idx]]
            neighbors[i][idx] = perm[idx]
            dists[i][idx] = dist * dist
        end
        progress && next!(pb)
    end
    progress && finish!(pb)
    return neighbors, dists
end

function validate_distance_matrix(D::AbstractMatrix)
    (size(D, 1) == size(D, 2) && issymmetric(D) && all(x -> x >= 0, D)) ||
        throw(ArgumentError("Distance matrix D must be symmetric and positive"))
end

function find_knn_kdtree(X::AbstractMatrix{T}, k::Int; progress::Bool=false) where T<:Number
    n = size(X, 1)
    progress && (pb = Progress(4; desc="Finding nearest neighbors"))
    Xt = permutedims(Float32.(X), (2, 1))   # Float32: halves memory bandwidth for distance eval
    progress && next!(pb)
    kdtree = KDTree(Xt, Euclidean())
    progress && next!(pb)
    neighbors = [Vector{Int}(undef, k) for _ in 1:n]
    dists = [Vector{Float64}(undef, k) for _ in 1:n]
    # KDTree is read-only during queries — parallel queries are safe.
    Threads.@threads for i in 1:n
        idxs_i, ds_i = knn(kdtree, view(Xt, :, i), k + 1, true)
        pos = 1
        for idx in eachindex(idxs_i)
            idxs_i[idx] == i && continue
            neighbors[i][pos] = idxs_i[idx]
            dists[i][pos] = ds_i[idx]^2
            pos += 1
            pos > k && break
        end
    end
    progress && (next!(pb); finish!(pb))
    return neighbors, dists
end

function find_knn_nndescent(X::AbstractMatrix{T}, k::Int; progress::Bool=false) where T<:Number
    n = size(X, 1)
    progress && (pb = Progress(2; desc="Finding nearest neighbors"))
    # NNDescent expects (d, n) — features as rows, points as columns
    Xf = permutedims(Float32.(X), (2, 1))
    progress && next!(pb)
    # nndescent returns approximate kNN without self-references; distances are actual (not squared)
    g = nndescent(Xf, k, Euclidean())
    idx_mat, dist_mat = knn_matrices(g)     # (k, n) matrices, 1-indexed, sorted ascending
    progress && (next!(pb); finish!(pb))
    neighbors = [Vector{Int}(undef, k) for _ in 1:n]
    dists     = [Vector{Float64}(undef, k) for _ in 1:n]
    @inbounds for i in 1:n
        for ki in 1:k
            neighbors[i][ki] = idx_mat[ki, i]
            dists[i][ki]     = Float64(dist_mat[ki, i])^2   # squared Euclidean
        end
    end
    return neighbors, dists
end

function find_knn_bruteforce(X::AbstractMatrix{T}, k::Int; progress::Bool=false) where T<:Number
    n = size(X, 1)
    n <= 0 && return [Int[], Float64[]]
    k = min(k, n - 1)
    Xf = Float32.(X)                         # Float32: 2× faster GEMM, halves memory
    Xnorms = vec(sum(abs2, Xf, dims=2))
    XX = Xf * Xf'
    neighbors = [Vector{Int}(undef, k) for _ in 1:n]
    dists = [Vector{Float64}(undef, k) for _ in 1:n]
    progress && (pb = Progress(n; desc="Finding nearest neighbors"))
    Threads.@threads for i in 1:n
        row = Vector{Float32}(undef, n)
        ni = Xnorms[i]
        @inbounds for j in 1:n
            row[j] = ifelse(j == i, Inf32, ni + Xnorms[j] - 2 * XX[i, j])
        end
        perm = partialsortperm(row, 1:k; initialized=false)
        for idx in 1:k
            neighbors[i][idx] = perm[idx]
            dists[i][idx] = row[perm[idx]]
        end
        progress && next!(pb)
    end
    progress && finish!(pb)
    return neighbors, dists
end

function find_knn_data(X::AbstractMatrix{T}, k::Int; progress::Bool=false) where T<:Number
    n = size(X, 1)
    if n <= 0
        return [Int[], Float64[]]
    end
    k = min(k, n - 1)
    k < 1 && return [Int[], Float64[]]
    n <= 15000 && return find_knn_bruteforce(X, k; progress=progress)
    # KD-trees degrade toward brute-force at high dimensions (curse of dimensionality).
    # NN-Descent (approximate kNN) is 5-20x faster for the typical t-SNE input of 20-50 PCA dims.
    size(X, 2) >= 20 && return find_knn_nndescent(X, k; progress=progress)
    return find_knn_kdtree(X, k; progress=progress)
end

function find_knn_data(X::AbstractMatrix{T}, dist::Function, k::Int; progress::Bool=false) where T<:Number
    n = size(X, 1)
    n <= 0 && return [Int[], Float64[]]
    k = min(k, n - 1)
    d = size(X, 2)
    neighbors = [Vector{Int}(undef, k) for _ in 1:n]
    dists = [Vector{Float64}(undef, k) for _ in 1:n]
    # Per-thread buffers — allocated once, reused every iteration.
    nt = _max_thread_id()
    xi_bufs  = [Vector{Float64}(undef, d) for _ in 1:nt]
    xj_bufs  = [Vector{Float64}(undef, d) for _ in 1:nt]
    row_bufs = [Vector{Float64}(undef, n) for _ in 1:nt]
    progress && (pb = Progress(n; desc="Finding nearest neighbors"))
    Threads.@threads for i in 1:n
        tid = Threads.threadid()
        xi  = xi_bufs[tid]
        xj  = xj_bufs[tid]
        row = row_bufs[tid]
        @inbounds for l in 1:d; xi[l] = X[i, l]; end
        @inbounds for j in 1:n
            if j == i
                row[j] = Float64(Inf)
            else
                for l in 1:d; xj[l] = X[j, l]; end
                row[j] = Float64(dist(xi, xj))^2
            end
        end
        perm = partialsortperm(row, 1:k; initialized=false)
        for idx in 1:k
            neighbors[i][idx] = perm[idx]
            dists[i][idx] = row[perm[idx]]
        end
        progress && next!(pb)
    end
    progress && finish!(pb)
    return neighbors, dists
end

function find_knn_data(X::AbstractMatrix{T}, dist::PreMetric, k::Int; progress::Bool=false) where T<:Number
    n = size(X, 1)
    n <= 0 && return [Int[], Float64[]]
    k = min(k, n - 1)
    return find_knn_bruteforce_metric(X, dist, k; progress=progress)
end

function find_knn_bruteforce_metric(X::AbstractMatrix{T}, dist::PreMetric, k::Int; progress::Bool=false) where T<:Number
    n = size(X, 1)
    neighbors = [Vector{Int}(undef, k) for _ in 1:n]
    dists = [Vector{Float64}(undef, k) for _ in 1:n]
    D = pairwise(dist, X', dims=2)
    nt = _max_thread_id()
    row_bufs = [Vector{Float64}(undef, n) for _ in 1:nt]
    progress && (pb = Progress(n; desc="Finding nearest neighbors"))
    Threads.@threads for i in 1:n
        tid = Threads.threadid()
        row = row_bufs[tid]
        # D[i,j] = dist(row_i, row_j): distances FROM point i (correct for asymmetric metrics)
        @inbounds for j in 1:n
            row[j] = j == i ? Float64(Inf) : Float64(D[i, j])^2
        end
        perm = partialsortperm(row, 1:k; initialized=false)
        for idx in 1:k
            neighbors[i][idx] = perm[idx]
            dists[i][idx] = row[perm[idx]]
        end
        progress && next!(pb)
    end
    progress && finish!(pb)
    return neighbors, dists
end

function find_knn_vector(X::AbstractVector, dist::Union{Function, PreMetric}, k::Int; progress::Bool=false)
    n = length(X)
    n <= 0 && return [Int[], Float64[]]
    k = min(k, n - 1)
    neighbors = [Vector{Int}(undef, k) for _ in 1:n]
    dists = [Vector{Float64}(undef, k) for _ in 1:n]
    progress && (pb = Progress(n; desc="Finding nearest neighbors"))
    Threads.@threads for i in 1:n
        row = Vector{Float64}(undef, n)
        xi = X[i]
        for j in 1:n
            row[j] = j == i ? Float64(Inf) : Float64(dist(xi, X[j]))^2
        end
        perm = partialsortperm(row, 1:k; initialized=false)
        for idx in 1:k
            neighbors[i][idx] = perm[idx]
            dists[i][idx] = row[perm[idx]]
        end
        progress && next!(pb)
    end
    progress && finish!(pb)
    return neighbors, dists
end

function sparse_perplexities!(P_rows::Vector{Vector{Pair{Int,Float64}}},
                              neighbors::Vector{Vector{Int}},
                              dists::Vector{Vector{Float64}},
                              perplexity::Float64, tol::Float64,
                              max_iter::Int; progress::Bool=false)
    n = length(P_rows)
    Htarget = log(perplexity)
    beta_values = Vector{Float64}(undef, n)
    progress && (pb = Progress(n; desc="Computing sparse perplexities"))

    Threads.@threads for i in 1:n
        ndists = dists[i]
        nidxs = neighbors[i]
        k = length(ndists)

        min_d = minimum(ndists)
        shifted = [ndists[j] - min_d for j in 1:k]

        beta = 1.0
        betamin = 0.0
        betamax = Float64(Inf)

        Pvals = zeros(Float64, k)

        for _ in 1:max_iter
            for j in 1:k
                Pvals[j] = exp(-beta * shifted[j])
            end
            sumP = sum(Pvals)
            if sumP <= 0 || !isfinite(sumP)
                sumP = 1.0
                Pvals[1] = 1.0
                for j in 2:k
                    Pvals[j] = 0.0
                end
                break
            end
            for j in 1:k
                Pvals[j] /= sumP
            end
            H = log(sumP) + beta * dot(shifted, Pvals)
            Hdiff = H - Htarget

            if abs(Hdiff) < tol
                break
            end

            if Hdiff > 0
                betamin = beta
                beta = isfinite(betamax) ? (beta + betamax) / 2 : beta * 2
            else
                betamax = beta
                beta = (beta + betamin) / 2
            end
        end

        empty!(P_rows[i])
        for j in 1:k
            if Pvals[j] > 0
                push!(P_rows[i], nidxs[j] => Float64(Pvals[j]))
            end
        end
        beta_values[i] = beta
        progress && next!(pb)
    end
    progress && finish!(pb)

    return P_rows, beta_values
end

function symmetrize_sparse_P!(P_rows::Vector{Vector{Pair{Int,Float64}}}; progress::Bool=false)
    n = length(P_rows)
    dicts = [Dict{Int,Float64}() for _ in 1:n]
    progress && (pb = Progress(2n; desc="Symmetrizing sparse perplexities"))

    for i in 1:n
        for (j, pval) in P_rows[i]
            dicts[i][j] = get(dicts[i], j, 0.0) + pval
            dicts[j][i] = get(dicts[j], i, 0.0) + pval
        end
        progress && next!(pb)
    end

    for i in 1:n
        P_rows[i] = [j => v for (j, v) in dicts[i]]
        progress && next!(pb)
    end
    progress && finish!(pb)
    return P_rows
end

function compute_P_sum(P_rows::Vector{Vector{Pair{Int,Float64}}})
    total = 0.0
    for row in P_rows
        for (_, v) in row
            total += v
        end
    end
    return total
end

function scale_sparse_P!(P_rows::Vector{Vector{Pair{Int,Float64}}}, scale::Float64)
    for row in P_rows
        for idx in eachindex(row)
            row[idx] = row[idx].first => row[idx].second * scale
        end
    end
    return P_rows
end

function sparse_rows_to_csr(P_rows::Vector{Vector{Pair{Int,Float64}}})
    n = length(P_rows)
    rowptr = Vector{Int}(undef, n + 1)
    rowptr[1] = 1
    nnz = 0
    for i in 1:n
        nnz += length(P_rows[i])
        rowptr[i + 1] = nnz + 1
    end

    colidx = Vector{Int}(undef, nnz)
    values = Vector{Float64}(undef, nnz)
    pos = 1
    for i in 1:n
        for (j, pval) in P_rows[i]
            colidx[pos] = j
            values[pos] = pval
            pos += 1
        end
    end
    return rowptr, colidx, values
end

function compute_sparse_P(X::Union{AbstractMatrix, AbstractVector},
                          distance::Union{Bool, Function, PreMetric},
                          perplexity::Number, tol::Float64,
                          max_iter::Int, verbose::Bool, progress::Bool=false)
    n = size(X, 1)
    k = min(Int(3 * perplexity), n - 1)
    k = max(k, 1)

    if verbose
        @info("Computing sparse P with k=$k nearest neighbors...")
    end

    neighbors = Vector{Int}[]
    dists = Vector{Float64}[]

    if distance === false
        neighbors, dists = find_knn_data(X, k; progress=progress)
    elseif distance === true
        validate_distance_matrix(X)
        neighbors, dists = find_knn_byrow(X, k; progress=progress)
    elseif isa(distance, Function)
        if isa(X, AbstractVector)
            neighbors, dists = find_knn_vector(X, distance, k; progress=progress)
        else
            neighbors, dists = find_knn_data(X, distance, k; progress=progress)
        end
    elseif isa(distance, PreMetric)
        if isa(X, AbstractVector)
            neighbors, dists = find_knn_vector(X, distance, k; progress=progress)
        else
            neighbors, dists = find_knn_data(X, distance, k; progress=progress)
        end
    else
        error("Unsupported distance type: $(typeof(distance))")
    end

    P_rows = [Vector{Pair{Int,Float64}}() for _ in 1:n]
    P_rows, beta = sparse_perplexities!(P_rows, neighbors, dists, Float64(perplexity), tol, max_iter; progress=progress)
    symmetrize_sparse_P!(P_rows; progress=progress)

    sumP = compute_P_sum(P_rows)

    if verbose
        @info("Sparse P computed: $(minimum(length.(P_rows))) to $(maximum(length.(P_rows))) non-zeros per row, sum=$sumP")
    end

    return P_rows, sumP, beta
end

function compute_KL_bh_fast(Y::Matrix{T}, P_rows::Vector{Vector{Pair{Int,Float64}}},
                              tree::SpaceNode, theta::Float64,
                              yi::Vector{Float64}, grad_buf::Vector{Float64}) where T<:Number
    n = size(Y, 1)
    ndims = size(Y, 2)
    z_ref = Ref(0.0)
    # Pre-allocate traversal stacks once to avoid O(n) heap allocations in the loop
    kl_stack = [zeros(Float64, ndims) for _ in 1:MAX_TREE_DEPTH]
    kl_z_stack = [Ref(0.0) for _ in 1:MAX_TREE_DEPTH]

    Z = 0.0
    for i in 1:n
        for d in 1:ndims
            yi[d] = Y[i, d]
        end
        walk_tree_into!(tree, Y, yi, theta, i, z_ref, grad_buf, kl_stack, kl_z_stack, 0)
        Z += z_ref[]
    end

    kldiv = 0.0
    for i in 1:n
        for d in 1:ndims
            yi[d] = Y[i, d]
        end
        for (j, p_ij) in P_rows[i]
            if j == i
                continue
            end
            dij = 0.0
            for d in 1:ndims
                dd = yi[d] - Y[j, d]
                dij += dd * dd
            end
            q_ij = 1.0 / (1.0 + dij)
            if p_ij > 0 && q_ij > 0
                kldiv += p_ij * log(p_ij / (q_ij / Z))
            end
        end
    end

    return kldiv
end

function optimize_bh_2d!(Y::Matrix{Float64}, dY::Matrix{Float64}, iY::Matrix{Float64},
                         gains::Matrix{Float64}, P_rows::Vector{Vector{Pair{Int,Float64}}},
                         P_rowptr::Vector{Int}, P_colidx::Vector{Int}, P_values::Vector{Float64},
                         max_iter::Integer, theta::Float64, min_gain::Float64, eta::Float64,
                         initial_momentum::Float64, final_momentum::Float64,
                         momentum_switch_iter::Integer, stop_cheat_iter::Integer,
                         cheat_scale::Float64, progress::Bool, max_depth::Int)
    n = size(Y, 1)
    current_scale = cheat_scale
    target_scale = 1.0
    thread_Z = zeros(Float64, _max_thread_id())
    thread_node_stack = [Int[] for _ in 1:length(thread_Z)]
    tree = FlatTree2D()
    Ymean = zeros(Float64, 1, 2)

    progress && (pb = Progress(max_iter; desc="Computing BH t-SNE"))
    progress_interval = max(max_iter ÷ 100, 1)

    for iter in 1:max_iter
        build_flat_tree_2d!(tree, Y, max_depth)
        stack_size = length(tree.count)
        for stack in thread_node_stack
            length(stack) < stack_size && resize!(stack, stack_size)
        end
        Z = compute_repulsive_forces_2d!(dY, Y, tree, theta, thread_Z, thread_node_stack)
        Zinv = 1.0 / Z

        Threads.@threads for i in 1:n
            @inbounds begin
                xi = Y[i, 1]
                yi = Y[i, 2]
                gx = -dY[i, 1] * Zinv
                gy = -dY[i, 2] * Zinv
                for pos in P_rowptr[i]:(P_rowptr[i + 1] - 1)
                    j = P_colidx[pos]
                    j == i && continue
                    dx = xi - Y[j, 1]
                    dy = yi - Y[j, 2]
                    q_ij = 1.0 / (1.0 + dx * dx + dy * dy)
                    pijq = P_values[pos] * q_ij
                    gx += pijq * dx
                    gy += pijq * dy
                end
                dY[i, 1] = 4.0 * gx
                dY[i, 2] = 4.0 * gy
            end
        end

        momentum = iter <= momentum_switch_iter ? initial_momentum : final_momentum
        @inbounds for idx in eachindex(gains)
            gains[idx] = max(ifelse((dY[idx] > 0) == (iY[idx] > 0),
                                    gains[idx] * 0.8,
                                    gains[idx] + 0.2),
                             min_gain)
            iY[idx] = momentum * iY[idx] - eta * (gains[idx] * dY[idx])
            Y[idx] += iY[idx]
        end

        Y .-= mean!(Ymean, Y)

        if current_scale != target_scale && iter >= min(max_iter, stop_cheat_iter)
            scale = target_scale / current_scale
            scale_sparse_P!(P_rows, scale)
            P_values .*= scale
            current_scale = target_scale
        end

        if progress && (iter == 1 || iter == max_iter || iter % progress_interval == 0)
            update!(pb, iter)
        end
    end

    progress && finish!(pb)
    return Y
end

function optimize_bh_3d!(Y::Matrix{Float64}, dY::Matrix{Float64}, iY::Matrix{Float64},
                         gains::Matrix{Float64}, P_rows::Vector{Vector{Pair{Int,Float64}}},
                         P_rowptr::Vector{Int}, P_colidx::Vector{Int}, P_values::Vector{Float64},
                         max_iter::Integer, theta::Float64, min_gain::Float64, eta::Float64,
                         initial_momentum::Float64, final_momentum::Float64,
                         momentum_switch_iter::Integer, stop_cheat_iter::Integer,
                         cheat_scale::Float64, progress::Bool, max_depth::Int)
    n = size(Y, 1)
    current_scale = cheat_scale
    target_scale = 1.0
    thread_Z = zeros(Float64, _max_thread_id())
    thread_node_stack = [Int[] for _ in 1:length(thread_Z)]
    tree = FlatTree3D()
    Ymean = zeros(Float64, 1, 3)

    progress && (pb = Progress(max_iter; desc="Computing BH t-SNE"))
    progress_interval = max(max_iter ÷ 100, 1)

    for iter in 1:max_iter
        build_flat_tree_3d!(tree, Y, max_depth)
        stack_size = length(tree.count)
        for stack in thread_node_stack
            length(stack) < stack_size && resize!(stack, stack_size)
        end
        Z = compute_repulsive_forces_3d!(dY, Y, tree, theta, thread_Z, thread_node_stack)
        Zinv = 1.0 / Z

        Threads.@threads for i in 1:n
            @inbounds begin
                xi = Y[i, 1]
                yi = Y[i, 2]
                zi = Y[i, 3]
                gx = -dY[i, 1] * Zinv
                gy = -dY[i, 2] * Zinv
                gz = -dY[i, 3] * Zinv
                for pos in P_rowptr[i]:(P_rowptr[i + 1] - 1)
                    j = P_colidx[pos]
                    j == i && continue
                    dx = xi - Y[j, 1]
                    dy = yi - Y[j, 2]
                    dz = zi - Y[j, 3]
                    q_ij = 1.0 / (1.0 + dx * dx + dy * dy + dz * dz)
                    pijq = P_values[pos] * q_ij
                    gx += pijq * dx
                    gy += pijq * dy
                    gz += pijq * dz
                end
                dY[i, 1] = 4.0 * gx
                dY[i, 2] = 4.0 * gy
                dY[i, 3] = 4.0 * gz
            end
        end

        momentum = iter <= momentum_switch_iter ? initial_momentum : final_momentum
        @inbounds for idx in eachindex(gains)
            gains[idx] = max(ifelse((dY[idx] > 0) == (iY[idx] > 0),
                                    gains[idx] * 0.8,
                                    gains[idx] + 0.2),
                             min_gain)
            iY[idx] = momentum * iY[idx] - eta * (gains[idx] * dY[idx])
            Y[idx] += iY[idx]
        end

        Y .-= mean!(Ymean, Y)

        if current_scale != target_scale && iter >= min(max_iter, stop_cheat_iter)
            scale = target_scale / current_scale
            scale_sparse_P!(P_rows, scale)
            P_values .*= scale
            current_scale = target_scale
        end

        if progress && (iter == 1 || iter == max_iter || iter % progress_interval == 0)
            update!(pb, iter)
        end
    end

    progress && finish!(pb)
    return Y
end

function tsne_bh(X::Union{AbstractMatrix, AbstractVector}, ndims::Integer,
                 reduce_dims::Integer, max_iter::Integer, perplexity::Number;
                 distance::Union{Bool, Function, PreMetric} = false,
                 min_gain::Number = 0.01, eta::Number = 200.0,
                 pca_init::Bool = false,
                 initial_momentum::Number = 0.5, final_momentum::Number = 0.8,
                 momentum_switch_iter::Integer = 250,
                 stop_cheat_iter::Integer = 250, cheat_scale::Number = 12.0,
                 theta::Number = 0.5,
                 max_depth::Integer = 7,
                 verbose::Bool = false, progress::Bool = true,
                 extended_output = false,
                 rng::AbstractRNG = Random.default_rng())

    ini_Y_with_X = false
    if isa(X, AbstractMatrix) && (distance !== true)
        verbose && @info("Initial X shape is $(size(X))")
        ndims < size(X, 2) || throw(DimensionMismatch("X has fewer dimensions ($(size(X,2))) than ndims=$ndims"))
        ini_Y_with_X = true
        X = X * (1.0 / std(X))
        if 0 < reduce_dims < size(X, 2)
            reduce_dims = max(reduce_dims, ndims)
            verbose && @info("Preprocessing the data using PCA...")
            X = pca(X, reduce_dims)
        end
    end

    n = size(X, 1)

    if pca_init && ini_Y_with_X
        verbose && @info("Using the first $ndims components of the data PCA as the initial layout...")
        if reduce_dims >= ndims
            Y = X[:, 1:ndims]
        else
            Y = pca(Matrix(X), ndims)
        end
    else
        verbose && @info("Starting with random layout...")
        Y = randn(rng, Float64, n, ndims)
    end

    ndims <= 4 || throw(ArgumentError("Barnes-Hut method does not support ndims > 4 (got ndims=$ndims). Use method=:exact instead."))
    max_depth >= 0 || throw(ArgumentError("max_depth must be non-negative"))
    max_depth <= MAX_TREE_DEPTH || throw(ArgumentError("max_depth must be <= $MAX_TREE_DEPTH"))

    dY = zeros(Float64, n, ndims)
    iY = zeros(Float64, n, ndims)
    gains = ones(Float64, n, ndims)

    verbose && @info("Computing sparse P via kNN...")
    P_rows, sumP, beta = compute_sparse_P(X, distance, Float64(perplexity), 1e-5, 50, verbose, progress)

    scale_sparse_P!(P_rows, cheat_scale / sumP)
    current_scale = cheat_scale
    target_scale = 1.0
    P_rowptr, P_colidx, P_values = sparse_rows_to_csr(P_rows)

    yi_buf = zeros(Float64, ndims)
    grad_buf = zeros(Float64, ndims)

    if ndims == 2
        optimize_bh_2d!(Y, dY, iY, gains, P_rows, P_rowptr, P_colidx, P_values,
                        max_iter, Float64(theta), Float64(min_gain), Float64(eta),
                        Float64(initial_momentum), Float64(final_momentum),
                        momentum_switch_iter, stop_cheat_iter, cheat_scale, progress,
                        Int(max_depth))
        verbose && @info("Final BH t-SNE complete.")

        if extended_output
            tree_final = build_spacetree(Y, max_depth)
            kldiv = compute_KL_bh_fast(Y, P_rows, tree_final, Float64(theta), yi_buf, grad_buf)
            return Y, beta, kldiv
        else
            return Y
        end
    elseif ndims == 3
        optimize_bh_3d!(Y, dY, iY, gains, P_rows, P_rowptr, P_colidx, P_values,
                        max_iter, Float64(theta), Float64(min_gain), Float64(eta),
                        Float64(initial_momentum), Float64(final_momentum),
                        momentum_switch_iter, stop_cheat_iter, cheat_scale, progress,
                        Int(max_depth))
        verbose && @info("Final BH t-SNE complete.")

        if extended_output
            tree_final = build_spacetree(Y, max_depth)
            kldiv = compute_KL_bh_fast(Y, P_rows, tree_final, Float64(theta), yi_buf, grad_buf)
            return Y, beta, kldiv
        else
            return Y
        end
    end

    max_tid = _max_thread_id()
    thread_yi = [zeros(Float64, ndims) for _ in 1:max_tid]
    thread_grad = [zeros(Float64, ndims) for _ in 1:max_tid]
    thread_Z = zeros(Float64, max_tid)
    thread_z_ref = [Ref(0.0) for _ in 1:max_tid]
    thread_stack = [[zeros(Float64, ndims) for _ in 1:MAX_TREE_DEPTH] for _ in 1:max_tid]
    thread_z_stack = [[Ref(0.0) for _ in 1:MAX_TREE_DEPTH] for _ in 1:max_tid]
    Ymean_buf = zeros(Float64, 1, ndims)

    progress && (pb = Progress(max_iter; desc="Computing BH t-SNE"))
    progress_interval = max(max_iter ÷ 100, 1)

    for iter in 1:max_iter
        tree = build_spacetree(Y, max_depth)
        fill!(thread_Z, 0.0)
        Threads.@threads for i in 1:n
            tid = Threads.threadid()
            local_yi = thread_yi[tid]
            local_grad = thread_grad[tid]
            for d in 1:ndims
                local_yi[d] = Y[i, d]
            end
            walk_tree_into!(tree, Y, local_yi, Float64(theta), i, thread_z_ref[tid], local_grad,
                           thread_stack[tid], thread_z_stack[tid], 0)
            thread_Z[tid] += thread_z_ref[tid][]
            for d in 1:ndims
                dY[i, d] = local_grad[d]
            end
        end
        Z = sum(thread_Z)

        Zinv = 1.0 / Z

        Threads.@threads for i in 1:n
            local_yi = thread_yi[Threads.threadid()]
            for d in 1:ndims
                local_yi[d] = Y[i, d]
            end
            row_dY = @view dY[i, :]
            for d in 1:ndims
                row_dY[d] = -row_dY[d] * Zinv
            end
            for (j, p_ij) in P_rows[i]
                if j == i
                    continue
                end
                dij = 0.0
                for d in 1:ndims
                    dd = Float64(Y[i, d] - Y[j, d])
                    dij += dd * dd
                end
                q_ij = 1.0 / (1.0 + dij)
                for d in 1:ndims
                    row_dY[d] += p_ij * q_ij * Float64(Y[i, d] - Y[j, d])
                end
            end
            for d in 1:ndims
                row_dY[d] *= 4.0
            end
        end

        momentum = iter <= momentum_switch_iter ? Float64(initial_momentum) : Float64(final_momentum)
        @inbounds for idx in eachindex(gains)
            gains[idx] = max(ifelse((dY[idx] > 0) == (iY[idx] > 0),
                                    gains[idx] * 0.8,
                                    gains[idx] + 0.2),
                             Float64(min_gain))
            iY[idx] = momentum * iY[idx] - Float64(eta) * (gains[idx] * dY[idx])
            Y[idx] += iY[idx]
        end

        Y .-= mean!(Ymean_buf, Y)

        if current_scale != target_scale && iter >= min(max_iter, stop_cheat_iter)
            scale_sparse_P!(P_rows, target_scale / current_scale)
            P_values .*= target_scale / current_scale
            current_scale = target_scale
        end

        if progress && (iter == 1 || iter == max_iter || iter % progress_interval == 0)
            update!(pb, iter)
        end
    end

    progress && finish!(pb)
    verbose && @info("Final BH t-SNE complete.")

    if extended_output
        tree_final = build_spacetree(Y, max_depth)
        kldiv = compute_KL_bh_fast(Y, P_rows, tree_final, Float64(theta), yi_buf, grad_buf)
    end

    if !extended_output
        return Y
    else
        return Y, beta, kldiv
    end
end

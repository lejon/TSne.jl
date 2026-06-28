const MAX_TREE_DEPTH = 30

mutable struct FlatTree2D
    count::Vector{Int}
    point::Vector{Int}
    child1::Vector{Int}
    child2::Vector{Int}
    child3::Vector{Int}
    child4::Vector{Int}
    cx::Vector{Float64}
    cy::Vector{Float64}
    half_width::Vector{Float64}
    sumx::Vector{Float64}
    sumy::Vector{Float64}
end

FlatTree2D() = FlatTree2D(Int[], Int[], Int[], Int[], Int[], Int[],
                          Float64[], Float64[], Float64[], Float64[], Float64[])

function reset!(tree::FlatTree2D)
    empty!(tree.count)
    empty!(tree.point)
    empty!(tree.child1)
    empty!(tree.child2)
    empty!(tree.child3)
    empty!(tree.child4)
    empty!(tree.cx)
    empty!(tree.cy)
    empty!(tree.half_width)
    empty!(tree.sumx)
    empty!(tree.sumy)
    return tree
end

function push_node!(tree::FlatTree2D, cx::Float64, cy::Float64, half_width::Float64)
    push!(tree.count, 0)
    push!(tree.point, 0)
    push!(tree.child1, 0)
    push!(tree.child2, 0)
    push!(tree.child3, 0)
    push!(tree.child4, 0)
    push!(tree.cx, cx)
    push!(tree.cy, cy)
    push!(tree.half_width, half_width)
    push!(tree.sumx, 0.0)
    push!(tree.sumy, 0.0)
    return length(tree.count)
end

@inline function get_child(tree::FlatTree2D, node::Int, bits::Int)
    bits == 0 && return tree.child1[node]
    bits == 1 && return tree.child2[node]
    bits == 2 && return tree.child3[node]
    return tree.child4[node]
end

@inline function set_child!(tree::FlatTree2D, node::Int, bits::Int, child::Int)
    if bits == 0
        tree.child1[node] = child
    elseif bits == 1
        tree.child2[node] = child
    elseif bits == 2
        tree.child3[node] = child
    else
        tree.child4[node] = child
    end
    return child
end

@inline function child_bits_2d(Y::AbstractMatrix, idx::Int, cx::Float64, cy::Float64)
    bits = 0
    Y[idx, 1] > cx && (bits |= 1)
    Y[idx, 2] > cy && (bits |= 2)
    return bits
end

function ensure_child!(tree::FlatTree2D, node::Int, bits::Int)
    child = get_child(tree, node, bits)
    child != 0 && return child

    child_half = tree.half_width[node] / 2
    child_cx = tree.cx[node] + ifelse((bits & 1) == 1, child_half, -child_half)
    child_cy = tree.cy[node] + ifelse((bits & 2) == 2, child_half, -child_half)
    child = push_node!(tree, child_cx, child_cy, child_half)
    return set_child!(tree, node, bits, child)
end

@inline has_children(tree::FlatTree2D, node::Int) =
    tree.child1[node] != 0 || tree.child2[node] != 0 ||
    tree.child3[node] != 0 || tree.child4[node] != 0

function insert_into_child!(tree::FlatTree2D, Y::AbstractMatrix, idx::Int, node::Int,
                            depth::Int, max_depth::Int)
    bits = child_bits_2d(Y, idx, tree.cx[node], tree.cy[node])
    child = ensure_child!(tree, node, bits)
    insert_point!(tree, Y, idx, child, depth + 1, max_depth)
end

function insert_point!(tree::FlatTree2D, Y::AbstractMatrix, idx::Int, node::Int,
                       depth::Int, max_depth::Int)
    @inbounds begin
        tree.count[node] += 1
        tree.sumx[node] += Y[idx, 1]
        tree.sumy[node] += Y[idx, 2]

        if !has_children(tree, node)
            old = tree.point[node]
            if old == 0 && tree.count[node] == 1
                tree.point[node] = idx
                return
            end

            if old != 0
                tree.point[node] = 0
                if depth >= max_depth
                    return
                end
                insert_into_child!(tree, Y, old, node, depth, max_depth)
                insert_into_child!(tree, Y, idx, node, depth, max_depth)
                return
            end

            depth >= max_depth && return
        end

        insert_into_child!(tree, Y, idx, node, depth, max_depth)
    end
    return
end

function build_flat_tree_2d!(tree::FlatTree2D, Y::Matrix{T}, max_depth::Integer = MAX_TREE_DEPTH) where T<:Number
    n = size(Y, 1)
    max_depth >= 0 || throw(ArgumentError("max_depth must be non-negative"))
    max_depth <= MAX_TREE_DEPTH || throw(ArgumentError("max_depth must be <= $MAX_TREE_DEPTH"))
    (minsx, maxsx) = extrema(view(Y, :, 1))
    (minsy, maxsy) = extrema(view(Y, :, 2))
    cx = Float64((minsx + maxsx) / 2)
    cy = Float64((minsy + maxsy) / 2)
    half_width = Float64(max(maxsx - cx, maxsy - cy, eps(Float64)))

    reset!(tree)
    push_node!(tree, cx, cy, half_width)
    for idx in 1:n
        insert_point!(tree, Y, idx, 1, 0, Int(max_depth))
    end
    return tree
end

function compute_repulsive_forces_2d!(dY::Matrix{Float64}, Y::Matrix{T},
                                      tree::FlatTree2D, theta::Float64,
                                      thread_Z::Vector{Float64},
                                      thread_node_stack::Vector{Vector{Int}}) where T<:Number
    n = size(Y, 1)
    fill!(thread_Z, 0.0)
    # Cache field refs once — avoids repeated struct dereference inside the hot inner loop.
    t_count  = tree.count;  t_point  = tree.point
    t_child1 = tree.child1; t_child2 = tree.child2
    t_child3 = tree.child3; t_child4 = tree.child4
    t_cx     = tree.cx;     t_cy     = tree.cy
    t_hw     = tree.half_width
    t_sumx   = tree.sumx;   t_sumy   = tree.sumy

    Threads.@threads :static for i in 1:n
        tid = Threads.threadid()
        stack = thread_node_stack[tid]
        sp = 1
        stack[1] = 1
        xi = Float64(Y[i, 1])
        yi = Float64(Y[i, 2])
        gx = 0.0
        gy = 0.0
        zi = 0.0

        @inbounds while sp > 0
            node = stack[sp]
            sp -= 1
            cnt = t_count[node]
            cnt == 0 && continue

            leaf = t_child1[node] == 0 && t_child2[node] == 0 &&
                   t_child3[node] == 0 && t_child4[node] == 0
            if leaf
                p = t_point[node]
                if p > 0
                    p == i && continue
                    dx = xi - Float64(Y[p, 1])
                    dy = yi - Float64(Y[p, 2])
                    d2 = dx * dx + dy * dy
                    q = 1.0 / (1.0 + d2)
                    zi += q
                    q2 = q * q
                    gx += q2 * dx
                    gy += q2 * dy
                    continue
                end
            end

            hw = t_hw[node]
            cmx = t_sumx[node] / cnt
            cmy = t_sumy[node] / cnt
            dx = xi - cmx
            dy = yi - cmy
            d2 = dx * dx + dy * dy
            contains_query = abs(xi - t_cx[node]) < hw &&
                             abs(yi - t_cy[node]) < hw

            if !contains_query && (leaf || hw * hw < theta * theta * d2)
                q = 1.0 / (1.0 + d2)
                zi += cnt * q
                q2 = q * q * cnt
                gx += q2 * dx
                gy += q2 * dy
            elseif leaf
                count_excl = cnt - 1
                count_excl <= 0 && continue
                ex_cmx = (t_sumx[node] - xi) / count_excl
                ex_cmy = (t_sumy[node] - yi) / count_excl
                dx = xi - ex_cmx
                dy = yi - ex_cmy
                d2 = dx * dx + dy * dy
                q = 1.0 / (1.0 + d2)
                zi += count_excl * q
                q2 = q * q * count_excl
                gx += q2 * dx
                gy += q2 * dy
            else
                c1 = t_child1[node]; c2 = t_child2[node]
                c3 = t_child3[node]; c4 = t_child4[node]
                c1 != 0 && (sp += 1; stack[sp] = c1)
                c2 != 0 && (sp += 1; stack[sp] = c2)
                c3 != 0 && (sp += 1; stack[sp] = c3)
                c4 != 0 && (sp += 1; stack[sp] = c4)
            end
        end

        dY[i, 1] = gx
        dY[i, 2] = gy
        thread_Z[tid] += zi
    end

    return sum(thread_Z)
end

mutable struct FlatTree3D
    count::Vector{Int}
    point::Vector{Int}
    child1::Vector{Int}
    child2::Vector{Int}
    child3::Vector{Int}
    child4::Vector{Int}
    child5::Vector{Int}
    child6::Vector{Int}
    child7::Vector{Int}
    child8::Vector{Int}
    cx::Vector{Float64}
    cy::Vector{Float64}
    cz::Vector{Float64}
    half_width::Vector{Float64}
    sumx::Vector{Float64}
    sumy::Vector{Float64}
    sumz::Vector{Float64}
end

FlatTree3D() = FlatTree3D(Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[],
                          Float64[], Float64[], Float64[], Float64[],
                          Float64[], Float64[], Float64[])

function reset!(tree::FlatTree3D)
    empty!(tree.count)
    empty!(tree.point)
    empty!(tree.child1)
    empty!(tree.child2)
    empty!(tree.child3)
    empty!(tree.child4)
    empty!(tree.child5)
    empty!(tree.child6)
    empty!(tree.child7)
    empty!(tree.child8)
    empty!(tree.cx)
    empty!(tree.cy)
    empty!(tree.cz)
    empty!(tree.half_width)
    empty!(tree.sumx)
    empty!(tree.sumy)
    empty!(tree.sumz)
    return tree
end

function push_node!(tree::FlatTree3D, cx::Float64, cy::Float64, cz::Float64, half_width::Float64)
    push!(tree.count, 0)
    push!(tree.point, 0)
    push!(tree.child1, 0)
    push!(tree.child2, 0)
    push!(tree.child3, 0)
    push!(tree.child4, 0)
    push!(tree.child5, 0)
    push!(tree.child6, 0)
    push!(tree.child7, 0)
    push!(tree.child8, 0)
    push!(tree.cx, cx)
    push!(tree.cy, cy)
    push!(tree.cz, cz)
    push!(tree.half_width, half_width)
    push!(tree.sumx, 0.0)
    push!(tree.sumy, 0.0)
    push!(tree.sumz, 0.0)
    return length(tree.count)
end

@inline function get_child(tree::FlatTree3D, node::Int, bits::Int)
    bits == 0 && return tree.child1[node]
    bits == 1 && return tree.child2[node]
    bits == 2 && return tree.child3[node]
    bits == 3 && return tree.child4[node]
    bits == 4 && return tree.child5[node]
    bits == 5 && return tree.child6[node]
    bits == 6 && return tree.child7[node]
    return tree.child8[node]
end

@inline function set_child!(tree::FlatTree3D, node::Int, bits::Int, child::Int)
    if bits == 0
        tree.child1[node] = child
    elseif bits == 1
        tree.child2[node] = child
    elseif bits == 2
        tree.child3[node] = child
    elseif bits == 3
        tree.child4[node] = child
    elseif bits == 4
        tree.child5[node] = child
    elseif bits == 5
        tree.child6[node] = child
    elseif bits == 6
        tree.child7[node] = child
    else
        tree.child8[node] = child
    end
    return child
end

@inline function child_bits_3d(Y::AbstractMatrix, idx::Int, cx::Float64, cy::Float64, cz::Float64)
    bits = 0
    Y[idx, 1] > cx && (bits |= 1)
    Y[idx, 2] > cy && (bits |= 2)
    Y[idx, 3] > cz && (bits |= 4)
    return bits
end

function ensure_child!(tree::FlatTree3D, node::Int, bits::Int)
    child = get_child(tree, node, bits)
    child != 0 && return child

    child_half = tree.half_width[node] / 2
    child_cx = tree.cx[node] + ifelse((bits & 1) == 1, child_half, -child_half)
    child_cy = tree.cy[node] + ifelse((bits & 2) == 2, child_half, -child_half)
    child_cz = tree.cz[node] + ifelse((bits & 4) == 4, child_half, -child_half)
    child = push_node!(tree, child_cx, child_cy, child_cz, child_half)
    return set_child!(tree, node, bits, child)
end

@inline has_children(tree::FlatTree3D, node::Int) =
    tree.child1[node] != 0 || tree.child2[node] != 0 ||
    tree.child3[node] != 0 || tree.child4[node] != 0 ||
    tree.child5[node] != 0 || tree.child6[node] != 0 ||
    tree.child7[node] != 0 || tree.child8[node] != 0

function insert_into_child!(tree::FlatTree3D, Y::AbstractMatrix, idx::Int, node::Int,
                            depth::Int, max_depth::Int)
    bits = child_bits_3d(Y, idx, tree.cx[node], tree.cy[node], tree.cz[node])
    child = ensure_child!(tree, node, bits)
    insert_point!(tree, Y, idx, child, depth + 1, max_depth)
end

function insert_point!(tree::FlatTree3D, Y::AbstractMatrix, idx::Int, node::Int,
                       depth::Int, max_depth::Int)
    @inbounds begin
        tree.count[node] += 1
        tree.sumx[node] += Y[idx, 1]
        tree.sumy[node] += Y[idx, 2]
        tree.sumz[node] += Y[idx, 3]

        if !has_children(tree, node)
            old = tree.point[node]
            if old == 0 && tree.count[node] == 1
                tree.point[node] = idx
                return
            end

            if old != 0
                tree.point[node] = 0
                if depth >= max_depth
                    return
                end
                insert_into_child!(tree, Y, old, node, depth, max_depth)
                insert_into_child!(tree, Y, idx, node, depth, max_depth)
                return
            end

            depth >= max_depth && return
        end

        insert_into_child!(tree, Y, idx, node, depth, max_depth)
    end
    return
end

function build_flat_tree_3d!(tree::FlatTree3D, Y::Matrix{T}, max_depth::Integer = MAX_TREE_DEPTH) where T<:Number
    n = size(Y, 1)
    max_depth >= 0 || throw(ArgumentError("max_depth must be non-negative"))
    max_depth <= MAX_TREE_DEPTH || throw(ArgumentError("max_depth must be <= $MAX_TREE_DEPTH"))
    (minsx, maxsx) = extrema(view(Y, :, 1))
    (minsy, maxsy) = extrema(view(Y, :, 2))
    (minsz, maxsz) = extrema(view(Y, :, 3))
    cx = Float64((minsx + maxsx) / 2)
    cy = Float64((minsy + maxsy) / 2)
    cz = Float64((minsz + maxsz) / 2)
    half_width = Float64(max(maxsx - cx, maxsy - cy, maxsz - cz, eps(Float64)))

    reset!(tree)
    push_node!(tree, cx, cy, cz, half_width)
    for idx in 1:n
        insert_point!(tree, Y, idx, 1, 0, Int(max_depth))
    end
    return tree
end

function compute_repulsive_forces_3d!(dY::Matrix{Float64}, Y::Matrix{T},
                                      tree::FlatTree3D, theta::Float64,
                                      thread_Z::Vector{Float64},
                                      thread_node_stack::Vector{Vector{Int}}) where T<:Number
    n = size(Y, 1)
    fill!(thread_Z, 0.0)
    # Cache field refs once — avoids repeated struct dereference inside the hot inner loop.
    t_count  = tree.count;  t_point  = tree.point
    t_child1 = tree.child1; t_child2 = tree.child2
    t_child3 = tree.child3; t_child4 = tree.child4
    t_child5 = tree.child5; t_child6 = tree.child6
    t_child7 = tree.child7; t_child8 = tree.child8
    t_cx     = tree.cx;     t_cy     = tree.cy;    t_cz     = tree.cz
    t_hw     = tree.half_width
    t_sumx   = tree.sumx;   t_sumy   = tree.sumy;  t_sumz   = tree.sumz

    Threads.@threads :static for i in 1:n
        tid = Threads.threadid()
        stack = thread_node_stack[tid]
        sp = 1
        stack[1] = 1
        xi = Float64(Y[i, 1])
        yi = Float64(Y[i, 2])
        zi_coord = Float64(Y[i, 3])
        gx = 0.0
        gy = 0.0
        gz = 0.0
        zi = 0.0

        @inbounds while sp > 0
            node = stack[sp]
            sp -= 1
            cnt = t_count[node]
            cnt == 0 && continue

            leaf = t_child1[node] == 0 && t_child2[node] == 0 &&
                   t_child3[node] == 0 && t_child4[node] == 0 &&
                   t_child5[node] == 0 && t_child6[node] == 0 &&
                   t_child7[node] == 0 && t_child8[node] == 0
            if leaf
                p = t_point[node]
                if p > 0
                    p == i && continue
                    dx = xi - Float64(Y[p, 1])
                    dy = yi - Float64(Y[p, 2])
                    dz = zi_coord - Float64(Y[p, 3])
                    d2 = dx * dx + dy * dy + dz * dz
                    q = 1.0 / (1.0 + d2)
                    zi += q
                    q2 = q * q
                    gx += q2 * dx
                    gy += q2 * dy
                    gz += q2 * dz
                    continue
                end
            end

            hw = t_hw[node]
            cmx = t_sumx[node] / cnt
            cmy = t_sumy[node] / cnt
            cmz = t_sumz[node] / cnt
            dx = xi - cmx
            dy = yi - cmy
            dz = zi_coord - cmz
            d2 = dx * dx + dy * dy + dz * dz
            contains_query = abs(xi - t_cx[node]) < hw &&
                             abs(yi - t_cy[node]) < hw &&
                             abs(zi_coord - t_cz[node]) < hw

            if !contains_query && (leaf || hw * hw < theta * theta * d2)
                q = 1.0 / (1.0 + d2)
                zi += cnt * q
                q2 = q * q * cnt
                gx += q2 * dx
                gy += q2 * dy
                gz += q2 * dz
            elseif leaf
                count_excl = cnt - 1
                count_excl <= 0 && continue
                ex_cmx = (t_sumx[node] - xi) / count_excl
                ex_cmy = (t_sumy[node] - yi) / count_excl
                ex_cmz = (t_sumz[node] - zi_coord) / count_excl
                dx = xi - ex_cmx
                dy = yi - ex_cmy
                dz = zi_coord - ex_cmz
                d2 = dx * dx + dy * dy + dz * dz
                q = 1.0 / (1.0 + d2)
                zi += count_excl * q
                q2 = q * q * count_excl
                gx += q2 * dx
                gy += q2 * dy
                gz += q2 * dz
            else
                c1 = t_child1[node]; c2 = t_child2[node]
                c3 = t_child3[node]; c4 = t_child4[node]
                c5 = t_child5[node]; c6 = t_child6[node]
                c7 = t_child7[node]; c8 = t_child8[node]
                c1 != 0 && (sp += 1; stack[sp] = c1)
                c2 != 0 && (sp += 1; stack[sp] = c2)
                c3 != 0 && (sp += 1; stack[sp] = c3)
                c4 != 0 && (sp += 1; stack[sp] = c4)
                c5 != 0 && (sp += 1; stack[sp] = c5)
                c6 != 0 && (sp += 1; stack[sp] = c6)
                c7 != 0 && (sp += 1; stack[sp] = c7)
                c8 != 0 && (sp += 1; stack[sp] = c8)
            end
        end

        dY[i, 1] = gx
        dY[i, 2] = gy
        dY[i, 3] = gz
        thread_Z[tid] += zi
    end

    return sum(thread_Z)
end

mutable struct SpaceNode
    n::Int
    point_idx::Int
    child_bits::Int
    indices::Vector{Int}
    center_of_mass::Vector{Float64}
    bbox_center::Vector{Float64}
    half_width::Float64
    children::Vector{SpaceNode}
end

function build_tree!(Y::Matrix{T}, indices::Vector{Int},
                     bbox_center::Vector{Float64}, half_width::Float64,
                     depth::Int = 0, child_bits::Int = 0,
                     max_depth::Int = MAX_TREE_DEPTH) where T<:Number
    n = length(indices)
    if n == 0
        return nothing
    end

    ncols = size(Y, 2)

    if n == 1
        idx = indices[1]
        yi = [Y[idx, d] for d in 1:ncols]
        return SpaceNode(1, idx, child_bits, [idx], yi, bbox_center, half_width, SpaceNode[])
    end

    center_of_mass = zeros(Float64, ncols)
    for idx in indices
        for d in 1:ncols
            center_of_mass[d] += Y[idx, d]
        end
    end
    for d in 1:ncols
        center_of_mass[d] /= n
    end

    if depth >= max_depth || half_width < 1e-15
        return SpaceNode(n, -1, child_bits, copy(indices), center_of_mass, bbox_center, half_width, SpaceNode[])
    end

    nchildren = 1 << ncols
    child_indices = [Int[] for _ in 1:nchildren]
    for idx in indices
        bits = 0
        for d in 1:ncols
            if Y[idx, d] > bbox_center[d]
                bits |= (1 << (d - 1))
            end
        end
        push!(child_indices[bits + 1], idx)
    end

    child_half = half_width / 2
    children = SpaceNode[]
    for (bits, cidxs) in enumerate(child_indices)
        if isempty(cidxs)
            continue
        end
        child_center = copy(bbox_center)
        bitval = bits - 1
        for d in 1:ncols
            child_center[d] += ifelse((bitval >> (d - 1)) & 1 == 1, child_half, -child_half)
        end
        child = build_tree!(Y, cidxs, child_center, child_half, depth + 1, bitval, max_depth)
        if child !== nothing
            push!(children, child)
        end
    end

    return SpaceNode(n, -1, child_bits, Int[], center_of_mass, bbox_center, half_width, children)
end

function build_spacetree(Y::Matrix{T}, max_depth::Integer = MAX_TREE_DEPTH) where T<:Number
    n = size(Y, 1)
    n == 0 && error("Empty Y matrix")
    max_depth >= 0 || throw(ArgumentError("max_depth must be non-negative"))
    max_depth <= MAX_TREE_DEPTH || throw(ArgumentError("max_depth must be <= $MAX_TREE_DEPTH"))

    ncols = size(Y, 2)
    ncols > 4 && error("Barnes-Hut tree not supported for ndims > 4 (got ndims=$ncols)")

    mins_maxs = [extrema(view(Y, :, d)) for d in 1:ncols]
    bbox_center = [(mn + mx) / 2 for (mn, mx) in mins_maxs]
    half_width = maximum(mx - bbox_center[d] for (d, (mn, mx)) in enumerate(mins_maxs))
    half_width = max(half_width, eps(typeof(half_width)))

    sums = [sum(Y[i, d] for d in 1:ncols) for i in 1:n]
    indices = sortperm(sums)

    root = build_tree!(Y, indices, bbox_center, half_width, 0, 0, Int(max_depth))
    return root
end

function walk_tree_into!(node::SpaceNode, Y::Matrix{T}, yi::Vector{Float64}, theta::Float64,
                         point_idx::Int, z_out::Ref{Float64}, grad::Vector{Float64}) where T<:Number
    ndims = length(yi)
    stack = [zeros(Float64, ndims) for _ in 1:MAX_TREE_DEPTH]
    z_stack = [Ref(0.0) for _ in 1:MAX_TREE_DEPTH]
    walk_tree_into!(node, Y, yi, theta, point_idx, z_out, grad, stack, z_stack, 0, true)
end

function add_aggregate_interaction!(count::Int, center::Vector{Float64}, yi::Vector{Float64},
                                    z_out::Ref{Float64}, grad::Vector{Float64})
    ndims = length(yi)
    d2 = 0.0
    for d in 1:ndims
        diff = yi[d] - center[d]
        d2 += diff * diff
    end

    q_ij = 1.0 / (1.0 + d2)
    z_out[] += count * q_ij
    q2 = q_ij * q_ij * count
    for d in 1:ndims
        grad[d] += q2 * (yi[d] - center[d])
    end
end

function add_direct_interactions!(node::SpaceNode, Y::Matrix{T}, yi::Vector{Float64},
                                  point_idx::Int, z_out::Ref{Float64},
                                  grad::Vector{Float64}) where T<:Number
    ndims = length(yi)
    for idx in node.indices
        idx == point_idx && continue

        d2 = 0.0
        for d in 1:ndims
            diff = yi[d] - Y[idx, d]
            d2 += diff * diff
        end

        q_ij = 1.0 / (1.0 + d2)
        z_out[] += q_ij
        q2 = q_ij * q_ij
        for d in 1:ndims
            grad[d] += q2 * (yi[d] - Y[idx, d])
        end
    end
end

function add_self_excluding_aggregate!(node::SpaceNode, yi::Vector{Float64},
                                       point_idx::Int, z_out::Ref{Float64},
                                       grad::Vector{Float64})
    count = node.n - 1
    count <= 0 && return

    center = copy(node.center_of_mass)
    if count != node.n
        for d in 1:length(yi)
            center[d] = (node.n * node.center_of_mass[d] - yi[d]) / count
        end
    end
    add_aggregate_interaction!(count, center, yi, z_out, grad)
end

function walk_tree_into!(node::SpaceNode, Y::Union{Nothing,AbstractMatrix},
                         yi::Vector{Float64}, theta::Float64,
                         point_idx::Int, z_out::Ref{Float64}, grad::Vector{Float64},
                         stack::Vector{Vector{Float64}},
                         z_stack::Vector{Base.RefValue{Float64}}, depth::Int)
    walk_tree_into!(node, Y, yi, theta, point_idx, z_out, grad, stack, z_stack, depth, true)
end

function walk_tree_into!(node::SpaceNode, Y::Union{Nothing,AbstractMatrix},
                         yi::Vector{Float64}, theta::Float64,
                         point_idx::Int, z_out::Ref{Float64}, grad::Vector{Float64},
                         stack::Vector{Vector{Float64}},
                         z_stack::Vector{Base.RefValue{Float64}}, depth::Int,
                         contains_query::Bool)
    ndims = length(yi)

    if node.n == 0
        z_out[] = 0.0
        fill!(grad, 0.0)
        return
    end

    if node.n == 1 && node.point_idx >= 0
        if node.point_idx == point_idx
            z_out[] = 0.0
            fill!(grad, 0.0)
            return
        end
        z_out[] = 0.0
        fill!(grad, 0.0)
        add_aggregate_interaction!(1, node.center_of_mass, yi, z_out, grad)
        return
    end

    d2 = 0.0
    for dd in 1:ndims
        diff = yi[dd] - node.center_of_mass[dd]
        d2 += diff * diff
    end
    if !contains_query && (d2 < 1e-30 || isempty(node.children) || node.half_width * node.half_width < theta * theta * d2)
        z_out[] = 0.0
        fill!(grad, 0.0)
        add_aggregate_interaction!(node.n, node.center_of_mass, yi, z_out, grad)
        return
    end

    if isempty(node.children)
        z_out[] = 0.0
        fill!(grad, 0.0)
        if Y === nothing
            add_self_excluding_aggregate!(node, yi, point_idx, z_out, grad)
        else
            add_direct_interactions!(node, Y, yi, point_idx, z_out, grad)
        end
        return
    end

    z_out[] = 0.0
    fill!(grad, 0.0)
    child_z = z_stack[depth + 1]
    child_grad = stack[depth + 1]
    query_bits = 0
    if contains_query
        for d in 1:ndims
            if yi[d] > node.bbox_center[d]
                query_bits |= (1 << (d - 1))
            end
        end
    end
    for child in node.children
        fill!(child_grad, 0.0)
        child_contains_query = contains_query && child.child_bits == query_bits
        walk_tree_into!(child, Y, yi, theta, point_idx, child_z, child_grad,
                       stack, z_stack, depth + 1, child_contains_query)
        z_out[] += child_z[]
        for dd in 1:ndims
            grad[dd] += child_grad[dd]
        end
    end
end

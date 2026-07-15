# FIt-SNE: FFT-accelerated t-SNE (Linderman et al. 2019, doi:10.1038/s41592-018-0308-4)
#
# Replaces the Barnes-Hut quadtree traversal with an interpolation + FFT convolution
# on a coarse grid.  Complexity per iteration: O(n + M² log M) where M = n_boxes × 3.
#
# Algorithm (2D, dof=1, standard Student-t kernel):
#   We need φ_d(y_i) = Σ_j charge_d(y_j) · K̃(y_i − y_j) for d = 0..3
#   with kernel K̃(Δx,Δy) = (1 + Δx² + Δy²)^{-2}
#   and charges:  d=0 → 1,   d=1 → y_j1,   d=2 → y_j2,   d=3 → ‖y_j‖²
#
#   Key identity: Σ_i [(1+‖y_i‖²)φ₀ − 2(y_i1·φ₁ + y_i2·φ₂) + φ₃] − n = Z
#
#   Repulsive gradient: dY[i,1] = y_i1·φ₀ − φ₁,   dY[i,2] = y_i2·φ₀ − φ₂

# ── Workspace ────────────────────────────────────────────────────────────────

mutable struct FFTWorkspace{P1, P2}
    n       :: Int
    M       :: Int    # n_boxes × n_interp  (grid size per dimension)
    n_boxes :: Int
    n_interp:: Int    # fixed = 3

    # 4 charge arrays, each (2M, 2M): spread into them, irfft writes result back in-place
    W_pad1  :: Matrix{Float64}
    W_pad2  :: Matrix{Float64}
    W_pad3  :: Matrix{Float64}
    W_pad4  :: Matrix{Float64}
    K_pad   :: Matrix{Float64}     # (2M, 2M)  kernel; rebuilt only when grid spacing changes
    K_hat   :: Matrix{ComplexF64}  # (M+1, 2M) rfft of K_pad
    W_hat   :: Matrix{ComplexF64}  # (M+1, 2M) rfft scratch (destroyed by irfft)

    # Per-point arrays — (4, n) layout: charges/PHI for all 4 fields at point i are consecutive
    PHI     :: Matrix{Float64}     # (4, n)
    charges :: Matrix{Float64}     # (4, n) — 1, y1, y2, ‖y‖²
    box_idx :: Matrix{Int}         # (n, 2)
    Lx      :: Matrix{Float64}     # (3, n)
    Ly      :: Matrix{Float64}     # (3, n)

    # Per-thread charge accumulation buffers: each thread scatters independently, then reduce
    thread_W1 :: Vector{Matrix{Float64}}   # nthreads × (2M, 2M)
    thread_W2 :: Vector{Matrix{Float64}}
    thread_W3 :: Vector{Matrix{Float64}}
    thread_W4 :: Vector{Matrix{Float64}}

    # Cached grid spacings: K_pad/K_hat are only rebuilt when spacing changes by >0.2%
    last_hx :: Float64
    last_hy :: Float64

    rfft_plan  :: P1
    irfft_plan :: P2
end

function FFTWorkspace(n::Int, n_boxes::Int, n_interp::Int = 3; flags::UInt32 = FFTW.PATIENT)
    M        = n_boxes * n_interp
    W_pad1   = zeros(Float64, 2M, 2M)
    W_pad2   = zeros(Float64, 2M, 2M)
    W_pad3   = zeros(Float64, 2M, 2M)
    W_pad4   = zeros(Float64, 2M, 2M)
    K_pad    = zeros(Float64, 2M, 2M)
    K_hat    = zeros(ComplexF64, M + 1, 2M)
    W_hat    = zeros(ComplexF64, M + 1, 2M)
    PHI      = zeros(Float64, 4, n)
    charges  = zeros(Float64, 4, n)
    box_idx  = zeros(Int,     n, 2)
    Lx       = zeros(Float64, 3, n)
    Ly       = zeros(Float64, 3, n)
    nt       = _max_thread_id()
    thread_W1 = [zeros(Float64, 2M, 2M) for _ in 1:nt]
    thread_W2 = [zeros(Float64, 2M, 2M) for _ in 1:nt]
    thread_W3 = [zeros(Float64, 2M, 2M) for _ in 1:nt]
    thread_W4 = [zeros(Float64, 2M, 2M) for _ in 1:nt]

    rfft_plan  = plan_rfft(W_pad1;      flags = flags)
    irfft_plan = plan_irfft(W_hat, 2M; flags = flags)

    return FFTWorkspace(n, M, n_boxes, n_interp,
                        W_pad1, W_pad2, W_pad3, W_pad4,
                        K_pad, K_hat, W_hat,
                        PHI, charges, box_idx, Lx, Ly,
                        thread_W1, thread_W2, thread_W3, thread_W4,
                        0.0, 0.0,
                        rfft_plan, irfft_plan)
end

# ── Grid helpers ─────────────────────────────────────────────────────────────

# Smallest integer ≥ n whose prime factors are all in {2, 3, 5, 7} (FFTW-optimal)
next_smooth_number(n::Int) = Base.nextprod((2, 3, 5, 7), n)

# Bounding-box ranges of the current embedding (raw, before padding).
function _embedding_ranges(Y::Matrix{Float64})
    x_min = Y[1, 1]; x_max = Y[1, 1]
    y_min = Y[1, 2]; y_max = Y[1, 2]
    @inbounds for i in 2:size(Y, 1)
        xi = Y[i, 1]; yi = Y[i, 2]
        xi < x_min && (x_min = xi); xi > x_max && (x_max = xi)
        yi < y_min && (y_min = yi); yi > y_max && (y_max = yi)
    end
    return x_max - x_min, y_max - y_min
end

# n_boxes is determined by the *smaller* of a range-based estimate and a dataset-size cap.
# The dataset-size cap (≈ sqrt(n)/3) keeps M²logM ≈ n logn so FFT beats BH at large n
# regardless of how visually spread the embedding gets.
function compute_n_boxes(n::Int, x_range::Float64, y_range::Float64)
    r = max(x_range, y_range)
    range_based = isfinite(r) ? ceil(Int, min(r * 1.5, 200.0)) : 200
    n_based     = ceil(Int, sqrt(Float64(n)) / 3)
    raw         = clamp(min(range_based, n_based), 50, 200)
    return next_smooth_number(raw)
end

# ── Core per-iteration function ───────────────────────────────────────────────

function compute_repulsive_forces_fft_2d!(dY::Matrix{Float64},
                                          Y::Matrix{Float64},
                                          ws::FFTWorkspace)
    n        = ws.n
    M        = ws.M
    n_boxes  = ws.n_boxes
    n_interp = ws.n_interp   # = 3

    # ── Step A: bounding box with 25% padding ────────────────────────────────
    x_min = Y[1, 1]; x_max = Y[1, 1]
    y_min = Y[1, 2]; y_max = Y[1, 2]
    @inbounds for i in 2:n
        xi = Y[i, 1]; yi = Y[i, 2]
        xi < x_min && (x_min = xi); xi > x_max && (x_max = xi)
        yi < y_min && (y_min = yi); yi > y_max && (y_max = yi)
    end
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    pad = 0.25
    x_min -= pad * x_range; x_max += pad * x_range
    y_min -= pad * y_range; y_max += pad * y_range
    wx = (x_max - x_min) / n_boxes
    wy = (y_max - y_min) / n_boxes
    hx = wx / n_interp
    hy = wy / n_interp

    # ── Step B: build K_pad only when grid spacing changed by >0.2% ──────────
    # K_pad[i,j] = f(di, dj) where di = min(i-1, 2M-(i-1)) ∈ {0..M}, same for dj.
    # Uses 4-fold symmetry: fill unique top-left (M+1)×(M+1) block, then mirror.
    K_pad     = ws.K_pad
    K_hat     = ws.K_hat
    rfft_plan = ws.rfft_plan
    hx_changed = abs(hx - ws.last_hx) / (ws.last_hx + 1e-12) > 2e-3
    hy_changed = abs(hy - ws.last_hy) / (ws.last_hy + 1e-12) > 2e-3
    if hx_changed || hy_changed
        @inbounds for j in 1:M + 1
            dj = j - 1; dj2hy2 = (dj * hy)^2
            for i in 1:M + 1
                di = i - 1
                K_pad[i, j] = (1.0 + (di * hx)^2 + dj2hy2)^(-2)
            end
        end
        # Mirror columns M+2..2M from columns M..2 (j mirrors to 2M+2-j)
        @inbounds for j in M + 2:2M
            j_src = 2M + 2 - j
            for i in 1:M + 1
                K_pad[i, j] = K_pad[i, j_src]
            end
        end
        # Mirror rows M+2..2M from rows M..2 (i mirrors to 2M+2-i)
        @inbounds for j in 1:2M
            for i in 2:M
                K_pad[2M + 2 - i, j] = K_pad[i, j]
            end
        end
        mul!(K_hat, rfft_plan, K_pad)
        ws.last_hx = hx
        ws.last_hy = hy
    end

    # ── Step C: per-point box assignment, Lagrange weights, charges (parallel) ─
    charges = ws.charges
    box_idx = ws.box_idx
    Lx      = ws.Lx
    Ly      = ws.Ly
    @inbounds Threads.@threads for i in 1:n
        xi = Y[i, 1]; yi2 = Y[i, 2]
        ux = (xi  - x_min) / wx
        uy = (yi2 - y_min) / wy
        bx = clamp(floor(Int, ux) + 1, 1, n_boxes)
        by = clamp(floor(Int, uy) + 1, 1, n_boxes)
        box_idx[i, 1] = bx
        box_idx[i, 2] = by
        xf = ux - (bx - 1)
        yf = uy - (by - 1)
        Lx[1, i] =  4.5 * (xf - 0.5) * (xf - 5.0/6.0)
        Lx[2, i] = -9.0 * (xf - 1.0/6.0) * (xf - 5.0/6.0)
        Lx[3, i] =  4.5 * (xf - 1.0/6.0) * (xf - 0.5)
        Ly[1, i] =  4.5 * (yf - 0.5) * (yf - 5.0/6.0)
        Ly[2, i] = -9.0 * (yf - 1.0/6.0) * (yf - 5.0/6.0)
        Ly[3, i] =  4.5 * (yf - 1.0/6.0) * (yf - 0.5)
        charges[2, i] = xi
        charges[3, i] = yi2
        charges[4, i] = xi * xi + yi2 * yi2
    end

    # ── Step D: parallel spread — each thread scatters into its own buffers ───
    # Avoids write conflicts: thread tid accumulates exclusively into thread_W*[tid].
    # Serial reduce sums thread buffers into W_pad1..4 for FFT input.
    thread_W1 = ws.thread_W1; thread_W2 = ws.thread_W2
    thread_W3 = ws.thread_W3; thread_W4 = ws.thread_W4
    @inbounds Threads.@threads for k in 1:length(thread_W1)
        fill!(thread_W1[k], 0.0)
        fill!(thread_W2[k], 0.0)
        fill!(thread_W3[k], 0.0)
        fill!(thread_W4[k], 0.0)
    end
    Threads.@threads :static for i in 1:n
        tid = Threads.threadid()
        tw1 = thread_W1[tid]; tw2 = thread_W2[tid]
        tw3 = thread_W3[tid]; tw4 = thread_W4[tid]
        bx = box_idx[i, 1]; by = box_idx[i, 2]
        c2 = charges[2, i]; c3 = charges[3, i]; c4 = charges[4, i]
        gx_base = (bx - 1) * n_interp
        gy_base = (by - 1) * n_interp
        for jy in 1:3
            gy  = gy_base + jy
            ly  = Ly[jy, i]
            for jx in 1:3
                gx = gx_base + jx
                w  = Lx[jx, i] * ly
                tw1[gx, gy] += w
                tw2[gx, gy] += w * c2
                tw3[gx, gy] += w * c3
                tw4[gx, gy] += w * c4
            end
        end
    end
    W_pad1 = ws.W_pad1; W_pad2 = ws.W_pad2
    W_pad3 = ws.W_pad3; W_pad4 = ws.W_pad4
    fill!(W_pad1, 0.0); fill!(W_pad2, 0.0)
    fill!(W_pad3, 0.0); fill!(W_pad4, 0.0)
    @inbounds for k in 1:length(thread_W1)
        W_pad1 .+= thread_W1[k]
        W_pad2 .+= thread_W2[k]
        W_pad3 .+= thread_W3[k]
        W_pad4 .+= thread_W4[k]
    end

    # ── Step E: 4 sequential FFT convolutions (single shared scratch buffer) ───
    W_hat      = ws.W_hat
    irfft_plan = ws.irfft_plan
    mul!(W_hat, rfft_plan, W_pad1); @inbounds W_hat .*= K_hat; mul!(W_pad1, irfft_plan, W_hat)
    mul!(W_hat, rfft_plan, W_pad2); @inbounds W_hat .*= K_hat; mul!(W_pad2, irfft_plan, W_hat)
    mul!(W_hat, rfft_plan, W_pad3); @inbounds W_hat .*= K_hat; mul!(W_pad3, irfft_plan, W_hat)
    mul!(W_hat, rfft_plan, W_pad4); @inbounds W_hat .*= K_hat; mul!(W_pad4, irfft_plan, W_hat)

    # ── Step F: single parallel gather — all 4 PHI values per point at once ──
    # W_pad1..4 now hold the convolution output (valid region [1:M, 1:M]).
    PHI = ws.PHI
    @inbounds Threads.@threads for i in 1:n
        phi1 = 0.0; phi2 = 0.0; phi3 = 0.0; phi4 = 0.0
        bx = box_idx[i, 1]; by = box_idx[i, 2]
        gx_base = (bx - 1) * n_interp
        gy_base = (by - 1) * n_interp
        for jy in 1:3
            gy  = gy_base + jy
            wly = Ly[jy, i]
            for jx in 1:3
                gx   = gx_base + jx
                w    = Lx[jx, i] * wly
                phi1 += w * W_pad1[gx, gy]
                phi2 += w * W_pad2[gx, gy]
                phi3 += w * W_pad3[gx, gy]
                phi4 += w * W_pad4[gx, gy]
            end
        end
        # PHI layout (4, n): all 4 fields for point i are consecutive
        PHI[1, i] = phi1
        PHI[2, i] = phi2
        PHI[3, i] = phi3
        PHI[4, i] = phi4
    end

    # ── Step G: compute Z (partition function) ────────────────────────────────
    Z = 0.0
    @inbounds for i in 1:n
        y1 = Y[i, 1]; y2 = Y[i, 2]
        Z += (1.0 + y1*y1 + y2*y2) * PHI[1, i] -
             2.0 * (y1 * PHI[2, i] + y2 * PHI[3, i]) +
             PHI[4, i]
    end
    Z -= n
    Z = max(Z, eps(Float64))

    # ── Step H: repulsive gradient ────────────────────────────────────────────
    @inbounds Threads.@threads for i in 1:n
        dY[i, 1] = Y[i, 1] * PHI[1, i] - PHI[2, i]
        dY[i, 2] = Y[i, 2] * PHI[1, i] - PHI[3, i]
    end

    return Z
end

# ── Iteration loop ────────────────────────────────────────────────────────────

function optimize_fft_2d!(Y        :: Matrix{Float64},
                          dY       :: Matrix{Float64},
                          iY       :: Matrix{Float64},
                          gains    :: Matrix{Float64},
                          P_rows   :: Vector{Vector{Pair{Int,Float64}}},
                          P_rowptr :: Vector{Int},
                          P_colidx :: Vector{Int},
                          P_values :: Vector{Float64},
                          max_iter :: Int,
                          min_gain :: Float64,
                          eta      :: Float64,
                          initial_momentum     :: Float64,
                          final_momentum       :: Float64,
                          momentum_switch_iter :: Int,
                          stop_cheat_iter      :: Int,
                          cheat_scale          :: Float64,
                          progress      :: Bool,
                          n_boxes_fixed :: Int)   # 0 = auto-adaptive from embedding range
    n             = size(Y, 1)
    init_boxes    = n_boxes_fixed > 0 ? n_boxes_fixed : compute_n_boxes(n, 8.0, 8.0)
    ws            = FFTWorkspace(n, init_boxes; flags = FFTW.PATIENT)
    Ymean         = zeros(Float64, 1, 2)
    current_scale = cheat_scale
    target_scale  = 1.0

    progress && (pb = Progress(max_iter; desc="Computing FFT t-SNE"))
    progress_interval = max(max_iter ÷ 100, 1)

    for iter in 1:max_iter
        # Grow the workspace when the embedding spreads beyond the current grid.
        # Never shrink — the PATIENT plan is expensive; rebuilds use ESTIMATE.
        if n_boxes_fixed == 0
            xr, yr  = _embedding_ranges(Y)
            desired  = compute_n_boxes(n, xr, yr)
            if desired > ws.n_boxes
                ws = FFTWorkspace(n, desired; flags = FFTW.ESTIMATE)
            end
        end

        Z    = compute_repulsive_forces_fft_2d!(dY, Y, ws)
        Zinv = 1.0 / Z

        Threads.@threads for i in 1:n
            @inbounds begin
                xi = Y[i, 1]; yi = Y[i, 2]
                gx = -dY[i, 1] * Zinv
                gy = -dY[i, 2] * Zinv
                for pos in P_rowptr[i]:(P_rowptr[i + 1] - 1)
                    j = P_colidx[pos]
                    j == i && continue
                    dx = xi - Y[j, 1]
                    dy = yi - Y[j, 2]
                    q_ij = 1.0 / (1.0 + dx*dx + dy*dy)
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
            iY[idx]  = momentum * iY[idx] - eta * (gains[idx] * dY[idx])
            Y[idx]  += iY[idx]
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

# ── Top-level entry point ─────────────────────────────────────────────────────

function tsne_fft(X            :: Union{AbstractMatrix, AbstractVector},
                  ndims        :: Integer,
                  reduce_dims  :: Integer,
                  max_iter     :: Integer,
                  perplexity   :: Number;
                  distance     :: Union{Bool, Function, PreMetric} = false,
                  min_gain     :: Number = 0.01,
                  eta          :: Number = 200.0,
                  pca_init     :: Bool = false,
                  initial_momentum     :: Number = 0.5,
                  final_momentum       :: Number = 0.8,
                  momentum_switch_iter :: Integer = 250,
                  stop_cheat_iter      :: Integer = 250,
                  cheat_scale          :: Number  = 12.0,
                  verbose       :: Bool = false,
                  progress      :: Bool = true,
                  extended_output      = false,
                  rng           :: AbstractRNG = Random.default_rng(),
                  n_boxes_per_dim :: Integer = 0)

    ndims == 2 || throw(ArgumentError(
        "FFT method (:fft) only supports ndims=2 (got ndims=$ndims). " *
        "Use method=:barneshut for other dimensions."))

    ini_Y_with_X = false
    if isa(X, AbstractMatrix) && (distance !== true)
        verbose && @info("Initial X shape is $(size(X))")
        ndims < size(X, 2) || throw(DimensionMismatch(
            "X has fewer dimensions ($(size(X,2))) than ndims=$ndims"))
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
            Y = Matrix{Float64}(X[:, 1:ndims])
        else
            Y = pca(Matrix(X), ndims)
        end
    else
        verbose && @info("Starting with random layout...")
        Y = randn(rng, Float64, n, ndims)
    end

    dY    = zeros(Float64, n, ndims)
    iY    = zeros(Float64, n, ndims)
    gains = ones(Float64, n, ndims)

    verbose && @info("Computing sparse P via kNN...")
    P_rows, sumP, beta = compute_sparse_P(X, distance, Float64(perplexity), 1e-5, 50, verbose, progress)

    scale_sparse_P!(P_rows, cheat_scale / sumP)
    P_rowptr, P_colidx, P_values = sparse_rows_to_csr(P_rows)

    n_boxes_fixed = n_boxes_per_dim > 0 ? next_smooth_number(Int(n_boxes_per_dim)) : 0
    if verbose
        if n_boxes_fixed > 0
            @info("FFT grid: n_boxes=$n_boxes_fixed (fixed), M=$(n_boxes_fixed*3), FFT size=$(2*n_boxes_fixed*3)×$(2*n_boxes_fixed*3)")
        else
            @info("FFT grid: auto-adaptive n_boxes (range+n-based, starting at $(compute_n_boxes(n, 8.0, 8.0)))")
        end
    end

    optimize_fft_2d!(Y, dY, iY, gains, P_rows, P_rowptr, P_colidx, P_values,
                     Int(max_iter), Float64(min_gain), Float64(eta),
                     Float64(initial_momentum), Float64(final_momentum),
                     Int(momentum_switch_iter), Int(stop_cheat_iter), Float64(cheat_scale),
                     progress, n_boxes_fixed)

    if !extended_output
        return Y
    else
        tree     = build_spacetree(Y, 7)
        yi_buf   = zeros(Float64, ndims)
        grad_buf = zeros(Float64, ndims)
        kldiv    = compute_KL_bh_fast(Y, P_rows, tree, 0.5, yi_buf, grad_buf)
        return Y, beta, kldiv
    end
end

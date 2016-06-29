module TSne

using ProgressMeter

# Numpy Math.sum => axis = 0 => sum down the columns. axis = 1 => sum along the rows
# Julia Base.sum => axis = 1 => sum down the columns. axis = 2 => sum along the rows

#
#  tsne.jl
#
#  This is a straight off Julia port of Laurens van der Maatens python implementation of tsne

export tsne

"""
    Compute the perplexity and the i-th column for a specific value of the precision of a Gaussian distribution.
"""
function Hbeta!(P::AbstractVector, D::AbstractVector, Doffset::Number, beta::Number)
    @inbounds @simd for j in eachindex(D)
        P[j] = exp(-beta * D[j])
    end
    sumP = sum(P)
    @assert (isfinite(sumP) && sumP > 0.0) "Degenerated P[$i]: sum=$sumP, beta=$beta"
    H = -beta*Doffset + log(sumP) + beta * dot(D, P) / sumP
    @assert isfinite(H) "Degenerated H"
    scale!(P, 1/sumP)
    return H
end

"""
    Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity.
"""
function x2p(X::Matrix, tol::Number = 1e-5, perplexity::Number = 30.0;
             max_iter::Integer = 50,
             verbose::Bool=false, progress::Bool=true)
    verbose && info("Computing pairwise distances...")
    (n, d) = size(X)
    sum_XX = sumabs2(X, 2)
    D = -2 * (X*X') .+ sum_XX .+ sum_XX'
    Di = zeros(n)
    P = zeros(n, n)
    Pcol = zeros(n)
    beta = ones(n)
    logU = log(perplexity)

    # Loop over all datapoints
    progress && (pb = Progress(n, "Computing point perplexities"))
    for i in 1:n
        progress && update!(pb, i)

        # Compute the Gaussian kernel and entropy for the current precision
        betai = 1.0
        betamin = 0.0
        betamax = Inf

        copy!(Di, slice(D, :, i))
        Di[i] = prevfloat(Inf) # exclude D[i,i] from minimum(), yet make it finite and exp(-D[i,i])==0.0
        minD = minimum(Di) # distance of i-th point to its closest neighbour
        @inbounds @simd for j in eachindex(Di)
            Di[j] -= minD
        end

        H = Hbeta!(Pcol, Di, minD, betai)
        Hdiff = H - logU

        # Evaluate whether the perplexity is within tolerance
        tries = 0
        while abs(Hdiff) > tol && tries < max_iter
            # If not, increase or decrease precision
            if Hdiff > 0
                betamin = betai
                betai = isfinite(betamax) ? (betai + betamax)/2 : betai*2
            else
                betamax = betai
                betai = (betai + betamin)/2
            end

            # Recompute the values
            H = Hbeta!(Pcol, Di, minD, betai)
            Hdiff = H - logU
            tries += 1
        end
        verbose && abs(Hdiff) > tol && warn("P[$i]: perplexity error is above tolerance: $(Hdiff)")
        # Set the final column of P
        @assert Pcol[i] == 0.0 "Diagonal probability P[$i,$i]=$(Pcol[i]) not zero"
        P[:, i] = Pcol
        beta[i] = betai
    end
    progress && finish!(pb)
    # Return final P-matrix
    verbose && info("Mean σ=$(mean(sqrt(1 ./ beta)))")
    return P
end

"""
    Runs PCA on the NxD array `X` in order to reduce its dimensionality to `ndims` dimensions.

    FIXME use PCA routine from JuliaStats?
"""
function pca{T}(X::Matrix{T}, ndims::Integer = 50)
    info("Preprocessing the data using PCA...")
    (n, d) = size(X)
    X = X - repmat(mean(X, 1), n, 1)
    C = (X' * X) ./ (size(X,1)-1)
    l, M = eig(C)
    sorder = sortperm(l, rev=true)
    M = M[:, sorder]::Matrix{T}
    Y = X * M[:, 1:min(d, ndims)]
    return Y
end

"""
    Apply t-SNE to `X`, i.e. embed its points into `ndims` dimensions
    preserving close neighbours.
    Different from the orginal implementation,
    the default is not to use PCA for initialization.
"""
function tsne(X::Matrix, ndims::Integer = 2, initial_dims::Integer = 0, max_iter::Integer = 1000, perplexity::Number = 30.0;
              min_gain::Number = 0.01, eta::Number = 200.0,
              initial_momentum::Number = 0.5, final_momentum::Number = 0.8, momentum_switch_iter::Integer = 250,
              stop_cheat_iter::Integer = 250, cheat_scale::Number = 12.0,
              verbose::Bool = false, progress::Bool=true)
    verbose && info("Initial X Shape is $(size(X))")

    # Initialize variables
    X = scale(X, 1.0/std(X))
    if initial_dims>0
        X = pca(X, initial_dims)
    end
    (n, d) = size(X)
    Y = randn(n, ndims)
    dY = zeros(n, ndims)
    iY = zeros(n, ndims)
    gains = ones(n, ndims)

    # Compute P-values
    P = x2p(X, 1e-5, perplexity, verbose=verbose, progress=progress)
    P = P + P'
    scale!(P, 1.0/sum(P))
    scale!(P, cheat_scale)  # early exaggeration
    P = max(P, 1e-12)
    L = similar(P)
    Ymean = zeros(1, ndims)
    sum_YY = zeros(n, 1)
    Lcolsums = zeros(n, 1)

    # Run iterations
    progress && (pb = Progress(max_iter, "Computing t-SNE"))
    Q = similar(P)
    for iter in 1:max_iter
        progress && update!(pb, iter)
        # Compute pairwise affinities
        sumabs2!(sum_YY, Y)
        # FIXME profiling indicates a lot of time is lost in copytri!()
        A_mul_Bt!(Q, Y, Y)
        @inbounds for j in 1:size(Q, 2)
            @simd for i in 1:(j-1)
                Q[j,i] = Q[i,j] = 1.0 / (1.0 - 2.0 * Q[i,j] + sum_YY[i] + sum_YY[j])
            end
        end
        sum_Q = sum(Q)

        # Compute gradient
        @inbounds @simd for i in eachindex(P)
            L[i] = (P[i] - Q[i]/sum_Q) * Q[i]
        end
        sum!(Lcolsums, L)
        for (i, ldiag) in enumerate(Lcolsums)
            L[i, i] -= ldiag
        end
        A_mul_B!(dY, L, Y)
        scale!(dY, -4.0)

        # Perform the update
        momentum = iter <= momentum_switch_iter ? initial_momentum : final_momentum
        @inbounds for i in eachindex(gains)
            flag = (dY[i] > 0) == (iY[i] > 0)
            gains[i] = max(flag ? gains[i] * 0.8 : gains[i] + 0.2, min_gain)
            iY[i] = momentum * iY[i] - eta * (gains[i] * dY[i])
            Y[i] += iY[i]
        end
        mean!(Ymean, Y)
        @inbounds for j in 1:size(Y, 2)
            YcolMean = Ymean[j]
            for i in 1:size(Y, 1)
                Y[i,j] -= YcolMean
            end
        end

        # Compute current value of cost function
        if verbose && mod((iter + 1), max(max_iter÷100, 1) ) == 0
            err = sum(pq -> pq[1] > 0.0 && pq[2] > 0.0 && sum_Q > 0.0 ?
                      pq[1]*log(pq[1]/pq[2]*sum_Q)::Float64 : 0.0,
                      zip(P, Q))
            info("Iteration #$(iter + 1): error is $err")
        end
        # stop cheating with P-values
        if iter == min(max_iter, stop_cheat_iter)
            scale!(P, 1/cheat_scale)
        end
    end
    progress && (finish!(pb))

    # Return solution
    return Y
end

end

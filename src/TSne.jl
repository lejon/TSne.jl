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
    Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution.
"""
function Hbeta(D, beta = 1.0)
    P = exp(-D * beta)
    sumP = sum(P)
    H = log(sumP) + beta * dot(D, P) / sumP
    scale!(P, 1/sumP)
    return (H, P)
end

"""
    Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity.
"""
function x2p(X::Matrix, tol::Number = 1e-5, perplexity::Number = 30.0;
             verbose::Bool=false, progress::Bool=true)
    verbose && info("Computing pairwise distances...")
    (n, d) = size(X)
    sum_XX = sumabs2(X, 2)
    D = (-2 * (X * X') .+ sum_XX)' .+ sum_XX
    P = zeros(n, n)
    beta = ones(n)
    logU = log(perplexity)

    # Loop over all datapoints
    range = collect(1:n)
    progress && (pb = Progress(n, "Computing point perplexities"))
    for i in 1:n
        progress && update!(pb, i)

        # Compute the Gaussian kernel and entropy for the current precision
        betai = beta[i]
        betamin = -Inf
        betamax =  Inf

        inds = range[range .!=i]
        Di = squeeze(D[i, inds], 1)
        (H, thisP) = Hbeta(Di, betai)

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol && tries < 50

            # If not, increase or decrease precision
            if Hdiff > 0
                betamin = betai
                betai = isfinite(betamax) ? (betai + betamax)/2 : betai*2
            else
                betamax = betai
                betai = isfinite(betamin) ? (betai + betamin)/2 : betai/2
            end

            # Recompute the values
            (H, thisP) = Hbeta(Di, betai)
            Hdiff = H - logU
            tries += 1
        end
        verbose && abs(Hdiff) > tol && warn("P[$i]: perplexity error is above tolerance: $(Hdiff)")
        # Set the final row of P
        P[i, inds] = thisP
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
    Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to `ndims` dimensions.
    Diffrent from orginal, default is to not use PCA
"""
function tsne(X::Matrix, ndims::Integer = 2, initial_dims::Integer = -1, max_iter::Integer = 1000, perplexity::Number = 30.0;
              verbose::Bool = false, progress::Bool=true)
    verbose && info("Initial X Shape is $(size(X))")

    # Initialize variables
    if initial_dims>0
        X = pca(X, initial_dims)
    end
    (n, d) = size(X)
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = randn(n, ndims)
    dY = zeros(n, ndims)
    iY = zeros(n, ndims)
    gains = ones(n, ndims)

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + P'
    scale!(P, 1.0/sum(P))
    scale!(P, 4)                        # early exaggeration
    P = max(P, 1e-12)
    L = similar(P)

    # Run iterations
    progress && (pb = Progress(max_iter, "Computing t-SNE"))
    for iter in 1:max_iter
        progress && update!(pb, iter)
        # Compute pairwise affinities
        sum_YY = sumabs2(Y, 2)
        Q = 1 ./ (1 + ((-2 * (Y * Y')) .+ sum_YY)' .+ sum_YY)
        # Setting diagonal to zero
        @inbounds for i in 1:size(Q, 1)
            Q[i,i] = 0.0
        end
        sum_Q = sum(Q)

        # Compute gradient
        @inbounds for i in eachindex(P)
            L[i] = (P[i] - Q[i]/sum_Q) * Q[i]
        end
        A_mul_B!(dY, (diagm(sum(L, 1)[:,]) - L), Y)
        scale!(dY, 4)

        # Perform the update
        momentum = iter <= 20 ? initial_momentum : final_momentum
        @inbounds for i in eachindex(gains)
            flag = (dY[i] > 0) == (iY[i] > 0)
            gains[i] = max(flag ? gains[i] * 0.8 : gains[i] + 0.2, min_gain)
            iY[i] = momentum * iY[i] - eta * (gains[i] * dY[i])
            Y[i] += iY[i]
        end
        Y = Y - repmat(mean(Y, 1), n, 1)

        # Compute current value of cost function
        if verbose && mod((iter + 1), max(max_iter÷100, 1) ) == 0
            err = sum(pq -> pq[1] > 0.0 && pq[2] > 0.0 && sum_Q > 0.0 ?
                      pq[1]*log(pq[1]/pq[2]*sum_Q)::Float64 : 0.0,
                      zip(P, Q))
            info("Iteration #$(iter + 1): error is $err")
        end
        # stop cheating with P-values
        if iter == min(max_iter, 100)
            scale!(P, 1/4)
        end
    end
    progress && (finish!(pb))

    # Return solution
    return Y
end

end

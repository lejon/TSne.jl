module TSne

using LinearAlgebra, Statistics, Distances, ProgressMeter
using Printf: @sprintf

export tsne

"""
Compute the point perplexities `P` given its squared distances to the other points `D`
and the precision of Gaussian distribution `beta`.
"""
function Hbeta!(P::AbstractVector, D::AbstractVector, beta::Number)
    @inbounds P .= exp.(-beta .* D)
    sumP = sum(P)
    @assert (isfinite(sumP) && sumP > 0.0) "Degenerated P: sum=$sumP, beta=$beta"
    H = log(sumP) + beta * dot(D, P) / sumP
    @assert isfinite(H) "Degenerated H"
    @inbounds P .*= 1/sumP
    return H
end

"""
    perplexities(D::AbstractMatrix, tol::Number = 1e-5, perplexity::Number = 30.0;
                 [keyword arguments])

Convert `n×n` squared distances matrix `D` into `n×n` perplexities matrix `P`.
Performs a binary search to get P-values in such a way that each conditional
Gaussian has the same perplexity.
"""
function perplexities(D::AbstractMatrix{T}, tol::Number = 1e-5, perplexity::Number = 30.0;
                      max_iter::Integer = 50,
                      verbose::Bool=false, progress::Bool=true) where T<:Number
    (issymmetric(D) && all(x -> x >= 0, D)) ||
        throw(ArgumentError("Distance matrix D must be symmetric and positive"))

    # initialize
    n = size(D, 1)
    P = fill(zero(T), n, n) # perplexities matrix
    beta = fill(one(T), n)  # vector of Normal distribution precisions for each point
    Htarget = log(perplexity) # the expected entropy
    Di = fill(zero(T), n)
    Pcol = fill(zero(T), n)

    # Loop over all datapoints
    progress && (pb = Progress(n, "Computing point perplexities"))
    for i in 1:n
        progress && update!(pb, i)

        # Compute the Gaussian kernel and entropy for the current precision
        betai = 1.0
        betamin = 0.0
        betamax = Inf

        copyto!(Di, view(D, :, i))
        Di[i] = prevfloat(Inf) # exclude D[i,i] from minimum(), yet make it finite and exp(-D[i,i])==0.0
        minD = minimum(Di) # distance of i-th point to its closest neighbour
        @inbounds Di .-= minD # entropy is invariant to offsetting Di, which helps to avoid overflow

        H = Hbeta!(Pcol, Di, betai)
        Hdiff = H - Htarget

        # Evaluate whether the perplexity is within tolerance
        tries = 0
        while abs(Hdiff) > tol && tries < max_iter
            # If not, increase or decrease precision
            if Hdiff > 0.0
                betamin = betai
                betai = isfinite(betamax) ? (betai + betamax)/2 : betai*2
            else
                betamax = betai
                betai = (betai + betamin)/2
            end

            # Recompute the values
            H = Hbeta!(Pcol, Di, betai)
            Hdiff = H - Htarget
            tries += 1
        end
        verbose && abs(Hdiff) > tol && warn("P[$i]: perplexity error is above tolerance: $(Hdiff)")
        # Set the final column of P
        @assert Pcol[i] == 0.0 "Diagonal probability P[$i,$i]=$(Pcol[i]) not zero"
        @inbounds P[:, i] .= Pcol
        beta[i] = betai
    end
    progress && finish!(pb)
    # Return final P-matrix
    verbose && @info(@sprintf("Mean σ=%.4f", mean(sqrt.(1 ./ beta))))
    return P, beta
end

"""
    pca(X::Matrix, ncols::Integer = 50)

Run PCA on `X` to reduce the number of its dimensions to `ndims`.

FIXME use PCA routine from JuliaStats?
"""
function pca(X::AbstractMatrix, ndims::Integer = 50)
    (n, d) = size(X)
    (d <= ndims) && return X
    Y = X .- mean(X, dims=1)
    C = Symmetric((Y' * Y) ./ (n-1))
    Ceig = eigen(C, (d-ndims+1):d) # take eigvects for top ndims largest eigvals
    return Y * reverse(Ceig.vectors, dims=2)
end

# K-L divergence element
kldivel(p, q) = ifelse(p > zero(p) && q > zero(q), p*log(p/q), zero(p))

# pairwise squared distance
# if X is the matrix of objects, then the distance between its rows
pairwisesqdist(X::AbstractMatrix, dist::Bool) =
    dist ? X.^2 : pairwise(SqEuclidean(), X')

pairwisesqdist(X::AbstractVector, dist::Union{Function, PreMetric}) =
    [dist(x, y)^2 for x in X, y in X] # note: some redundant calc since dist should be symmetric

pairwisesqdist(X::AbstractMatrix, dist::Function) =
    [dist(x, y)^2 for x in eachrow(X), y in eachrow(X)] # note: some redundant calc since dist should be symmetric

pairwisesqdist(X::AbstractMatrix, dist::PreMetric) =
    pairwise(dist, X').^2 # use Distances

"""
    tsne(X::Union{AbstractMatrix, AbstractVector}, ndims::Integer=2, reduce_dims::Integer=0,
         max_iter::Integer=1000, perplexity::Number=30.0; [keyword arguments])

Apply t-SNE (t-Distributed Stochastic Neighbor Embedding) to `X`,
i.e. embed its points (rows) into `ndims` dimensions preserving close neighbours.

Returns the points×`ndims` matrix of calculated embedded coordinates.

Different from the orginal implementation,
the default is not to use PCA for initialization.

### Arguments
* `distance` if `true`, specifies that `X` is a distance matrix,
  if of type `Function` or `Distances.SemiMetric`, specifies the function to
  use for calculating the distances between the rows
  (or elements, if `X` is a vector) of `X`
* `reduce_dims` the number of the first dimensions of `X` PCA to use for t-SNE,
  if 0, all available dimension are used
* `pca_init` whether to use the first `ndims` of `X` PCA as the initial t-SNE layout,
  if `false` (the default), the method is initialized with the random layout
* `max_iter` how many iterations of t-SNE to do
* `perplexity` the number of "effective neighbours" of a datapoint,
  typical values are from 5 to 50, the default is 30
* `verbose` output informational and diagnostic messages
* `progress` display progress meter during t-SNE optimization
* `min_gain`, `eta`, `initial_momentum`, `final_momentum`, `momentum_switch_iter`,
  `stop_cheat_iter`, `cheat_scale` low-level parameters of t-SNE optimization
* `extended_output` if `true`, returns a tuple of embedded coordinates matrix,
  point perplexities and final Kullback-Leibler divergence

See also [Original t-SNE implementation](https://lvdmaaten.github.io/tsne).
"""
function tsne(X::Union{AbstractMatrix, AbstractVector}, ndims::Integer = 2, reduce_dims::Integer = 0,
              max_iter::Integer = 1000, perplexity::Number = 30.0;
              distance::Union{Bool, Function, SemiMetric} = false,
              min_gain::Number = 0.01, eta::Number = 200.0, pca_init::Bool = false,
              initial_momentum::Number = 0.5, final_momentum::Number = 0.8, momentum_switch_iter::Integer = 250,
              stop_cheat_iter::Integer = 250, cheat_scale::Number = 12.0,
              verbose::Bool = false, progress::Bool=true,
              extended_output = false)
    # preprocess X
    ini_Y_with_X = false
    if isa(X, AbstractMatrix) && (distance !== true)
        verbose && @info("Initial X shape is $(size(X))")
        ndims < size(X, 2) || throw(DimensionMismatch("X has fewer dimensions ($(size(X,2))) than ndims=$ndims"))

        ini_Y_with_X = true
        X = X * (1.0/std(X)::eltype(X)) # note that X is copied
        if 0<reduce_dims<size(X, 2)
            reduce_dims = max(reduce_dims, ndims)
            verbose && @info("Preprocessing the data using PCA...")
            X = pca(X, reduce_dims)
        end
    end
    n = size(X, 1)
    # Initialize embedding
    if pca_init && ini_Y_with_X
        verbose && @info("Using the first $ndims components of the data PCA as the initial layout...")
        if reduce_dims >= ndims
            Y = X[:, 1:ndims] # reuse X PCA
        else
            @assert reduce_dims <= 0 # no X PCA
            Y = pca(X, ndims)
        end
    else
        verbose && @info("Starting with random layout...")
        Y = randn(n, ndims)
    end

    dY = fill!(similar(Y), 0)     # gradient vector
    iY = fill!(similar(Y), 0)     # momentum vector
    gains = fill!(similar(Y), 1)  # how much momentum is affected by gradient

    # Compute P-values
    verbose && (distance !== true) && @info("Computing pairwise distances...")
    D = pairwisesqdist(X, distance)
    P, beta = perplexities(D, 1e-5, perplexity,
                           verbose=verbose, progress=progress)
    P .+= P' # make P symmetric
    P .*= cheat_scale/sum(P) # normalize + early exaggeration
    sum_P = cheat_scale

    # Run iterations
    progress && (pb = Progress(max_iter, "Computing t-SNE"))
    Q = fill!(similar(P), 0)     # temp upper-tri matrix with 1/(1 + (Y[i]-Y[j])²)
    Ymean = similar(Y, 1, ndims) # average for each embedded dimension
    sum_YY = similar(Y, n, 1)    # square norms of embedded points
    L = fill!(similar(P), 0)     # temp upper-tri matrix for KLdiv gradient calculation
    Lcolsums = similar(L, n, 1)  # sum(Symmetric(L), 2)
    last_kldiv = NaN
    for iter in 1:max_iter
        # Compute pairwise affinities
        BLAS.syrk!('U', 'N', 1.0, Y, 0.0, Q) # Q=YY', updates only the upper tri of Q
        @inbounds for i in 1:size(Q, 2)
            sum_YY[i] = Q[i, i]
        end
        sum_Q = 0.0
        @inbounds for j in 1:size(Q, 2)
            sum_YYj_p1 = 1.0 + sum_YY[j]
            Qj = view(Q, :, j)
            Qj[j] = 0.0
            for i in 1:(j-1)
                sqdist_p1 = sum_YYj_p1 - 2.0 * Qj[i] + sum_YY[i]
                @fastmath Qj[i] = ifelse(sqdist_p1 > 1.0, 1.0 / sqdist_p1, 1.0)
                sum_Q += Qj[i]
            end
        end
        sum_Q *= 2 # the diagonal and lower-tri part of Q is zero

        # Compute the gradient
        inv_sum_Q = 1.0 / sum_Q
        fill!(Lcolsums, 0.0) # column sums
        # fill the upper triangle of L (gradient)
        @inbounds for j in 1:size(L, 2)
            Lj = view(L, :, j)
            Pj = view(P, :, j)
            Qj = view(Q, :, j)
            Lsumj = 0.0
            for i in 1:(j-1)
                @fastmath Lj[i] = l = (Pj[i] - Qj[i]*inv_sum_Q) * Qj[i]
                Lcolsums[i] += l
                Lsumj += l
            end
            Lcolsums[j] += Lsumj
        end
        @inbounds for (i, ldiag) in enumerate(Lcolsums)
            L[i, i] = -ldiag
        end
        # dY = -4LY
        BLAS.symm!('L', 'U', -4.0, L, Y, 0.0, dY)

        # Perform the update
        momentum = iter <= momentum_switch_iter ? initial_momentum : final_momentum
        @inbounds for i in eachindex(gains)
            gains[i] = max(ifelse((dY[i] > 0) == (iY[i] > 0),
                                  gains[i] * 0.8,
                                  gains[i] + 0.2),
                           min_gain)
            iY[i] = momentum * iY[i] - eta * (gains[i] * dY[i])
            Y[i] += iY[i]
        end
        @inbounds Y .-= mean!(Ymean, Y)

        # stop cheating with P-values
        if sum_P != 1.0 && iter >= min(max_iter, stop_cheat_iter)
            P .*= 1/sum_P
            sum_P = 1.0
        end
        # Compute the current value of cost function
        if !isfinite(last_kldiv) || iter == max_iter ||
            (progress && mod(iter, max(max_iter÷20, 10)) == 0)
            local kldiv = 0.0
            @inbounds for j in 1:size(P, 2)
                Pj = view(P, :, j)
                Qj = view(Q, :, j)
                kldiv_j = 0.0
                for i in 1:(j-1)
                    # P and Q are symmetric (only the upper triangle used)
                    @fastmath kldiv_j += kldivel(Pj[i], Qj[i])
                end
                kldiv += 2*kldiv_j + kldivel(Pj[j], Q[j])
            end
            last_kldiv = kldiv/sum_P + log(sum_Q/sum_P) # adjust wrt P and Q scales
        end
        progress && update!(pb, iter,
                            showvalues = Dict(:KL_divergence => @sprintf("%.4f%s", last_kldiv,
                                                                         iter <= stop_cheat_iter ? " (warmup)" : "")))
    end
    progress && (finish!(pb))
    verbose && @info(@sprintf("Final t-SNE KL-divergence=%.4f", last_kldiv))

    # Return solution
    if !extended_output
        return Y
    else
        return Y, beta, last_kldiv
    end
end

end

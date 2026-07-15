## Run t-SNE on 1.3M mouse brain cells (10x Genomics E18)
##
## Usage:
##   julia --project=examples -t auto examples/demo-mousebrain.jl bh
##   julia --project=examples -t auto examples/demo-mousebrain.jl fft
##
## The HDF5 file is expected at:  data/1M_neurons_filtered_gene_bc_matrices_h5.h5
## Download with:
##   curl -L -o data/1M_neurons_filtered_gene_bc_matrices_h5.h5 \
##     https://s3-us-west-2.amazonaws.com/10x.files/samples/cell/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5

using TSne, HDF5, SparseArrays, Statistics, LinearAlgebra, Random, Printf, TSVD
using Gadfly, Cairo, DataFrames

const DATA_FILE   = joinpath(@__DIR__, "..", "data", "1M_neurons_filtered_gene_bc_matrices_h5.h5")
const N_PCS       = 50
const N_HVG       = 3_000
const PERPLEXITY  = 30
const MAX_ITER    = 1000
const MAX_CELLS   = 1_000_000

# ── Data loading ─────────────────────────────────────────────────────────────

function load_10x_h5(path::String; max_cells::Int = 0)
    println("Loading $path …")
    t = @elapsed begin
        X = h5open(path, "r") do f
            grp_name = haskey(f, "matrix") ? "matrix" : first(keys(f))
            grp = f[grp_name]

            shape   = read(grp["shape"])
            n_genes = Int(shape[1])
            n_cells_total = Int(shape[2])

            n_read = if max_cells > 0 && max_cells < n_cells_total
                max_cells
            else
                n_cells_total
            end

            indptr_raw = vec(read(grp["indptr"]))
            nnz = indptr_raw[n_read + 1]
            indptr = indptr_raw[1:n_read + 1] .+ 1
            indices = Int.(grp["indices"][1:nnz]) .+ 1
            data    = Float32.(grp["data"][1:nnz])

            println("  Shape: $n_genes genes × $n_read cells")
            println("  Non-zeros: $nnz ($(round(100.0*nnz/n_genes/n_read, digits=2))% dense)")
            SparseMatrixCSC{Float32, Int}(n_genes, n_read, indptr, indices, data)
        end
    end
    @printf "  Loaded in %.1f s\n" t
    return X
end

function log_normalize!(X::SparseMatrixCSC{Float32})
    println("Log-normalizing …")
    @inbounds for j in 1:size(X, 2)
        col_sum = 0f0
        for idx in nzrange(X, j)
            col_sum += X.nzval[idx]
        end
        scale = col_sum > 0f0 ? 10_000f0 / col_sum : 1f0
        for idx in nzrange(X, j)
            X.nzval[idx] = log1p(X.nzval[idx] * scale)
        end
    end
    return X
end

function select_hvg(X::SparseMatrixCSC{Float32}, n_hvg::Int)
    println("Selecting top $n_hvg highly variable genes …")
    n_genes, n_cells = size(X)
    gene_mean = zeros(Float64, n_genes)
    gene_sq   = zeros(Float64, n_genes)
    @inbounds for j in 1:n_cells
        for idx in nzrange(X, j)
            i = X.rowval[idx]; v = Float64(X.nzval[idx])
            gene_mean[i] += v
            gene_sq[i]   += v * v
        end
    end
    gene_var = gene_sq ./ n_cells .- (gene_mean ./ n_cells).^2
    top_idx  = partialsortperm(gene_var, 1:n_hvg, rev=true)
    sort!(top_idx)

    keep = falses(n_genes)
    for idx in top_idx
        keep[idx] = true
    end
    old2new = zeros(Int, n_genes)
    for (new_i, old_i) in enumerate(top_idx)
        old2new[old_i] = new_i
    end

    new_colptr = Vector{Int}(undef, n_cells + 1)
    new_colptr[1] = 1
    @inbounds for j in 1:n_cells
        cnt = 0
        for idx in nzrange(X, j)
            if keep[X.rowval[idx]]
                cnt += 1
            end
        end
        new_colptr[j + 1] = new_colptr[j] + cnt
    end
    total_nnz = new_colptr[end] - 1
    new_rowval = Vector{Int}(undef, total_nnz)
    new_nzval  = Vector{Float32}(undef, total_nnz)
    pos = 1
    @inbounds for j in 1:n_cells
        for idx in nzrange(X, j)
            i = X.rowval[idx]
            keep[i] || continue
            new_rowval[pos] = old2new[i]
            new_nzval[pos]  = X.nzval[idx]
            pos += 1
        end
    end
    return SparseMatrixCSC{Float32, Int}(n_hvg, n_cells, new_colptr, new_rowval, new_nzval)
end

function pca_sparse(X::SparseMatrixCSC{Float32}, n_pcs::Int)
    println("Computing truncated SVD ($n_pcs PCs) …")
    t = @elapsed U, s, V = tsvd(X, n_pcs; maxiter=4000, tolconv=1e-4)
    @printf "  SVD done in %.1f s\n" t
    return Matrix{Float64}(V .* s')
end

# ── Main ──────────────────────────────────────────────────────────────────────

length(ARGS) == 1 || error("Usage: julia --project=examples -t auto examples/demo-mousebrain.jl {bh|fft}")

method_str = ARGS[1]
method = if method_str == "bh"
    :barneshut
elseif method_str == "fft"
    :fft
else
    error("Unknown method \"$method_str\". Use \"bh\" or \"fft\".")
end

isfile(DATA_FILE) || error("""
Dataset not found: $DATA_FILE
Download with:
  mkdir -p data && curl -L -o data/1M_neurons_filtered_gene_bc_matrices_h5.h5 \\
    https://s3-us-west-2.amazonaws.com/10x.files/samples/cell/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5
""")

println("=" ^ 60)
println(" Mouse brain t-SNE  (method=:$method_str, cells=$MAX_CELLS, iters=$MAX_ITER)")
println("=" ^ 60)
println("Julia threads: $(Threads.nthreads())")
println()

X_raw  = load_10x_h5(DATA_FILE; max_cells = MAX_CELLS)
log_normalize!(X_raw)
X_hvg  = select_hvg(X_raw, N_HVG)
X_raw  = nothing; GC.gc()
X_pca  = pca_sparse(X_hvg, N_PCS)
X_hvg  = nothing; GC.gc()
n_cells = size(X_pca, 1)
println("Loaded $n_cells cells, projecting to t-SNE …\n")

Random.seed!(42)
t_total = @elapsed Y = tsne(X_pca, 2, -1, MAX_ITER, PERPLEXITY;
                            method   = method,
                            progress = true,
                            verbose  = true)

println()
@printf "  Done in %.1f s  →  ~%.0f ms/iter\n" t_total (t_total * 1000 / MAX_ITER)
@printf "  Y shape: %d × %d\n" size(Y)...

outfile = "mousebrain_$(method_str).png"
println("Plotting to $outfile …")
df = DataFrame(x=Y[:, 1], y=Y[:, 2])
p = plot(df, x=:x, y=:y, Geom.point,
         Guide.title("Mouse brain t-SNE ($(method_str), $(n_cells) cells, $(MAX_ITER) iters)"),
         Coord.cartesian(fixed=false),
         Theme(point_size=1pt))
draw(PNG(outfile, 1200px, 900px), p)
println("Saved $outfile")
t-SNE (t-Stochastic Neighbor Embedding)
=======================================

[![Build Status](https://github.com/lejon/TSne.jl/workflows/CI/badge.svg)](https://github.com/lejon/TSne.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/lejon/TSne.jl/badge.svg)](https://coveralls.io/github/lejon/TSne.jl)

Julia implementation of L.J.P. van der Maaten and G.E. Hintons [t-SNE visualisation technique](https://lvdmaaten.github.io/tsne/).

## Installation

```julia
julia> Pkg.add("TSne")
```

## Basic API

```julia
tsne(X, ndims, reduce_dims, max_iter, perplexity; [keyword arguments])
```

Apply t-SNE (t-Distributed Stochastic Neighbor Embedding) to `X`, embedding its rows into `ndims` dimensions while preserving close neighbours. Returns the points×`ndims` matrix of embedded coordinates.

**Positional arguments**

| Argument | Description |
|---|---|
| `X` | `AbstractMatrix` (rows = observations, columns = features) or `AbstractVector` of observations |
| `ndims` | Dimension of the embedded space (typically 2 or 3) |
| `reduce_dims` | Number of leading PCA dimensions of `X` to use; `0` uses all |
| `max_iter` | Number of optimization iterations |
| `perplexity` | Related to the number of effective nearest neighbours; typical values 5–50 |

**Keyword arguments**

| Argument | Default | Description |
|---|---|---|
| `method` | `:exact` | `:exact` (O(n²)) or `:barneshut` (O(n log n)) |
| `theta` | `0.5` | Barnes-Hut opening angle. Lower = more accurate, range 0.2–0.8 |
| `max_depth` | `7` | Barnes-Hut tree depth limit. Only used when `method = :barneshut` |
| `distance` | `false` | `true` if `X` is a precomputed distance matrix; or a `Function`/`Distances.SemiMetric` |
| `pca_init` | `false` | Initialise from the first `ndims` PCA components instead of random |
| `reduce_dims` | `0` | PCA pre-reduction dimensionality (`0` = no reduction) |
| `verbose` | `false` | Print informational messages |
| `progress` | `true` | Show progress meter |
| `extended_output` | `false` | Return `(Y, beta, kl_divergence)` instead of just `Y` |
| `rng` | `Random.default_rng()` | RNG for reproducible initialisation; pass `MersenneTwister(seed)` for a fixed seed |
| `min_gain`, `eta`, `initial_momentum`, `final_momentum`, `momentum_switch_iter`, `stop_cheat_iter`, `cheat_scale` | | Low-level optimiser parameters |

## Methods

### Exact t-SNE (`method = :exact`)

The default. Computes pairwise affinities exactly — O(n²) in both time and memory. Suitable for up to a few thousand points.

```julia
Y = tsne(X, 2, 50, 1000, 30.0)
```

### Barnes-Hut t-SNE (`method = :barneshut`)

O(n log n) approximation using a quadtree (2D), octree (3D), or generic space tree. Supports multi-threading (`julia -t auto`). Recommended for large datasets.

```julia
Y = tsne(X, 2, 50, 1000, 30.0; method=:barneshut, theta=0.5)
```

Supports `ndims` up to 4. Use `method=:exact` for higher dimensions.

## Example

```julia
using TSne, Statistics, MLDatasets

rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())

alldata, allabels = MNIST(split=:train)[:];
alldata = Float64.(alldata);
data = reshape(permutedims(alldata[:, :, 1:2500], (3, 1, 2)),
               2500, size(alldata, 1)*size(alldata, 2));
X = rescale(data, dims=1);

Y = tsne(X, 2, 50, 1000, 20.0; method=:barneshut)

using Plots
theplot = scatter(Y[:,1], Y[:,2], marker=(2,2,:auto,stroke(0)), color=Int.(allabels[1:size(Y,1)]))
Plots.pdf(theplot, "myplot.pdf")
```

![](example.png)

## Running the example scripts

The scripts in `examples/` and `scripts/` use additional packages (Gadfly, Cairo, MLDatasets, RDatasets) that are not installed as part of TSne.jl. Each directory has its own `Project.toml`. Activate it with `--project`:

```bash
# One-time setup
julia --project=scripts -e "import Pkg; Pkg.instantiate()"

# Run a script
julia --project=scripts -t auto scripts/mnist_bh.jl
```

Similarly for `examples/`:

```bash
julia --project=examples -e "import Pkg; Pkg.instantiate()"
julia --project=examples examples/demo.jl iris
```

## Benchmarks

```bash
julia --project -t auto benchmark/bm_compare.jl   # fast before/after comparison (~2 min)
julia --project -t auto benchmark/benchmark.jl     # full suite
```

## Command line usage

```bash
julia examples/demo-csv.jl haveheader --labelcol=5 examples/iris-headers.csv
```

## See also

 * [Some tips working with t-SNE](http://lejon.github.io)
 * [How to Use t-SNE Effectively](http://distill.pub/2016/misread-tsne/)

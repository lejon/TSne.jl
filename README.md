t-SNE (t-Stochastic Neighbor Embedding)
=======================================

[![Build Status](https://github.com/lejon/TSne.jl/workflows/CI/badge.svg)](https://github.com/lejon/TSne.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/lejon/TSne.jl/badge.svg)](https://coveralls.io/github/lejon/TSne.jl)

Julia implementation of L.J.P. van der Maaten and G.E. Hintons [t-SNE visualisation technique](https://lvdmaaten.github.io/tsne/).

The scripts in the `examples` folder require `Plots`, `MLDatasets` and `RDatasets` Julia packages.

## Installation

  `julia> Pkg.add("TSne")`

## Basic API usage
`tsne(X, ndim, reduce_dims, max_iter, perplexit; [keyword arguments])`
         
Apply t-SNE (t-Distributed Stochastic Neighbor Embedding) to `X`,
i.e. embed its points (rows) into `ndims` dimensions preserving close neighbours.
Returns the points×`ndims` matrix of calculated embedded coordinates.

- `X`: AbstractMatrix or AbstractVector. If `X` is a matrix, then rows are observations and columns are features.
- `ndims`: Dimension of the embedded space.
- `reduce_dims` the number of the first dimensions of `X` PCA to use for t-SNE,
  if 0, all available dimension are used
- `max_iter`: Maximum number of iterations for the optimization
- `perplexity': The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results

**Optional Arguments**
* `distance` if `true`, specifies that `X` is a distance matrix,
  if of type `Function` or `Distances.SemiMetric`, specifies the function to
  use for calculating the distances between the rows
  (or elements, if `X` is a vector) of `X`
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
  
### Example usage
```jl
using TSne, Statistics, MLDatasets

rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())

alldata, allabels = MNIST.traindata(Float64);
data = reshape(permutedims(alldata[:, :, 1:2500], (3, 1, 2)),
               2500, size(alldata, 1)*size(alldata, 2));
# Normalize the data, this should be done if there are large scale differences in the dataset
X = rescale(data, dims=1);

Y = tsne(X, 2, 50, 1000, 20.0);

using Plots
theplot = scatter(Y[:,1], Y[:,2], marker=(2,2,:auto,stroke(0)), color=Int.(allabels[1:size(Y,1)]))
Plots.pdf(theplot, "myplot.pdf")
```

![](example.png)

## Command line usage

```julia demo-csv.jl haveheader --labelcol=5 iris-headers.csv```

Creates `myplot.pdf` with t-SNE result visualized using `Gadfly.jl`.

## See also
 * [Some tips working with t-SNE](http://lejon.github.io)
 * [How to Use t-SNE Effectively](http://distill.pub/2016/misread-tsne/)

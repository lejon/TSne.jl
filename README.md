[![Travis](https://travis-ci.org/lejon/TSne.jl.svg?branch=master)](https://travis-ci.org/lejon/TSne.jl)
[![Coveralls](https://coveralls.io/repos/github/lejon/TSne.jl/badge.svg?branch=master)](https://coveralls.io/github/lejon/TSne.jl?branch=master)

t-SNE (t-Stochastic Neighbor Embedding)
=======================================

Julia implementation of L.J.P. van der Maaten and G.E. Hintons [t-SNE visualisation technique](https://lvdmaaten.github.io/tsne/).

The scripts in the `examples` folder require `Gadfly`, `MNIST` and `RDatasets` Julia packages.

## Installation

  `julia> Pkg.add("TSne")`

## Basic API usage

```jl
using TSne, MNIST

rescale(A, dim::Integer=1) = (A .- mean(A, dim)) ./ max.(std(A, dim), eps())

data, labels = traindata()
data = convert(Matrix{Float64}, data[:, 1:2500])'
# Normalize the data, this should be done if there are large scale differences in the dataset
X = rescale(data, 1)

Y = tsne(X, 2, 50, 1000, 20.0)

using Gadfly
theplot = plot(x=Y[:,1], y=Y[:,2], color=string.(labels[1:size(Y,1)]))
draw(PDF("myplot.pdf", 4inch, 3inch), theplot)
```

![](example.png)

## Command line usage

```julia demo-csv.jl haveheader --labelcol=5 iris-headers.csv```

Creates `myplot.pdf` with t-SNE result visualized using `Gadfly.jl`.

## See also
 * [Some tips working with t-SNE](http://lejon.github.io)
 * [How to Use t-SNE Effectively](http://distill.pub/2016/misread-tsne/)

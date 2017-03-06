[![Travis](https://travis-ci.org/lejon/TSne.jl.svg?branch=master)](https://travis-ci.org/lejon/TSne.jl)
[![Coveralls](https://coveralls.io/repos/github/lejon/TSne.jl/badge.svg?branch=master)](https://coveralls.io/github/lejon/TSne.jl?branch=master)

Julia t-SNE
===========

Julia port of L.J.P. van der Maaten and G.E. Hintons T-SNE visualisation technique.

Please observe, that it is not extensively tested.

The examples in the 'examples' dir requires you to have Gadfly and RDatasets installed

**Please note:** At some point something changed in Julia which caused poor results, it took a while before I noted this but now  I have updated the implementation so that it works again. See the link below for images rendered using this implementation.

For some tips working with t-sne [Klick here] (http://lejon.github.io)

## Basic installation:

  `julia> Pkg.clone("git://github.com/lejon/TSne.jl.git")`

## Basic API usage:

```jl
using TSne, MNIST

function rescale(A, dim::Integer=1)
    res = A .- mean(A, dim)
    res ./= map!(x -> x > 0.0 ? x : 1.0, std(A, dim))
    res
end

data, labels = traindata()
data = convert(Matrix{Float64}, data[:, 1:2500])'
# Normalize the data, this should be done if there are large scale differences in the dataset
X = rescale(data, 1)

Y = tsne(X, 2, 50, 1000, 20.0)

using Gadfly
labels = convert(Vector{String}, labels[1:2500])
theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)
draw(PDF("myplot.pdf", 4inch, 3inch), theplot)
```

![](example.png)

## Stand Alone Usage

```julia demo-csv.jl haveheader --labelcol=5 iris-headers.csv```

Creates myplot.pdf with TSne result visuallized using Gadfly.

julia-tsne
==========

Julia port of L.J.P. van der Maaten and G.E. Hintons T-SNE visualisation technique.

Please observe, that it is not extensively tested. 

The examples in the 'examples' dir requires you to have Gadfly and RDatasets installed

**Please note:** At some point something changed in Julia which caused poor results, it took a while before I noted this but now  I have updated the implementation so that it works again. See the link below for images rendered using this implementation.

Basic installation: 

  `julia> Pkg.clone("git://github.com/lejon/TSne.jl.git")`
  
Basic usage: 
  
```jl
using TSne, MNIST

data, labels = traindata()
data = data'
data = data[1:2500,:]
Xcenter = data - mean(data)
Xstd = std(data)
X = Xcenter / Xstd

Y = tsne(X, 2, 50, 1000, 20.0)

using Gadfly
theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)
draw(PDF("myplot.pdf", 4inch, 3inch), theplot)
```

![](example.png)

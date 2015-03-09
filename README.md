julia-tsne
==========

Julia port of L.J.P. van der Maaten and G.E. Hintons T-SNE visualisation technique.

Please observe, that it is not extensively tested. 

The examples in the 'examples' dir requires you to have Gadfly and RDatasets installed

**Please note:** At some point something changed in Julia which caused poor results, it took a while before I noted this but now  I have updated the implementation so that it works again. See the link below for images rendered using this implementation.

[Resulting graph](http://lejon.github.io/TSne/)

Basic installation: 

  `julia> Pkg.clone("git://github.com/lejon/TSne.jl.git")`
  
Basic usage: 
  
`using TSne`

`using Gadfly`

`X = readcsv("mnist2500_X_reformatted.txt",Float64)`

`labelf = open ("mnist2500_labels.txt")`

`labels = readlines(labelf)`

`labels = map((x)->chomp(x), labels)`

`Y = tsne(X, 2, 50, 1000, 20.0)`

`writecsv("mnist2500_tsne.csv",Y)`

`theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)`

`draw(PDF("myplot.pdf", 4inch, 3inch), theplot)`

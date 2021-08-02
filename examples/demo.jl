using Gadfly
using Cairo
using TSne
using Statistics: mean, std
using CSV
using DataFrames: DataFrame
using DelimitedFiles: writedlm

"""
Normalize `A` columns, so that the mean and standard deviation
of each column are 0 and 1, resp.
"""
function rescale(A, dim::Integer=1)
    res = A .- mean(A, dims=dim)
    res ./= map(x -> x > 0.0 ? x : 1.0, std(A, dims=dim))
    res
end

if length(ARGS)==0
    println("usage:\n\tjulia demo.jl iris\n\tjulia demo.jl mnist")
    exit(0)
end

if ARGS[1] == "iris"
    using RDatasets
    println("Using Iris dataset.")
    iris = dataset("datasets","iris")
    X = hcat(eachcol(iris)[1:4]...)
    labels = iris[:, 5]
    plotname = "iris"
    initial_dims = -1
    iterations = 1500
    perplexity = 15
elseif ARGS[1] == "mnist"
    using MLDatasets
    println("Using MNIST dataset.")
    X, labels = MNIST.traindata(Float64);
    npts = min(2500, size(X, 3), length(labels))
    labels = string.(labels[1:npts])
    X = rescale(transpose(reshape(X[:, :, 1:npts], (size(X, 1)*size(X, 2), :))))
    plotname = "mnist"
    initial_dims = 50
    iterations = 1000
    perplexity = 20
else
    error("Unknown dataset \"", ARGS[1], "\"")
end

println("X dimensions are: ", size(X))
Y = tsne(X, 2, initial_dims, iterations, perplexity)
println("Y dimensions are: ", size(Y))

CSV.write(plotname*"_tsne_out.csv", DataFrame(Y), writeheader=false)
open("labels.txt", "w") do io
    writedlm(io, (labels,), '\n')
end

theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)
draw(PDF(plotname*".pdf", 4inch, 3inch), theplot)
#draw(SVG(plotname*".svg", 4inch, 3inch), theplot)

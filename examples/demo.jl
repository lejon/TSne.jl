using Gadfly
using TSne

function normalize(A)
	for col in 1:size(A)[2]
        	std(A[:,col]) == 0 && continue 
        	A[:,col] = (A[:,col]-mean(A[:,col])) / std(A[:,col])
	end
	A
end

if length(ARGS)==0 
    println("usage:\n\tjulia demo.jl iris\n\tjulia demo.jl mnist")
    exit(0)
end

use_iris = ARGS[1] == "iris"
lables = ()

if use_iris
    using RDatasets
    println("Using Iris dataset.")
    iris = dataset("datasets","iris")
    X = float(convert(Array,iris[:,1:4]))
    labels = iris[:,5]
    plotname = "iris"
    initial_dims = -1
    iterations = 1500
    perplexity = 15
else
    using MNIST
    X, labels = traindata()
    labels = labels[1:2500]
    X = X'
    X = X[1:2500,:]
    X = normalize(X)
    plotname = "mnist"
    initial_dims = 50
    iterations = 1000
    perplexity = 20
end

println("X dimensions are: " * string(size(X)))
Y = tsne(X, 2, initial_dims, iterations, perplexity)
println("Y dimensions are: " * string(size(Y)))

writecsv(plotname*"_tsne_out.csv",Y)
lbloutfile = open("labels.txt", "w")
writedlm(lbloutfile,labels)
close(lbloutfile)

theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)

draw(PDF(plotname*".pdf", 4inch, 3inch), theplot)
#draw(SVG(plotname*".svg", 4inch, 3inch), theplot)

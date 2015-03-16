using Gadfly 
using TSne

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
	X = array(iris[:,1:4])
	labels = iris[:,5]
	plotname = "iris"
	initial_dims = -1
	iterations = 1500
	perplexity = 15
else
	using MNIST
	X, labels = traindata()
	X = X'
	X = X[1:2500,:]
	Xcenter = X - mean(X)
	Xstd = std(X)
	X = Xcenter / Xstd
	plotname = "mnist"
	initial_dims = 50
	iterations = 1000
	perplexity = 20
end

println("X dimensions are: " * string(size(X)))
Y = tsne(X, 2, initial_dims, iterations, perplexity)
println("Y dimensions are: " * string(size(Y)))

theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)

writecsv(plotname*"_tsne_out.csv",Y)
draw(PDF(plotname*".pdf", 4inch, 3inch), theplot)
draw(SVG(plotname*".svg", 4inch, 3inch), theplot)

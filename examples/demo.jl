using RDatasets
using Gadfly 
using TSne

use_iris = false
lables = ()

if use_iris
	println("Using Iris dataset.")
	iris = data("datasets","iris")
	X = matrix(iris[:,2:5])
	labels = iris[:,6]
	plotname = "iris"
	initial_dims = -1
	iterations = 1500
	perplexity = 15
else
	println("Using MNIST dataset.")
	mnist = readcsv("mnist2500_X_reformatted.txt",Float64)
	X = mnist
	labelf = open ("mnist2500_labels.txt")
	labels = readlines(labelf)
	labels = map((x)->chomp(x), labels)
	plotname = "mnist"
	initial_dims = 50
	iterations = 1000
	perplexity = 30
end


#X = randn(5, 3)

println("X dimensions are: " * string(size(X)))
Y = tsne(X, 2, initial_dims, iterations, perplexity)
#Y = pca(X,2)
println("Y dimensions are: " * string(size(Y)))

theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)

draw(PDF(plotname*".pdf", 4inch, 3inch), theplot)
draw(SVG(plotname*".svg", 4inch, 3inch), theplot)
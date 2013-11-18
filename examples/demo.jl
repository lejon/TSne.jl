using RDatasets
using Gadfly 
using TSne

use_iris = true
lables = ()

if use_iris
	iris = data("datasets","iris")
	X = matrix(iris[:,2:5])
	labels = iris[:,6]
	plotname = "iris.pdf"
else
	mnist = readcsv("mnist2500_X_reformatted.txt",Float64)
	X = mnist
	labelf = open ("mnist2500_labels.txt")
	labels = readlines(labelf)
	labels = map((x)->chomp(x), labels)
	plotname = "mnist.pdf"
end

println("X dimensions are: " * string(size(X)))
Y = tsne(X, 2, -1, 1000, 30.0)
#Y = pca(X,2)
println("Y dimensions are: " * string(size(Y)))

theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)

draw(PDF(plotname, 4inch, 3inch), theplot)

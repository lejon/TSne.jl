using TSne
using DataFrames

doc = """Use t-SNE to generate a PDF called myplot.pdf from an input CSV file. Default assumption is to have no header and no labels. If these are available in the CSV these must be given as arguments.

Usage:
  demo-csv.jl <filename>
  demo-csv.jl [--labelcol=<col>] <filename>
  demo-csv.jl haveheader [--labelcol=<col>] <filename>
  demo-csv.jl [--labelcolname=<colname>] <filename>
  demo-csv.jl haveheader [--labelcolname=<colname>] <filename>


Options:
  -h --help     Show this screen.
  --version     Show version.
  --filename=   Path to CSV file 
  --noheader    The CSV file does not have a header row	
  --nolabel    The CSV file does not have a label column 

"""

using DocOpt

arguments = docopt(doc, version=v"2.0.0")
dump(arguments)

if nothing==arguments["--labelcol"]
	lblcol = -1
else 
	lblcol = parse(Int64,arguments["--labelcol"])
end

df = readtable(arguments["<filename>"],header = nothing!=arguments["haveheader"])

if nothing!=arguments["--labelcolname"]
	lblcol = find(x -> x==symbol(arguments["--labelcolname"]),names(df))[1]
end

println("Data is $df")
if lblcol>0
	labels = df[:,lblcol]
end

dataset = df[filter(x -> x!=lblcol,1:ncol(df)),]
data = convert(Array,dataset)
# Normalize the data, this should be done if there are large scale differences in the dataset
Xcenter = data - mean(data)
Xstd = std(data)
X = Xcenter / Xstd

# Run t-SNE
Y = tsne(X, 2, 50, 1000, 20.0)

using Gadfly
if lblcol>0
	theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)
else 
	theplot = plot(x=Y[:,1], y=Y[:,2])
end
draw(PDF("myplot.pdf", 8inch, 6inch), theplot)


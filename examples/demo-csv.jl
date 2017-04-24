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

"""
Normalize `A` columns, so that the mean and standard deviation
of each column are 0 and 1, resp.
"""
function rescale(A, dim::Integer=1)
    res = A .- mean(A, dim)
    res ./= map!(x -> x > 0.0 ? x : 1.0, std(A, dim))
    res
end

using DocOpt

arguments = docopt(doc, version=v"2.0.0")
dump(arguments)

if nothing==arguments["--labelcol"]
    lblcol = -1
else
    lblcol = parse(Int64, arguments["--labelcol"])
end

df = readtable(arguments["<filename>"], header = nothing!=arguments["haveheader"])

if nothing!=arguments["--labelcolname"]
    lblcol = findfirst(x -> x==symbol(arguments["--labelcolname"]), names(df))
end

println("Data is $df")
labels = lblcol > 0 ? df[:, lblcol] : nothing

dataset = df[filter(x -> x!=lblcol, 1:ncol(df)), :]
data = convert(Matrix{Float64}, dataset)
# Normalize the data, this should be done if there are large scale differences in the dataset
X = rescale(data)

# Run t-SNE
Y = tsne(X, 2, 50, 1000, 20.0)

using Gadfly
theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)
draw(PDF("myplot.pdf", 8inch, 6inch), theplot)

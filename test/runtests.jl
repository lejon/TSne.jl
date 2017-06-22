using RDatasets, TSne, MNIST
using Base.Test

my_tests = [
  "test_tsne.jl",
]

for t in my_tests
  include(t)
end

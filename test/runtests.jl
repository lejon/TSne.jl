using FactCheck, RDatasets, TSne, MNIST

my_tests = [
  "test_tsne.jl",
]

for t in my_tests
  include(t)
end

exitstatus()

using RDatasets, TSne, MNIST
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

my_tests = [
  "test_tsne.jl",
]

for t in my_tests
  include(t)
end

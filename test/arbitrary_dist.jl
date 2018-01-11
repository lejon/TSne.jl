using TSne

struct MyType
    x::Float64
end

function test_tsne()
    δ(x::MyType, y::MyType) = abs(x.x - y.x) % 1.0
    data = [MyType(rand()) for i = 1:100]
    tsne(data, δ)
end

@testset "tsne()" begin
    @testset "API tests" begin
        iris = dataset("datasets", "iris")
        X = convert(Matrix{Float64}, iris[:, 1:4])
        Y = tsne(X, ndims=2, maxiter=10, verbose=true)
        @test size(Y) == (150, 2)
        Y = tsne(X, ndims=3, maxiter=10, verbose=false)
        @test size(Y) == (150, 3)
        tsne(X, ndims=2, maxiter=10, verbose=true, progress=false)
        Y = tsne(X, ndims=2, maxiter=10, inilayout=:pca, cheat_scale=1.0, progress=false)
        @test size(Y) == (150, 2)
        Y, beta, kldiv = tsne(X, ndims=2, maxiter=10, inilayout=:pca, cheat_scale=1.0, progress=false, extended_output=true)
        @test size(Y) == (150, 2)
        @test beta isa AbstractVector
        @test length(beta) == 150
        @test isfinite(kldiv)

        @testset "distance = true" begin
            @test_throws ArgumentError tsne(X, ndims=3, maxiter=10, distance=true, verbose=false)
            XX = pairwise(CosineDist(), X')
            Y = tsne(XX, ndims=3, maxiter=10, distance=true, verbose=false)
            @test size(Y) == (150, 3)
        end

        @testset "distance isa Function" begin
            Y = tsne(X, ndims=3, maxiter=10, distance=cityblock, verbose=false)
            @test size(Y) == (150, 3)
        end

        @testset "X isa Vector, distance isa Function" begin
            Y = tsne([view(X, i, :) for i in 1:size(X, 1)],
                     ndims=3, maxiter=10, distance=cityblock, verbose=false)
            @test size(Y) == (150, 3)
        end

        @testset "distance isa Distances.Metric" begin
            Y = tsne(X, ndims=3, maxiter=10, distance=Minkowski(0.5), verbose=false)
            @test size(Y) == (150, 3)
        end
    end

    @testset "Iris dataset" begin
        iris = dataset("datasets", "iris")
        X = convert(Matrix{Float64}, iris[:, 1:4])

        # embed in 3D
        Y3d = tsne(X, ndims=3, maxiter=10, progress=false)
        @test size(Y3d) == (150, 3)

        # embed in 2D
        Y2d = tsne(X, ndims=2, maxiter=10, progress=false)
        @test size(Y2d) == (150, 2)
    end

    @testset "MNIST.traindata() dataset" begin
        Random.seed!(345678)
        train_data, labels = MLDatasets.with_accept(true) do
                MNIST.traindata(Float64)
        end
        X_labels = labels[1:2500] .+ 1
        X = reshape(permutedims(train_data[:, :, 1:2500], (3, 1, 2)),
                    2500, size(train_data, 1)*size(train_data, 2))
        X .-= mean(X, dims=1)
        X ./= std(X, dims=1)

        Y = tsne(X, ndims=2, reduce_dims=50, maxiter=2000, perplexity=20, progress=true, progress=true)
        @test size(Y) == (2500, 2)
    end
end

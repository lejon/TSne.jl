@testset "tsne()" begin
    @testset "API tests" begin
        iris = dataset("datasets", "iris")
        X = convert(Matrix{Float64}, iris[:, 1:4])
        Y = tsne(X, 2, -1, 10, 15, verbose=true)
        @test size(Y) == (150, 2)
        Y = tsne(X, 3, -1, 10, 15, verbose=false)
        @test size(Y) == (150, 3)
        tsne(X, 2, -1, 10, 15, verbose=true, progress=false)
        tsne(X, 2, -1, 10, 15, verbose=false, progress=false)
        Y = tsne(X, 2, -1, 10, 15, pca_init=true, cheat_scale=1.0, progress=false)
        @test size(Y) == (150, 2)
        Y, beta, kldiv = tsne(X, 2, -1, 10, 15, pca_init=true, cheat_scale=1.0, progress=false, extended_output=true)
        @test size(Y) == (150, 2)
        @test beta isa AbstractVector
        @test length(beta) == 150
        @test isfinite(kldiv)

        @testset "distance = true" begin
            @test_throws ArgumentError tsne(X, 3, -1, 10, 15, distance=true, verbose=false)
            XX = pairwise(CosineDist(), X')
            Y = tsne(XX, 3, -1, 10, 15, distance=true, verbose=false)
            @test size(Y) == (150, 3)
        end

        @testset "distance isa Function" begin
            Y = tsne(X, 3, -1, 10, 15, distance=cityblock, verbose=false)
            @test size(Y) == (150, 3)
        end

        @testset "X isa Vector, distance isa Function" begin
            Y = tsne([view(X, i, :) for i in 1:size(X, 1)],
                     3, -1, 10, 15, distance=cityblock, verbose=false)
            @test size(Y) == (150, 3)
        end

        @testset "distance isa Distances.Metric" begin
            Y = tsne(X, 3, -1, 10, 15, distance=Minkowski(0.5), verbose=false)
            @test size(Y) == (150, 3)
        end
    end

    @testset "Iris dataset" begin
        iris = dataset("datasets", "iris")
        X = convert(Matrix{Float64}, iris[:, 1:4])
        # embed in 3D
        Y3d = tsne(X, 3, -1, 100, 15, progress=false)
        @test size(Y3d) == (150, 3)
        # embed in 2D
        Y2d = tsne(X, 2, 50, 50, 20, progress=false)
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

        Y = tsne(X, 2, 50, 2000, 20, progress=true)
        @test size(Y) == (2500, 2)
    end
end

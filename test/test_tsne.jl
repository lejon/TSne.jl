@testset "tsne()" begin
    @testset "API tests" begin
        iris = dataset("datasets", "iris")
        X = convert(Matrix, iris[:, 1:4])
        Y = tsne(X, 2, -1, 10, 15, verbose=true)
        @test size(Y) == (150, 2)
        Y = tsne(X, 3, -1, 10, 15, verbose=false)
        @test size(Y) == (150, 3)
        tsne(X, 2, -1, 10, 15, verbose=true, progress=false)
        tsne(X, 2, -1, 10, 15, verbose=false, progress=false)
        Y = tsne(X, 2, -1, 10, 15, pca_init=true, cheat_scale=1.0, progress=false)
        @test size(Y) == (150, 2)
    end

    @testset "Iris dataset" begin
        iris = dataset("datasets", "iris")
        X = convert(Matrix, iris[:, 1:4])
        # embed in 3D
        Y3d = tsne(X, 3, -1, 1500, 15, progress=false)
        @test size(Y3d) == (150, 3)
        # embed in 2D
        Y2d = tsne(X, 2, 50, 50, 20, progress=false)
        @test size(Y2d) == (150, 2)
    end

    @testset "MNIST.traindata() dataset" begin
        train_data, labels = MNIST.traindata()
        X = train_data[:, 1:2500]'
        Xcenter = X - mean(X)
        Xstd = std(X)
        X = Xcenter / Xstd
        Y = tsne(X, 2, 50, 30, 20, progress=true)
        @test size(Y) == (2500, 2)
    end
end

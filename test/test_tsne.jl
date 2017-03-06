@testset "tsne()" begin
    @testset "API tests" begin
        iris = dataset("datasets","iris")
        X = convert(Matrix, iris[:, 1:4])
        @testset "verbose=true" begin
            Y = tsne(X, 2, -1, 10, 15, verbose=true)
            @test size(Y) == (150, 2)
        end
        @testset "verbose=false" begin
            Y = tsne(X, 3, -1, 10, 15, verbose=false)
            @test size(Y) == (150, 3)
        end
        @testset "no progress bar" begin
            tsne(X, 2, -1, 10, 15, verbose=true, progress=false)
        end
        @testset "no progress bar, verbose=false" begin
            tsne(X, 2, -1, 10, 15, verbose=false, progress=false)
        end
        @testset "PCA for initial layout" begin
            Y = tsne(X, 2, -1, 10, 15, pca_init=true, cheat_scale=1.0, progress=false)
            @test size(Y) == (150, 2)
        end
    end

    @testset "Iris dataset" begin
        iris = dataset("datasets","iris")
        X = convert(Matrix, iris[:, 1:4])
        @testset "embed in 3D" begin
            Y = tsne(X, 3, -1, 1500, 15, progress=false)
            @test size(Y) == (150, 3)
        end
        @testset "embed in 2D" begin
            Y = tsne(X, 2, 50, 50, 20, progress=false)
            @test size(Y) == (150, 2)
        end
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

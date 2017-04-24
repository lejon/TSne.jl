@testset "tsne()" begin
    @testset "API tests" begin
        info("t1")
        iris = dataset("datasets", "iris")
        info("t2")
        X = convert(Matrix, iris[:, 1:4])
        info("t3")
        Y = tsne(X, 2, -1, 10, 15, verbose=true)
        info("t4")
        @test size(Y) == (150, 2)
        info("t5")
        Y = tsne(X, 3, -1, 10, 15, verbose=false)
        info("t6")
        @test size(Y) == (150, 3)
        info("t7")
        tsne(X, 2, -1, 10, 15, verbose=true, progress=false)
        info("t8")
        tsne(X, 2, -1, 10, 15, verbose=false, progress=false)
        info("t9")
        Y = tsne(X, 2, -1, 10, 15, pca_init=true, cheat_scale=1.0, progress=false)
        info("t10")
        @test size(Y) == (150, 2)
        info("t11")
    end

    @testset "Iris dataset" begin
        info("t12")
        iris = dataset("datasets", "iris")
        info("t13")
        X = convert(Matrix, iris[:, 1:4])
        # embed in 3D
        info("t14")
        Y3d = tsne(X, 3, -1, 1500, 15, progress=false)
        info("t15")
        @test size(Y3d) == (150, 3)
        # embed in 2D
        info("t16")
        Y2d = tsne(X, 2, 50, 50, 20, progress=false)
        info("t17")
        @test size(Y2d) == (150, 2)
    end

    @testset "MNIST.traindata() dataset" begin
        info("t18")
        train_data, labels = MNIST.traindata()
        info("t19")
        X = train_data[:, 1:2500]'
        info("t20")
        Xcenter = X - mean(X)
        info("t21")
        Xstd = std(X)
        info("t22")
        X = Xcenter / Xstd
        info("t23")
        Y = tsne(X, 2, 50, 30, 20, progress=true)
        info("t24")
        @test size(Y) == (2500, 2)
        info("t25")
    end
end

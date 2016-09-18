facts("tsne()") do
    context("API tests") do
        iris = dataset("datasets","iris")
        X = convert(Matrix, iris[:, 1:4])
        context("verbose=true") do
            Y = tsne(X, 2, -1, 10, 15, verbose=true)
            @fact size(Y) --> (150, 2)
        end
        context("verbose=false") do
            Y = tsne(X, 3, -1, 10, 15, verbose=false)
            @fact size(Y) --> (150, 3)
        end
        context("no progress bar") do
            tsne(X, 2, -1, 10, 15, verbose=true, progress=false)
        end
        context("no progress bar, verbose=false") do
            tsne(X, 2, -1, 10, 15, verbose=false, progress=false)
        end
        context("PCA for initial layout") do
            Y = tsne(X, 2, -1, 10, 15, pca_init=true, cheat_scale=1.0, progress=false)
            @fact size(Y) --> (150, 2)
        end
    end

    context("Iris dataset") do
        iris = dataset("datasets","iris")
        X = convert(Matrix, iris[:, 1:4])
        context("embed in 3D") do
            Y = tsne(X, 3, -1, 1500, 15, progress=false)
            @fact size(Y) --> (150, 3)
        end
        context("embed in 2D") do
            Y = tsne(X, 2, 50, 50, 20, progress=false)
            @fact size(Y) --> (150, 2)
        end
    end

    context("MNIST.traindata() dataset") do
        train_data, labels = MNIST.traindata()
        X = train_data[:, 1:2500]'
        Xcenter = X - mean(X)
        Xstd = std(X)
        X = Xcenter / Xstd
        Y = tsne(X, 2, 50, 30, 20, progress=true)
        @fact size(Y) --> (2500, 2)
    end
end

@testset "tsne()" begin
    @testset "API tests" begin
        iris = dataset("datasets", "iris")
        X = hcat(eachcol(iris)[1:4]...)
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
            XX = pairwise(CosineDist(), X', dims=2)
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
        X = hcat(eachcol(iris)[1:4]...)
        # embed in 3D
        Y3d = tsne(X, 3, -1, 100, 15, progress=false)
        @test size(Y3d) == (150, 3)
        # embed in 2D
        Y2d = tsne(X, 2, 50, 50, 20, progress=false)
        @test size(Y2d) == (150, 2)
    end

    @testset "Barnes-Hut method" begin
        iris = dataset("datasets", "iris")
        X = hcat(eachcol(iris)[1:4]...)

        @testset "basic API" begin
            Y = tsne(X, 2, -1, 10, 15, method=:barneshut, theta=0.5, verbose=false, progress=false)
            @test size(Y) == (150, 2)
            Y = tsne(X, 3, -1, 10, 15, method=:barneshut, theta=0.5, verbose=false, progress=false)
            @test size(Y) == (150, 3)
        end

        @testset "extended_output" begin
            Y, beta, kldiv = tsne(X, 2, -1, 10, 15, method=:barneshut, theta=0.5,
                                  verbose=false, progress=false, extended_output=true)
            @test size(Y) == (150, 2)
            @test isfinite(kldiv)
            @test kldiv > 0
            @test beta isa AbstractVector
            @test length(beta) == 150
            @test all(isfinite, beta)
            @test all(>(0), beta)
        end

        @testset "extended_output compatibility with exact" begin
            Random.seed!(42)
            exact = tsne(X, 2, -1, 10, 15, verbose=false, progress=false,
                         extended_output=true)
            Random.seed!(42)
            bh = tsne(X, 2, -1, 10, 15, method=:barneshut, theta=0.5,
                      verbose=false, progress=false, extended_output=true)
            @test length(exact) == length(bh) == 3
            @test size(exact[1]) == size(bh[1]) == (150, 2)
            @test exact[2] isa AbstractVector
            @test bh[2] isa AbstractVector
            @test length(exact[2]) == length(bh[2]) == 150
            @test isfinite(exact[3])
            @test isfinite(bh[3])
        end

        @testset "pca_init" begin
            Y = tsne(X, 2, -1, 10, 15, method=:barneshut, theta=0.5,
                     pca_init=true, verbose=false, progress=false)
            @test size(Y) == (150, 2)
        end

        @testset "theta consistency" begin
            Random.seed!(42)
            Y_lo = tsne(X, 2, -1, 50, 15, method=:barneshut, theta=0.1, verbose=false, progress=false)
            Random.seed!(42)
            Y_hi = tsne(X, 2, -1, 50, 15, method=:barneshut, theta=0.8, verbose=false, progress=false)
            @test size(Y_lo) == size(Y_hi)
            @test all(isfinite, Y_lo)
            @test all(isfinite, Y_hi)
            # different θ values should give different embeddings
            @test !isapprox(Y_lo, Y_hi, atol=1e-4)
        end

        @testset "bounded tree depth" begin
            Random.seed!(42)
            Y = tsne(X, 2, -1, 10, 15, method=:barneshut, theta=0.5,
                     max_depth=4, verbose=false, progress=false)
            @test size(Y) == (150, 2)
            @test all(isfinite, Y)
            Y3 = tsne(X, 3, -1, 10, 15, method=:barneshut, theta=0.5,
                      max_depth=4, verbose=false, progress=false)
            @test size(Y3) == (150, 3)
            @test all(isfinite, Y3)
            @test_throws ArgumentError tsne(X, 2, -1, 10, 15, method=:barneshut,
                                            max_depth=-1, verbose=false, progress=false)
        end

        @testset "distance function" begin
            Y = tsne(X, 2, -1, 10, 15, method=:barneshut, theta=0.5,
                     distance=cityblock, verbose=false, progress=false)
            @test size(Y) == (150, 2)
        end

        @testset "distance metric" begin
            Y = tsne(X, 2, -1, 10, 15, method=:barneshut, theta=0.5,
                     distance=Minkowski(0.5), verbose=false, progress=false)
            @test size(Y) == (150, 2)
        end

        @testset "distance = true (precomputed)" begin
            @test_throws ArgumentError tsne(X, 2, -1, 10, 15, method=:barneshut, theta=0.5,
                                            distance=true, verbose=false, progress=false)
            XX = pairwise(CosineDist(), X', dims=2)
            Y = tsne(XX, 2, -1, 10, 15, method=:barneshut, theta=0.5,
                     distance=true, verbose=false, progress=false)
            @test size(Y) == (150, 2)
        end

        @testset "tree excludes self interactions" begin
            Ydup = [0.0 0.0;
                    0.0 0.0;
                    1.0 0.0]
            tree = TSne.build_spacetree(Ydup)
            yi = zeros(2)
            grad = zeros(2)
            z_ref = Ref(0.0)
            for i in 1:size(Ydup, 1)
                yi .= view(Ydup, i, :)
                TSne.walk_tree_into!(tree, Ydup, yi, 0.0, i, z_ref, grad)
                exact_z = sum(j == i ? 0.0 : 1.0 / (1.0 + sum(abs2, Ydup[i, :] .- Ydup[j, :]))
                              for j in 1:size(Ydup, 1))
                @test z_ref[] ≈ exact_z
            end
        end

        @testset "small theta (θ=0.01) no crash" begin
            Y = tsne(X, 2, -1, 20, 15, method=:barneshut, theta=0.01,
                     verbose=false, progress=false)
            @test size(Y) == (150, 2)
            @test all(isfinite, Y)
        end

        @testset "large theta (θ=0.8) no crash" begin
            Y = tsne(X, 2, -1, 20, 15, method=:barneshut, theta=0.8,
                     verbose=false, progress=false)
            @test size(Y) == (150, 2)
            @test all(isfinite, Y)
        end
    end

    @testset "rng reproducibility" begin
        iris = dataset("datasets", "iris")
        X = hcat(eachcol(iris)[1:4]...)

        @testset "exact method" begin
            Y1 = tsne(X, 2, -1, 10, 15, rng=MersenneTwister(42), progress=false)
            Y2 = tsne(X, 2, -1, 10, 15, rng=MersenneTwister(42), progress=false)
            @test Y1 == Y2
            Y3 = tsne(X, 2, -1, 10, 15, rng=MersenneTwister(99), progress=false)
            @test Y1 != Y3
        end

        @testset "barneshut method" begin
            Y1 = tsne(X, 2, -1, 10, 15, method=:barneshut, rng=MersenneTwister(42), progress=false)
            Y2 = tsne(X, 2, -1, 10, 15, method=:barneshut, rng=MersenneTwister(42), progress=false)
            @test Y1 == Y2
            Y3 = tsne(X, 2, -1, 10, 15, method=:barneshut, rng=MersenneTwister(99), progress=false)
            @test Y1 != Y3
        end
    end

    @testset "FFT method" begin
        iris = dataset("datasets", "iris")
        X = hcat(eachcol(iris)[1:4]...)

        @testset "basic API" begin
            Y = tsne(X, 2, -1, 10, 15, method=:fft, verbose=false, progress=false)
            @test size(Y) == (150, 2)
            @test all(isfinite, Y)
        end

        @testset "ndims != 2 throws" begin
            @test_throws ArgumentError tsne(X, 3, -1, 10, 15, method=:fft,
                                            verbose=false, progress=false)
        end

        @testset "extended_output" begin
            Y, beta, kldiv = tsne(X, 2, -1, 10, 15, method=:fft,
                                  verbose=false, progress=false, extended_output=true)
            @test size(Y) == (150, 2)
            @test all(isfinite, Y)
            @test beta isa AbstractVector
            @test length(beta) == 150
            @test all(isfinite, beta)
            @test isfinite(kldiv) && kldiv > 0
        end

        @testset "pca_init" begin
            Y = tsne(X, 2, -1, 10, 15, method=:fft,
                     pca_init=true, verbose=false, progress=false)
            @test size(Y) == (150, 2)
            @test all(isfinite, Y)
        end

        @testset "rng reproducibility" begin
            Y1 = tsne(X, 2, -1, 10, 15, method=:fft,
                      rng=MersenneTwister(42), progress=false)
            Y2 = tsne(X, 2, -1, 10, 15, method=:fft,
                      rng=MersenneTwister(42), progress=false)
            @test Y1 == Y2
            Y3 = tsne(X, 2, -1, 10, 15, method=:fft,
                      rng=MersenneTwister(99), progress=false)
            @test Y1 != Y3
        end

        @testset "distance function" begin
            Y = tsne(X, 2, -1, 10, 15, method=:fft,
                     distance=cityblock, verbose=false, progress=false)
            @test size(Y) == (150, 2)
            @test all(isfinite, Y)
        end

        @testset "distance metric (PreMetric)" begin
            Y = tsne(X, 2, -1, 10, 15, method=:fft,
                     distance=Minkowski(0.5), verbose=false, progress=false)
            @test size(Y) == (150, 2)
            @test all(isfinite, Y)
        end

        @testset "n_boxes_per_dim override" begin
            Y = tsne(X, 2, -1, 10, 15, method=:fft,
                     n_boxes_per_dim=20, verbose=false, progress=false)
            @test size(Y) == (150, 2)
            @test all(isfinite, Y)
        end

        @testset "distance = true (precomputed)" begin
            @test_throws ArgumentError tsne(X, 2, -1, 10, 15, method=:fft,
                                            distance=true, verbose=false, progress=false)
            XX = pairwise(CosineDist(), X', dims=2)
            Y = tsne(XX, 2, -1, 10, 15, method=:fft,
                     distance=true, verbose=false, progress=false)
            @test size(Y) == (150, 2)
            @test all(isfinite, Y)
        end

        @testset "FFT φ and Z unit test" begin
            # Compare FFT-interpolated φ₀..φ₃ and Z against exact O(n²) brute-force
            n = 50
            rng_local = MersenneTwister(1234)
            Y = randn(rng_local, Float64, n, 2)
            dY = zeros(Float64, n, 2)
            n_boxes = 30
            ws = TSne.FFTWorkspace(n, n_boxes)
            Z_fft = TSne.compute_repulsive_forces_fft_2d!(dY, Y, ws)

            # Exact sums
            phi_exact = zeros(Float64, n, 4)
            for i in 1:n
                for j in 1:n
                    d2 = (Y[i,1]-Y[j,1])^2 + (Y[i,2]-Y[j,2])^2
                    k = (1.0 + d2)^(-2)
                    phi_exact[i, 1] += k * 1.0
                    phi_exact[i, 2] += k * Y[j, 1]
                    phi_exact[i, 3] += k * Y[j, 2]
                    phi_exact[i, 4] += k * (Y[j,1]^2 + Y[j,2]^2)
                end
            end

            Z_exact = 0.0
            for i in 1:n, j in 1:n
                i == j && continue
                Z_exact += 1.0 / (1.0 + (Y[i,1]-Y[j,1])^2 + (Y[i,2]-Y[j,2])^2)
            end

            for d in 1:4
                phi_fft_d = ws.PHI[d, :]   # layout is (4, n)
                phi_ex_d  = phi_exact[:, d]
                relerr = maximum(abs.(phi_fft_d .- phi_ex_d) ./ (abs.(phi_ex_d) .+ 1e-10))
                @test relerr < 0.05
            end
            @test abs(Z_fft - Z_exact) / Z_exact < 0.05
        end

        @testset "cluster separation" begin
            # Two well-separated clusters in input should separate in embedding
            rng_local = MersenneTwister(777)
            n_per = 50
            X_clust = vcat(randn(rng_local, n_per, 5),
                           randn(rng_local, n_per, 5) .+ 20.0)
            Y = tsne(X_clust, 2, -1, 100, 10, method=:fft,
                     rng=MersenneTwister(1), progress=false)
            c1 = mean(Y[1:n_per, :], dims=1)
            c2 = mean(Y[(n_per+1):end, :], dims=1)
            centroid_dist = sqrt(sum((c1 .- c2).^2))
            @test centroid_dist > 1.0
        end
    end

    @testset "MNIST dataset" begin
        Random.seed!(345678)
        train_data, labels = MNIST(split=:train)[:]
        train_data = Float64.(train_data)
        X = reshape(permutedims(train_data[:, :, 1:500], (3, 1, 2)),
                    500, size(train_data, 1)*size(train_data, 2))
        X .-= mean(X, dims=1)
        X ./= map(x -> ifelse(x > 0, x, 1.0), std(X, dims=1))

        Y = tsne(X, 2, 50, 10, 20, progress=false)
        @test size(Y) == (500, 2)
        @test all(isfinite, Y)
    end
end

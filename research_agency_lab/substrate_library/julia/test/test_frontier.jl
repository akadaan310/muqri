# Mathematical identities the frontier layer must satisfy (run: julia --project=. test/test_frontier.jl).
using Test, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "QaariLab.jl"))
using .QaariLab
const Q = QaariLab

randspd(rng, d) = (A = randn(rng, d, d); Matrix(Symmetric(A * A' + 0.1I)))

@testset "frontier math" begin
    rng = MersenneTwister(7)
    @testset "Bures–Wasserstein barycenter" begin
        # commuting (diagonal) case: C̄^½ = Σ wᵢ Cᵢ^½
        Cs = [Matrix(Diagonal(rand(rng, 3) .+ 0.2)) for _ in 1:4]
        w = rand(rng, 4); w ./= sum(w)
        _, C̄, _ = Q.bw_barycenter([zeros(3) for _ in 1:4], Cs, w)
        @test C̄ ≈ (sum(w[i] * sqrt.(Cs[i]) for i in 1:4))^2 atol = 1e-9
        # general case satisfies the fixed-point equation
        Cs = [randspd(rng, 4) for _ in 1:5]; w = fill(0.2, 5)
        _, C̄, _ = Q.bw_barycenter([zeros(4) for _ in 1:5], Cs, w)
        Ch = Q.psd_sqrt(C̄)
        @test C̄ ≈ sum(w[i] * Matrix(Q.psd_sqrt(Ch * Cs[i] * Ch)) for i in 1:5) rtol = 1e-8
    end
    @testset "W2 and AIRM" begin
        A, B = randspd(rng, 3), randspd(rng, 3)
        m = randn(rng, 3)
        @test Q.w2_gauss(m, A, m, A)[3] ≈ 0 atol = 1e-9
        # 1-D: W2² = (μ1−μ2)² + (σ1−σ2)²
        @test Q.w2_gauss([1.0], fill(4.0, 1, 1), [0.0], fill(1.0, 1, 1))[3] ≈ 1 + 1 atol = 1e-12
        @test Q.airm_dist(A, B) ≈ Q.airm_dist(B, A) rtol = 1e-9
        G = randn(rng, 3, 3) + 3I          # affine invariance: d(GAGᵀ, GBGᵀ) = d(A, B)
        @test Q.airm_dist(G * A * G', G * B * G') ≈ Q.airm_dist(A, B) rtol = 1e-7
        @test norm(Q.le_vec(A) - Q.le_vec(B)) ≈ norm(Matrix(Q.logm_spd(A)) - Matrix(Q.logm_spd(B))) rtol = 1e-9
    end
    @testset "Karcher mean of two = AIRM geodesic midpoint" begin
        A, B = randspd(rng, 3), randspd(rng, 3)
        Ah, Ahi = Q.psd_sqrt(A), Q.psd_invsqrt(A)
        mid = Ah * Q.psd_sqrt(Ahi * B * Ahi) * Ah
        @test Q.karcher_mean([A, B], [0.5, 0.5]) ≈ mid rtol = 1e-7
    end
    @testset "OGK" begin
        X = randn(rng, 4000, 3) * [2.0 0 0; 0.5 1 0; 0 0 0.3] .+ [10.0 -3 1]
        μ, Σ = Q.ogk(X)
        @test μ ≈ vec(mean(X; dims = 1)) atol = 0.1
        @test Σ ≈ cov(X) rtol = 0.1
        Xc = vcat(X, fill(1e3, 200, 3))     # 5 % gross outliers barely move it
        μc, _ = Q.ogk(Xc)
        @test μc ≈ μ atol = 0.15
    end
    @testset "conformal" begin
        @test Q.conformal_threshold(1:9, 0.2) == (8.0, true)       # ⌈10·0.8⌉ = 8
        @test Q.conformal_threshold(1:7, 0.1) == (7.0, false)      # ⌈8·0.9⌉ = 8 > 7
    end
    @testset "Sinkhorn divergence respects the ground metric" begin
        M = [abs(i - j) for i in 1:5, j in 1:5] ./ 4.0
        a = [1.0, 0, 0, 0, 0] .+ 1e-9
        near, far = [0, 1.0, 0, 0, 0] .+ 1e-9, [0, 0, 0, 0, 1.0] .+ 1e-9
        @test abs(Q.sinkhorn_divergence(a, a, M)) < 1e-6
        @test 0 < Q.sinkhorn_divergence(a, near, M) < Q.sinkhorn_divergence(a, far, M)
    end

    @testset "EVT / GPD tail thresholds (A6)" begin
        # inverse CDF of the GPD, so we can simulate tails with a known shape
        gpd_q(u, ξ, σ) = abs(ξ) < 1e-12 ? -σ * log(1 - u) : (σ / ξ) * ((1 - u)^(-ξ) - 1)

        # 1. parameter recovery on a heavy right tail (ξ > 0 — deliberate elongation, e.g. madd)
        for (ξ0, σ0) in ((0.3, 2.0), (0.15, 1.0))
            y = [gpd_q(rand(rng), ξ0, σ0) for _ in 1:20_000]
            ξ, σ = Q.gpd_fit(y)
            @test isapprox(ξ, ξ0; atol = 0.05)
            @test isapprox(σ, σ0; rtol = 0.12)
        end

        # 2. bounded tail (ξ < 0) is recovered with the right sign
        y = [gpd_q(rand(rng), -0.25, 1.5) for _ in 1:20_000]
        ξ, σ = Q.gpd_fit(y)
        @test ξ < 0
        @test isapprox(ξ, -0.25; atol = 0.06)

        # 3. exponential limit ξ → 0: scale is the mean
        y = -log.(1 .- rand(rng, 20_000)) .* 3.0
        ξ, σ = Q.gpd_fit(y)
        @test abs(ξ) < 0.06
        @test isapprox(σ, 3.0; rtol = 0.1)

        # 4. degenerate inputs never throw and never return a NaN threshold
        @test Q.gpd_fit(Float64[]) == (0.0, NaN) || isnan(Q.gpd_fit(Float64[])[2])
        q, _, _, _, _ = Q.pot_threshold([1.0, 2.0, 3.0], 1e-2)
        @test isfinite(q)

        # 5. the whole point: a smaller exceedance risk must push the cut further out, and on a
        #    heavy-tailed sample the EVT cut must sit ABOVE the Gaussian μ+3σ rule that it replaces
        scores = [gpd_q(rand(rng), 0.35, 1.0) for _ in 1:5_000]
        q1, ξ1, _, _, _ = Q.pot_threshold(scores, 1e-2)
        q2, _, _, _, _ = Q.pot_threshold(scores, 1e-3)
        @test q2 > q1
        @test ξ1 > 0                                   # ξ reports the non-Gaussianity
        @test q1 > mean(scores) + 3 * std(scores) * 0.5
        # empirical coverage: ~1 % of the sample should exceed the 1 % cut
        @test 0.002 < count(>(q1), scores) / length(scores) < 0.03
    end
end

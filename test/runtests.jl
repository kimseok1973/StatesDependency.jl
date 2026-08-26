using StatesDependency
using Test
using Random
using Statistics

# Fast settings so the whole suite stays under a couple of minutes.
const FAST = (K = 2, R = 4_000, burnin = 1_500, thin = 2, nchains = 2, verbose = false)

@testset "StatesDependency.jl" begin

    @testset "build_panel" begin
        # two households, brands in rows, occasions in columns
        A = [1.0 0.0 0.0
             0.0 1.0 1.0
             0.0 0.0 0.0]
        B = [0.0 0.0 2.0 0.0
             1.0 0.0 0.0 1.0
             0.0 3.0 0.0 0.0]
        p = build_panel([A, B])
        @test p.B == 3
        @test n_households(p) == 2
        @test p.choices[1] == [1, 2, 2]
        @test p.choices[2] == [2, 3, 1, 2]
        @test n_occasions(p) == 7
        @test n_used(p) == 5
        @test lagged_repeat_rate(p) ≈ 1 / 5

        # all-zero column is a no-purchase occasion and is dropped
        C = [1.0 0.0 0.0
             0.0 0.0 1.0
             0.0 0.0 0.0]
        q = build_panel([C, C])
        @test q.choices[1] == [1, 2]
        @test q.dropped_zero_columns == 2

        # a column with several positive rows
        D = [1.0 3.0
             0.0 2.0
             0.0 0.0]
        @test build_panel([D, D]).choices[1] == [1, 1]
        @test build_panel([D, D]; tie_rule = :argmax).ambiguous_columns == 2
        @test_throws ArgumentError build_panel([D, D]; tie_rule = :error)

        # households with fewer than 2 occasions are dropped
        short = reshape([1.0, 0.0, 0.0], 3, 1)
        r = build_panel([A, short])
        @test n_households(r) == 1
        @test r.dropped_households == 1

        # 3-D input
        X3 = zeros(3, 4, 5)
        for h in 1:5, t in 1:4
            X3[mod1(h + t, 3), t, h] = 1.0
        end
        @test n_households(build_panel(X3)) == 5

        # errors
        @test_throws DimensionMismatch build_panel([A, zeros(2, 3)])
        @test_throws ArgumentError build_panel([-A, A])
        @test_throws ArgumentError build_panel([A]; tie_rule = :nonsense)
        @test_throws ArgumentError build_panel([reshape([1.0, 0.0], 2, 1)])
        @test_throws DimensionMismatch build_panel([A, B]; brand_names = ["a", "b"])
    end

    @testset "shuffle_panel" begin
        sim = simulate_panel(H = 30, B = 3, T = 8, gamma = 1.0, seed = 3)
        p  = build_panel(sim.X)
        sp = shuffle_panel(p, Xoshiro(1))
        @test n_households(sp) == n_households(p)
        for h in 1:n_households(p)
            @test sort(sp.choices[h]) == sort(p.choices[h])   # same basket
        end
        @test n_used(sp) == n_used(p)
    end

    @testset "simulate_panel" begin
        sim = simulate_panel(H = 20, B = 5, T = 7, gamma = 0.5, seed = 9)
        @test length(sim.X) == 20
        @test all(M -> size(M) == (5, 7), sim.X)
        @test all(M -> all(sum(M; dims = 1) .== 1), sim.X)

        rag = simulate_panel(H = 15, B = 3, T = 4:9, seed = 2)
        @test length(unique(size.(rag.X, 2))) > 1

        z = simulate_panel(H = 15, B = 3, T = 6, zero_column_prob = 0.5, seed = 4)
        @test any(M -> any(iszero, sum(M; dims = 1)), z.X)
        @test build_panel(z.X).dropped_zero_columns > 0

        q = simulate_panel(H = 10, B = 3, T = 5, quantity = :random,
                           max_quantity = 4, seed = 5)
        @test maximum(maximum.(q.X)) > 1

        @test_throws ArgumentError simulate_panel(B = 1)
        @test_throws ArgumentError simulate_panel(quantity = :bogus)

        # more state dependence must raise the raw repeat rate
        lo = lagged_repeat_rate(build_panel(simulate_panel(H = 400, B = 4, T = 15,
                                                           gamma = 0.0, seed = 11).X))
        hi = lagged_repeat_rate(build_panel(simulate_panel(H = 400, B = 4, T = 15,
                                                           gamma = 1.5, seed = 11).X))
        @test hi > lo
    end

    @testset "diagnostics" begin
        rng = Xoshiro(7)
        iid = randn(rng, 2000, 3)
        @test split_rhat(iid) < 1.02
        @test ess(iid) > 3000

        # a slow random walk must look badly mixed
        rw = cumsum(randn(rng, 2000, 3); dims = 1)
        @test split_rhat(rw) > 1.05
        @test ess(rw) < ess(iid)
    end

    @testset "recovery: true state dependence" begin
        sim = simulate_panel(H = 350, B = 4, T = 14, gamma = 1.0, K = 2, seed = 21)
        res = DHRTests(sim.X; FAST...)

        @test res isa DHRTestResult
        # The estimator has a mild upward bias in short panels (incidental
        # parameters), so we check the neighbourhood of the truth, not coverage.
        @test 0.7 < res.gamma_mean < 1.4
        @test res.gamma_ci[1] > 0                          # excludes zero
        @test res.excess_ci[1] > 0                         # survives the placebo
        @test res.delta_dic < 0                            # SD model preferred
        @test res.verdict === :state_dependence
        @test res.odds_ratio ≈ exp(res.gamma_mean)
        @test res.delta_pp > 0
        @test isfinite(res.gamma_rhat) && res.gamma_rhat < 1.1
        @test 0.05 < res.accept_gamma < 0.95
        @test length(gamma_draws(res)) == res.settings.nchains *
              length((res.settings.burnin+1):res.settings.thin:res.settings.R)

        s = summarize(res)
        @test s.gamma ≈ res.gamma_mean
        @test s.verdict === :state_dependence

        io = IOBuffer()
        show(io, MIME"text/plain"(), res)
        out = String(take!(io))
        @test occursin("EXCESS", out)
        @test occursin("verdict", out)
    end

    @testset "null: heterogeneity only" begin
        sim = simulate_panel(H = 350, B = 4, T = 14, gamma = 0.0, K = 2, seed = 22)
        res = DHRTests(sim.X; FAST...)

        @test res.excess_ci[1] < 0 < res.excess_ci[2]      # no excess over placebo
        @test res.verdict === :no_evidence
        @test abs(res.gamma_mean) < 0.4
    end

    @testset "no placebo / no null model" begin
        sim = simulate_panel(H = 120, B = 3, T = 10, gamma = 0.8, seed = 23)
        res = DHRTests(sim.X; K = 1, R = 2_000, burnin = 800, thin = 2,
                       nchains = 1, placebo = false, compare_null = false,
                       verbose = false)
        @test res.placebo === nothing
        @test res.excess === nothing
        @test res.dic_nosd === nothing
        @test res.verdict in (:state_dependence, :no_evidence, :inconclusive)
        io = IOBuffer(); show(io, res)
        @test occursin("DHRTestResult", String(take!(io)))
    end

    @testset "covariates" begin
        sim = simulate_panel(H = 120, B = 3, T = 8, gamma = 0.6, seed = 24)
        rng = Xoshiro(24)
        cov = [randn(rng, 3, 1, size(M, 2)) for M in sim.X]
        res = DHRTests(sim.X; covariates = cov, K = 1, R = 2_000, burnin = 800,
                       thin = 2, nchains = 1, placebo = false, compare_null = false,
                       verbose = false)
        @test length(res.beta_mean) == 1
        @test res.beta_ci[1][1] < res.beta_mean[1] < res.beta_ci[1][2]
        # pure noise: the coefficient should not be distinguishable from zero
        @test res.beta_ci[1][1] < 0 < res.beta_ci[1][2]
    end

    @testset "argument validation" begin
        sim = simulate_panel(H = 20, B = 3, T = 6, seed = 25)
        @test_throws ArgumentError DHRTests(sim.X; level = 1.5, verbose = false)
        @test_throws ArgumentError DHRTests(sim.X; K = 0, verbose = false)
        @test_throws ArgumentError DHRTests(sim.X; R = 100, burnin = 200, verbose = false)
        @test_throws ArgumentError DHRTests(sim.X; R = 100, burnin = 50, thin = 100,
                                            verbose = false)
    end
end

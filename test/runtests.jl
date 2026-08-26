using StatesDependency
using Test
using Random
using Statistics
using Distributions
using InverseFunctions: inverse

const HAS_BIJECTORS = Base.find_package("Bijectors") !== nothing
HAS_BIJECTORS && @eval using Bijectors

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

    @testset "window_shuffle" begin
        y = rand(Xoshiro(5), 1:4, 200)
        z = window_shuffle(y, 20, Xoshiro(6))
        @test length(z) == length(y)
        @test sort(z) == sort(y)                       # same basket overall

        # local composition is preserved up to the window: counts over a span
        # much longer than the window barely move
        cnt(v, lo, hi) = [count(==(b), v[lo:hi]) for b in 1:4]
        @test sum(abs, cnt(z, 1, 100) - cnt(y, 1, 100)) <= 40

        # a window as long as the series is just a global shuffle
        @test sort(window_shuffle(y, length(y), Xoshiro(7))) == sort(y)
        @test_throws ArgumentError window_shuffle(y, 1, Xoshiro(8))
    end

    @testset "single household mode" begin
        one = simulate_panel(H = 1, B = 4, T = 400, gamma = 1.0, K = 1, seed = 61)
        r = DHRTests(one.X[1]; mode = :single, nperm = 199, verbose = false)

        @test r isa DHRSingleResult
        @test r.n_occasions == 400
        @test r.n_used == 399
        @test r.converged
        @test r.gamma > 0
        @test r.ci_profile[1] < r.gamma < r.ci_profile[2]
        @test r.ci_profile[1] > 0
        @test r.lr > 0 && 0 <= r.lr_pvalue <= 1
        @test r.p_window < 0.05
        @test r.verdict === :state_dependence
        @test length(r.null_window) <= r.nperm
        # profile and Wald agree in a long history
        @test abs(r.ci_profile[1] - r.ci_wald[1]) < 0.1

        s = summarize(r)
        @test s.gamma ≈ r.gamma
        @test s.verdict === :state_dependence

        sd = sampling_distribution(r)
        @test sd isa Normal && sd.μ ≈ r.gamma && sd.σ ≈ r.se
        nw = null_distribution(r, :window)
        @test nw isa PosteriorSample
        @test length(draws(nw)) == length(r.null_window)
        @test null_distribution(r, :global) isa PosteriorSample
        @test_throws ArgumentError null_distribution(r, :bogus)

        io = IOBuffer(); show(io, MIME"text/plain"(), r)
        out = String(take!(io))
        @test occursin("window shuffle", out)
        @test occursin("profile", out)
        io = IOBuffer(); show(io, r)
        @test occursin("DHRSingleResult", String(take!(io)))

        # a zero-order history must not be called state dependent
        null = simulate_panel(H = 1, B = 4, T = 400, gamma = 0.0, K = 1, seed = 62)
        rn = DHRTests(null.X[1]; mode = :single, nperm = 199, verbose = false)
        @test rn.verdict !== :state_dependence
        @test rn.ci_profile[1] < 0 < rn.ci_profile[2]

        # drifting tastes without state dependence: the window null must not
        # call it state dependence even though the raw repeat rate is high
        dr = simulate_panel(H = 1, B = 4, T = 400, gamma = 0.0, K = 1,
                            drift_sd = 0.15, seed = 63)
        rd = DHRTests(dr.X[1]; mode = :single, nperm = 199, window = 20,
                      verbose = false)
        @test rd.verdict in (:nonstationarity, :no_evidence)
        # the mechanism: a window shuffle keeps the drift-induced clustering, so
        # its null sits above the global-shuffle null, and the observed estimate
        # is judged against the higher bar
        @test mean(rd.null_window) > mean(rd.null_global)

        # input handling
        @test DHRTests(one.X; mode = :single, nperm = 99, verbose = false) isa
              DHRSingleResult
        @test_throws ArgumentError DHRTests(simulate_panel(H = 3, B = 3, T = 50).X;
                                            mode = :single, verbose = false)
        @test_throws ArgumentError DHRTests(one.X[1]; mode = :bogus, verbose = false)
        @test_throws ArgumentError DHRTests(one.X[1]; mode = :single, window = 5_000,
                                            verbose = false)
        @test_throws ArgumentError DHRTests(one.X[1]; mode = :single, nperm = 5,
                                            verbose = false)
        onebrand = reshape(repeat([1.0, 0.0, 0.0], 40), 3, 40)
        @test_throws ArgumentError DHRTests(onebrand; mode = :single, verbose = false)
    end

    @testset "posterior as a Distribution" begin
        sim = simulate_panel(H = 150, B = 4, T = 12, gamma = 0.8, seed = 41)
        res = DHRTests(sim.X; K = 1, R = 3_000, burnin = 1_000, thin = 2,
                       nchains = 2, verbose = false)

        d = posterior(res, :gamma)
        @test d isa PosteriorSample
        @test length(draws(d)) == length(gamma_draws(res))
        @test mean(d) ≈ res.gamma_mean
        @test std(d) ≈ res.gamma_sd
        @test median(d) ≈ res.gamma_median
        @test quantile(d, 0.025) ≈ res.gamma_ci[1]
        @test quantile(d, 0.975) ≈ res.gamma_ci[2]
        @test effective_size(d) ≈ res.gamma_ess

        # sampling
        rng = Xoshiro(4)
        x = rand(rng, d, 5_000)
        @test length(x) == 5_000
        @test all(v -> minimum(d) <= v <= maximum(d), x)
        @test abs(mean(x) - mean(d)) < 4 * std(d)          # resampling is unbiased
        @test rand(Xoshiro(9), d) == rand(Xoshiro(9), d)   # reproducible

        # empirical cdf
        @test cdf(d, minimum(d) - 1) == 0
        @test cdf(d, maximum(d)) == 1
        @test 0.4 < cdf(d, median(d)) < 0.6

        # kernel density: positive, finite, and integrates to one
        @test isfinite(logpdf(d, mean(d)))
        @test pdf(d, mean(d)) > 0
        @test logpdf(d, mean(d) + 50 * std(d)) < logpdf(d, mean(d))
        @test isfinite(logpdf(d, mean(d) + 50 * std(d)))   # no underflow to -Inf
        lo, hi = mean(d) - 8 * std(d), mean(d) + 8 * std(d)
        grid = range(lo, hi; length = 2001)
        mass = sum(pdf.(Ref(d), grid)) * step(grid)
        @test 0.98 < mass < 1.02

        # parametric approximation
        n = fit(Normal, d)
        @test n isa Normal
        @test n.μ ≈ mean(d)
        @test isfinite(logpdf(n, mean(d)))
        @test abs(quantile(n, 0.975) - quantile(d, 0.975)) < 0.5 * std(d)
        @test fit(LogNormal, posterior(res, :odds_ratio)) isa LogNormal

        # the other quantities
        @test mean(posterior(res, :excess)) ≈ res.excess
        @test quantile(posterior(res, :excess), 0.025) ≈ res.excess_ci[1]
        @test mean(posterior(res, :placebo)) ≈ res.placebo.mean
        @test mean(posterior(res, :odds_ratio)) ≈ mean(exp.(gamma_draws(res)))
        @test mean(posterior(res, :lift)) ≈ res.delta_pp
        @test length(lift_share(res)) == 4
        @test sum(lift_share(res)) ≈ 1

        @test_throws ArgumentError posterior(res, :nonsense)
        @test_throws ArgumentError posterior(res, :beta, 1)   # no covariates fitted

        io = IOBuffer(); show(io, d)
        @test occursin("PosteriorSample(:gamma", String(take!(io)))

        # without a placebo fit there is no excess posterior
        res2 = DHRTests(sim.X; K = 1, R = 2_000, burnin = 800, thin = 2, nchains = 1,
                        placebo = false, compare_null = false, verbose = false)
        @test_throws ArgumentError posterior(res2, :excess)
        @test_throws ArgumentError posterior(res2, :placebo)
    end

    @testset "LiftBijector" begin
        b = LiftBijector(0.25)
        @test b(0.0) == 0.0                                # no state dependence, no lift
        @test b(1.0) > 0 && b(-1.0) < 0
        @test b(2.0) > b(1.0)                              # monotone increasing
        @test b(10_000.0) ≈ 0.75 atol = 1e-6               # saturates at 1 - share

        ib = inverse(b)
        @test ib isa InverseLiftBijector
        for g in (-2.0, -0.3, 0.0, 0.63, 1.7)
            @test ib(b(g)) ≈ g atol = 1e-10
        end
        @test inverse(ib) isa LiftBijector

        # analytic log|Jacobian| against a central difference
        for g in (-1.0, 0.0, 0.63, 1.5)
            h = 1e-6
            num = log(abs((b(g + h) - b(g - h)) / (2h)))
            @test lift_logabsdetjac(b, g) ≈ num atol = 1e-6
            @test lift_logabsdetjac(ib, b(g)) ≈ -lift_logabsdetjac(b, g) atol = 1e-10
        end

        @test_throws ArgumentError LiftBijector(0.0)
        @test_throws ArgumentError LiftBijector(1.0)
        @test_throws ArgumentError LiftBijector(-0.2)
        @test_throws DomainError inverse(LiftBijector(0.25))(0.9)
    end

    if HAS_BIJECTORS
        @testset "Bijectors extension" begin
            @test Base.get_extension(StatesDependency,
                                     :StatesDependencyBijectorsExt) !== nothing

            sim = simulate_panel(H = 120, B = 4, T = 10, gamma = 0.8, seed = 42)
            res = DHRTests(sim.X; K = 1, R = 2_000, burnin = 800, thin = 2,
                           nchains = 1, compare_null = false, verbose = false)
            d = posterior(res, :gamma)
            n = fit(Normal, d)
            b = LiftBijector(0.25)

            @test Bijectors.bijector(d) === identity

            # the empirical posterior composes directly
            t1 = Bijectors.transformed(d, exp)
            @test isfinite(rand(Xoshiro(1), t1))
            @test isfinite(logpdf(t1, exp(mean(d))))

            # the parametric fit plus our bijector gives an analytic logpdf
            t2 = Bijectors.transformed(n, b)
            @test isfinite(logpdf(t2, b(mean(d))))
            s = rand(Xoshiro(2), t2, 20_000)
            @test abs(mean(s) - b(mean(d))) < 5 * std(s) / sqrt(length(s)) + 1e-3
            @test all(v -> -0.25 < v < 0.75, s)

            _, lad = Bijectors.with_logabsdet_jacobian(b, 0.63)
            @test lad ≈ lift_logabsdetjac(b, 0.63)
            ib = Bijectors.inverse(b)
            @test ib isa InverseLiftBijector
            _, ladi = Bijectors.with_logabsdet_jacobian(ib, b(0.63))
            @test ladi + lad ≈ 0 atol = 1e-10

            t3 = lift_distribution(res, 0.25)
            @test isfinite(logpdf(t3, b(mean(d))))
            t4 = lift_distribution(res, 0.25; family = nothing)
            @test isfinite(rand(Xoshiro(3), t4))
        end
    else
        @info "Bijectors.jl not in this environment - extension tests skipped"
        @testset "no-Bijectors fallback" begin
            sim = simulate_panel(H = 40, B = 3, T = 8, seed = 43)
            res = DHRTests(sim.X; K = 1, R = 1_000, burnin = 400, thin = 2,
                           nchains = 1, placebo = false, compare_null = false,
                           verbose = false)
            @test_throws ArgumentError lift_distribution(res, 0.25)
        end
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

# Carrying the posterior downstream instead of a point estimate.
#
#     julia --project=. examples/posterior_distributions.jl
#
# The last section needs Bijectors.jl; it is skipped if you do not have it.

using StatesDependency, Distributions, Random, Statistics, Printf

sim = simulate_panel(H = 300, B = 4, T = 12, gamma = 0.8, seed = 17)
res = DHRTests(sim.X; verbose = false)

println("=== posterior objects ===")
for s in (:gamma, :placebo, :excess, :odds_ratio, :lift)
    println("  ", posterior(res, s))
end

d = posterior(res, :gamma)
@printf("\nP(gamma > 0)        = %.4f   (= 1 - cdf(d, 0))\n", 1 - cdf(d, 0.0))
@printf("95%% interval        = [%.3f, %.3f]\n", quantile(d, 0.025), quantile(d, 0.975))
@printf("effective sample    = %.0f out of %d draws\n", effective_size(d), length(draws(d)))

println("\n=== propagating the uncertainty ===")
# What is the posterior of the lift for a brand holding a 30% share?
rng = Xoshiro(1)
g  = rand(rng, d, 100_000)
b  = LiftBijector(0.30)
li = b.(g)
@printf("lift for a 30%% brand: mean %+.4f, 95%% [%+.4f, %+.4f]\n",
        mean(li), quantile(li, 0.025), quantile(li, 0.975))

println("\n=== parametric approximation (what to use inside a sampler) ===")
n = fit(Normal, d)
println("  fit(Normal, gamma)     = ", n)
println("  fit(LogNormal, exp(g)) = ", fit(LogNormal, posterior(res, :odds_ratio)))
@printf("  normal vs empirical quantiles: 2.5%% %+.4f, 97.5%% %+.4f\n",
        quantile(n, 0.025) - quantile(d, 0.025),
        quantile(n, 0.975) - quantile(d, 0.975))
println("  -> use `n` as a prior in another model; it has a cheap logpdf")

if Base.find_package("Bijectors") !== nothing
    @eval using Bijectors
    println("\n=== with Bijectors.jl ===")
    t = transformed(n, LiftBijector(0.30))
    println("  transformed(Normal, LiftBijector(0.30)) = ", typeof(t))
    @printf("  rand   = %+.4f\n", rand(Xoshiro(2), t))
    @printf("  logpdf = %+.4f\n", logpdf(t, mean(li)))
    @printf("  mean of 100k draws = %+.4f  (vs %.4f above)\n",
            mean(rand(Xoshiro(3), t, 100_000)), mean(li))
    println("  lift_distribution(res, 0.30) does the same in one call")
else
    println("\n(Bijectors.jl not installed - skipping the transformed-distribution part)")
end

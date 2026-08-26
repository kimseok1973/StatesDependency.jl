"""
    StatesDependency

Statistical tests for state dependence in consecutive brand purchases, following
Dube, Hitsch & Rossi (2010).

The entry point is [`DHRTests`](@ref). It takes brand (row) x purchase-occasion
(column) quantity matrices, fits a hierarchical Bayes multinomial logit with a
flexible (finite normal mixture) heterogeneity distribution and a common lagged
choice coefficient, and reports that coefficient with a credible interval
together with an order-shuffled placebo that says how much of it heterogeneity
alone can produce.

```julia
using StatesDependency

sim = simulate_panel(H = 400, B = 4, T = 12, gamma = 0.8, seed = 1)
res = DHRTests(sim.X)
```
"""
module StatesDependency

using LinearAlgebra
using Random
using Statistics
using Printf
using Distributions
using InverseFunctions

export DHRTests, DHRTestResult, DHRSingleResult
export PurchasePanel, build_panel, shuffle_panel, lagged_repeat_rate, choice_matrix
export n_households, n_occasions, n_used, n_covariates
export simulate_panel, dummy_data
export fit_hbmnl, HBMNLFit
export split_rhat, ess
export summarize, gamma_draws
export window_shuffle, sampling_distribution, null_distribution
export posterior, PosteriorSample, effective_size, draws, lift_share
export LiftBijector, InverseLiftBijector, lift_distribution, lift_logabsdetjac

include("panel.jl")
include("diagnostics.jl")
include("transforms.jl")
include("hbmnl.jl")
include("simulate.jl")
include("dhrtests.jl")
include("posterior.jl")
include("single.jl")

end # module

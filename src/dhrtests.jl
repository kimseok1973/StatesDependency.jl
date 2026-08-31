# ---------------------------------------------------------------------------
# dhrtests.jl -- the user-facing test
# ---------------------------------------------------------------------------

"""
    DHRTestResult

Result of [`DHRTests`](@ref). Printing it gives the full report; the fields are
there for programmatic use.

Key fields
- `gamma_mean`, `gamma_ci`, `p_positive` : posterior summary of the common
  state-dependence coefficient.
- `odds_ratio`, `odds_ratio_ci` : `exp(gamma)`, the multiplicative effect on the
  choice odds of the brand bought last time.
- `delta_pp`, `delta_pp_ci` : share-weighted change in choice probability
  (percentage points) induced by having bought the brand last time.
- `placebo` : the same summary from the order-shuffled panel (`nothing` when
  `placebo = false`).
- `excess`, `excess_ci` : posterior of `gamma - gamma_placebo`. This is the
  quantity to read: it is the part of the lagged-choice effect that cannot be
  reproduced by heterogeneity alone.
- `dic_sd`, `dic_nosd`, `delta_dic` : model comparison against the nested model
  without state dependence.
- `verdict` : `:state_dependence`, `:inconclusive` or `:no_evidence`.
"""
struct DHRTestResult
    n_brands::Int
    n_households::Int
    n_occasions::Int
    n_used::Int
    repeat_rate::Float64
    level::Float64

    gamma_mean::Float64
    gamma_median::Float64
    gamma_sd::Float64
    gamma_ci::Tuple{Float64,Float64}
    p_positive::Float64
    gamma_ess::Float64
    gamma_rhat::Float64

    odds_ratio::Float64
    odds_ratio_ci::Tuple{Float64,Float64}
    delta_pp::Float64
    delta_pp_ci::Tuple{Float64,Float64}

    placebo::Union{Nothing,NamedTuple}
    excess::Union{Nothing,Float64}
    excess_ci::Union{Nothing,Tuple{Float64,Float64}}
    p_excess_positive::Union{Nothing,Float64}

    dic_sd::Float64
    dic_nosd::Union{Nothing,Float64}
    delta_dic::Union{Nothing,Float64}
    lml_sd::Float64
    lml_nosd::Union{Nothing,Float64}

    beta_mean::Vector{Float64}
    beta_ci::Vector{Tuple{Float64,Float64}}

    excess_draws::Union{Nothing,Vector{Float64}}
    lift_draws::Vector{Float64}
    shares::Vector{Float64}

    verdict::Symbol
    accept_alpha::Float64
    accept_gamma::Float64
    fits::NamedTuple
    settings::NamedTuple
end

"""
    DHRTests(X; kwargs...)

Test for state dependence in consecutive purchases, following
Dube, Hitsch & Rossi (2010, *Quantitative Marketing and Economics* 8:417-445),
"State dependence and alternative explanations for consumer inertia".

# Input

`X` is a brand (row) x purchase-occasion (column) quantity matrix per household:

* a `Vector` of `B x T_h` matrices (ragged `T_h` allowed),
* a `B x T x H` array, or
* an already built [`PurchasePanel`](@ref).

Columns must be in chronological order -- the whole test is about what the
*previous* column contained. An all-zero column is a no-purchase occasion and is
dropped; a column with several positive rows is resolved by `tie_rule`.

You can also hand it the choices directly, one element per occasion, and the
indicator matrices are built for you:

* brand **labels** -- `["A", "B", "A"]` becomes `[1 0 1; 0 1 0]`. Strings,
  symbols and a `CategoricalVector` all work; rows come out in sorted label
  order (level order for a categorical).
* brand **codes** -- `[2, 2, 1, 1, 2]` becomes `[0 0 1 1 0; 1 1 0 0 1]`.
* a `Vector` of such vectors, one per household, with the levels pooled across
  households so everybody shares the row order.

`missing` inside a sequence is an occasion with no purchase. `brand_names` fixes
the level set and its order; `choice_matrix(build_panel(X))` shows how a
sequence was read.

# What is estimated

A hierarchical Bayes multinomial logit,

    v_hjt = alpha_hj + gamma * 1{ j == y_{h,t-1} } + x_hjt' beta,
    alpha_h ~ sum_k pi_k N(mu_k, Sigma_k),

with the first occasion of every household treated as a given initial
condition. `gamma` is common across households on purpose: with the handful of
occasions a typical panel gives per household there is no information to
identify household-specific state dependence, and forcing it degrades hold-out
fit.

# Why the placebo matters

A flexible heterogeneity distribution can soak up a lot of apparent inertia, but
not all of it, and a positive `gamma` on its own is not proof: the estimator has
a small-sample bias in that direction. `DHRTests` therefore re-fits the *same*
model to a panel whose occasions have been randomly re-ordered within each
household. That panel has identical cross-sectional composition and zero true
state dependence, so `gamma - gamma_placebo` (`excess`) is the honest test
statistic. The verdict requires that interval to exclude zero.

# Main keyword arguments

| keyword | default | meaning |
|---|---|---|
| `K` | `2` | mixture components in the heterogeneity distribution |
| `R`, `burnin`, `thin` | `12000, 4000, 5` | MCMC sweeps, burn-in, thinning |
| `nchains` | `2` | independent chains (gives split-Rhat) |
| `level` | `0.95` | credible-interval level |
| `placebo` | `true` | fit the order-shuffled placebo |
| `compare_null` | `true` | also fit the nested model without `gamma` (gives DIC) |
| `covariates` | `nothing` | `Vector` of `B x L x T_h` arrays (price, display, ...) |
| `brand_names` | `nothing` | row labels |
| `tie_rule` | `:argmax` | how to read a column with several positive rows |
| `seed` | `20260826` | RNG seed |
| `verbose` | `true` | progress output |

# One long history instead of a panel: `mode = :single`

`DHRTests(X; mode = :single)` takes a **single** `B x T` matrix and runs a
completely different estimator: a fixed-effect conditional logit, a
likelihood-ratio test with a profile interval, and a permutation null that
shuffles occasions **inside short windows**. It returns a
[`DHRSingleResult`](@ref).

With one household there is no cross-sectional heterogeneity left to confound
the lagged-choice effect. What replaces it is the household's own drift over
time, which a global order shuffle cannot separate from inertia -- hence the
window null. `window` (occasions per window, default `clamp(T/8, 8, 25)`) and
`nperm` (default 499) control it.

The window null *assumes* tastes are constant inside a window. `:single` mode
also checks that assumption, with a stationarity pre-test on brand shares across
`n_blocks` stretches of the sequence, at several block scales at once
(default `[3, 6, 12, 24]`, trimmed to those giving blocks of 15+ occasions) -- see
[`stationarity_test`](@ref). If it rejects, the verdict is `:nonstationarity`
whatever the window null says, because a moving alpha is exactly the case the
window null cannot handle. This is the two-stage procedure of Bass, Givon,
Kalwani, Reibstein & Wright (1984): stationarity first, order second.

Rough guide to how long the history has to be (B = 4, 5% level): about 200
occasions for 80% power at `gamma = 0.5`, about 100 for `gamma = 1.0`. Below 50
the estimate is badly biased downwards and the test has almost no power.

# Example

```julia
using StatesDependency

sim = simulate_panel(H = 400, B = 4, T = 12, gamma = 0.8, seed = 1)
res = DHRTests(sim.X)          # prints a full report
res.gamma_ci                   # (0.61, 0.99)
res.verdict                    # :state_dependence

# one household with a long history
one = simulate_panel(H = 1, B = 4, T = 400, gamma = 0.8, seed = 2)
DHRTests(one.X[1]; mode = :single)
```
"""
function DHRTests(X;
                  K::Int = 2,
                  R::Int = 12_000,
                  burnin::Int = 4_000,
                  thin::Int = 5,
                  nchains::Int = 2,
                  level::Real = 0.95,
                  placebo::Bool = true,
                  compare_null::Bool = true,
                  covariates = nothing,
                  brand_names = nothing,
                  tie_rule::Symbol = :argmax,
                  min_occasions::Int = 2,
                  drop_unused::Bool = false,
                  seed::Integer = 20260826,
                  verbose::Bool = true,
                  mode::Symbol = :panel,
                  window::Union{Nothing,Int} = nothing,
                  nperm::Int = 499,
                  n_blocks = nothing,   # nothing | Integer | iterable of Integer
                  kwargs...)

    0 < level < 1 || throw(ArgumentError("level must be in (0,1)"))
    mode in (:panel, :single) ||
        throw(ArgumentError("mode must be :panel or :single, got :$mode"))

    # In :single mode a bare B x T matrix is exactly what is expected, so the
    # "treated as ONE household" warning would be noise.
    Xin = (mode === :single && X isa AbstractMatrix) ? [X] : X

    panel = Xin isa PurchasePanel ? Xin :
            build_panel(Xin; brand_names, covariates, tie_rule, min_occasions,
                        drop_unused)

    if mode === :single
        covariates === nothing || throw(ArgumentError(
            "covariates are not supported in :single mode yet"))
        if verbose
            @printf("DHRTests (single household): %d brands, %d occasions (%d used)\n",
                    panel.B, n_occasions(panel), n_used(panel))
        end
        return _dhr_single(panel; level, window, nperm, n_blocks, seed, verbose)
    end

    if verbose
        @printf("DHRTests: %d households, %d brands, %d occasions (%d used)\n",
                n_households(panel), panel.B, n_occasions(panel), n_used(panel))
        panel.dropped_households > 0 &&
            @printf("  dropped %d household(s) with fewer than %d occasions\n",
                    panel.dropped_households, min_occasions)
        panel.dropped_zero_columns > 0 &&
            @printf("  dropped %d all-zero column(s)\n", panel.dropped_zero_columns)
        panel.ambiguous_columns > 0 &&
            @printf("  %d column(s) had several positive rows (tie_rule=:%s)\n",
                    panel.ambiguous_columns, tie_rule)
        println("  fitting model with state dependence ...")
    end

    fit_sd = fit_hbmnl(panel; K, R, burnin, thin, nchains, sd = true, seed,
                       verbose, label = "sd", kwargs...)

    fit_null = nothing
    if compare_null
        verbose && println("  fitting nested model without state dependence ...")
        fit_null = fit_hbmnl(panel; K, R, burnin, thin, nchains, sd = false,
                             seed = seed + 7, verbose, label = "null", kwargs...)
    end

    fit_pl = nothing
    if placebo
        verbose && println("  fitting order-shuffled placebo ...")
        sp = shuffle_panel(panel, Xoshiro(UInt64(seed) + 0xBEEF))
        fit_pl = fit_hbmnl(sp; K, R, burnin, thin, nchains, sd = true,
                           seed = seed + 13, verbose, label = "placebo", kwargs...)
    end

    g = vec(fit_sd.gamma)
    gci = _ci(g, level)

    # share-weighted change in choice probability, per draw
    shares = _mean_shares(panel, fit_sd.alpha_mean)
    dpp = [100 * _delta_share(shares, gd) for gd in g]

    pl = nothing
    exc = excci = pexc = nothing
    exc_draws = nothing
    if fit_pl !== nothing
        gp  = vec(fit_pl.gamma)
        pl  = (mean = mean(gp), median = median(gp), sd = std(gp),
               ci = _ci(gp, level), p_positive = mean(>(0), gp),
               ess = ess(fit_pl.gamma[1, :, :]), rhat = split_rhat(fit_pl.gamma[1, :, :]))
        # gamma and gamma_placebo come from independent posteriors, so pairing
        # draws at random is a draw from the posterior of their difference.
        rng = Xoshiro(UInt64(seed) + 0xF00D)
        n   = max(length(g), length(gp))
        d   = [g[rand(rng, 1:length(g))] - gp[rand(rng, 1:length(gp))] for _ in 1:n]
        exc_draws = d
        exc = mean(d); excci = _ci(d, level); pexc = mean(>(0), d)
    end

    ddic = fit_null === nothing ? nothing : fit_sd.dic - fit_null.dic

    L = n_covariates(panel)
    bci = Tuple{Float64,Float64}[]
    if L > 0
        for l in 1:L
            push!(bci, _ci(vec(fit_sd.beta[l, :, :]), level))
        end
    end

    ci_excludes_zero = gci[1] > 0 || gci[2] < 0
    verdict = if !ci_excludes_zero
        :no_evidence
    elseif fit_pl !== nothing && !(excci[1] > 0 || excci[2] < 0)
        :no_evidence
    elseif fit_null !== nothing && ddic >= 0
        :inconclusive
    else
        :state_dependence
    end

    return DHRTestResult(
        panel.B, n_households(panel), n_occasions(panel), n_used(panel),
        lagged_repeat_rate(panel), float(level),
        mean(g), median(g), std(g), gci, mean(>(0), g),
        ess(fit_sd.gamma[1, :, :]), split_rhat(fit_sd.gamma[1, :, :]),
        exp(mean(g)), (exp(gci[1]), exp(gci[2])),
        mean(dpp), _ci(dpp, level),
        pl, exc, excci, pexc,
        fit_sd.dic,
        fit_null === nothing ? nothing : fit_null.dic, ddic,
        fit_sd.lml_nr,
        fit_null === nothing ? nothing : fit_null.lml_nr,
        fit_sd.beta_mean, bci,
        exc_draws, dpp, shares,
        verdict, fit_sd.accept_alpha, fit_sd.accept_gamma,
        (; sd = fit_sd, null = fit_null, placebo = fit_pl),
        (; K, R, burnin, thin, nchains, level, seed, placebo, compare_null))
end

# average brand shares implied by the posterior-mean household intercepts
function _mean_shares(panel::PurchasePanel, alpha::Matrix{Float64})
    B = panel.B
    s = zeros(B)
    v = Vector{Float64}(undef, B)
    H = size(alpha, 2)
    for h in 1:H
        for j in 1:B
            v[j] = j < B ? alpha[j, h] : 0.0
        end
        m = maximum(v); tot = 0.0
        for j in 1:B
            v[j] = exp(v[j] - m); tot += v[j]
        end
        s .+= v ./ tot
    end
    return s ./ H
end

# share-weighted lift in choice probability from having bought the brand last time
function _delta_share(shares::Vector{Float64}, g::Float64)
    d = 0.0
    for s in shares
        (0 < s < 1) || continue
        d += s * (_logistic(g + _logit(s)) - s)
    end
    return d
end

_stars(lo, hi) = (lo > 0 || hi < 0) ? "*" : " "

function Base.show(io::IO, ::MIME"text/plain", r::DHRTestResult)
    pct = round(Int, 100 * r.level)
    println(io, "State dependence test  (Dube, Hitsch & Rossi 2010)")
    println(io, "=" ^ 68)
    @printf(io, "panel        : %d households x %d brands, %d occasions (%d used)\n",
            r.n_households, r.n_brands, r.n_occasions, r.n_used)
    @printf(io, "raw repeat   : %.3f of consecutive occasions repeat the brand\n",
            r.repeat_rate)
    @printf(io, "model        : HB multinomial logit, %d-component normal mixture,\n",
            r.settings.K)
    @printf(io, "               %d chains x %d sweeps (burn-in %d, thin %d)\n",
            r.settings.nchains, r.settings.R, r.settings.burnin, r.settings.thin)
    println(io, "-" ^ 68)
    @printf(io, "gamma (state dependence)  %8.3f   %d%% CI [%7.3f, %7.3f] %s\n",
            r.gamma_mean, pct, r.gamma_ci[1], r.gamma_ci[2],
            _stars(r.gamma_ci...))
    @printf(io, "  P(gamma > 0)            %8.3f   ESS %.0f, Rhat %.3f\n",
            r.p_positive, r.gamma_ess, r.gamma_rhat)
    @printf(io, "  odds ratio exp(gamma)   %8.3f   %d%% CI [%7.3f, %7.3f]\n",
            r.odds_ratio, pct, r.odds_ratio_ci[1], r.odds_ratio_ci[2])
    @printf(io, "  choice prob. lift       %+7.2f pp  %d%% CI [%+6.2f, %+6.2f]\n",
            r.delta_pp, pct, r.delta_pp_ci[1], r.delta_pp_ci[2])
    if r.placebo !== nothing
        println(io, "-" ^ 68)
        @printf(io, "placebo (order shuffled)  %8.3f   %d%% CI [%7.3f, %7.3f]\n",
                r.placebo.mean, pct, r.placebo.ci[1], r.placebo.ci[2])
        @printf(io, "EXCESS  gamma - placebo   %8.3f   %d%% CI [%7.3f, %7.3f] %s\n",
                r.excess, pct, r.excess_ci[1], r.excess_ci[2], _stars(r.excess_ci...))
        @printf(io, "  P(excess > 0)           %8.3f\n", r.p_excess_positive)
    end
    if r.dic_nosd !== nothing
        println(io, "-" ^ 68)
        @printf(io, "DIC   with SD %10.1f   without SD %10.1f   delta %+8.1f\n",
                r.dic_sd, r.dic_nosd, r.delta_dic)
        @printf(io, "log ML (Newton-Raftery, read with care)  %.1f vs %.1f\n",
                r.lml_sd, r.lml_nosd)
    end
    if !isempty(r.beta_mean)
        println(io, "-" ^ 68)
        for l in eachindex(r.beta_mean)
            @printf(io, "beta[%d]                   %8.3f   %d%% CI [%7.3f, %7.3f] %s\n",
                    l, r.beta_mean[l], pct, r.beta_ci[l][1], r.beta_ci[l][2],
                    _stars(r.beta_ci[l]...))
        end
    end
    println(io, "=" ^ 68)
    msg = r.verdict === :state_dependence ?
            "state dependence: the lagged-choice effect survives the placebo" :
          r.verdict === :no_evidence ?
            "no evidence: heterogeneity alone reproduces the observed inertia" :
            "inconclusive: gamma is positive but the model comparison does not agree"
    @printf(io, "verdict: %s\n         %s\n", r.verdict, msg)
    if !isnan(r.gamma_rhat) && r.gamma_rhat > 1.01
        println(io, "WARNING: Rhat > 1.01 -- increase R or nchains before believing this.")
    end
    @printf(io, "acceptance: alpha %.2f, gamma %.2f\n", r.accept_alpha, r.accept_gamma)
end

Base.show(io::IO, r::DHRTestResult) =
    print(io, "DHRTestResult(gamma=", round(r.gamma_mean; digits = 3),
          ", CI=", round.(r.gamma_ci; digits = 3), ", verdict=:", r.verdict, ")")

"""
    summarize(r::DHRTestResult)

Flat `NamedTuple` of the headline numbers, convenient for collecting many runs
into a table.
"""
function summarize(r::DHRTestResult)
    return (; n_households = r.n_households, n_brands = r.n_brands,
              n_used = r.n_used, repeat_rate = r.repeat_rate,
              gamma = r.gamma_mean, gamma_lo = r.gamma_ci[1], gamma_hi = r.gamma_ci[2],
              p_positive = r.p_positive, odds_ratio = r.odds_ratio,
              delta_pp = r.delta_pp,
              placebo = r.placebo === nothing ? NaN : r.placebo.mean,
              excess = r.excess === nothing ? NaN : r.excess,
              excess_lo = r.excess_ci === nothing ? NaN : r.excess_ci[1],
              excess_hi = r.excess_ci === nothing ? NaN : r.excess_ci[2],
              delta_dic = r.delta_dic === nothing ? NaN : r.delta_dic,
              rhat = r.gamma_rhat, ess = r.gamma_ess, verdict = r.verdict)
end

"""
    gamma_draws(r::DHRTestResult)

Pooled posterior draws of the state-dependence coefficient.
"""
gamma_draws(r::DHRTestResult) = vec(r.fits.sd.gamma)

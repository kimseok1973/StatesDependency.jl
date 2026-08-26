# ---------------------------------------------------------------------------
# posterior.jl -- expose posterior draws as Distributions objects
# ---------------------------------------------------------------------------

"""
    PosteriorSample <: ContinuousUnivariateDistribution

The posterior of one scalar quantity, represented by the MCMC draws themselves.
No parametric assumption is made: `rand` resamples the draws, `quantile` and
`cdf` read the empirical distribution function, `mean`/`std`/`var` are the
sample moments.

Get one with [`posterior`](@ref):

```julia
d = posterior(res, :gamma)
rand(d, 10_000)          # propagate the uncertainty downstream
quantile(d, 0.975)
cdf(d, 0.0)              # posterior mass below zero
```

`pdf` and `logpdf` are available through a Gaussian kernel density estimate.
They cost `O(ndraws)` per evaluation and smooth the tails, so they are meant for
plotting and light use. **Inside a sampler, fit a parametric approximation
instead** -- `fit` forwards to `Distributions.fit`, so any univariate family
works:

```julia
using Distributions
n = fit(Normal, posterior(res, :gamma))      # Normal(0.740, 0.042)
ln = fit(LogNormal, posterior(res, :odds_ratio))
```

`effective_size(d)` reports the effective sample size behind the draws. MCMC
draws are autocorrelated, so resampling `n` values from a chain with an ESS of
`m < n` does not give `n` independent samples -- do not quote it as such.
"""
struct PosteriorSample <: ContinuousUnivariateDistribution
    draws::Vector{Float64}
    sorted::Vector{Float64}
    name::Symbol
    ess::Float64
end

function PosteriorSample(draws::AbstractVector{<:Real}, name::Symbol = :theta;
                         ess::Real = NaN)
    d = collect(Float64.(draws))
    isempty(d) && throw(ArgumentError("no draws for $name"))
    return PosteriorSample(d, sort(d), name, float(ess))
end

draws(d::PosteriorSample) = d.draws

"""
    effective_size(d::PosteriorSample)

Effective sample size behind the draws (`NaN` when unknown).
"""
effective_size(d::PosteriorSample) = d.ess

Base.length(d::PosteriorSample) = 1
Distributions.minimum(d::PosteriorSample) = first(d.sorted)
Distributions.maximum(d::PosteriorSample) = last(d.sorted)
Distributions.insupport(d::PosteriorSample, x::Real) = isfinite(x)

Base.rand(rng::AbstractRNG, d::PosteriorSample) = d.draws[rand(rng, 1:length(d.draws))]

Statistics.mean(d::PosteriorSample)   = mean(d.draws)
Statistics.var(d::PosteriorSample)    = var(d.draws)
Statistics.std(d::PosteriorSample)    = std(d.draws)
Statistics.median(d::PosteriorSample) = median(d.draws)

Distributions.quantile(d::PosteriorSample, q::Real) = quantile(d.sorted, q; sorted = true)

function Distributions.cdf(d::PosteriorSample, x::Real)
    return searchsortedlast(d.sorted, x) / length(d.sorted)
end

# Gaussian KDE, Silverman's rule on the effective sample size
function _bandwidth(d::PosteriorSample)
    n = isfinite(d.ess) && d.ess > 1 ? d.ess : length(d.draws)
    s = std(d.draws)
    q = quantile(d.sorted, 0.75; sorted = true) - quantile(d.sorted, 0.25; sorted = true)
    a = q > 0 ? min(s, q / 1.349) : s
    return 1.06 * a * n^(-0.2)
end

# Evaluated in log space so the far tails give a finite (very negative) value
# rather than underflowing to -Inf.
function Distributions.logpdf(d::PosteriorSample, x::Real)
    h = _bandwidth(d)
    h > 0 || return x == first(d.draws) ? Inf : -Inf
    n = length(d.draws)
    m = -Inf
    @inbounds for v in d.draws
        e = -0.5 * ((x - v) / h)^2
        e > m && (m = e)
    end
    s = 0.0
    @inbounds for v in d.draws
        s += exp(-0.5 * ((x - v) / h)^2 - m)
    end
    return m + log(s) - log(n * h * sqrt(2pi))
end

Distributions.pdf(d::PosteriorSample, x::Real) = exp(logpdf(d, x))

"""
    fit(D, d::PosteriorSample)

Fit a parametric family to the draws, forwarding to `Distributions.fit`. Use
this when you need a cheap analytic `logpdf` -- for example to carry the
posterior of `gamma` into another model as a prior.

```julia
fit(Normal, posterior(res, :gamma))
```
"""
Distributions.fit(D::Type{<:UnivariateDistribution}, d::PosteriorSample) = fit(D, d.draws)

function Base.show(io::IO, d::PosteriorSample)
    lo, hi = quantile(d, 0.025), quantile(d, 0.975)
    print(io, "PosteriorSample(:", d.name, ", n=", length(d.draws),
          ", mean=", round(mean(d); digits = 4),
          ", sd=", round(std(d); digits = 4),
          ", 95%=[", round(lo; digits = 4), ", ", round(hi; digits = 4), "]",
          isfinite(d.ess) ? ", ess=" * string(round(Int, d.ess)) : "", ")")
end

Base.show(io::IO, ::MIME"text/plain", d::PosteriorSample) = show(io, d)

"""
    posterior(res::DHRTestResult, what::Symbol, index::Int = 1)

Return the posterior of one quantity as a [`PosteriorSample`](@ref).

| `what` | quantity |
|---|---|
| `:gamma` | the state-dependence coefficient |
| `:placebo` | the same coefficient on the order-shuffled panel |
| `:excess` | `gamma - gamma_placebo`, the statistic the verdict is based on |
| `:odds_ratio` | `exp(gamma)` |
| `:lift` | share-weighted change in choice probability, percentage points |
| `:beta` | covariate coefficient number `index` |

```julia
d = posterior(res, :gamma)
rand(d, 10_000)
using Distributions; fit(Normal, d)
```
"""
function posterior(res::DHRTestResult, what::Symbol, index::Int = 1)
    if what === :gamma
        return PosteriorSample(vec(res.fits.sd.gamma), :gamma; ess = res.gamma_ess)
    elseif what === :placebo
        res.fits.placebo === nothing &&
            throw(ArgumentError("this result has no placebo fit; rerun DHRTests with placebo = true"))
        return PosteriorSample(vec(res.fits.placebo.gamma), :placebo; ess = res.placebo.ess)
    elseif what === :excess
        res.excess_draws === nothing &&
            throw(ArgumentError("this result has no placebo fit; rerun DHRTests with placebo = true"))
        e = min(res.gamma_ess, res.placebo.ess)
        return PosteriorSample(res.excess_draws, :excess; ess = e)
    elseif what === :odds_ratio
        return PosteriorSample(exp.(vec(res.fits.sd.gamma)), :odds_ratio; ess = res.gamma_ess)
    elseif what === :lift
        return PosteriorSample(res.lift_draws, :lift; ess = res.gamma_ess)
    elseif what === :beta
        L = size(res.fits.sd.beta, 1)
        1 <= index <= L ||
            throw(ArgumentError("beta index $index out of range (L = $L)"))
        return PosteriorSample(vec(res.fits.sd.beta[index, :, :]), :beta)
    else
        throw(ArgumentError("unknown quantity :$what; use :gamma, :placebo, :excess, " *
                            ":odds_ratio, :lift or :beta"))
    end
end

"""
    lift_share(res::DHRTestResult)

The brand shares implied by the posterior-mean household intercepts, in the
order of the input rows. These are the weights behind `posterior(res, :lift)`
and the shares a `LiftBijector` (available when Bijectors.jl is loaded) needs.
"""
lift_share(res::DHRTestResult) = copy(res.shares)

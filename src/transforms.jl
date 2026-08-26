# ---------------------------------------------------------------------------
# transforms.jl -- the package's derived quantities as invertible transforms
# ---------------------------------------------------------------------------

"""
    LiftBijector(share)

The monotone map from the state-dependence coefficient to the **change in the
choice probability of a brand with base share `share`**:

    g  ->  s * exp(g) / (1 - s + s * exp(g))  -  s

It is a plain callable, invertible through `InverseFunctions.inverse`, so it
composes with anything in that ecosystem. Load Bijectors.jl and it becomes a
bijector proper -- `transformed(d, LiftBijector(s))` then gives a `Distribution`
with a `logpdf`, which is what you need to carry the lift into another model:

```julia
using Bijectors, Distributions
b = LiftBijector(0.25)                       # a brand holding a 25% share
d = transformed(fit(Normal, posterior(res, :gamma)), b)
rand(d, 1000)                                 # lift in probability units
logpdf(d, 0.1)
```

For the share-weighted aggregate across all brands there is no closed-form
inverse, so use the exact draws instead: `posterior(res, :lift)` (in percentage
points). Shares come from [`lift_share`](@ref).
"""
struct LiftBijector
    share::Float64
    function LiftBijector(s::Real)
        (0 < s < 1) || throw(ArgumentError("share must be in (0,1), got $s"))
        return new(float(s))
    end
end

"""
    InverseLiftBijector(share)

Inverse of [`LiftBijector`](@ref); build it with `inverse(LiftBijector(s))`.
"""
struct InverseLiftBijector
    share::Float64
    function InverseLiftBijector(s::Real)
        (0 < s < 1) || throw(ArgumentError("share must be in (0,1), got $s"))
        return new(float(s))
    end
end

# The map is a logistic shift: p = logistic(g + logit(share)). Written that way
# it stays finite for |g| large, which the algebraic form s*e/(1-s+s*e) does not
# (it gives Inf/Inf = NaN once exp(g) overflows).
@inline _logistic(x::Real) = x >= 0 ? inv(1 + exp(-x)) : (e = exp(x); e / (1 + e))
@inline _logit(p::Real) = log(p) - log1p(-p)

lift_probability(b::LiftBijector, g::Real) = _logistic(g + _logit(b.share))

(b::LiftBijector)(g::Real) = lift_probability(b, g) - b.share

function (b::InverseLiftBijector)(l::Real)
    s = b.share
    p = l + s
    (0 < p < 1) || throw(DomainError(l, "lift $l puts the probability outside (0,1) " *
                                        "for a base share of $s"))
    return _logit(p) - _logit(s)
end

InverseFunctions.inverse(b::LiftBijector) = InverseLiftBijector(b.share)
InverseFunctions.inverse(b::InverseLiftBijector) = LiftBijector(b.share)

"""
    lift_logabsdetjac(b::LiftBijector, g)

`log |d lift / d gamma|` at `g`. Used by the Bijectors extension; exposed
because it is also the sensitivity of the lift to the coefficient.
"""
function lift_logabsdetjac(b::LiftBijector, g::Real)
    p = lift_probability(b, g)
    return log(p) + log1p(-p)          # d lift / d gamma = p (1 - p)
end

function lift_logabsdetjac(b::InverseLiftBijector, l::Real)
    g = b(l)
    return -lift_logabsdetjac(LiftBijector(b.share), g)
end

"""
    lift_distribution(res, share; family = Normal)

The posterior of the choice-probability lift for a brand of base share `share`,
as a transformed `Distribution` with a `logpdf`.

**Requires Bijectors.jl**: run `using Bijectors` first. Without it, use the
exact draws through `posterior(res, :lift)` instead.
"""
function lift_distribution(::Any, args...; kwargs...)
    throw(ArgumentError(
        "lift_distribution needs Bijectors.jl -- run `using Bijectors` first, " *
        "or use posterior(res, :lift) for the exact draws."))
end

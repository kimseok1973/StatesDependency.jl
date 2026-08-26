module StatesDependencyBijectorsExt

using StatesDependency
using StatesDependency: PosteriorSample, LiftBijector, InverseLiftBijector,
                        lift_logabsdetjac, lift_distribution
using Bijectors
using Distributions

# A PosteriorSample carries draws of an already-unconstrained quantity, so the
# bijector to unconstrained space is the identity. Declaring it lets a
# PosteriorSample be used wherever Bijectors asks a distribution for its
# transform (e.g. as a prior in a Turing model).
Bijectors.bijector(::PosteriorSample) = identity

# --- make LiftBijector a first-class bijector ------------------------------

Bijectors.transform(b::LiftBijector, g::Real) = b(g)
Bijectors.transform(b::InverseLiftBijector, l::Real) = b(l)

# `Bijectors.inverse` is `InverseFunctions.inverse`, already defined in the core
# package for these two types -- redefining it here would be an overwrite.

Bijectors.logabsdetjac(b::LiftBijector, g::Real) = lift_logabsdetjac(b, g)
Bijectors.logabsdetjac(b::InverseLiftBijector, l::Real) = lift_logabsdetjac(b, l)

Bijectors.with_logabsdet_jacobian(b::LiftBijector, g::Real) =
    (b(g), lift_logabsdetjac(b, g))
Bijectors.with_logabsdet_jacobian(b::InverseLiftBijector, l::Real) =
    (b(l), lift_logabsdetjac(b, l))

Bijectors.is_monotonically_increasing(::LiftBijector) = true
Bijectors.is_monotonically_increasing(::InverseLiftBijector) = true

# --- convenience -----------------------------------------------------------

"""
    lift_distribution(res, share; family = Normal)

Posterior of the choice-probability lift for a brand whose base share is
`share`, as a transformed distribution carrying a `logpdf`.

`family` is the parametric approximation fitted to the `gamma` draws before the
transform; `Normal` is a good fit in practice (the posterior of `gamma` is very
close to Gaussian). Pass `family = nothing` to transform the empirical draws
instead, in which case `logpdf` goes through the kernel density estimate of
[`PosteriorSample`](@ref) and is correspondingly slow.
"""
function StatesDependency.lift_distribution(res::DHRTestResult, share::Real;
                                            family = Normal)
    d = posterior(res, :gamma)
    base = family === nothing ? d : Distributions.fit(family, d)
    return transformed(base, LiftBijector(share))
end

end # module

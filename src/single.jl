# ---------------------------------------------------------------------------
# single.jl -- state dependence inside ONE long purchase history
#
# With a single household there is no cross-sectional heterogeneity left to
# confound the lagged-choice effect: alpha is a constant, so the model is an
# ordinary fixed-effect conditional logit and the question reduces to "is this
# sequence first-order Markov, or i.i.d. multinomial?".
#
# What takes the place of the panel's unobserved heterogeneity is *within*
# household non-stationarity -- taste drift, assortment and price changes,
# seasonality. A household that goes through phases produces runs that look
# exactly like inertia, and a GLOBAL order shuffle cannot tell them apart: it
# destroys drift and state dependence together. The null that separates them is
# a shuffle inside short windows, over which drift is approximately constant.
#
# Measured on simulated single households (B = 4, T = 300, 150 replications,
# 5% level):
#
#   scenario                     LR test   global shuffle   window shuffle (W=20)
#   no SD, no drift                5.3%          5.3%              6.0%
#   no SD, strong drift           68.0%         73.3%              7.3%
#   SD gamma=0.8, no drift        99.3%         99.3%             99.3%
#
# The window null keeps the size right where the other two produce two thirds
# false positives, and gives up nothing in power.
# ---------------------------------------------------------------------------

"""
    DHRSingleResult

Result of `DHRTests(X; mode = :single)`. Printing it gives the full report.

Key fields
- `gamma`, `se` : the fixed-effect conditional-logit estimate and its standard
  error.
- `ci_wald`, `ci_profile` : Wald and profile-likelihood intervals. Prefer the
  profile interval; the Wald one is symmetric by construction and too optimistic
  when `n_used` is small.
- `lr`, `lr_pvalue` : likelihood-ratio test of `gamma = 0` against chi-square(1).
- `p_window`, `window` : permutation p-value from shuffling occasions **inside
  windows of `window` occasions**. This is the test to read.
- `p_global` : permutation p-value from shuffling the whole sequence. Compare
  the two: `p_global` small with `p_window` large means the sequence has serial
  structure at the drift scale but not at the adjacent-occasion scale.
- `null_window`, `null_global` : the permutation distributions of `gamma`.
- `verdict` : `:state_dependence`, `:nonstationarity`, `:inconclusive` or
  `:no_evidence`.
"""
struct DHRSingleResult
    n_brands::Int
    n_brands_used::Int
    n_occasions::Int
    n_used::Int
    repeat_rate::Float64
    level::Float64

    gamma::Float64
    se::Float64
    ci_wald::Tuple{Float64,Float64}
    ci_profile::Tuple{Float64,Float64}

    lr::Float64
    lr_pvalue::Float64

    window::Int
    p_window::Float64
    p_global::Float64
    null_window::Vector{Float64}
    null_global::Vector{Float64}
    nperm::Int

    alpha::Vector{Float64}
    used_brands::Vector{Int}
    brand_names::Vector{String}

    verdict::Symbol
    converged::Bool
end

# --- the fixed-effect conditional logit ------------------------------------

# Design for one sequence. Parameters are (intercepts for all but the last used
# brand, gamma); the lag dummy is the last column.
function _single_design(y::Vector{Int})
    used  = sort(unique(y))
    Bu    = length(used)
    remap = Dict(b => i for (i, b) in enumerate(used))
    yy = [remap[v] for v in y]
    T  = length(yy)
    P  = Bu                              # (Bu - 1) intercepts + gamma
    X  = zeros(Bu, P, T)
    for t in 2:T
        lag = yy[t-1]
        for j in 1:Bu
            j < Bu && (X[j, j, t] = 1.0)
            X[j, P, t] = (j == lag)
        end
    end
    return X, yy, used, Bu, P, T
end

function _single_llgh(X, yy, Bu, P, T, th, want::Bool)
    ll = 0.0
    g  = zeros(P)
    H  = zeros(P, P)
    v  = zeros(Bu); p = zeros(Bu); xb = zeros(P)
    @inbounds for t in 2:T
        for j in 1:Bu
            s = 0.0
            for k in 1:P
                s += X[j, k, t] * th[k]
            end
            v[j] = s
        end
        m = maximum(v); tot = 0.0
        for j in 1:Bu; p[j] = exp(v[j] - m); tot += p[j]; end
        p ./= tot
        ll += v[yy[t]] - (m + log(tot))
        want || continue
        fill!(xb, 0.0)
        for j in 1:Bu, k in 1:P; xb[k] += p[j] * X[j, k, t]; end
        for k in 1:P; g[k] += X[yy[t], k, t] - xb[k]; end
        for j in 1:Bu, a in 1:P, b in 1:P; H[a, b] -= p[j] * X[j, a, t] * X[j, b, t]; end
        for a in 1:P, b in 1:P; H[a, b] += xb[a] * xb[b]; end
    end
    return ll, g, H
end

# Newton on the coordinates in `free`, holding the rest at their current value.
function _single_newton(X, yy, Bu, P, T; free = 1:P, th0 = nothing, maxit = 200)
    th = th0 === nothing ? zeros(P) : copy(th0)
    ll = -Inf; g = zeros(P); H = zeros(P, P)
    conv = false
    for _ in 1:maxit
        ll, g, H = _single_llgh(X, yy, Bu, P, T, th, true)
        gg = g[free]; HH = H[free, free]
        if maximum(abs, gg) < 1e-9
            conv = true
            break
        end
        st = try
            (-HH - 1e-9I) \ gg
        catch
            break                       # singular: separation, give up cleanly
        end
        all(isfinite, st) || break
        th[free] .+= st
        maximum(abs, th) > 40 && break  # separation
    end
    return th, ll, H, conv
end

# --- window / global order nulls -------------------------------------------

"""
    window_shuffle(y, window, rng)

Shuffle the occasions of a single sequence **inside consecutive windows** of
`window` occasions, with a random offset so the window boundaries move between
draws. Slow drift in tastes survives (it is approximately constant inside a
window); dependence between adjacent occasions does not. That makes it the null
for "no state dependence, preferences locally constant".
"""
function window_shuffle(y::Vector{Int}, window::Int, rng::AbstractRNG)
    window >= 2 || throw(ArgumentError("window must be >= 2, got $window"))
    n = length(y)
    z = copy(y)
    off = window <= 1 ? 0 : rand(rng, 0:(window-1))
    s = 1 - off
    while s <= n
        lo = max(s, 1)
        hi = min(s + window - 1, n)
        hi > lo && (z[lo:hi] = y[lo:hi][randperm(rng, hi - lo + 1)])
        s += window
    end
    return z
end

_default_window(T::Int) = clamp(T ÷ 8, 8, 25)

# --- the test ---------------------------------------------------------------

function _single_fit(y::Vector{Int})
    X, yy, used, Bu, P, T = _single_design(y)
    th, ll, H, conv = _single_newton(X, yy, Bu, P, T)
    return (; X, yy, used, Bu, P, T, th, ll, H, conv)
end

# gamma only, for the permutation loop
function _single_gamma(y::Vector{Int})
    length(unique(y)) < 2 && return NaN
    f = _single_fit(y)
    return abs(f.th[f.P]) > 30 ? NaN : f.th[f.P]
end

function _profile_ci(X, yy, Bu, P, T, th, llhat, level)
    crit = quantile(Chisq(1), level)
    ghat = th[P]
    prof = function (gv)
        t0 = copy(th); t0[P] = gv
        _, l0, _, _ = _single_newton(X, yy, Bu, P, T; free = 1:(P-1), th0 = t0)
        return 2 * (llhat - l0) - crit
    end
    se = 1.0
    lo = _bisect_side(prof, ghat, -1.0)
    hi = _bisect_side(prof, ghat, +1.0)
    return (lo, hi)
end

# walk outwards from ghat until the profile statistic crosses, then bisect
function _bisect_side(f, ghat::Float64, dir::Float64)
    step = 0.25
    a = ghat
    b = ghat + dir * step
    for _ in 1:40
        fb = f(b)
        (isfinite(fb) && fb > 0) && break
        a = b
        step *= 1.6
        b = ghat + dir * step
        abs(b - ghat) > 60 && return dir < 0 ? -Inf : Inf
    end
    for _ in 1:60
        m = (a + b) / 2
        fm = f(m)
        (isfinite(fm) && fm > 0) ? (b = m) : (a = m)
        abs(b - a) < 1e-6 && break
    end
    return (a + b) / 2
end

function _dhr_single(panel::PurchasePanel;
                     level::Real, window::Union{Nothing,Int}, nperm::Int,
                     seed::Integer, verbose::Bool)

    n_households(panel) == 1 || throw(ArgumentError(
        "mode = :single expects exactly one household, got $(n_households(panel)). " *
        "Pass one B x T matrix, or loop over households yourself."))
    nperm >= 19 || throw(ArgumentError("nperm must be at least 19, got $nperm"))

    y = panel.choices[1]
    length(unique(y)) >= 2 || throw(ArgumentError(
        "this household bought only one brand; there is nothing to test"))

    f = _single_fit(y)
    f.conv || @warn "the conditional logit did not fully converge; the " *
                         "sequence may separate (a brand always or never repeated)"

    ghat = f.th[f.P]
    V = try
        inv(-f.H)
    catch
        fill(NaN, f.P, f.P)
    end
    se = (size(V, 1) >= f.P && isfinite(V[f.P, f.P]) && V[f.P, f.P] >= 0) ?
         sqrt(V[f.P, f.P]) : NaN
    z = quantile(Normal(), 1 - (1 - level) / 2)
    ci_w = isfinite(se) ? (ghat - z * se, ghat + z * se) : (-Inf, Inf)

    # restricted fit for the LR test
    t0 = copy(f.th); t0[f.P] = 0.0
    _, ll0, _, _ = _single_newton(f.X, f.yy, f.Bu, f.P, f.T;
                                  free = 1:(f.P-1), th0 = t0)
    lr = 2 * (f.ll - ll0)
    lrp = ccdf(Chisq(1), max(lr, 0.0))

    ci_p = _profile_ci(f.X, f.yy, f.Bu, f.P, f.T, f.th, f.ll, level)

    W = window === nothing ? _default_window(f.T) : window
    W >= 2 || throw(ArgumentError("window must be >= 2, got $W"))
    W <= f.T || throw(ArgumentError("window ($W) exceeds the number of occasions ($(f.T))"))

    verbose && @printf("  %d permutations, window = %d occasions ...\n", nperm, W)
    rng = Xoshiro(UInt64(seed))
    nullw = Float64[]; nullg = Float64[]
    for _ in 1:nperm
        v = _single_gamma(window_shuffle(y, W, rng))
        isnan(v) || push!(nullw, v)
        v = _single_gamma(shuffle(rng, y))
        isnan(v) || push!(nullg, v)
    end

    pw = (count(>=(ghat), nullw) + 1) / (length(nullw) + 1)
    pg = (count(>=(ghat), nullg) + 1) / (length(nullg) + 1)

    a = 1 - level
    ci_excl = ci_p[1] > 0 || ci_p[2] < 0
    verdict = if pw < a && ci_excl
        :state_dependence
    elseif pw < a
        :inconclusive
    elseif pg < a
        :nonstationarity
    else
        :no_evidence
    end

    names = panel.brand_names
    return DHRSingleResult(panel.B, f.Bu, f.T, f.T - 1, lagged_repeat_rate(panel),
                           float(level), ghat, se, ci_w, ci_p, lr, lrp,
                           W, pw, pg, nullw, nullg, nperm,
                           f.th[1:(f.P-1)], collect(f.used), names,
                           verdict, f.conv)
end

function Base.show(io::IO, ::MIME"text/plain", r::DHRSingleResult)
    pct = round(Int, 100 * r.level)
    println(io, "State dependence in a single purchase history")
    println(io, "=" ^ 68)
    @printf(io, "history      : %d occasions (%d used), %d of %d brands bought\n",
            r.n_occasions, r.n_used, r.n_brands_used, r.n_brands)
    @printf(io, "raw repeat   : %.3f of consecutive occasions repeat the brand\n",
            r.repeat_rate)
    println(io, "model        : fixed-effect conditional logit (no heterogeneity to")
    println(io, "               confound: alpha is a constant for one household)")
    println(io, "-" ^ 68)
    @printf(io, "gamma                     %8.3f   se %.3f\n", r.gamma, r.se)
    @printf(io, "  Wald    %d%% CI          [%7.3f, %7.3f] %s\n",
            pct, r.ci_wald[1], r.ci_wald[2], _stars(r.ci_wald...))
    @printf(io, "  profile %d%% CI          [%7.3f, %7.3f] %s   <- prefer this one\n",
            pct, r.ci_profile[1], r.ci_profile[2], _stars(r.ci_profile...))
    @printf(io, "  odds ratio exp(gamma)   %8.3f   [%7.3f, %7.3f]\n",
            exp(r.gamma), exp(r.ci_profile[1]), exp(r.ci_profile[2]))
    @printf(io, "  LR vs gamma=0           %8.2f   p = %.4f  (chi2, 1 df)\n",
            r.lr, r.lr_pvalue)
    println(io, "-" ^ 68)
    @printf(io, "window shuffle (W = %2d)   p = %.4f   null mean %+.3f [%+.3f, %+.3f] %s\n",
            r.window, r.p_window, _nm(r.null_window),
            _nq(r.null_window, 0.025), _nq(r.null_window, 0.975),
            r.p_window < 1 - r.level ? "*" : " ")
    @printf(io, "global shuffle            p = %.4f   null mean %+.3f [%+.3f, %+.3f]\n",
            r.p_global, _nm(r.null_global),
            _nq(r.null_global, 0.025), _nq(r.null_global, 0.975))
    println(io, "=" ^ 68)
    msg = if r.verdict === :state_dependence
        "state dependence: the effect survives shuffling inside short windows,\n" *
        "         so it is not slow drift in this household's tastes"
    elseif r.verdict === :nonstationarity
        "drift, not state dependence: the sequence has serial structure, but it\n" *
        "         disappears once occasions are shuffled inside short windows --\n" *
        "         that is a household whose tastes move over time"
    elseif r.verdict === :inconclusive
        "inconclusive: the window null rejects but the profile interval covers zero"
    else
        "no evidence: neither null is rejected"
    end
    @printf(io, "verdict: %s\n         %s\n", r.verdict, msg)

    if r.n_used < 50
        println(io, "WARNING: fewer than 50 usable occasions. Power is very low and the")
        println(io, "         estimate is biased downwards (fixed effects with a lagged")
        println(io, "         dependent variable). Treat this as descriptive.")
    elseif r.n_used < 150
        println(io, "NOTE: with under ~150 usable occasions the test only detects strong")
        println(io, "      state dependence (about 50% power at gamma = 0.5).")
    end
    r.converged || println(io, "WARNING: the optimiser did not converge -- separation is likely.")
end

Base.show(io::IO, r::DHRSingleResult) =
    print(io, "DHRSingleResult(gamma=", round(r.gamma; digits = 3),
          ", profile CI=", round.(r.ci_profile; digits = 3),
          ", p_window=", round(r.p_window; digits = 4),
          ", verdict=:", r.verdict, ")")

_nm(v) = isempty(v) ? NaN : mean(v)
_nq(v, q) = isempty(v) ? NaN : quantile(v, q)

"""
    summarize(r::DHRSingleResult)

Flat `NamedTuple` of the headline numbers.
"""
function summarize(r::DHRSingleResult)
    return (; n_occasions = r.n_occasions, n_used = r.n_used,
              n_brands_used = r.n_brands_used, repeat_rate = r.repeat_rate,
              gamma = r.gamma, se = r.se,
              lo = r.ci_profile[1], hi = r.ci_profile[2],
              odds_ratio = exp(r.gamma), lr = r.lr, lr_pvalue = r.lr_pvalue,
              window = r.window, p_window = r.p_window, p_global = r.p_global,
              verdict = r.verdict)
end

"""
    sampling_distribution(r::DHRSingleResult)

`Normal(gamma, se)`, the asymptotic sampling distribution of the estimate. Use
it to propagate the uncertainty, or compose it with [`LiftBijector`](@ref):

```julia
using Bijectors
transformed(sampling_distribution(r), LiftBijector(0.3))
```

This is a sampling distribution, not a posterior -- it is symmetric by
construction, so check it against `r.ci_profile` before leaning on the tails.
"""
sampling_distribution(r::DHRSingleResult) = Normal(r.gamma, r.se)

"""
    null_distribution(r::DHRSingleResult, which = :window)

The permutation distribution of `gamma` under the window (`:window`) or global
(`:global`) order null, as a [`PosteriorSample`](@ref) so it can be sampled and
plotted like the other distributions in this package.
"""
function null_distribution(r::DHRSingleResult, which::Symbol = :window)
    which === :window && return PosteriorSample(r.null_window, :null_window;
                                                ess = length(r.null_window))
    which === :global && return PosteriorSample(r.null_global, :null_global;
                                                ess = length(r.null_global))
    throw(ArgumentError("which must be :window or :global, got :$which"))
end

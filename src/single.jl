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
#
# The window null ASSUMES tastes are constant inside a window; it does not check
# it. Bass, Givon, Kalwani, Reibstein & Wright (1984) test stationarity FIRST and
# only test the order of the process on the sequences that pass. We do the same:
# `p_stationarity` below is that pre-test, and a rejection gates the verdict.
#
# The statistic that separates a moving alpha from a large gamma is the marginal
# distribution. With alpha fixed, the chain is stationary whatever gamma is, so
# brand shares are the same in every stretch of the sequence. If alpha moves, the
# shares move. That is the only observable difference between the two.
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
- `p_stationarity` : the stationarity pre-test. Brand shares are compared across
  contiguous stretches of the sequence at several block scales (`n_blocks`),
  against a parametric bootstrap null in which alpha is constant and gamma
  equals its estimate. A small `p_stationarity` means the household's tastes
  moved, so the window null's assumption fails and the verdict is
  `:nonstationarity` regardless of `p_window`. `NaN` when the fit did not
  converge or the history is too short to block.
- `n_blocks`, `chisq_blocks`, `p_blocks` : the block scales scanned, the
  chi-square at each, and the per-scale p-values. The scale with the smallest
  `p_blocks` says how fast the tastes moved -- a coarse split for slow drift, a
  fine one for quick switching between phases.
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

    n_blocks::Vector{Int}
    chisq_blocks::Vector{Float64}
    p_blocks::Vector{Float64}
    p_stationarity::Float64
    null_stationarity::Vector{Float64}

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

# --- stationarity of alpha --------------------------------------------------
#
# Bass, Givon, Kalwani, Reibstein & Wright (1984), "An Investigation into the
# Order of the Brand Choice Process", Marketing Science 3(4): test stationarity
# first, test the order of the process only on what passes. A runs-type
# statistic cannot tell drifting tastes from inertia -- both shorten the runs --
# so the order of the two tests is what does the work, not either one alone.
#
# Statistic: Pearson chi-square for homogeneity of brand shares across M
# contiguous blocks of the sequence.
#
# Null distribution: NOT chi-square. State dependence inflates the variance of a
# block share (the occasions inside a block are not independent), so the
# asymptotic reference over-rejects whenever gamma > 0. Measured at T = 400,
# B = 4, 200 replications, alpha genuinely constant:
#
#   gamma      bootstrap null    asymptotic chi-square
#   0.0             6.0%                  6.0%
#   0.8             7.0%                 31.0%
#   1.5             5.0%                 63.5%
#
# So we simulate from the fitted model itself -- alpha constant, gamma at its
# estimate, first occasion held at the observed one -- which is exactly the null
# "tastes are constant, whatever the state dependence is".
#
# Several block scales at once. Power depends on matching the block length to the
# speed of the taste change: slow drift shows up in a coarse split, fast regime
# switching only in a fine one, and each is nearly blind to the other. The size
# of the bootstrap test holds at every M, so we scan a grid and calibrate the
# smallest p-value by the same bootstrap draws (a min-p / Westfall-Young step).

# Block counts to scan. Blocks shorter than ~15 occasions carry too little
# information, so the grid is trimmed to what T can support.
function _default_block_grid(T::Int)
    g = [m for m in (3, 6, 12, 24) if T ÷ m >= 15]
    isempty(g) && (g = T >= 24 ? [2] : Int[])
    return g
end

# M contiguous, near-equal ranges covering 1:T
function _blocks(T::Int, M::Int)
    M >= 2 || throw(ArgumentError("n_blocks must be >= 2, got $M"))
    M <= T || throw(ArgumentError("n_blocks ($M) exceeds the number of occasions ($T)"))
    edges = [round(Int, (m - 1) * T / M) + 1 for m in 1:M]
    push!(edges, T + 1)
    return [edges[m]:(edges[m+1]-1) for m in 1:M]
end

# Pearson chi-square for homogeneity of brand shares across blocks. Cells whose
# expected count is zero (a brand never bought in this draw) contribute nothing;
# the bootstrap null sees the same convention, so it stays calibrated.
function _block_chisq(y::Vector{Int}, Bu::Int, blocks)
    M = length(blocks)
    n = zeros(Int, M, Bu)
    @inbounds for (m, rg) in enumerate(blocks), t in rg
        n[m, y[t]] += 1
    end
    N = sum(n)
    N == 0 && return 0.0
    rows = vec(sum(n; dims = 2))
    cols = vec(sum(n; dims = 1))
    x2 = 0.0
    @inbounds for m in 1:M, b in 1:Bu
        e = rows[m] * cols[b] / N
        e > 0 && (x2 += (n[m, b] - e)^2 / e)
    end
    return x2
end

# One sequence from the fitted conditional logit. Brand Bu is the baseline, so
# `alpha` holds the first Bu-1 intercepts.
function _simulate_single(alpha::Vector{Float64}, g::Float64, Bu::Int, T::Int,
                          y1::Int, rng::AbstractRNG)
    z = Vector{Int}(undef, T)
    z[1] = y1
    v = zeros(Bu); p = zeros(Bu)
    @inbounds for t in 2:T
        lag = z[t-1]
        for j in 1:Bu
            v[j] = (j < Bu ? alpha[j] : 0.0) + (j == lag ? g : 0.0)
        end
        m = maximum(v); tot = 0.0
        for j in 1:Bu
            p[j] = exp(v[j] - m); tot += p[j]
        end
        u = rand(rng) * tot
        acc = 0.0; pick = Bu
        for j in 1:Bu
            acc += p[j]
            if u <= acc
                pick = j
                break
            end
        end
        z[t] = pick
    end
    return z
end

# Monte-Carlo p-value of x against a column of null statistics (larger = more
# extreme), counting the draw itself so it can never be zero.
_mc_p(nulls::AbstractVector{Float64}, x::Float64, n::Int) =
    (count(>=(x), nulls) + 1) / (n + 1)

function _stationarity(yy::Vector{Int}, Bu::Int, alpha::Vector{Float64},
                       g::Float64, grid::Vector{Int}, nboot::Int, rng::AbstractRNG)
    T = length(yy)
    K = length(grid)
    K >= 1 || return (Float64[], NaN, Float64[], Float64[])

    blocksets = [_blocks(T, M) for M in grid]
    obs = [_block_chisq(yy, Bu, bs) for bs in blocksets]

    # nullstat[b, k] : block chi-square of bootstrap draw b at grid point k
    nullstat = Matrix{Float64}(undef, nboot, K)
    @inbounds for b in 1:nboot
        z = _simulate_single(alpha, g, Bu, T, yy[1], rng)
        for k in 1:K
            nullstat[b, k] = _block_chisq(z, Bu, blocksets[k])
        end
    end

    p_each = [_mc_p(view(nullstat, :, k), obs[k], nboot) for k in 1:K]
    K == 1 && return p_each, p_each[1], obs, vec(nullstat[:, 1])

    # min-p across scales, calibrated against the same draws
    tobs = minimum(p_each)
    tnull = Vector{Float64}(undef, nboot)
    @inbounds for b in 1:nboot
        m = Inf
        for k in 1:K
            pk = _mc_p(view(nullstat, :, k), nullstat[b, k], nboot)
            pk < m && (m = pk)
        end
        tnull[b] = m
    end
    p = (count(<=(tobs), tnull) + 1) / (nboot + 1)
    return p_each, p, obs, tnull
end

_block_grid(T::Int, n_blocks) =
    n_blocks === nothing ? _default_block_grid(T) :
    n_blocks isa Integer ? [Int(n_blocks)] : sort(unique(Int.(collect(n_blocks))))

"""
    stationarity_test(y; n_blocks = nothing, nboot = 499, seed = 20260826)

Test whether one household's **preferences** stayed put over its purchase
sequence, separately from how much state dependence it has.

`y` is a single history in any of the forms [`build_panel`](@ref) accepts: a
`B x T` matrix, a vector of labels, or a vector of brand codes.

The statistic is a Pearson chi-square for homogeneity of brand shares across
contiguous stretches of the sequence. The reference distribution is a
**parametric bootstrap** from the fitted conditional logit with a *constant*
alpha and gamma at its estimate -- not the chi-square distribution, which
over-rejects badly whenever `gamma > 0` because occasions inside a block are
then correlated (measured at T = 400, B = 4: 31% at `gamma = 0.8` and 63% at
`gamma = 1.5`, against 5-7% for the bootstrap).

Why this separates the two things a long sequence confounds: with alpha fixed
the chain is stationary *whatever gamma is*, so brand shares do not move; if
alpha moves, they do. That is the only observable difference between the two.

`n_blocks` is the block scale. Power depends on matching it to the speed of the
change -- slow drift needs a coarse split, quick switching between phases a fine
one -- so by default several scales are scanned (`[3, 6, 12, 24]`, trimmed to
those giving blocks of at least 15 occasions) and the smallest p-value is
calibrated against the same bootstrap draws. Pass an integer or a vector to fix
the grid yourself.

Returns a `NamedTuple` `(; pvalue, n_blocks, statistic, p_blocks, gamma, null)`,
where `statistic` and `p_blocks` are per-scale and `pvalue` is the combined one.
A small `pvalue` means tastes moved, and a window-shuffle test of state
dependence on this sequence is not trustworthy.

```julia
drifting = simulate_panel(H = 1, B = 4, T = 400, gamma = 0.0, drift_sd = 0.25, seed = 3)
stationarity_test(drifting.X[1]).pvalue      # small
```

Follows the two-stage procedure of Bass, Givon, Kalwani, Reibstein & Wright
(1984), *An Investigation into the Order of the Brand Choice Process*,
Marketing Science 3(4), 267-287: stationarity first, order of the process
second. A runs-type statistic on its own cannot tell drifting tastes from
inertia -- both shorten the runs -- so it is the order of the two tests that
does the work.
"""
function stationarity_test(y; n_blocks = nothing,
                           nboot::Int = 499, seed::Integer = 20260826)
    nboot >= 19 || throw(ArgumentError("nboot must be at least 19, got $nboot"))
    panel = y isa PurchasePanel ? y : build_panel(y isa AbstractMatrix ? [y] : y)
    n_households(panel) == 1 || throw(ArgumentError(
        "stationarity_test expects exactly one household, got $(n_households(panel))"))
    yy = panel.choices[1]
    length(unique(yy)) >= 2 || throw(ArgumentError(
        "this household bought only one brand; there is nothing to test"))

    f = _single_fit(yy)
    g = f.th[f.P]
    grid = _block_grid(f.T, n_blocks)
    blank = (; pvalue = NaN, n_blocks = grid, statistic = Float64[],
               p_blocks = Float64[], gamma = g, null = Float64[])

    if !(f.conv && isfinite(g) && abs(g) <= 30)
        @warn "the conditional logit did not converge; the bootstrap null would " *
              "be built from meaningless parameters"
        return blank
    end
    isempty(grid) && (@warn "history too short to split into blocks"; return blank)

    p_each, p, obs, nulls = _stationarity(f.yy, f.Bu, f.th[1:(f.P-1)], g, grid,
                                          nboot, Xoshiro(UInt64(seed)))
    return (; pvalue = p, n_blocks = grid, statistic = obs, p_blocks = p_each,
              gamma = g, null = nulls)
end

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
                     n_blocks::Union{Nothing,Int}, seed::Integer, verbose::Bool)

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

    # stationarity pre-test: is the window null's "tastes constant" assumption
    # even tenable for this household?
    grid = _block_grid(f.T, n_blocks)
    ok = f.conv && isfinite(ghat) && abs(ghat) <= 30 && !isempty(grid)
    if ok
        verbose && @printf("  stationarity: blocks %s, %d bootstrap draws ...\n",
                           string(grid), nperm)
        pbl, ps, x2, nulls = _stationarity(f.yy, f.Bu, f.th[1:(f.P-1)], ghat,
                                           grid, nperm, rng)
    else
        pbl, ps, x2, nulls = Float64[], NaN, Float64[], Float64[]
    end

    a = 1 - level
    ci_excl = ci_p[1] > 0 || ci_p[2] < 0
    # A moving alpha invalidates the window null, so it is decided first.
    verdict = if isfinite(ps) && ps < a
        :nonstationarity
    elseif pw < a && ci_excl
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
                           grid, x2, pbl, ps, nulls,
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
    println(io, "-" ^ 68)
    if isfinite(r.p_stationarity)
        @printf(io, "stationarity of alpha     p = %.4f %s\n", r.p_stationarity,
                r.p_stationarity < 1 - r.level ? "*" : " ")
        println(io, "  (brand shares across stretches of the sequence, vs a bootstrap")
        println(io, "   null holding alpha constant at gamma = gamma-hat)")
        for (i, M) in enumerate(r.n_blocks)
            @printf(io, "  %2d blocks of %3d          p = %.4f   chi2 %7.1f\n",
                    M, r.n_occasions ÷ M, r.p_blocks[i], r.chisq_blocks[i])
        end
        # only meaningful once the test has actually rejected
        if length(r.n_blocks) > 1 && r.p_stationarity < 1 - r.level
            k = argmin(r.p_blocks)
            ties = count(==(r.p_blocks[k]), r.p_blocks)
            if ties == 1
                @printf(io, "  smallest p at %d blocks (~%d occasions): tastes move on about that scale\n",
                        r.n_blocks[k], r.n_occasions ÷ r.n_blocks[k])
            else
                println(io, "  every scale rejects, so the drift is not confined to one time scale")
            end
        end
    else
        println(io, "stationarity of alpha     not computed (fit did not converge, or")
        println(io, "                          the history is too short to block)")
    end
    println(io, "=" ^ 68)
    msg = if r.verdict === :state_dependence
        "state dependence: the effect survives shuffling inside short windows,\n" *
        "         so it is not slow drift in this household's tastes"
    elseif r.verdict === :nonstationarity && isfinite(r.p_stationarity) &&
           r.p_stationarity < 1 - r.level
        "tastes moved: this household's brand shares are not the same across\n" *
        "         the sequence, so alpha is not a constant and the window null's\n" *
        "         assumption fails. gamma is not identified here -- read it as a\n" *
        "         description, not as evidence of state dependence"
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
              n_blocks = r.n_blocks, p_stationarity = r.p_stationarity,
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

`:stationarity` gives the bootstrap distribution of the block chi-square
statistic instead -- note that this one is a distribution of the test statistic,
not of `gamma`.
"""
function null_distribution(r::DHRSingleResult, which::Symbol = :window)
    which === :window && return PosteriorSample(r.null_window, :null_window;
                                                ess = length(r.null_window))
    which === :global && return PosteriorSample(r.null_global, :null_global;
                                                ess = length(r.null_global))
    if which === :stationarity
        isempty(r.null_stationarity) && throw(ArgumentError(
            "the stationarity test was not computed for this result"))
        return PosteriorSample(r.null_stationarity, :null_stationarity;
                               ess = length(r.null_stationarity))
    end
    throw(ArgumentError(
        "which must be :window, :global or :stationarity, got :$which"))
end

# ---------------------------------------------------------------------------
# brandwise.jl -- does state dependence differ BY BRAND?
#
#     v_hjt = alpha_hj + gamma_j * 1{ j == y_{h,t-1} } + x_hjt' beta
#     gamma_j ~ N(gamma-bar, tau^2)
#
# Two things make this harder than it looks.
#
# 1. THE BIAS IS BRAND-SPECIFIC. Fitting free gamma_j to a panel with true
#    (1.2, 0.8, 0.4, 0.0) and ignoring heterogeneity returns roughly
#    (1.96, 1.41, 1.35, 0.22): brand 3 picks up +0.95 while brand 2 picks up
#    +0.61, so two brands whose truth differs by a factor of two come out nearly
#    tied -- and with SEs near 0.075, that tie looks *precise*. Reading the raw
#    ordering of gamma_j is therefore not safe. Every brand needs its own
#    placebo, and EXCESS_j = gamma_j - gamma_j^placebo is what carries meaning.
#
# 2. B TESTS INSTEAD OF ONE. Two defences. The hierarchical prior shrinks the
#    brands toward a common value, so a brand only separates when its own data
#    insist (Gelman, Hill & Yajima 2012). On top of that the per-brand verdicts
#    are selected by Bayesian FDR rather than one-at-a-time thresholds.
#
# The information available for gamma_j is the number of occasions on which
# brand j was the PREVIOUS choice, which scales with its share. Empirically
# SE(gamma_j) ~ 2.6 / sqrt(n_lag), so ~300 lagged occasions are needed before a
# brand's own number is worth reading. Brands below that are reported as
# :underpowered rather than :no_evidence -- absence of evidence, not evidence of
# absence.
# ---------------------------------------------------------------------------

"""
    BrandwiseResult

Result of [`brandwise_test`](@ref). Printing it gives the full report.

Key fields
- `gamma`, `gamma_ci` : per-brand state-dependence coefficient.
- `placebo` : the same coefficient from the order-shuffled panel. The bias is
  brand-specific, so this is subtracted brand by brand.
- `excess`, `excess_ci`, `p_excess` : `gamma_j - placebo_j`. **This is what to
  read**, never the raw `gamma`.
- `tau`, `tau_ci` : the between-brand SD of the gammas.
- `tau_placebo`, `p_tau` : the same spread on the order-shuffled panel, and
  `P(tau > tau_placebo)`. This is the test of "do the brands differ at all",
  and it gates everything below it.
- `differ` : whether the brands differ at all.
- `lag_occasions` : occasions on which each brand was the previous choice, i.e.
  how much information that brand's `gamma_j` rests on.
- `verdicts` : per brand, one of `:state_dependence`, `:no_evidence`,
  `:underpowered`.
"""
struct BrandwiseResult
    n_brands::Int
    n_households::Int
    n_occasions::Int
    n_used::Int
    level::Float64
    fdr_q::Float64

    brand_names::Vector{String}
    shares::Vector{Float64}
    lag_occasions::Vector{Int}

    gamma::Vector{Float64}
    gamma_ci::Vector{Tuple{Float64,Float64}}
    placebo::Vector{Float64}
    excess::Vector{Float64}
    excess_ci::Vector{Tuple{Float64,Float64}}
    p_excess::Vector{Float64}
    excess_draws::Vector{Vector{Float64}}

    tau::Float64
    tau_ci::Tuple{Float64,Float64}
    tau_placebo::Float64
    p_tau::Float64
    dic_brandwise::Float64
    dic_common::Union{Nothing,Float64}
    delta_dic::Union{Nothing,Float64}
    differ::Bool

    ess::Vector{Float64}
    rhat::Vector{Float64}
    verdicts::Vector{Symbol}
    min_lag::Int
    fits::NamedTuple
end

# Bayesian FDR. local fdr_j = P(excess_j <= 0); sort ascending and keep the
# largest set whose running mean stays under q. Controls the expected share of
# false positives among the selected brands, which is the relevant guarantee
# when B verdicts are reported side by side.
function _bayes_fdr(p_positive::Vector{Float64}, q::Float64)
    lfdr = 1 .- p_positive
    ord  = sortperm(lfdr)
    keep = falses(length(lfdr))
    run  = 0.0
    for (i, j) in enumerate(ord)
        run += lfdr[j]
        run / i <= q || break
        keep[j] = true
    end
    return keep
end

"""
    brandwise_test(X; kwargs...)

Test whether state dependence differs **by brand**, and which brands have it.

Fits the same hierarchical Bayes multinomial logit as [`DHRTests`](@ref) but with
one coefficient per brand,

```
v_hjt = alpha_hj + gamma_j * 1{j == y_{h,t-1}} + x_hjt'beta,
gamma_j ~ N(gamma-bar, tau^2)
```

and re-fits it to an order-shuffled panel so each brand gets **its own placebo**.
That is not a refinement: the small-sample bias differs across brands strongly
enough to invert the ordering of the raw `gamma_j`, so `EXCESS_j` is the only
quantity worth reading.

Read the report top-down:

1. **`tau`, and DIC vs the common-gamma model.** If the brands do not differ,
   stop -- the per-brand numbers are then just noise around one value.
2. **`EXCESS_j` per brand**, with verdicts selected by Bayesian FDR at `fdr_q`.
3. **`lag_occasions`**, the information behind each brand. Below `min_lag`
   (default 300) a brand is `:underpowered`, not `:no_evidence`.

Keyword arguments are those of [`DHRTests`](@ref) (`K`, `R`, `burnin`, `thin`,
`nchains`, `level`, `covariates`, `brand_names`, `seed`, `verbose`, ...) plus:

| keyword | default | meaning |
|---|---|---|
| `fdr_q` | `0.05` | Bayesian FDR level for the per-brand verdicts |
| `min_lag` | `300` | below this many lagged occasions a brand is `:underpowered` |
| `compare_common` | `true` | also fit the common-gamma model, for the DIC comparison |
| `tau_scale` | `0.5` | scale of the half-normal prior on `tau` |

```julia
sim = simulate_panel(H = 800, B = 4, T = 20, gamma = 0.8, seed = 1)
r = brandwise_test(sim.X)
r.excess, r.verdicts
```

A caution on using this for promotion planning: `gamma_j` is on the log-odds
scale, so the same `gamma_j` buys a different lift in probability depending on
the brand's base share (largest near a 40% share). A small brand can have the
highest `gamma_j` and still be the worst place to spend.
"""
function brandwise_test(X;
                        K::Int = 2,
                        R::Int = 12_000,
                        burnin::Int = 4_000,
                        thin::Int = 5,
                        nchains::Int = 2,
                        level::Real = 0.95,
                        fdr_q::Real = 0.05,
                        min_lag::Int = 300,
                        compare_common::Bool = true,
                        tau_scale::Float64 = 0.5,
                        covariates = nothing,
                        brand_names = nothing,
                        tie_rule::Symbol = :argmax,
                        min_occasions::Int = 2,
                        drop_unused::Bool = false,
                        seed::Integer = 20260826,
                        verbose::Bool = true)

    0 < level < 1 || throw(ArgumentError("level must be in (0,1)"))
    0 < fdr_q < 1 || throw(ArgumentError("fdr_q must be in (0,1)"))

    panel = X isa PurchasePanel ? X :
            build_panel(X; brand_names, covariates, tie_rule, min_occasions,
                        drop_unused)
    B = panel.B
    B >= 2 || throw(ArgumentError("need at least 2 brands"))

    if verbose
        @printf("brandwise_test: %d households, %d brands, %d occasions (%d used)\n",
                n_households(panel), B, n_occasions(panel), n_used(panel))
    end

    lagidx, _ = _lag_partition(panel)
    lagn = [length(lagidx[b]) for b in 1:B]

    verbose && println("  fitting brandwise model ...")
    fit_sd = fit_hbmnl(panel; K, R, burnin, thin, nchains, sd = true, seed,
                       verbose = false, brandwise = true,
                       tau_scale, label = "brandwise")

    verbose && println("  fitting order-shuffled placebo ...")
    shuf = shuffle_panel(panel, Xoshiro(UInt64(seed) + 977))
    fit_pl = fit_hbmnl(shuf; K, R, burnin, thin, nchains, sd = true,
                       seed = seed + 5501, verbose = false, brandwise = true,
                       tau_scale, label = "placebo")

    fit_cm = nothing
    if compare_common
        verbose && println("  fitting common-gamma model (for the DIC comparison) ...")
        fit_cm = fit_hbmnl(panel; K, R, burnin, thin, nchains, sd = true,
                           seed = seed + 131, verbose = false, brandwise = false,
                           label = "common")
    end

    a  = 1 - level
    lo, hi = a / 2, 1 - a / 2

    gam   = Vector{Float64}(undef, B)
    gci   = Vector{Tuple{Float64,Float64}}(undef, B)
    plc   = Vector{Float64}(undef, B)
    exc   = Vector{Float64}(undef, B)
    eci   = Vector{Tuple{Float64,Float64}}(undef, B)
    pex   = Vector{Float64}(undef, B)
    edr   = Vector{Vector{Float64}}(undef, B)
    essv  = Vector{Float64}(undef, B)
    rhv   = Vector{Float64}(undef, B)

    for b in 1:B
        gs = fit_sd.gamma[b, :, :]
        ps = fit_pl.gamma[b, :, :]
        gv = vec(gs); pv = vec(ps)
        n  = min(length(gv), length(pv))
        d  = gv[1:n] .- pv[1:n]

        gam[b] = mean(gv)
        gci[b] = (quantile(gv, lo), quantile(gv, hi))
        plc[b] = mean(pv)
        exc[b] = mean(d)
        eci[b] = (quantile(d, lo), quantile(d, hi))
        pex[b] = mean(d .> 0)
        edr[b] = d
        essv[b] = ess(gs)
        rhv[b]  = split_rhat(gs)
    end

    tv  = vec(fit_sd.tau)
    tau = mean(tv)
    tci = (quantile(tv, lo), quantile(tv, hi))

    # Do the brands differ? NOT by DIC: on nested models with useless extra
    # parameters, delta-DIC < 0 happens about half the time, which measured out
    # at a 50% false-positive rate. Compare tau against the tau the SAME
    # estimator manufactures on the order-shuffled panel, where by construction
    # every brand has the same (zero) state dependence.
    tvp = vec(fit_pl.tau)
    ntt = min(length(tv), length(tvp))
    p_tau = mean(tv[1:ntt] .> tvp[1:ntt])
    differ = p_tau > 1 - a

    ddic = fit_cm === nothing ? nothing : fit_sd.dic - fit_cm.dic

    # Two hurdles, as in the scalar test: the credible interval must exclude
    # zero AND the brand must survive Bayesian FDR across the B brands.
    keep = _bayes_fdr(pex, float(fdr_q))
    verdicts = Vector{Symbol}(undef, B)
    for b in 1:B
        verdicts[b] = lagn[b] < min_lag                 ? :underpowered :
                      (keep[b] && eci[b][1] > 0)        ? :state_dependence :
                                                          :no_evidence
    end

    shares = zeros(Float64, B)
    tot = 0
    for y in panel.choices, t in eachindex(y)
        shares[y[t]] += 1; tot += 1
    end
    shares ./= tot

    return BrandwiseResult(B, n_households(panel), n_occasions(panel),
                           n_used(panel), float(level), float(fdr_q),
                           panel.brand_names, shares, lagn,
                           gam, gci, plc, exc, eci, pex, edr,
                           tau, tci, mean(tvp), p_tau, fit_sd.dic,
                           fit_cm === nothing ? nothing : fit_cm.dic, ddic, differ,
                           essv, rhv, verdicts, min_lag,
                           (; sd = fit_sd, placebo = fit_pl, common = fit_cm))
end

function Base.show(io::IO, ::MIME"text/plain", r::BrandwiseResult)
    pct = round(Int, 100 * r.level)
    println(io, "Brand-wise state dependence")
    println(io, "=" ^ 76)
    @printf(io, "panel        : %d households x %d brands, %d occasions (%d used)\n",
            r.n_households, r.n_brands, r.n_occasions, r.n_used)
    println(io, "model        : HB multinomial logit, gamma_j ~ N(gamma-bar, tau^2),")
    println(io, "               each brand against its OWN order-shuffled placebo")
    println(io, "-" ^ 76)

    # step 1 -- do the brands differ at all?
    @printf(io, "do brands differ?   tau %.3f  vs placebo %.3f   P(>) = %.3f  %s\n",
            r.tau, r.tau_placebo, r.p_tau,
            r.differ ? "-> yes" : "-> NO")
    @printf(io, "                    tau %d%% CI [%.3f, %.3f]\n",
            pct, r.tau_ci[1], r.tau_ci[2])
    if r.delta_dic !== nothing
        @printf(io, "                    (DIC brandwise %.1f  common %.1f  delta %+.1f -- shown\n",
                r.dic_brandwise, r.dic_common, r.delta_dic)
        println(io, "                     for reference only; DIC is not a test)")
    end
    println(io, "-" ^ 76)

    @printf(io, "%-14s %6s %8s %8s %8s %9s  %-16s %6s\n",
            "brand", "share", "lag occ", "gamma", "placebo", "EXCESS",
            "$(pct)% CI", "P(>0)")
    for b in 1:r.n_brands
        mark = r.verdicts[b] === :state_dependence ? "*" :
               r.verdicts[b] === :underpowered ? "?" : " "
        @printf(io, "%-14s %5.1f%% %8d %8.3f %8.3f %8.3f  [%6.2f,%6.2f] %6.3f %s\n",
                first(r.brand_names[b], 14), 100 * r.shares[b], r.lag_occasions[b],
                r.gamma[b], r.placebo[b], r.excess[b],
                r.excess_ci[b][1], r.excess_ci[b][2], r.p_excess[b], mark)
    end
    println(io, "-" ^ 76)

    sel = [r.brand_names[b] for b in 1:r.n_brands
           if r.verdicts[b] === :state_dependence]
    und = [r.brand_names[b] for b in 1:r.n_brands
           if r.verdicts[b] === :underpowered]
    @printf(io, "* state dependence at FDR %.0f%%: %s\n", 100 * r.fdr_q,
            isempty(sel) ? "(none)" : join(sel, ", "))
    if !isempty(und)
        @printf(io, "? too little data to say (< %d lagged occasions): %s\n",
                r.min_lag, join(und, ", "))
    end
    println(io, "=" ^ 76)

    if !r.differ && r.delta_dic !== nothing
        println(io, "READ THIS FIRST: the brands do not measurably differ. Use the common")
        println(io, "                 gamma from DHRTests and ignore the per-brand split.")
    end
    println(io, "EXCESS is gamma minus that brand's own placebo. Do not read the raw")
    println(io, "gamma column on its own -- its bias differs across brands and can")
    println(io, "invert the ordering.")
    if any(x -> x > 1.01, r.rhat)
        println(io, "WARNING: Rhat > 1.01 for at least one brand. Run more sweeps.")
    end
end

Base.show(io::IO, r::BrandwiseResult) =
    print(io, "BrandwiseResult(", r.n_brands, " brands, tau=",
          round(r.tau; digits = 3), ", differ=", r.differ, ", verdicts=",
          r.verdicts, ")")

"""
    summarize(r::BrandwiseResult)

One `NamedTuple` per brand, as a `Vector`.
"""
function summarize(r::BrandwiseResult)
    [(; brand = r.brand_names[b], share = r.shares[b],
        lag_occasions = r.lag_occasions[b],
        gamma = r.gamma[b], placebo = r.placebo[b], excess = r.excess[b],
        lo = r.excess_ci[b][1], hi = r.excess_ci[b][2],
        p_excess = r.p_excess[b], ess = r.ess[b], rhat = r.rhat[b],
        verdict = r.verdicts[b]) for b in 1:r.n_brands]
end

"""
    posterior(r::BrandwiseResult, brand)

Posterior draws of `EXCESS_j` for one brand, as a [`PosteriorSample`](@ref).
`brand` is an index or a brand name.
"""
function posterior(r::BrandwiseResult, brand)
    b = brand isa Integer ? Int(brand) : findfirst(==(String(brand)), r.brand_names)
    b === nothing && throw(ArgumentError("no brand named $brand"))
    1 <= b <= r.n_brands || throw(ArgumentError("brand index out of range"))
    return PosteriorSample(r.excess_draws[b], Symbol("excess_", r.brand_names[b]);
                           ess = r.ess[b])
end

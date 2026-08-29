# ---------------------------------------------------------------------------
# hbmnl.jl -- hierarchical Bayes multinomial logit with a finite normal-mixture
#             heterogeneity distribution and a COMMON state-dependence
#             coefficient, estimated by Gibbs / random-walk Metropolis.
#
# Utility of brand j for household h on occasion t (t >= 2):
#
#     v_hjt = alpha_hj + gamma * 1{ j == y_{h,t-1} } + x_hjt' * beta
#
# with alpha_hB == 0 (last brand is the reference) and
#
#     alpha_h ~ sum_k pi_k N(mu_k, Sigma_k).
#
# gamma is deliberately NOT household specific: with a handful of occasions per
# household there is no room to identify heterogeneous state dependence, and
# doing so degrades hold-out fit (Dube, Hitsch & Rossi 2010; and our own
# replication on the dunnhumby beer panel).
# ---------------------------------------------------------------------------

"""
    HBMNLFit

Posterior draws and fit statistics from [`fit_hbmnl`](@ref).
"""
struct HBMNLFit
    gamma::Matrix{Float64}        # ndraws x nchains (0 x nchains when sd = false)
    beta::Array{Float64,3}        # L x ndraws x nchains
    loglik::Matrix{Float64}       # ndraws x nchains
    alpha_mean::Matrix{Float64}   # p x H, pooled posterior mean
    gamma_mean::Float64
    beta_mean::Vector{Float64}
    dic::Float64
    p_D::Float64
    lml_nr::Float64               # Newton-Raftery (harmonic mean) -- read with care
    accept_alpha::Float64
    accept_gamma::Float64
    K::Int
    sd::Bool
    ndraws::Int
    nchains::Int
end

# --- small numerical helpers ----------------------------------------------

@inline function _logsumexp!(v::Vector{Float64})
    m = -Inf
    @inbounds for x in v
        x > m && (m = x)
    end
    isfinite(m) || return m
    s = 0.0
    @inbounds for x in v
        s += exp(x - m)
    end
    return m + log(s)
end

@inline function _logmvn(x::AbstractVector{Float64}, mu::Vector{Float64},
                         L::LowerTriangular{Float64,Matrix{Float64}},
                         logdet2::Float64, work::Vector{Float64})
    p = length(x)
    @inbounds for i in 1:p
        work[i] = x[i] - mu[i]
    end
    ldiv!(L, work)
    q = 0.0
    @inbounds for i in 1:p
        q += work[i]^2
    end
    return -0.5 * (p * log(2pi) + logdet2 + q)
end

# log-likelihood of one household given its parameters
function _ll_household(y::Vector{Int}, Xc, alpha::AbstractVector{Float64},
                       gamma::Float64, beta::Vector{Float64}, B::Int,
                       buf::Vector{Float64})
    ll = 0.0
    L = length(beta)
    @inbounds for t in 2:length(y)
        lastb = y[t-1]
        for j in 1:B
            v = j < B ? alpha[j] : 0.0
            j == lastb && (v += gamma)
            if L > 0
                for l in 1:L
                    v += Xc[j, l, t] * beta[l]
                end
            end
            buf[j] = v
        end
        ll += buf[y[t]] - _logsumexp!(buf)
    end
    return ll
end

function _ll_total(panel::PurchasePanel, alpha::Matrix{Float64}, gamma::Float64,
                   beta::Vector{Float64}, buf::Vector{Float64})
    s = 0.0
    for h in eachindex(panel.choices)
        Xc = panel.covariates === nothing ? nothing : panel.covariates[h]
        s += _ll_household(panel.choices[h], Xc, view(alpha, :, h), gamma, beta,
                           panel.B, buf)
    end
    return s
end

_sym(A) = Matrix(Symmetric((A .+ A') ./ 2))

# --- fast conditional for gamma -------------------------------------------
#
# Only the lagged brand's utility depends on gamma, so for occasion (h,t)
#
#     sum_j exp(v_j) = S_ht + A_ht * (exp(gamma) - 1)
#
# with S_ht = sum_j exp(alpha_hj + x_hjt'beta) and A_ht the term of the brand
# bought at t-1. Caching (S, A, repeat?) once per sweep turns each Metropolis
# step on gamma into one pass of scalar logs instead of a full multinomial
# recomputation, which buys us many gamma updates per sweep and a far better
# effective sample size for the one parameter the whole test is about.
function _fill_gamma_cache!(S::Vector{Float64}, A::Vector{Float64},
                            rep::Vector{Bool}, panel::PurchasePanel,
                            alpha::Matrix{Float64}, beta::Vector{Float64})
    B = panel.B
    L = length(beta)
    idx = 0
    @inbounds for h in eachindex(panel.choices)
        y  = panel.choices[h]
        Xc = panel.covariates === nothing ? nothing : panel.covariates[h]
        if L == 0
            s = 0.0
            for j in 1:B
                s += exp(j < B ? alpha[j, h] : 0.0)
            end
            for t in 2:length(y)
                idx += 1
                lag = y[t-1]
                S[idx]   = s
                A[idx]   = exp(lag < B ? alpha[lag, h] : 0.0)
                rep[idx] = y[t] == lag
            end
        else
            for t in 2:length(y)
                idx += 1
                lag = y[t-1]
                s = 0.0; a = 0.0
                for j in 1:B
                    v = j < B ? alpha[j, h] : 0.0
                    for l in 1:L
                        v += Xc[j, l, t] * beta[l]
                    end
                    e = exp(v)
                    s += e
                    j == lag && (a = e)
                end
                S[idx] = s; A[idx] = a; rep[idx] = y[t] == lag
            end
        end
    end
    return idx
end

function _gamma_logpost(S, A, n, nrep::Int, g::Float64, prior_sd::Float64)
    e = exp(g) - 1
    acc = 0.0
    @inbounds for i in 1:n
        acc += log(S[i] + A[i] * e)
    end
    return g * nrep - acc - 0.5 * (g / prior_sd)^2
end

"""
    fit_hbmnl(panel; K, R, burnin, thin, nchains, sd, seed, verbose, ...)

Fit the hierarchical multinomial logit above to a [`PurchasePanel`](@ref).

Keyword arguments
- `K` : number of normal mixture components in the heterogeneity distribution.
- `R`, `burnin`, `thin` : total sweeps, discarded sweeps, thinning.
- `nchains` : independent chains (used for the split-Rhat diagnostic).
- `sd` : include the state-dependence term. `false` fits the nested no-SD model.
- `gamma_prior_sd` : prior SD of the normal on `gamma` **and on every element
  of `beta`** -- one knob covers both, so widen it if a covariate is on a
  scale where 10 is tight.
- `dir_prior`, `kappa0`, `nu0_add` : hyperparameters. The
  defaults follow Dube, Hitsch & Rossi (2010): `Dir(0.5/K)`, `mu_k | Sigma_k ~
  N(0, 16 Sigma_k)`, `Sigma_k ~ IW(p+3, (p+3) I)`.
"""
function fit_hbmnl(panel::PurchasePanel;
                   K::Int = 2,
                   R::Int = 12_000,
                   burnin::Int = 4_000,
                   thin::Int = 5,
                   nchains::Int = 2,
                   sd::Bool = true,
                   seed::Integer = 20260826,
                   verbose::Bool = false,
                   dir_prior::Float64 = -1.0,
                   kappa0::Float64 = 1 / 16,
                   nu0_add::Int = 3,
                   gamma_prior_sd::Float64 = 10.0,
                   n_gamma_sweeps::Int = 10,
                   label::AbstractString = "")

    K >= 1 || throw(ArgumentError("K must be >= 1"))
    burnin < R || throw(ArgumentError("burnin must be smaller than R"))
    thin >= 1 || throw(ArgumentError("thin must be >= 1"))
    nchains >= 1 || throw(ArgumentError("nchains must be >= 1"))

    a0 = dir_prior > 0 ? dir_prior : 0.5 / K
    H  = n_households(panel)
    p  = panel.B - 1
    L  = n_covariates(panel)

    keep = collect((burnin+1):thin:R)
    nd   = length(keep)
    nd > 10 || throw(ArgumentError("only $nd draws would be kept; increase R or reduce thin"))

    gamma_out  = Matrix{Float64}(undef, sd ? nd : 0, nchains)
    beta_out   = Array{Float64,3}(undef, L, nd, nchains)
    ll_out     = Matrix{Float64}(undef, nd, nchains)
    alpha_out  = Array{Float64,3}(undef, p, H, nchains)
    acc_a      = zeros(nchains)
    acc_g      = zeros(nchains)

    chains = 1:nchains
    runner = function (c)
        rng = Xoshiro(UInt64(seed) + UInt64(1000c))
        g, b, ll, am, aa, ag = _run_chain(panel, K, R, burnin, keep, sd, rng, verbose,
                                          a0, kappa0, nu0_add, gamma_prior_sd,
                                          n_gamma_sweeps, p, L, H, label, c)
        sd && (gamma_out[:, c] = g)
        L > 0 && (beta_out[:, :, c] = b)
        ll_out[:, c]      = ll
        alpha_out[:, :, c] = am
        acc_a[c] = aa
        acc_g[c] = ag
        return nothing
    end

    if nchains > 1 && Threads.nthreads() > 1
        Threads.@threads for c in chains
            runner(c)
        end
    else
        for c in chains
            runner(c)
        end
    end

    alpha_mean = dropdims(sum(alpha_out; dims = 3); dims = 3) ./ nchains
    gamma_mean = sd ? mean(gamma_out) : 0.0
    beta_mean  = L > 0 ? vec(mean(reshape(beta_out, L, :); dims = 2)) : Float64[]

    buf   = Vector{Float64}(undef, panel.B)
    ll_at = _ll_total(panel, alpha_mean, gamma_mean, beta_mean, buf)
    Dbar  = -2 * mean(ll_out)
    Dhat  = -2 * ll_at
    p_D   = Dbar - Dhat
    dic   = Dbar + p_D

    negll = vec(-ll_out)
    m     = maximum(negll)
    lml   = -(m + log(sum(exp.(negll .- m)) / length(negll)))

    return HBMNLFit(gamma_out, beta_out, ll_out, alpha_mean, gamma_mean, beta_mean,
                    dic, p_D, lml, mean(acc_a), mean(acc_g), K, sd, nd, nchains)
end

function _run_chain(panel, K, R, burnin, keep, sd, rng, verbose,
                    a0, kappa0, nu0_add, gamma_prior_sd, n_gamma_sweeps,
                    p, L, H, label, c)

    B    = panel.B
    nu0  = float(p + nu0_add)
    Psi0 = Matrix{Float64}(I, p, p) .* (p + nu0_add)
    mu0  = zeros(p)

    alpha = zeros(p, H)
    z     = rand(rng, 1:K, H)
    pik   = fill(1 / K, K)
    mu    = [zeros(p) for _ in 1:K]
    Sig   = [Matrix{Float64}(I, p, p) for _ in 1:K]
    cholL = [LowerTriangular(Matrix{Float64}(I, p, p)) for _ in 1:K]
    logdt = zeros(K)
    propL = [LowerTriangular(Matrix{Float64}(I, p, p)) for _ in 1:K]

    gamma = 0.0
    beta  = zeros(L)

    s_a = fill(2.38 / sqrt(p), H)
    s_g = 0.3
    s_b = L > 0 ? 0.3 : 0.0

    nused = n_used(panel)
    Sg    = Vector{Float64}(undef, nused)
    Ag    = Vector{Float64}(undef, nused)
    repg  = Vector{Bool}(undef, nused)

    buf   = Vector{Float64}(undef, B)
    work  = Vector{Float64}(undef, p)
    prop  = Vector{Float64}(undef, p)
    llh   = Vector{Float64}(undef, H)
    for h in 1:H
        Xc = panel.covariates === nothing ? nothing : panel.covariates[h]
        llh[h] = _ll_household(panel.choices[h], Xc, view(alpha, :, h), gamma, beta, B, buf)
    end

    na = 0; da = 0; ng = 0; dg = 0
    nd = length(keep)
    g_out  = Vector{Float64}(undef, sd ? nd : 0)
    b_out  = Array{Float64,2}(undef, L, nd)
    ll_out = Vector{Float64}(undef, nd)
    a_acc  = zeros(p, H)
    kpos   = 1
    wts    = Vector{Float64}(undef, K)

    for r in 1:R
        # --- 1. household intercepts (random-walk Metropolis) --------------
        for k in 1:K
            propL[k] = LowerTriangular(Matrix(cholesky(_sym(Sig[k])).L))
        end
        for h in 1:H
            k  = z[h]
            ah = view(alpha, :, h)
            randn!(rng, work)
            mul!(prop, propL[k], work)
            @inbounds for i in 1:p
                prop[i] = ah[i] + s_a[h] * prop[i]
            end
            Xc  = panel.covariates === nothing ? nothing : panel.covariates[h]
            llp = _ll_household(panel.choices[h], Xc, prop, gamma, beta, B, buf)
            lpr_new = _logmvn(prop, mu[k], cholL[k], logdt[k], work)
            lpr_old = _logmvn(ah,   mu[k], cholL[k], logdt[k], work)
            acc = (llp + lpr_new) - (llh[h] + lpr_old)
            ok  = log(rand(rng)) < acc
            if ok
                @inbounds for i in 1:p
                    alpha[i, h] = prop[i]
                end
                llh[h] = llp
            end
            da += 1; na += ok
            if r <= burnin
                s_a[h] *= exp((ok - 0.30) / (r^0.6))
                s_a[h] = clamp(s_a[h], 1e-3, 50.0)
            end
        end

        # --- 2. common covariate coefficients ------------------------------
        if L > 0
            bp = beta .+ s_b .* randn(rng, L)
            llnew = 0.0
            for h in 1:H
                Xc = panel.covariates[h]
                llnew += _ll_household(panel.choices[h], Xc, view(alpha, :, h), gamma,
                                       bp, B, buf)
            end
            llold = sum(llh)
            acc = (llnew - 0.5 * sum(abs2, bp ./ gamma_prior_sd)) -
                  (llold - 0.5 * sum(abs2, beta ./ gamma_prior_sd))
            ok = log(rand(rng)) < acc
            if ok
                beta = bp
                for h in 1:H
                    llh[h] = _ll_household(panel.choices[h], panel.covariates[h],
                                           view(alpha, :, h), gamma, beta, B, buf)
                end
            end
            if r <= burnin
                s_b *= exp((ok - 0.30) / (r^0.6))
                s_b = clamp(s_b, 1e-4, 10.0)
            end
        end

        # --- 3. common state-dependence coefficient ------------------------
        if sd
            nu = _fill_gamma_cache!(Sg, Ag, repg, panel, alpha, beta)
            nrep = 0
            @inbounds for i in 1:nu
                nrep += repg[i]
            end
            lp = _gamma_logpost(Sg, Ag, nu, nrep, gamma, gamma_prior_sd)
            for _ in 1:n_gamma_sweeps
                gp  = gamma + s_g * randn(rng)
                lpp = _gamma_logpost(Sg, Ag, nu, nrep, gp, gamma_prior_sd)
                ok  = log(rand(rng)) < lpp - lp
                if ok
                    gamma = gp; lp = lpp
                end
                dg += 1; ng += ok
                if r <= burnin
                    s_g *= exp((ok - 0.30) / (r^0.6))
                    s_g = clamp(s_g, 1e-4, 10.0)
                end
            end
            for h in 1:H
                Xc = panel.covariates === nothing ? nothing : panel.covariates[h]
                llh[h] = _ll_household(panel.choices[h], Xc, view(alpha, :, h),
                                       gamma, beta, B, buf)
            end
        end

        # --- 4. mixture allocation ----------------------------------------
        if K > 1
            for h in 1:H
                ah = view(alpha, :, h)
                for k in 1:K
                    wts[k] = log(pik[k] + 1e-300) + _logmvn(ah, mu[k], cholL[k], logdt[k], work)
                end
                mx = maximum(wts)
                tot = 0.0
                for k in 1:K
                    wts[k] = exp(wts[k] - mx); tot += wts[k]
                end
                u = rand(rng) * tot
                cum = 0.0; zk = K
                for k in 1:K
                    cum += wts[k]
                    if u <= cum
                        zk = k; break
                    end
                end
                z[h] = zk
            end
            cnt = zeros(K)
            for h in 1:H
                cnt[z[h]] += 1
            end
            pik = rand(rng, Dirichlet(cnt .+ a0))
        else
            fill!(z, 1)
        end

        # --- 5. mixture component parameters (Normal-Inverse-Wishart) ------
        for k in 1:K
            idx = findall(==(k), z)
            nk  = length(idx)
            if nk == 0
                Sig[k] = _sym(rand(rng, InverseWishart(nu0, Matrix(Psi0))))
                Lk = cholesky(Sig[k] ./ kappa0).L
                mu[k] = mu0 .+ Lk * randn(rng, p)
            else
                Ak = @view alpha[:, idx]
                ab = vec(sum(Ak; dims = 2)) ./ nk
                S  = zeros(p, p)
                for i in idx
                    d = alpha[:, i] .- ab
                    S .+= d * d'
                end
                kn = kappa0 + nk
                mn = (kappa0 .* mu0 .+ nk .* ab) ./ kn
                d0 = ab .- mu0
                Psn = Psi0 .+ S .+ (kappa0 * nk / kn) .* (d0 * d0')
                Sig[k] = _sym(rand(rng, InverseWishart(nu0 + nk, _sym(Psn))))
                Lk = cholesky(Sig[k] ./ kn).L
                mu[k] = mn .+ Lk * randn(rng, p)
            end
            F = cholesky(Sig[k])
            cholL[k] = LowerTriangular(Matrix(F.L))
            logdt[k] = logdet(F)
        end

        # --- 6. store ------------------------------------------------------
        if kpos <= nd && r == keep[kpos]
            sd && (g_out[kpos] = gamma)
            L > 0 && (b_out[:, kpos] = beta)
            ll_out[kpos] = sum(llh)
            a_acc .+= alpha
            kpos += 1
        end

        if verbose && (r % max(1, R ÷ 10) == 0)
            @printf("  [%s chain %d] sweep %d/%d  loglik=%.1f  gamma=%.3f\n",
                    label, c, r, R, sum(llh), gamma)
            flush(stdout)
        end
    end

    a_acc ./= nd
    return g_out, b_out, ll_out, a_acc, na / max(da, 1), dg == 0 ? NaN : ng / dg
end

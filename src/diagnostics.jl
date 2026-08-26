# ---------------------------------------------------------------------------
# diagnostics.jl -- split-Rhat and effective sample size
# ---------------------------------------------------------------------------

"""
    split_rhat(X)

Split-Rhat for a `ndraws x nchains` matrix of draws. Each chain is split in half
first, so a single chain still yields a usable diagnostic. Values above ~1.01
mean the chains have not mixed.
"""
function split_rhat(X::AbstractMatrix{<:Real})
    nd, nc = size(X)
    nd < 4 && return NaN
    n = nd ÷ 2
    segs = Vector{Vector{Float64}}()
    for c in 1:nc
        push!(segs, Float64.(X[1:n, c]))
        push!(segs, Float64.(X[(nd-n+1):nd, c]))
    end
    m = length(segs)
    means = [mean(s) for s in segs]
    vars  = [var(s) for s in segs]
    W = mean(vars)
    W <= 0 && return NaN
    Bn = n * var(means)
    varplus = ((n - 1) * W + Bn) / n
    return sqrt(varplus / W)
end

"""
    ess(X)

Effective sample size for a `ndraws x nchains` matrix, using Geyer's initial
positive sequence on the chain-averaged autocorrelation (the same construction
Stan uses).
"""
function ess(X::AbstractMatrix{<:Real})
    nd, nc = size(X)
    nd < 8 && return NaN
    Y = Float64.(X)
    means = [mean(view(Y, :, c)) for c in 1:nc]
    vars  = [var(view(Y, :, c)) for c in 1:nc]
    W = mean(vars)
    W <= 0 && return NaN
    Bn = nd * var(means)
    varplus = ((nd - 1) * W + Bn) / nd

    maxlag = min(nd - 2, 500)
    rho = Vector{Float64}(undef, maxlag + 1)
    rho[1] = 1.0
    for t in 1:maxlag
        acov = 0.0
        for c in 1:nc
            s = 0.0
            @inbounds for i in 1:(nd-t)
                s += (Y[i, c] - means[c]) * (Y[i+t, c] - means[c])
            end
            acov += s / nd
        end
        acov /= nc
        rho[t+1] = 1 - (W - acov) / varplus
    end

    # Geyer initial positive sequence on successive pairs
    total = 0.0
    t = 1
    while t + 1 <= maxlag
        pair = rho[t+1] + rho[t+2]
        pair <= 0 && break
        total += pair
        t += 2
    end
    tau = 1 + 2 * total
    tau <= 0 && return NaN
    return nd * nc / tau
end

_quantile2(v::AbstractVector{<:Real}, q) = quantile(collect(Float64.(v)), q)

function _ci(v::AbstractVector{<:Real}, level::Real)
    a = (1 - level) / 2
    return (_quantile2(v, a), _quantile2(v, 1 - a))
end

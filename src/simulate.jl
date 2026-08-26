# ---------------------------------------------------------------------------
# simulate.jl -- dummy data generator
# ---------------------------------------------------------------------------

"""
    simulate_panel(; H, B, T, gamma, K, ...)

Generate a dummy purchase panel with a *known* amount of state dependence, in
exactly the input format [`DHRTests`](@ref) expects: a `Vector` of `B x T`
matrices, brands in the rows, purchase occasions in chronological order in the
columns, quantities in the cells.

Keyword arguments
- `H`, `B`, `T` : households, brands, occasions per household. Pass a
  `UnitRange` (or any collection) to `T` to draw a ragged panel.
- `gamma` : true state-dependence coefficient. `0.0` gives a zero-order panel
  with heterogeneity only -- the null case the test must not reject.
- `K`, `spread`, `alpha_sd` : the heterogeneity distribution is a `K`-component
  normal mixture; component means are drawn `N(0, spread^2 I)` and each
  component has covariance `alpha_sd^2 I`.
- `quantity` : `:unit` writes a 1 for the chosen brand, `:random` writes an
  integer in `1:max_quantity`.
- `zero_column_prob` : probability that an extra all-zero (no purchase) column
  is inserted, to exercise the column handling in [`build_panel`](@ref).
- `warmup` : occasions simulated and discarded before recording, so that the
  recorded sequence starts from the stationary distribution of the choice
  process.

Returns a `NamedTuple` `(; X, alpha, gamma, z, mu, Sigma, weights, choices)`
where `X` is the input matrix list and the rest is the ground truth.

```julia
sim = simulate_panel(H = 400, B = 4, T = 12, gamma = 0.8, seed = 1)
res = DHRTests(sim.X)
```
"""
function simulate_panel(; H::Int = 300,
                          B::Int = 4,
                          T = 10,
                          gamma::Real = 0.8,
                          K::Int = 2,
                          spread::Real = 1.2,
                          alpha_sd::Real = 0.8,
                          weights = nothing,
                          quantity::Symbol = :unit,
                          max_quantity::Int = 3,
                          zero_column_prob::Real = 0.0,
                          warmup::Int = 20,
                          seed::Integer = 1)

    H >= 1 || throw(ArgumentError("H must be >= 1"))
    B >= 2 || throw(ArgumentError("B must be >= 2"))
    K >= 1 || throw(ArgumentError("K must be >= 1"))
    quantity in (:unit, :random) ||
        throw(ArgumentError("quantity must be :unit or :random"))

    rng = Xoshiro(UInt64(seed))
    p   = B - 1

    w = weights === nothing ? fill(1 / K, K) : collect(Float64.(weights))
    length(w) == K || throw(DimensionMismatch("weights must have K entries"))
    w ./= sum(w)

    mu    = [spread .* randn(rng, p) for _ in 1:K]
    Sigma = [Matrix{Float64}(I, p, p) .* alpha_sd^2 for _ in 1:K]

    Tvec = T isa Integer ? fill(Int(T), H) : [rand(rng, T) for _ in 1:H]

    alpha   = zeros(p, H)
    z       = Vector{Int}(undef, H)
    choices = Vector{Vector{Int}}(undef, H)
    X       = Vector{Matrix{Float64}}(undef, H)
    v       = Vector{Float64}(undef, B)
    g       = float(gamma)

    for h in 1:H
        k = _sample_categorical(rng, w)
        z[h] = k
        alpha[:, h] = mu[k] .+ alpha_sd .* randn(rng, p)

        Th   = Tvec[h]
        last = 0
        seq  = Vector{Int}(undef, Th)
        for t in 1:(warmup + Th)
            for j in 1:B
                v[j] = (j < B ? alpha[j, h] : 0.0) + (j == last ? g : 0.0)
            end
            m = maximum(v)
            s = 0.0
            for j in 1:B
                v[j] = exp(v[j] - m); s += v[j]
            end
            v ./= s
            c = _sample_categorical(rng, v)
            last = c
            t > warmup && (seq[t-warmup] = c)
        end
        choices[h] = seq

        cols = Vector{Vector{Float64}}()
        for t in 1:Th
            if zero_column_prob > 0 && rand(rng) < zero_column_prob
                push!(cols, zeros(B))
            end
            col = zeros(B)
            col[seq[t]] = quantity === :unit ? 1.0 : float(rand(rng, 1:max_quantity))
            push!(cols, col)
        end
        X[h] = reduce(hcat, cols)
    end

    return (; X, alpha, gamma = g, z, mu, Sigma, weights = w, choices)
end

function _sample_categorical(rng::AbstractRNG, w::AbstractVector{<:Real})
    u = rand(rng) * sum(w)
    c = 0.0
    @inbounds for i in eachindex(w)
        c += w[i]
        u <= c && return i
    end
    return length(w)
end

"""
    dummy_data(; kwargs...)

Convenience wrapper returning only the input matrices of [`simulate_panel`](@ref).
"""
dummy_data(; kwargs...) = simulate_panel(; kwargs...).X

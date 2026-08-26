# ---------------------------------------------------------------------------
# panel.jl -- input (matrices, label sequences, brand codes) -> choice panel
# ---------------------------------------------------------------------------

"""
    PurchasePanel

A time-ordered brand-choice panel, built from brand (row) x purchase-occasion
(column) quantity matrices, or from sequences of brand labels / brand codes.

Fields
- `B` : number of brands
- `choices` : `choices[h][t]` is the brand index chosen by household `h` at its
  `t`-th purchase occasion. Occasions are in chronological order.
- `covariates` : optional `covariates[h]` of size `B x L x T_h`, aligned with
  `choices[h]`. `nothing` when no covariates were supplied.
- `brand_names` : labels for the rows.
- `dropped_households` / `dropped_zero_columns` / `ambiguous_columns` /
  `dropped_brands` : bookkeeping from `build_panel`.

Note that the *first* occasion of every household is not used for estimation:
following Dube, Hitsch & Rossi (2010) the initial condition is taken as given.
`n_used(panel)` reports the number of occasions that actually enter the
likelihood.
"""
struct PurchasePanel
    B::Int
    choices::Vector{Vector{Int}}
    covariates::Union{Nothing,Vector{Array{Float64,3}}}
    brand_names::Vector{String}
    dropped_households::Int
    dropped_zero_columns::Int
    ambiguous_columns::Int
    dropped_brands::Vector{String}
end

n_households(p::PurchasePanel) = length(p.choices)
n_occasions(p::PurchasePanel)  = isempty(p.choices) ? 0 : sum(length, p.choices)
n_used(p::PurchasePanel)       = isempty(p.choices) ? 0 : sum(t -> length(t) - 1, p.choices)
n_covariates(p::PurchasePanel) = p.covariates === nothing ? 0 : size(p.covariates[1], 2)

function Base.show(io::IO, p::PurchasePanel)
    print(io, "PurchasePanel(B=", p.B, ", households=", n_households(p),
          ", occasions=", n_occasions(p), ", used=", n_used(p), ")")
end

# ---------------------------------------------------------------------------
# input normalisation
# ---------------------------------------------------------------------------

_all_matrices(v::AbstractVector) = !isempty(v) && all(x -> x isa AbstractMatrix, v)
_all_vectors(v::AbstractVector)  = !isempty(v) && all(x -> x isa AbstractVector, v)

# Any accepted input -> (vector of B x T quantity matrices, brand names or nothing)
function _input_matrices(X, brand_names)
    if X isa AbstractArray{<:Real,3}
        return [Matrix{Float64}(@view X[:, :, h]) for h in axes(X, 3)], nothing

    elseif X isa AbstractMatrix
        eltype(X) <: Real || throw(ArgumentError(
            "a matrix input must hold quantities; got element type $(eltype(X))"))
        @warn "A single B x T matrix was supplied: it is treated as ONE household. " *
              "Pass a Vector of matrices (or a B x T x H array) for a real panel."
        return [Matrix{Float64}(X)], nothing

    elseif X isa AbstractVector
        isempty(X) && throw(ArgumentError("the input is empty"))
        if _all_matrices(X)
            return [Matrix{Float64}(M) for M in X], nothing
        elseif _all_vectors(X)
            return _sequences_to_matrices(collect(X), brand_names)   # many households
        else
            return _sequences_to_matrices([X], brand_names)          # one household
        end
    end
    throw(ArgumentError("unsupported input type $(typeof(X)); pass quantity " *
                        "matrices, a B x T x H array, or sequences of brand " *
                        "labels or brand codes"))
end

"""
    _sequences_to_matrices(seqs, brand_names)

Turn choice sequences into indicator matrices. A sequence is one household, in
chronological order; `missing` marks an occasion with no purchase.

* integers are brand **codes**: `[2, 2, 1, 1, 2]` becomes
  `[0 0 1 1 0; 1 1 0 0 1]`;
* anything else is a brand **label**: `["A", "B", "A"]` becomes
  `[1 0 1; 0 1 0]`, with the rows in sorted label order.

Levels are pooled across households, so every household ends up with the same
row order. Pass `brand_names` to fix the level set and its order yourself.
"""
function _sequences_to_matrices(seqs::Vector, brand_names)
    for (h, s) in pairs(seqs)
        s isa AbstractVector ||
            throw(ArgumentError("element $h of the input is not a sequence"))
        isempty(s) && throw(ArgumentError("household $h has an empty sequence"))
    end

    flat = collect(Iterators.flatten(seqs))
    vals = collect(skipmissing(flat))
    isempty(vals) && throw(ArgumentError("the input contains no purchases"))

    if all(x -> x isa Integer, vals)
        codes = Int.(vals)
        minimum(codes) >= 1 || throw(ArgumentError(
            "brand codes must be >= 1, got $(minimum(codes))"))
        B = brand_names === nothing ? maximum(codes) : length(brand_names)
        maximum(codes) <= B || throw(ArgumentError(
            "brand code $(maximum(codes)) exceeds the $B brand names supplied"))
        names = brand_names === nothing ? ["brand$(j)" for j in 1:B] :
                String.(collect(brand_names))
        return [_seq_matrix(s, B, Int) for s in seqs], names
    end

    if brand_names === nothing
        lev = try
            sort(unique(vals))            # CategoricalValue sorts by its level order
        catch
            unique(vals)                  # not orderable: keep order of appearance
        end
        idx = Dict(l => i for (i, l) in enumerate(lev))
        return [_seq_matrix(s, length(lev), x -> idx[x]) for s in seqs], string.(lev)
    else
        names = String.(collect(brand_names))
        idx = Dict(n => i for (i, n) in enumerate(names))
        f = function (x)
            k = string(x)
            haskey(idx, k) || throw(ArgumentError(
                "brand \"$k\" is not among the brand_names supplied"))
            return idx[k]
        end
        return [_seq_matrix(s, length(names), f) for s in seqs], names
    end
end

function _seq_matrix(s, B::Int, tocode)
    M = zeros(Float64, B, length(s))
    for (t, x) in enumerate(s)
        x === missing && continue        # no purchase on this occasion
        M[tocode(x), t] = 1.0
    end
    return M
end

# ---------------------------------------------------------------------------
# build_panel
# ---------------------------------------------------------------------------

"""
    build_panel(X; brand_names, covariates, tie_rule, min_occasions, drop_unused)

Convert the input into a [`PurchasePanel`](@ref).

# Accepted input

**Quantity matrices** -- brands in the rows, purchase occasions in chronological
order in the columns:

* a `Vector` of `B x T_h` matrices (one per household, ragged `T_h` allowed),
* a `B x T x H` array, or
* a single `B x T` matrix (one household).

**Choice sequences** -- one element per occasion, in chronological order:

* a vector of brand **labels** (strings, symbols, a `CategoricalVector`, ...):
  `["A", "B", "A"]` becomes `[1 0 1; 0 1 0]`, rows in sorted label order;
* a vector of brand **codes**: `[2, 2, 1, 1, 2]` becomes `[0 0 1 1 0; 1 1 0 0 1]`;
* a `Vector` of such vectors, one per household. Levels are pooled across
  households so the row order is the same for everybody.

`missing` inside a sequence marks an occasion with no purchase. Pass
`brand_names` to fix the level set and its order rather than inferring it.

# Column handling (matrix input)

* all-zero column  -> no purchase, the occasion is dropped;
* one positive row -> that brand is the choice;
* several positive rows -> resolved by `tie_rule`
  (`:argmax` (default, largest quantity wins, lowest index breaks a tie),
   `:error`, or `:drop`).

Households left with fewer than `min_occasions` (default `2`) occasions are
dropped, because at least one lagged choice is needed.

A brand that nobody ever buys -- easy to end up with when a `CategoricalVector`
carries unused levels, or when `brand_names` lists more brands than the data
contains -- always produces a warning, because its intercept is not identified
by the likelihood and is filled in by the prior alone. Pass `drop_unused = true`
to remove such brands from the panel instead. (`mode = :single` drops them by
itself: a conditional logit cannot carry them at all.)
"""
function build_panel(X;
                     brand_names::Union{Nothing,AbstractVector} = nothing,
                     covariates = nothing,
                     tie_rule::Symbol = :argmax,
                     min_occasions::Int = 2,
                     drop_unused::Bool = false)

    tie_rule in (:argmax, :error, :drop) ||
        throw(ArgumentError("tie_rule must be :argmax, :error or :drop, got $tie_rule"))

    mats, inferred_names = _input_matrices(X, brand_names)
    isempty(mats) && throw(ArgumentError("no households in the input"))

    B = size(first(mats), 1)
    B >= 2 || throw(ArgumentError("need at least 2 brands (rows), got $B"))
    for (h, M) in pairs(mats)
        size(M, 1) == B || throw(DimensionMismatch(
            "household $h has $(size(M,1)) rows, expected $B"))
        any(x -> x < 0 || !isfinite(x), M) &&
            throw(ArgumentError("household $h contains a negative or non-finite quantity"))
    end

    names = if inferred_names !== nothing
        inferred_names                                  # already resolved
    elseif brand_names === nothing
        ["brand$(j)" for j in 1:B]
    else
        String.(collect(brand_names))
    end
    length(names) == B || throw(DimensionMismatch(
        "brand_names has $(length(names)) entries, expected $B"))

    covs = covariates === nothing ? nothing : _check_covariates(covariates, mats, B)

    choices  = Vector{Vector{Int}}()
    kept_cov = covs === nothing ? nothing : Vector{Array{Float64,3}}()
    zero_cols = 0
    ambig     = 0
    dropped   = 0

    for (h, M) in pairs(mats)
        T = size(M, 2)
        seq  = Int[]
        keep = Int[]
        for t in 1:T
            col = @view M[:, t]
            nz  = count(>(0), col)
            if nz == 0
                zero_cols += 1
                continue
            elseif nz > 1
                ambig += 1
                if tie_rule === :error
                    throw(ArgumentError(
                        "household $h occasion $t has $nz brands with a positive " *
                        "quantity; pass tie_rule=:argmax or :drop"))
                elseif tie_rule === :drop
                    continue
                end
            end
            push!(seq, argmax(col))
            push!(keep, t)
        end
        if length(seq) < min_occasions
            dropped += 1
            continue
        end
        push!(choices, seq)
        if covs !== nothing
            push!(kept_cov, covs[h][:, :, keep])
        end
    end

    isempty(choices) && throw(ArgumentError(
        "no household has at least $min_occasions purchase occasions; " *
        "state dependence cannot be identified"))

    dropped_brands = String[]
    seen = falses(B)
    for y in choices, j in y
        seen[j] = true
    end
    if !all(seen)
        unused = names[.!seen]
        if drop_unused
            keepb = findall(seen)
            dropped_brands = unused
            length(keepb) >= 2 || throw(ArgumentError(
                "only $(length(keepb)) brand(s) are ever bought; nothing to test"))
            @warn "dropping $(length(unused)) brand(s) nobody ever buys: " *
                  join(unused, ", ")
            remap = Dict(b => i for (i, b) in enumerate(keepb))
            for y in choices
                for t in eachindex(y)
                    y[t] = remap[y[t]]
                end
            end
            if kept_cov !== nothing
                for h in eachindex(kept_cov)
                    kept_cov[h] = kept_cov[h][keepb, :, :]
                end
            end
            names = names[keepb]
            B = length(keepb)
        else
            @warn "$(length(unused)) brand(s) are never bought: " *
                  join(unused, ", ") *
                  ". Their intercepts are not identified by the likelihood, only " *
                  "by the prior. Pass drop_unused=true to remove them."
        end
    end

    return PurchasePanel(B, choices, kept_cov, names, dropped, zero_cols, ambig,
                         dropped_brands)
end

build_panel(p::PurchasePanel; kwargs...) = p

function _check_covariates(covariates, mats, B)
    covariates isa AbstractVector ||
        throw(ArgumentError("covariates must be a Vector of B x L x T arrays"))
    length(covariates) == length(mats) || throw(DimensionMismatch(
        "covariates has $(length(covariates)) households, expected $(length(mats))"))
    out = Vector{Array{Float64,3}}(undef, length(mats))
    L = size(first(covariates), 2)
    for h in eachindex(mats)
        A = Array{Float64,3}(covariates[h])
        size(A, 1) == B || throw(DimensionMismatch("covariates[$h] must have $B rows"))
        size(A, 2) == L || throw(DimensionMismatch("covariates[$h] must have $L columns"))
        size(A, 3) == size(mats[h], 2) || throw(DimensionMismatch(
            "covariates[$h] has $(size(A,3)) occasions, expected $(size(mats[h],2))"))
        out[h] = A
    end
    return out
end

"""
    shuffle_panel(panel, rng)

Return a copy of `panel` with the purchase occasions of each household randomly
re-ordered. Every household keeps exactly the same basket of choices, so all
cross-sectional heterogeneity survives -- only the *order* is destroyed. Fitting
the same model to the shuffled panel therefore gives the placebo distribution of
the state-dependence coefficient.
"""
function shuffle_panel(p::PurchasePanel, rng::AbstractRNG)
    ch = Vector{Vector{Int}}(undef, n_households(p))
    cv = p.covariates === nothing ? nothing :
         Vector{Array{Float64,3}}(undef, n_households(p))
    for h in eachindex(p.choices)
        perm  = randperm(rng, length(p.choices[h]))
        ch[h] = p.choices[h][perm]
        cv === nothing || (cv[h] = p.covariates[h][:, :, perm])
    end
    return PurchasePanel(p.B, ch, cv, p.brand_names,
                         p.dropped_households, p.dropped_zero_columns,
                         p.ambiguous_columns, p.dropped_brands)
end

"""
    lagged_repeat_rate(panel)

Share of estimation occasions on which the household repeats the brand it bought
on the previous occasion. A purely descriptive number: a zero-order Dirichlet
process with strong heterogeneity also produces a high value, which is exactly
why the placebo comparison in [`DHRTests`](@ref) is needed.
"""
function lagged_repeat_rate(p::PurchasePanel)
    hit = 0; tot = 0
    for y in p.choices, t in 2:length(y)
        tot += 1
        hit += (y[t] == y[t-1])
    end
    return tot == 0 ? NaN : hit / tot
end

"""
    choice_matrix(panel, h = 1)

The `B x T` indicator matrix of household `h`, i.e. the canonical form of
whatever was passed to [`build_panel`](@ref). Useful to check that a sequence
was read the way you expected.

```julia
choice_matrix(build_panel(["A", "B", "A"]))   # [1 0 1; 0 1 0]
```
"""
function choice_matrix(p::PurchasePanel, h::Int = 1)
    1 <= h <= n_households(p) ||
        throw(ArgumentError("household $h out of range (1:$(n_households(p)))"))
    y = p.choices[h]
    M = zeros(Float64, p.B, length(y))
    for (t, j) in enumerate(y)
        M[j, t] = 1.0
    end
    return M
end

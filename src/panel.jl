# ---------------------------------------------------------------------------
# panel.jl -- input matrices -> time-ordered brand choice panel
# ---------------------------------------------------------------------------

"""
    PurchasePanel

A time-ordered brand-choice panel, built from brand (row) x purchase-occasion
(column) quantity matrices.

Fields
- `B` : number of brands (= number of rows of the input matrices)
- `choices` : `choices[h][t]` is the brand index chosen by household `h` at its
  `t`-th purchase occasion. Occasions are in chronological order.
- `covariates` : optional `covariates[h]` of size `B x L x T_h`, aligned with
  `choices[h]`. `nothing` when no covariates were supplied.
- `brand_names` : labels for the rows.
- `dropped_households` / `dropped_zero_columns` / `ambiguous_columns` :
  bookkeeping from `build_panel`.

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
end

n_households(p::PurchasePanel) = length(p.choices)
n_occasions(p::PurchasePanel)  = isempty(p.choices) ? 0 : sum(length, p.choices)
n_used(p::PurchasePanel)       = isempty(p.choices) ? 0 : sum(t -> length(t) - 1, p.choices)
n_covariates(p::PurchasePanel) = p.covariates === nothing ? 0 : size(p.covariates[1], 2)

function Base.show(io::IO, p::PurchasePanel)
    print(io, "PurchasePanel(B=", p.B, ", households=", n_households(p),
          ", occasions=", n_occasions(p), ", used=", n_used(p), ")")
end

# --- helpers ---------------------------------------------------------------

_as_matrix_list(X::AbstractVector{<:AbstractMatrix}) = collect(X)

function _as_matrix_list(X::AbstractArray{<:Real,3})
    # B x T x H
    return [Matrix{Float64}(@view X[:, :, h]) for h in axes(X, 3)]
end

function _as_matrix_list(X::AbstractMatrix{<:Real})
    @warn "A single B x T matrix was supplied: it is treated as ONE household. " *
          "Pass a Vector of matrices (or a B x T x H array) for a real panel."
    return [Matrix{Float64}(X)]
end

"""
    build_panel(X; brand_names, covariates, tie_rule, min_occasions)

Convert brand x occasion quantity matrices into a [`PurchasePanel`](@ref).

`X` may be

* a `Vector` of `B x T_h` matrices (one per household, ragged `T_h` allowed),
* a `B x T x H` array, or
* a single `B x T` matrix (one household).

Rows are brands, columns are purchase occasions **in chronological order**, and
each entry is the quantity of that brand bought on that occasion.

Column handling
* all-zero column  -> no purchase, the occasion is dropped;
* one positive row -> that brand is the choice;
* several positive rows -> resolved by `tie_rule`
  (`:argmax` (default, largest quantity wins, lowest index breaks a tie),
   `:error`, or `:drop`).

Households left with fewer than `min_occasions` (default `2`) occasions are
dropped, because at least one lagged choice is needed.
"""
function build_panel(X;
                     brand_names::Union{Nothing,AbstractVector} = nothing,
                     covariates = nothing,
                     tie_rule::Symbol = :argmax,
                     min_occasions::Int = 2)

    tie_rule in (:argmax, :error, :drop) ||
        throw(ArgumentError("tie_rule must be :argmax, :error or :drop, got $tie_rule"))

    mats = _as_matrix_list(X)
    isempty(mats) && throw(ArgumentError("no households in the input"))

    B = size(first(mats), 1)
    B >= 2 || throw(ArgumentError("need at least 2 brands (rows), got $B"))
    for (h, M) in pairs(mats)
        size(M, 1) == B || throw(DimensionMismatch(
            "household $h has $(size(M,1)) rows, expected $B"))
        any(x -> x < 0 || !isfinite(x), M) &&
            throw(ArgumentError("household $h contains a negative or non-finite quantity"))
    end

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

    names = brand_names === nothing ? ["brand$(j)" for j in 1:B] :
            String.(collect(brand_names))
    length(names) == B || throw(DimensionMismatch(
        "brand_names has $(length(names)) entries, expected $B"))

    return PurchasePanel(B, choices, kept_cov, names, dropped, zero_cols, ambig)
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
                         p.ambiguous_columns)
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

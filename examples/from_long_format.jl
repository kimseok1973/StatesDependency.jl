# Turning a real transaction log into the input DHRTests expects.
#
# Most panels arrive as one row per purchase:
#
#     household, date (or week), brand, quantity
#
# `to_matrices` below folds that into one brand x occasion matrix per household,
# with the columns sorted by date. It has no dependencies beyond the stdlib, so
# you can paste it into your own script and drop the CSV reader of your choice in
# front of it.
#
#     julia --project=. examples/from_long_format.jl

using StatesDependency

"""
    to_matrices(household, when, brand, qty; brands)

Fold a long transaction log into `(X, hh_ids, brands)` where `X[i]` is the
`B x T` quantity matrix of household `hh_ids[i]`, columns sorted by `when`.

Several rows sharing a `(household, when)` pair are treated as ONE purchase
occasion, i.e. a shopping basket: their quantities are summed into the same
column. That is the right unit for a brand-choice model -- one trip, one choice.
"""
function to_matrices(household, when, brand, qty; brands = sort(unique(brand)))
    bidx = Dict(b => i for (i, b) in enumerate(brands))
    B    = length(brands)

    # household -> occasion key -> column of quantities
    baskets = Dict{Any,Dict{Any,Vector{Float64}}}()
    for i in eachindex(household)
        h = household[i]
        hh = get!(baskets, h, Dict{Any,Vector{Float64}}())
        col = get!(hh, when[i], zeros(B))
        col[bidx[brand[i]]] += float(qty[i])
    end

    hh_ids = sort(collect(keys(baskets)))
    X = Vector{Matrix{Float64}}(undef, length(hh_ids))
    for (i, h) in enumerate(hh_ids)
        occ  = sort(collect(keys(baskets[h])))          # chronological order
        X[i] = reduce(hcat, [baskets[h][o] for o in occ])
    end
    return X, hh_ids, brands
end

# --- a toy log, standing in for your CSV ----------------------------------
sim = simulate_panel(H = 300, B = 4, T = 12, gamma = 0.7, seed = 3)

household = Int[]; when = Int[]; brand = String[]; qty = Float64[]
names = ["Alpha", "Beta", "Gamma", "Delta"]
for (h, M) in pairs(sim.X), t in axes(M, 2), j in axes(M, 1)
    M[j, t] == 0 && continue
    push!(household, h); push!(when, t)
    push!(brand, names[j]); push!(qty, M[j, t])
end

X, hh_ids, brands = to_matrices(household, when, brand, qty; brands = names)
println("rebuilt ", length(X), " households, brands = ", brands)

res = DHRTests(X; brand_names = brands, verbose = false)
show(stdout, MIME"text/plain"(), res)
println()

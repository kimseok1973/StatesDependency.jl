# One household with a long purchase history.
#
#     julia --project=. examples/single_household.jl
#
# Three sequences, all from a single household, all with a high raw repeat rate.
# Only one of them is actually state dependent. The window-shuffle null is what
# tells them apart; the classical LR test and the global shuffle do not.

using StatesDependency, Statistics, Printf

cases = (("true state dependence      ", 0.8, 0.00),
         ("no SD, tastes drift        ", 0.0, 0.15),
         ("no SD, stable tastes       ", 0.0, 0.00))

println("first, what the raw numbers look like:\n")
@printf("%-28s  repeat rate\n", "sequence")
for (name, g, ds) in cases
    sim = simulate_panel(H = 1, B = 4, T = 400, gamma = g, K = 1,
                         drift_sd = ds, seed = 77)
    y = build_panel(sim.X[1] |> m -> [m])
    @printf("%-28s  %.3f\n", name, lagged_repeat_rate(y))
end
println("\n-> the repeat rate cannot separate inertia from drifting tastes.\n")

for (name, g, ds) in cases
    println("#"^70)
    println("# ", name, "  (simulated gamma = ", g, ", drift sd = ", ds, ")")
    println("#"^70)
    sim = simulate_panel(H = 1, B = 4, T = 400, gamma = g, K = 1,
                         drift_sd = ds, seed = 77)
    r = DHRTests(sim.X[1]; mode = :single, window = 20, nperm = 499,
                 verbose = false)
    show(stdout, MIME"text/plain"(), r)
    println()
end

println("\nthe permutation nulls, side by side (last case above):")
sim = simulate_panel(H = 1, B = 4, T = 400, gamma = 0.0, K = 1,
                     drift_sd = 0.15, seed = 77)
r = DHRTests(sim.X[1]; mode = :single, window = 20, nperm = 499, verbose = false)
@printf("  observed gamma      %+.3f\n", r.gamma)
@printf("  global-shuffle null %+.3f  -> p = %.3f  (drift destroyed, bar too low)\n",
        mean(r.null_global), r.p_global)
@printf("  window-shuffle null %+.3f  -> p = %.3f  (drift kept, bar honest)\n",
        mean(r.null_window), r.p_window)

# Quickstart: dummy data in, verdict out.
#
#     julia --project=. examples/quickstart.jl
#
# Run it twice on purpose: once on a panel with real state dependence, once on a
# panel that has heterogeneity only. The second is the case a naive repeat-rate
# statistic gets wrong.

using StatesDependency

for (label, g) in (("true state dependence", 0.8), ("heterogeneity only", 0.0))
    println("\n", "#"^70)
    println("# ", label, "   (simulated gamma = ", g, ")")
    println("#"^70)

    sim = simulate_panel(H = 400, B = 4, T = 12, gamma = g, K = 2, seed = 7)

    # What the input looks like: one B x T matrix per household.
    println("\nhousehold 1, brands (rows) x occasions (columns):")
    display(sim.X[1])
    println()

    res = DHRTests(sim.X; verbose = false)
    show(stdout, MIME"text/plain"(), res)

    println("\nheadline numbers as a NamedTuple:")
    println(summarize(res))
end

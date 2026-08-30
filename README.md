# StatesDependency.jl

[![CI](https://github.com/kimseok1973/StatesDependency.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/kimseok1973/StatesDependency.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Statistical tests for **state dependence in consecutive brand purchases**, following
Dubé, Hitsch & Rossi (2010), *State dependence and alternative explanations for
consumer inertia*, Quantitative Marketing and Economics 8(4), 417–445.

Input is the simplest thing a panel gives you: one **brand (row) × purchase-occasion
(column)** quantity matrix per household — or just the sequence of brands each
household bought, as labels, a `CategoricalVector`, or integer codes. One call
returns the state-dependence coefficient with a credible interval, an
order-shuffled placebo, and a verdict.

There is also a mode for **one household with a long history**
([`mode = :single`](#one-long-history-instead-of-a-panel-mode--single)) — a
fixed-effect conditional logit with a within-window permutation null, and a
[stationarity pre-test](#checking-the-assumption-instead-of-making-it-p_stationarity)
that refuses to call a household inertial when its tastes are simply moving.

```julia
using StatesDependency

sim = simulate_panel(H = 400, B = 4, T = 12, gamma = 0.8, seed = 1)
res = DHRTests(sim.X)
```

```
State dependence test  (Dube, Hitsch & Rossi 2010)
====================================================================
panel        : 400 households x 4 brands, 4800 occasions (4400 used)
raw repeat   : 0.658 of consecutive occasions repeat the brand
model        : HB multinomial logit, 2-component normal mixture,
               2 chains x 12000 sweeps (burn-in 4000, thin 5)
--------------------------------------------------------------------
gamma (state dependence)     0.826   95% CI [  0.719,   0.934] *
  P(gamma > 0)               1.000   ESS 812, Rhat 1.004
  odds ratio exp(gamma)      2.284   95% CI [  2.052,   2.545]
  choice prob. lift        +17.94 pp  95% CI [+16.02, +19.83]
--------------------------------------------------------------------
placebo (order shuffled)     0.004   95% CI [ -0.131,   0.140]
EXCESS  gamma - placebo      0.822   95% CI [  0.646,   0.997] *
  P(excess > 0)              1.000
--------------------------------------------------------------------
DIC   with SD     8021.4   without SD     8154.8   delta   -133.4
====================================================================
verdict: state_dependence
```

---

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/kimseok1973/StatesDependency.jl")
```

Julia 1.9+. The only non-stdlib dependencies are `Distributions` and
`InverseFunctions`. Bijectors.jl is optional (a package extension).

## Input format

### Quantity matrices

`X` is a `Vector` of `B × T_h` matrices — one per household, ragged `T_h` allowed —
or a `B × T × H` array.

* **Rows are brands.** Same order in every household.
* **Columns are purchase occasions, in chronological order.** The whole test is about
  what the *previous* column contained, so the column order is the data.
* **Cells are quantities.** An all-zero column is a no-purchase occasion and is
  dropped; a column with several positive rows is resolved by `tie_rule`
  (`:argmax` by default, or `:error` / `:drop`).

### Choice sequences

You can also pass the choices themselves, one element per occasion, and the
indicator matrices are built for you:

```julia
build_panel(["A", "B", "A"])       # -> [1 0 1
                                   #     0 1 0]
build_panel([2, 2, 1, 1, 2])       # -> [0 0 1 1 0
                                   #     1 1 0 0 1]
```

* **Labels** — strings, symbols, characters, or a `CategoricalVector`. Rows come
  out in sorted label order; for a categorical, in its declared level order
  (`categorical(x; levels = ["B", "A"])` keeps `B` first, and `ordered = true`
  works too). No dependency on CategoricalArrays.jl is involved — it works
  through `sort`/`unique`.
* **Codes** — positive integers indexing the brands directly, so `2` is row 2.
* **A `Vector` of such vectors** is a panel, one element per household. Levels are
  pooled across households, so everybody ends up with the same row order.
* `missing` inside a sequence is an occasion with no purchase; it becomes an
  all-zero column and is dropped like any other.

`brand_names` fixes the level set and its order instead of inferring it, and
`choice_matrix(build_panel(X))` shows exactly how a sequence was read.

A brand that nobody ever buys — easy to end up with from unused categorical
levels — warns, because its intercept is identified only by the prior. Pass
`drop_unused = true` to remove it.

Households with fewer than two occasions are dropped, and **the first occasion of
every household is not used** — its lagged choice is unobserved, so the initial
condition is taken as given, exactly as in DHR.

## What is estimated

A hierarchical Bayes multinomial logit with a flexible heterogeneity distribution.

At every purchase occasion the model hands each brand a **utility** — a score for
how likely that brand is to be picked. A random error is added on top, and the
brand with the largest total wins. With the error drawn from a Gumbel
distribution this is exactly the logit choice probability
$e^{v_j} / \sum_k e^{v_k}$.

$$v_{hjt} = \alpha_{hj} + \gamma \cdot \mathbb{1}\{j = y_{h,t-1}\} + x_{hjt}'\beta,
\qquad \alpha_h \sim \sum_{k=1}^{K} \pi_k \, N(\mu_k, \Sigma_k)$$

* $\alpha_{hj}$ — household $h$'s fixed taste for brand $j$. **This is the
  zero-order part.** It comes from a finite normal mixture, so the shape of the
  heterogeneity is not forced into a single bell.
* $\gamma$ — the **state-dependence coefficient**. Take this term away and each
  brand is left with a score that never moves: the model collapses back to
  zero-order.
* $x_{hjt}'\beta$ — optional covariates (price, display, ...).

**Only differences in utility matter.** Adding the same number to every brand
changes no choice probability, so one brand is pinned at $\alpha_{hB} \equiv 0$.
That is also why the scale is log-odds — $\gamma = 0.8$ means "0.8 higher on the
log-odds scale", and hence why `exp(gamma)` is reported as an odds ratio.

Sampling is Gibbs with random-walk Metropolis blocks (household intercepts,
$\gamma$, $\beta$) and a Normal-Inverse-Wishart / Dirichlet update for the
mixture. Priors follow DHR: $\text{Dir}(0.5/K)$,
$\mu_k \mid \Sigma_k \sim N(0, 16\Sigma_k)$, $\Sigma_k \sim IW(p+3, (p+3)I)$.

**$\gamma$ is common across households on purpose.** With the handful of occasions a
typical panel gives per household there is no information to identify
household-specific state dependence; forcing it degrades hold-out fit.

## Why the placebo is the actual test

A high raw repeat rate proves nothing — a zero-order Dirichlet process with strong
heterogeneity produces one too. A positive $\hat\gamma$ proves less than it looks,
either, because the estimator is biased in that direction in short panels.

So `DHRTests` re-fits the **same model** to a panel whose occasions have been randomly
re-ordered within each household. That panel has identical cross-sectional composition
and, by construction, zero state dependence. The statistic to read is

```
excess = gamma - gamma_placebo
```

and the verdict requires its credible interval to exclude zero. In our own replication
on a beer panel this mattered a lot: $\gamma = 0.65$ against a placebo of $0.11$.

The verdict is `:state_dependence` only when all three agree — the $\gamma$ interval
excludes zero, the excess interval excludes zero, and DIC prefers the model with
$\gamma$. Otherwise it is `:inconclusive` or `:no_evidence`.

## One long history instead of a panel: `mode = :single`

```julia
one = simulate_panel(H = 1, B = 4, T = 400, gamma = 0.8, seed = 2)
DHRTests(one.X[1]; mode = :single)
```

```
State dependence in a single purchase history
====================================================================
history      : 400 occasions (399 used), 4 of 4 brands bought
raw repeat   : 0.393 of consecutive occasions repeat the brand
model        : fixed-effect conditional logit (no heterogeneity to
               confound: alpha is a constant for one household)
--------------------------------------------------------------------
gamma                        0.625   se 0.104
  Wald    95% CI          [  0.422,   0.828] *
  profile 95% CI          [  0.420,   0.827] *   <- prefer this one
  odds ratio exp(gamma)      1.868   [  1.522,   2.286]
  LR vs gamma=0              34.24   p = 0.0000  (chi2, 1 df)
--------------------------------------------------------------------
window shuffle (W = 25)   p = 0.0020   null mean +0.066 [-0.169, +0.276] *
global shuffle            p = 0.0020   null mean -0.016 [-0.244, +0.179]
--------------------------------------------------------------------
stationarity of alpha     p = 0.4540
  (brand shares across stretches of the sequence, vs a bootstrap
   null holding alpha constant at gamma = gamma-hat)
   3 blocks of 133          p = 0.3540   chi2     9.5
   6 blocks of  66          p = 0.2420   chi2    26.5
  12 blocks of  33          p = 0.3760   chi2    49.6
  24 blocks of  16          p = 0.6920   chi2    87.3
====================================================================
verdict: state_dependence
         state dependence: the effect survives shuffling inside short windows,
         so it is not slow drift in this household's tastes
```

With a single household there is nothing left for cross-sectional heterogeneity
to hide in — `alpha` is a constant — so the whole hierarchical apparatus is
unnecessary. `mode = :single` fits a **fixed-effect conditional logit** by
Newton's method, tests `gamma = 0` by likelihood ratio, and reports a
**profile-likelihood interval**. It returns a `DHRSingleResult`.

**The confound changes form, it does not disappear.** What replaces heterogeneity
is the household's own drift — tastes that move over months, a changing
assortment, seasonality. A household that goes through phases produces runs that
look exactly like inertia, and the global order shuffle used in panel mode
cannot separate them: it destroys drift and state dependence together. So
`mode = :single` shuffles occasions **inside short windows** instead. Over a
window of ~20 occasions drift is approximately constant, so a window shuffle is
a null for "no state dependence, preferences locally constant".

Measured on simulated single households (B = 4, T = 300, 120 replications, 5%
level, rejection rates):

| scenario | LR test | global shuffle | **window shuffle** |
|---|---:|---:|---:|
| no SD, no drift | 6.7% | 5.8% | **4.2%** |
| no SD, mild drift | 13.3% | 12.5% | **3.3%** |
| no SD, **strong drift** | 67.2% | 73.1% | **10.1%** |
| SD `gamma = 0.8`, no drift | 96.7% | 97.5% | **95.8%** |
| SD `gamma = 0.8`, mild drift | 94.1% | 94.1% | **92.4%** |

Both classical tests call two thirds of drifting households state dependent. The
window null keeps the size near nominal and gives up essentially no power. Both
p-values are reported, and the contrast between them is itself the diagnostic:

| `p_global` | `p_window` | verdict |
|---|---|---|
| small | small | `:state_dependence` |
| small | large | `:nonstationarity` — serial structure at the drift scale only |
| large | large | `:no_evidence` |

### Checking the assumption instead of making it: `p_stationarity`

The window null *assumes* tastes are constant inside a window. It does not check
it, and when tastes move faster than the window it fails badly — with tastes
stepping to a new phase every 10 occasions, the window null calls 98% of
`gamma = 0` households state dependent, and no choice of window length fixes it.

So `:single` mode now runs a **stationarity pre-test first**, following Bass,
Givon, Kalwani, Reibstein & Wright (1984): test stationarity, and test the order
of the process only on what passes. A runs-type statistic cannot tell drifting
tastes from inertia — both shorten the runs — so it is the *order* of the two
tests that does the work.

The statistic that separates them is the **marginal distribution**. With `alpha`
fixed the chain is stationary whatever `gamma` is, so brand shares are the same
in every stretch of the sequence; if `alpha` moves, they move. That is the only
observable difference between the two.

- **statistic** — Pearson chi-square for homogeneity of brand shares across
  contiguous blocks of the sequence.
- **null** — *not* the chi-square distribution. State dependence correlates the
  occasions inside a block, so the asymptotic reference over-rejects whenever
  `gamma > 0`. Instead the null is a parametric bootstrap from the fitted model
  itself: `alpha` constant, `gamma` at its estimate.

| `gamma`, `alpha` genuinely constant | bootstrap null | asymptotic chi-square |
|---:|---:|---:|
| 0.0 | 6.0% | 6.0% |
| 0.8 | 7.0% | **31.0%** |
| 1.5 | 5.0% | **63.5%** |

*(T = 400, B = 4, 200 replications, 5% level. Both columns should read 5%.)*

Power depends on matching the block length to the speed of the change, and the
two ends are nearly blind to each other — so several scales are scanned
(`[3, 6, 12, 24]` blocks by default) and the smallest p-value is calibrated
against the same bootstrap draws. The size holds at every scale, so the scan
costs nothing.

What the gate buys, end to end (T = 400, W = 20, 60 replications, share of
households called `:state_dependence`):

| truth | window null alone | **with the pre-test** |
|---|---:|---:|
| `gamma = 0`, tastes step every 40 occasions | 18.3% | **0.0%** |
| `gamma = 0`, tastes step every 10 occasions | 98.3% | **5.0%** |
| `gamma = 0`, random-walk drift `sd = 0.10` | 10.0% | **0.0%** |
| `gamma = 0.8`, `alpha` constant | 91.7% | **90.0%** |
| `gamma = 0.5`, `alpha` constant | 81.7% | **75.0%** |

A rejection makes the verdict `:nonstationarity` whatever `p_window` says,
because a moving `alpha` is exactly the case the window null cannot handle. Read
that verdict as *"`gamma` is not identified for this household"*, not as
*"there is no state dependence"* — the pre-test protects the conclusion, it does
not recover the estimate.

Here is the same report on a household with **no state dependence at all**, whose
tastes drift. The classical reading of the top half is "inertia": `gamma = 0.33`,
LR `p = 0.009`, the interval clear of zero. The bottom half is what stops it.

```
gamma                        0.330   se 0.123
  profile 95% CI          [  0.085,   0.567] *   <- prefer this one
  LR vs gamma=0               6.90   p = 0.0086  (chi2, 1 df)
--------------------------------------------------------------------
window shuffle (W = 25)   p = 0.4520   null mean +0.305 [+0.076, +0.519]
global shuffle            p = 0.0080   null mean -0.015 [-0.285, +0.237]
--------------------------------------------------------------------
stationarity of alpha     p = 0.0020 *
   3 blocks of 133          p = 0.0020   chi2    60.6
   6 blocks of  66          p = 0.0020   chi2    79.5
  12 blocks of  33          p = 0.0020   chi2   106.0
  24 blocks of  16          p = 0.0020   chi2   155.2
  every scale rejects, so the drift is not confined to one time scale
====================================================================
verdict: nonstationarity
         tastes moved: this household's brand shares are not the same across
         the sequence, so alpha is not a constant and the window null's
         assumption fails. gamma is not identified here -- read it as a
         description, not as evidence of state dependence
```

Note the window null's mean sitting at `+0.305`: the drift-induced clustering
survives a within-window shuffle, so the bar the estimate is judged against rises
to meet it, and `p_window = 0.45`. The pre-test then says why.

```julia
r = DHRTests(y; mode = :single)
r.p_stationarity                 # small => tastes moved, gamma not identified
r.n_blocks, r.p_blocks           # which time scale it broke on

stationarity_test(y)             # standalone, same test
stationarity_test(y; n_blocks = 12)      # fix the scale yourself
stationarity_test(y; n_blocks = [4, 8])  # or the grid
```

The knobs, all optional:

| keyword | default | meaning |
|---|---|---|
| `n_blocks` | `[3, 6, 12, 24]`, trimmed to blocks of 15+ occasions | block scales to scan |
| `nperm` (on `DHRTests`) | `499` | bootstrap draws, shared with the permutation nulls |
| `nboot` (on `stationarity_test`) | `499` | bootstrap draws |
| `seed` | `20260826` | RNG seed |

**How long does the history have to be?** With B = 4, at the 5% level:

| occasions | size at `gamma = 0` | power at `gamma = 0.5` | power at `gamma = 1.0` | bias in `gamma` |
|---:|---:|---:|---:|---:|
| 25 | 6.6% | 10.6% | 36.0% | −1.15 |
| 50 | 5.3% | 25.1% | 72.8% | −0.38 |
| 100 | 5.8% | 49.5% | 90.5% | −0.31 |
| 200 | 5.0% | 79.7% | 98.8% | −0.02 |
| 400 | 3.2% | 95.0% | 99.5% | −0.07 |
| 1000 | 5.5% | 99.0% | 99.5% | −0.00 |

About 200 occasions is the practical floor. Note the bias runs the *other* way
from the panel case: fixed effects with a lagged dependent variable bias `gamma`
downwards (Nickell), and it dies off as `1/T`. Below 50 occasions the estimate is
not worth reading; the report says so.

Two things the numbers cannot fix. Households with 200+ occasions in one category
are extreme heavy buyers, so **the result is selected**. And it answers "is *this*
household inertial", not "are consumers inertial".

```julia
r = DHRTests(one.X[1]; mode = :single, window = 20, nperm = 999)
r.ci_profile                      # prefer this over r.ci_wald
r.p_window, r.p_global
sampling_distribution(r)          # Normal(gamma, se), composes with LiftBijector
null_distribution(r, :window)     # the permutation distribution itself
```

## API

| function | purpose |
|---|---|
| `DHRTests(X; ...)` | the panel test; returns a `DHRTestResult` |
| `DHRTests(X; mode = :single)` | one long history; returns a `DHRSingleResult` |
| `simulate_panel(; ...)` | dummy panel with a known `gamma` |
| `dummy_data(; ...)` | just the input matrices |
| `build_panel(X; ...)` | matrices or sequences → `PurchasePanel` |
| `choice_matrix(p, h)` | the `B × T` indicator matrix a sequence was read as |
| `shuffle_panel(p, rng)` | the order-shuffled placebo panel |
| `window_shuffle(y, w, rng)` | the within-window order null for one sequence |
| `stationarity_test(y)` | is this household's `alpha` constant? run before trusting `p_window` |
| `sampling_distribution(r)` | `Normal(gamma, se)` for a single-household result |
| `null_distribution(r, :window)` | the permutation distribution of `gamma` |
| `lagged_repeat_rate(p)` | descriptive repeat share |
| `fit_hbmnl(panel; ...)` | the sampler on its own |
| `summarize(res)` | flat `NamedTuple` of headline numbers |
| `gamma_draws(res)` | pooled posterior draws |
| `posterior(res, :gamma)` | any reported quantity as a `Distribution` |
| `effective_size(d)` | ESS behind a `PosteriorSample` |
| `LiftBijector(share)` | `gamma` → choice-probability lift, invertible |
| `lift_distribution(res, share)` | the same as a `Distribution` (needs Bijectors.jl) |
| `lift_share(res)` | implied brand shares |
| `split_rhat`, `ess` | MCMC diagnostics |

Main keyword arguments of `DHRTests`:

| keyword | default | meaning |
|---|---|---|
| `K` | `2` | mixture components in the heterogeneity distribution |
| `R`, `burnin`, `thin` | `12000, 4000, 5` | MCMC sweeps, burn-in, thinning |
| `nchains` | `2` | independent chains (gives split-Rhat) |
| `level` | `0.95` | credible-interval level |
| `placebo` | `true` | fit the order-shuffled placebo |
| `compare_null` | `true` | also fit the nested model without `gamma` |
| `covariates` | `nothing` | `Vector` of `B × L × T_h` arrays (price, display, …) |
| `tie_rule` | `:argmax` | how to read a column with several positive rows |
| `seed` | `20260826` | RNG seed |

With covariates, `covariates[h][j, l, t]` is variable `l` for brand `j` on occasion `t`
of household `h`, aligned with the columns of `X[h]` **before** any column is dropped.

## Reading the output

* `gamma` — log-odds effect of having bought the brand on the previous occasion.
* `odds ratio` — `exp(gamma)`; the multiplicative effect on the choice odds.
* `choice prob. lift` — share-weighted change in choice probability, in percentage
  points. The number to quote to a non-technical audience.
* `EXCESS` — the part that heterogeneity alone cannot reproduce. **Read this one.**
* `DIC` — negative delta favours the state-dependence model.
* The Newton-Raftery log marginal likelihood is printed for completeness. It is a
  harmonic-mean estimator: a single bad draw dominates it. Never read it alone.

Always check `Rhat` and `ESS` before believing a result. A warning is printed when
`Rhat > 1.01`; raise `R` or `nchains` if you see it.

**Known small-sample behaviour.** With a dozen or so occasions per household the
estimator sits a little above the truth — around `+0.1` on `gamma` for a
350-household, 4-brand, 14-occasion panel. This is ordinary incidental-parameter
bias, and it is the second reason the placebo is there: a positive point estimate is
not by itself evidence, only an excess over the shuffled panel is. Do not read the
last digit of `gamma` as a structural quantity.

### What the odds ratio actually says

```
odds ratio exp(gamma)      2.214   95% CI [  2.023,   2.423]
```

reads as: **having bought the brand on the previous occasion multiplies its choice
odds by 2.21.** Odds, not probability.

Since it is `exp(gamma)`, this line is just the `gamma` line transformed —
`gamma = log(2.214) = 0.795`, and the interval is the `gamma` interval
`[0.705, 0.885]` exponentiated end for end, because the map is monotone. What
matters is that **the interval does not straddle 1**: `1` is `gamma = 0`, no
effect.

Precisely, for the same household with the same tastes, comparing an occasion
where brand `j` was bought last time against one where it was not,

```
P(j) / P(anything else)   is multiplied by 2.21
```

and the same factor applies against any single rival brand, since a multinomial
logit preserves those ratios.

**Converting to probability depends on the brand's base share.** The same odds
ratio of 2.21 works out very differently across the size distribution:

| base share | probability after buying it last time | lift | ratio |
|---:|---:|---:|---:|
| 5% | 10.4% | +5.4 pt | 2.09× |
| 10% | 19.7% | +9.7 pt | 1.97× |
| 20% | 35.6% | +15.6 pt | 1.78× |
| 30% | 48.7% | +18.7 pt | 1.62× |
| 40% | 59.6% | **+19.6 pt** | 1.49× |
| 50% | 68.9% | +18.9 pt | 1.38× |
| 70% | 83.8% | +13.8 pt | 1.20× |

Small brands get the biggest *multiple*, mid-sized brands (around a 40% share)
the biggest *lift in percentage points*. The `choice prob. lift` line in the
report is exactly this lift, averaged over brands and weighted by share. Read at
the ends of the interval, a 25%-share brand gains between +15.3 and +19.7 points.

**Do not read this line on its own.** An odds ratio above 1 is not evidence of
state dependence: leftover heterogeneity and the short-panel bias push it above 1
even when the truth is `gamma = 0`. The line to judge by is `EXCESS`. If the
placebo's own odds ratio sits near 1.0, then nearly all of the 2.21 is real.

You can also take it as a distribution:

```julia
posterior(res, :odds_ratio)       # the posterior of exp(gamma)
lift_distribution(res, 0.25)      # probability lift for a 25%-share brand (needs Bijectors)
```

## Posteriors, not just intervals

Every reported quantity is also available as a `Distribution`, so you can sample
from it and carry the uncertainty downstream instead of re-deriving it from a
point estimate and an interval. **Which function you call depends on which
result you have** — the two modes estimate different things, so they hand back
different objects:

| result | function | what it is |
|---|---|---|
| `DHRTestResult` (panel) | `posterior(res, :gamma)` | a genuine posterior, the MCMC draws |
| `DHRSingleResult` (`mode = :single`) | `sampling_distribution(r)` | `Normal(gamma, se)`, the asymptotic sampling distribution |
| `DHRSingleResult` | `null_distribution(r, :window)` | the permutation null, as draws |
| `DHRSingleResult` | `null_distribution(r, :stationarity)` | the bootstrap null of the block chi-square |

`posterior` is not defined for a `DHRSingleResult` (it would be a lie — there is
no posterior there, the single-household estimator is maximum likelihood), and
`sampling_distribution` is not defined for a `DHRTestResult`.

### From a panel result

```julia
using Distributions

d = posterior(res, :gamma)      # PosteriorSample <: ContinuousUnivariateDistribution
rand(d, 10_000)                 # resample the actual draws
quantile(d, 0.975)              # identical to res.gamma_ci[2]
cdf(d, 0.0)                     # posterior mass below zero
```

`PosteriorSample` holds the MCMC draws themselves — no parametric assumption.
`pdf`/`logpdf` come from a Gaussian kernel density estimate, which costs
`O(ndraws)` per call and is meant for plotting. **Inside a sampler, fit a family
first**; `fit` forwards to `Distributions.fit`, so any univariate family works:

```julia
n  = fit(Normal, posterior(res, :gamma))          # Normal(0.740, 0.042)
ln = fit(LogNormal, posterior(res, :odds_ratio))
```

The `Normal` fit is a good approximation in practice — on a 350-household panel
the posterior of `gamma` had skewness `+0.03` and excess kurtosis `+0.01`, and
the fitted normal matched the empirical quantiles to within `0.15` standard
deviations out to the 0.5% and 99.5% points.

| `posterior(res, …)` | quantity |
|---|---|
| `:gamma` | the state-dependence coefficient |
| `:placebo` | the same coefficient on the order-shuffled panel |
| `:excess` | `gamma - gamma_placebo` — what the verdict is based on |
| `:odds_ratio` | `exp(gamma)` |
| `:lift` | share-weighted change in choice probability, percentage points |
| `:beta, l` | covariate coefficient `l` |

`gamma` and `gamma_placebo` come from independent posteriors, so `:excess` pairs
draws at random — that is a draw from the posterior of the difference.

**Effective sample size.** MCMC draws are autocorrelated. `effective_size(d)`
reports the ESS; on the defaults a chain of 3200 draws typically carries an ESS
near 570. Resampling 10,000 values from it does not give 10,000 independent
samples — it is fine for propagating uncertainty, but do not quote it as a
sample size.

### From a single-household result

`mode = :single` is maximum likelihood, not MCMC, so what comes back is the
asymptotic sampling distribution of the estimate:

```julia
r = DHRTests(y; mode = :single, nperm = 999)

d = sampling_distribution(r)             # Normal(0.1098, 0.2850)
rand(d, 10_000)
quantile(d, 0.025), quantile(d, 0.975)   # (-0.449, 0.668)
ccdf(d, 0.0)                             # P(gamma > 0) = 0.65
logpdf(d, 0.0)                           # analytic, cheap enough for a sampler
```

**It is a sampling distribution, not a posterior**, and it is symmetric by
construction. Check it against `r.ci_profile` before leaning on the tails — in
the example above the profile interval is `[-0.461, 0.664]` against the Wald
`[-0.449, 0.668]`, so nothing is lost, but that is not guaranteed in a short
history.

The permutation nulls come back as `PosteriorSample`, the same empirical type
the panel mode uses:

```julia
nw = null_distribution(r, :window)   # PosteriorSample(:null_window, n=999, mean=-0.173, 95%=[-0.713, 0.350])
ng = null_distribution(r, :global)   # PosteriorSample(:null_global, n=999, mean=-0.051, 95%=[-0.713, 0.508])

rand(nw, 1000)
r.null_window                        # the raw Vector{Float64}, if you prefer
```

Derived quantities are easiest by transforming draws:

```julia
or = exp.(rand(sampling_distribution(r), 100_000))     # odds ratio
mean(or), quantile(or, [0.025, 0.975])                 # 1.162, [0.638, 1.944]
```

### With Bijectors.jl

Bijectors is a **weak dependency**: the core package stays on `Distributions`
alone, and loading Bijectors lights up the transform machinery.

```julia
using Bijectors

transformed(posterior(res, :gamma), exp)            # empirical base, works as-is
transformed(fit(Normal, posterior(res, :gamma)), exp)   # analytic logpdf
```

For `exp` you do not really need Bijectors — `exp` of a normal is exactly
`LogNormal`. Bijectors earns its place for transforms with no named family, and
the package ships one: `LiftBijector(share)` maps `gamma` to the change in the
choice probability of a brand holding base share `share`,

$$g \;\longmapsto\; \mathrm{logistic}\!\left(g + \mathrm{logit}(s)\right) - s$$

with a closed-form inverse and log-Jacobian, so the result is a real
`Distribution`:

```julia
b = LiftBijector(0.25)                    # a brand holding a 25% share
d = lift_distribution(res, 0.25)          # == transformed(fit(Normal, γ), b)
rand(d, 1000)                             # lift in probability units
logpdf(d, 0.10)
```

The same works from a single-household result — `sampling_distribution(r)` is an
ordinary `Normal`, so `transformed(sampling_distribution(r), LiftBijector(0.30))`
composes just as well.

`LiftBijector` is a plain callable in the core package and implements
`InverseFunctions.inverse`, so it also composes outside the Bijectors ecosystem.
For the share-weighted aggregate across all brands there is no closed-form
inverse — use the exact draws, `posterior(res, :lift)`. Shares are available
through `lift_share(res)`.

## Cost

Roughly linear in households × occasions × brands × sweeps. A 400-household,
4-brand, 12-occasion panel at the defaults takes about a minute for all three fits
on one core. `nchains` runs on threads: start Julia with `-t auto`.

## Tests

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite covers input parsing, the placebo construction, the diagnostics, and two
statistical checks that matter: a panel simulated with `gamma = 1.0` must give an
interval covering `1.0` and a verdict of `:state_dependence`, and a panel simulated
with `gamma = 0.0` — heterogeneity only — must give an excess interval covering zero
and a verdict of `:no_evidence`.

## Reference

Dubé, J.-P., Hitsch, G. J., & Rossi, P. E. (2010). State dependence and alternative
explanations for consumer inertia. *The RAND Journal of Economics*, 41(3), 417–445.

---

# 日本語

**連続購買の状態依存性**（前回買ったブランドを次も選びやすいか）を統計的に判定する
Julia パッケージです。[Dubé, Hitsch & Rossi (2010)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1756-2171.2010.00106.x) に依拠しています。

入力は世帯ごとの **ブランド（行）× 購買機会（列）の数量マトリクス** 1 枚だけ。
`DHRTests(X)` を呼ぶと、状態依存係数 γ と信用区間、順序シャッフルのプラセボ、
そして判定が返ります。

**1 世帯の長い履歴**を使うモード（[`mode = :single`](#パネルではなく-1-世帯の長い履歴を使う-mode--single)）
もあります。固定効果条件付きロジット ＋ 窓内シャッフル null に加えて、
[定常性の事前検定](#仮定を置くのではなく検定する-p_stationarity)が入っており、
**嗜好が動いているだけの世帯を「慣性あり」と判定しない**ようになっています。

## 使い方

```julia
using StatesDependency

# ダミーデータ（真の γ = 0.8）
sim = simulate_panel(H = 400, B = 4, T = 12, gamma = 0.8, seed = 1)
res = DHRTests(sim.X)

res.gamma_mean      # γ の事後平均
res.gamma_ci        # 95% 信用区間
res.excess_ci       # γ − プラセボ の区間 ← これを読む
res.verdict         # :state_dependence / :inconclusive / :no_evidence
```

実データなら、世帯ごとに `B × T` 行列を作って `Vector` に入れるだけです。

```julia
X = [hh1, hh2, hh3, ...]      # 各要素は B × T_h の Matrix
res = DHRTests(X; brand_names = ["A", "B", "C", "D"])
```

## 入力の約束

* **行はブランド**。全世帯で同じ順序にすること。
* **列は購買機会で、時系列順**。「直前の列に何が入っていたか」が検定の本体なので、
  列の順序そのものがデータです。
* **セルは数量**。全ゼロ列は「買わなかった機会」として落とします。
  1 列に複数のブランドが立っている場合は `tie_rule`（既定 `:argmax`）で解決します。
* 購買機会が 2 回未満の世帯は落とします。また **各世帯の最初の機会は推定に使いません**
  （ラグが観測されないため、初期条件は所与とする。DHR と同じ扱い）。

### 選択系列をそのまま渡す

行列を組まずに、機会ごとの選択そのものを渡せます。指示行列は内部で作ります。

```julia
build_panel(["A", "B", "A"])       # -> [1 0 1
                                   #     0 1 0]
build_panel([2, 2, 1, 1, 2])       # -> [0 0 1 1 0
                                   #     1 1 0 0 1]
```

* **ラベル** … 文字列・シンボル・文字・`CategoricalVector`。行はラベルのソート順、
  カテゴリカルなら**宣言した水準順**（`categorical(x; levels = ["B","A"])` なら B が先。
  `ordered = true` も可）。CategoricalArrays.jl への依存はありません（`sort`/`unique`
  だけで動きます）。
* **コード** … 1 以上の整数。`2` はそのまま 2 行目です。
* **その Vector** … 1 要素 1 世帯のパネルになります。水準は全世帯でプールするので、
  行の順序は全員共通です。
* 系列中の `missing` は「買わなかった機会」。全ゼロ列になって落とされます。

`brand_names` を渡せば水準集合と順序を固定できます。
`choice_matrix(build_panel(X))` でどう読まれたか確認できます。

**誰も買っていないブランド**があると警告が出ます（カテゴリカルの未使用水準で起きがち）。
その切片は尤度では識別されず事前分布だけで決まるためです。`drop_unused = true` で除去できます。

## 何を推定しているか

異質性を柔軟にした階層ベイズ多項ロジットです。

購買機会ごとに、世帯 $h$ がブランド $j$ に与える**効用**（そのブランドの選ばれ
やすさを表すスコア）を次の形で置きます。ここにランダムな誤差が乗り、合計が
一番大きくなったブランドが選ばれる、と考えます。誤差に Gumbel 分布を仮定すると、
そのままロジットの選択確率 $e^{v_j} / \sum_k e^{v_k}$ になります。

$$v_{hjt} = \alpha_{hj} + \gamma \cdot 1\{j = y_{h,t-1}\} + x_{hjt}'\beta, \qquad
\alpha_h \sim \sum_k \pi_k N(\mu_k, \Sigma_k)$$

* $\alpha_{hj}$ … 世帯 $h$ のブランド $j$ に対する固定された選好。**ここがゼロ次の
  部分**です。有限正規混合から引くので、異質性の形が単峰に縛られません
* $\gamma$ … **状態依存係数**。この項を取り除くと、残るのはブランドごとに動かない
  スコアだけになり、モデルはゼロ次に戻ります
* $x_{hjt}'\beta$ … 任意の共変量（価格、山積みなど）

**効用は差だけが意味を持ちます。** 全ブランドに同じ数を足しても選択確率は変わらない
ので、基準ブランドを $\alpha_{hB} \equiv 0$ に固定します。尺度が対数オッズになるのも
これが理由で、$\gamma = 0.8$ は「対数オッズが 0.8 高い」という意味です。`exp(gamma)`
をオッズ比として報告しているのもここから来ています。

推定は Gibbs ＋ ランダムウォーク Metropolis（世帯切片、$\gamma$、$\beta$ のブロック）と、
混合分布に対する正規逆ウィシャート／ディリクレ更新です。事前分布は DHR と同じ
（`Dir(0.5/K)`、`μ_k|Σ_k ~ N(0, 16Σ_k)`、`Σ_k ~ IW(p+3, (p+3)I)`）。

**γ は意図的に世帯共通** にしています。1 世帯あたりの購買機会が数回しかない
パネルでは γ の異質性まで識別する余力がなく、世帯別 γ はホールドアウト適合を
むしろ悪化させます。

## プラセボが検定の本体

素の反復率が高いことは何の証拠にもなりません。異質性の強いゼロ次
Dirichlet 過程でも同じくらい高くなります。γ̂ が正であることも、短いパネルでは
推定量自体が正に偏るので、それだけでは足りません。

そこで `DHRTests` は、**世帯内で購買機会の順序だけをランダムに入れ替えた**パネルに
同じモデルを当てます。断面の構成は完全に同一で、真の状態依存はゼロです。読むべき
統計量は

```
excess = γ − γ_placebo
```

で、この区間が 0 を跨がないことを判定の必要条件にしています。
実際、手元のビールパネルの再現では γ = 0.65 に対してプラセボが 0.11 でした。

判定が `:state_dependence` になるのは、γ の区間が 0 を除き、excess の区間も 0 を除き、
かつ DIC が状態依存モデルを選んだときだけです。

## 出力の読み方

* `gamma` … 前回購買がもたらす対数オッズの押し上げ
* `odds ratio` … `exp(γ)`。選択オッズが何倍になるか
* `choice prob. lift` … 選択確率の変化（シェア加重、%pt）。非専門家に出す数字
* `EXCESS` … 異質性だけでは再現できない部分。**ここを読む**
* `DIC` … 負なら状態依存モデルが優位
* Newton-Raftery の対数周辺尤度は参考値です。調和平均推定量なので単一の悪い draw に
  支配されます。**単独で読まないこと**

`Rhat` と `ESS` は必ず確認してください。`Rhat > 1.01` のときは警告が出ます。

### オッズ比の読み方

```
odds ratio exp(gamma)      2.214   95% CI [  2.023,   2.423]
```

これは「**前回そのブランドを買っていると、そのブランドの選択オッズが 2.21 倍になる**」
という意味です。確率が 2.21 倍ではなく、**オッズ**が 2.21 倍です。

`exp(γ)` なので、この行は `gamma` の行を変換しただけのものです。
γ = log(2.214) = **0.795**、CI は γ の区間 `[0.705, 0.885]` をそのまま指数変換した
ものです（単調変換なので端点が対応します）。**区間が 1 を跨いでいない**のが要点で、
1 は γ = 0＝効果なしに当たります。

定義を正確に言うと、同じ世帯・同じ嗜好のもとで、ブランド j を前回買った場合と
買わなかった場合を比べて

```
P(j) / P(他のどれか)  が  2.21 倍になる
```

です。分母が「他の特定のブランド k」でも同じ倍率になります（多項ロジットなので
比が保たれる）。

**確率に直すとシェア依存です。** 同じオッズ比 2.21 でも、ベースシェアによって
効き方がまったく違います。

| ベースシェア | 前回買った後の確率 | 上昇幅 | 倍率 |
|---:|---:|---:|---:|
| 5% | 10.4% | +5.4pt | 2.09倍 |
| 10% | 19.7% | +9.7pt | 1.97倍 |
| 20% | 35.6% | +15.6pt | 1.78倍 |
| 30% | 48.7% | +18.7pt | 1.62倍 |
| 40% | 59.6% | **+19.6pt** | 1.49倍 |
| 50% | 68.9% | +18.9pt | 1.38倍 |
| 70% | 83.8% | +13.8pt | 1.20倍 |

**小ブランドほど「倍率」が大きく、中位ブランド（シェア 40% 付近）ほど「上昇幅」が
大きい**という形です。レポートの `choice prob. lift` 行は、この上昇幅をシェアで
加重平均したものです。CI の端で見ると、シェア 25% のブランドなら +15.3〜+19.7pt の
幅になります。

**この数字を単独で読まないでください。** オッズ比が 1 より大きいことは「状態依存が
ある」証拠にはなりません。異質性の吸収し残しと短パネルの上方バイアスで、真の γ が 0 でも
正に出ます。判断材料は `EXCESS` の行です。プラセボ側のオッズ比が 1.0 付近なら、
この 2.21 倍のうちほぼ全部が本物ということになります。

分布として取り出すこともできます:

```julia
posterior(res, :odds_ratio)       # exp(γ) の事後
lift_distribution(res, 0.25)      # シェア 25% のブランドの確率上昇（要 Bijectors）
```

## パネルではなく 1 世帯の長い履歴を使う: `mode = :single`

```julia
one = simulate_panel(H = 1, B = 4, T = 400, gamma = 0.8, seed = 2)
DHRTests(one.X[1]; mode = :single)
```

```
State dependence in a single purchase history
====================================================================
history      : 400 occasions (399 used), 4 of 4 brands bought
raw repeat   : 0.393 of consecutive occasions repeat the brand
model        : fixed-effect conditional logit (no heterogeneity to
               confound: alpha is a constant for one household)
--------------------------------------------------------------------
gamma                        0.625   se 0.104
  Wald    95% CI          [  0.422,   0.828] *
  profile 95% CI          [  0.420,   0.827] *   <- prefer this one
  odds ratio exp(gamma)      1.868   [  1.522,   2.286]
  LR vs gamma=0              34.24   p = 0.0000  (chi2, 1 df)
--------------------------------------------------------------------
window shuffle (W = 25)   p = 0.0020   null mean +0.066 [-0.169, +0.276] *
global shuffle            p = 0.0020   null mean -0.016 [-0.244, +0.179]
--------------------------------------------------------------------
stationarity of alpha     p = 0.4540
  (brand shares across stretches of the sequence, vs a bootstrap
   null holding alpha constant at gamma = gamma-hat)
   3 blocks of 133          p = 0.3540   chi2     9.5
   6 blocks of  66          p = 0.2420   chi2    26.5
  12 blocks of  33          p = 0.3760   chi2    49.6
  24 blocks of  16          p = 0.6920   chi2    87.3
====================================================================
verdict: state_dependence
```

世帯が 1 つなら、世帯間の観測されない異質性が隠れる場所はありません（α は定数）。
階層モデルは不要になるので、`mode = :single` は **固定効果条件付きロジット**を
Newton 法で当て、尤度比で `gamma = 0` を検定し、**プロファイル尤度区間**を返します
（戻り値は `DHRSingleResult`）。

**交絡は消えるのではなく形を変えます。** 異質性の代わりに来るのが、その世帯自身の
**嗜好変化**（数か月単位の好みの変化、配架の変化、季節性）です。時期によって
好みが動く世帯は、慣性とまったく同じ見た目の連続を作ります。パネル版で使っている
全体シャッフルではこれを分離できません（嗜好変化も状態依存もまとめて壊すので）。
そこで `mode = :single` は **短い窓の中だけでシャッフル**します。20 機会程度の窓の
中なら嗜好変化はほぼ一定なので、**「状態依存なし・嗜好は局所的に一定」という帰無仮説の
もとでの γ の分布**をこれで作れます。

以下この分布を **窓内 null**（窓内シャッフルの帰無分布）と短く呼びます。実体は
シャッフルした系列に同じ推定を当てて得た γ を `nperm` 個ならべたもので、
`r.null_window` にそのまま入っています。

ダミーの単一世帯での実測（B = 4、T = 300、120 反復、5% 水準、棄却率）:

| シナリオ | LR 検定 | 全体シャッフル | **窓内シャッフル** |
|---|---:|---:|---:|
| 状態依存なし・嗜好変化なし | 6.7% | 5.8% | **4.2%** |
| 状態依存なし・弱い嗜好変化 | 13.3% | 12.5% | **3.3%** |
| 状態依存なし・**強い嗜好変化** | 67.2% | 73.1% | **10.1%** |
| 状態依存 `gamma = 0.8`・嗜好変化なし | 96.7% | 97.5% | **95.8%** |
| 状態依存 `gamma = 0.8`・弱い嗜好変化 | 94.1% | 94.1% | **92.4%** |

古典的な 2 つの検定は、嗜好が変化しているだけの世帯の 3 分の 2 を「状態依存あり」と
判定します。窓内 null はサイズをほぼ名目通りに保ちつつ、検出力をほとんど落としません。
両方の p 値を出力しており、**その対比自体が診断になります**:

| `p_global` | `p_window` | 判定 |
|---|---|---|
| 小 | 小 | `:state_dependence` |
| 小 | 大 | `:nonstationarity` — 系列構造は嗜好変化の尺度にしかない |
| 大 | 大 | `:no_evidence` |

### 仮定を置くのではなく検定する: `p_stationarity`

窓内 null は「窓の中で嗜好は一定」を**仮定**しています。検定はしていません。
そして嗜好が窓より速く動くと大きく破綻します。10 機会ごとに嗜好が切り替わる世帯では、
`gamma = 0` にもかかわらず窓内 null が 98% を「状態依存あり」と判定し、
**窓長をどう選んでも直りません**。

そこで `:single` モードは**定常性の事前検定**を先に走らせます。Bass, Givon, Kalwani,
Reibstein & Wright (1984) の手続き ── まず定常性を検定し、通った系列についてのみ
次数を検定する ── に従います。ランズ型の統計量は嗜好変化と慣性を区別できません
（どちらもランを短くする）。効いているのは**2 つの検定の順序**であって、個々の検定
ではありません。

両者を分ける観測量は**周辺分布**です。α が固定なら `gamma` がいくつであろうと連鎖は
定常なので、系列のどの区間を取ってもブランドシェアは同じになります。α が動けば動く。
**これが両者の唯一の観測可能な差**です。

- **統計量** ── 系列を連続したブロックに分け、ブロック間のブランドシェアの均質性を
  見る Pearson カイ二乗。
- **帰無分布** ── カイ二乗分布では**ありません**。状態依存はブロック内の機会を相関
  させるので、`gamma > 0` だと漸近分布は過剰棄却します。代わりに、推定済みモデル
  自体からのパラメトリックブートストラップ（α 一定、`gamma` は推定値）を使います。

| α は本当に一定、`gamma` = | ブートストラップ帰無分布 | 漸近カイ二乗 |
|---:|---:|---:|
| 0.0 | 6.0% | 6.0% |
| 0.8 | 7.0% | **31.0%** |
| 1.5 | 5.0% | **63.5%** |

*(T = 400、B = 4、200 反復、5% 水準。どちらの列も 5% になるべき場面です。)*

検出力はブロック長を嗜好変化の速さに合わせられるかで決まり、**速い変化と遅い変化は
互いにほぼ盲目**です。そこで複数の尺度（既定は `[3, 6, 12, 24]` ブロック）を走査し、
最小 p 値を同じブートストラップ標本で較正します。サイズはどの尺度でも保たれるので、
走査のコストはありません。

ゲートの効果（T = 400、W = 20、60 反復、`:state_dependence` と判定された割合）:

| 真の姿 | 窓内 null のみ | **事前検定あり** |
|---|---:|---:|
| `gamma = 0`、40 機会ごとに嗜好が切替 | 18.3% | **0.0%** |
| `gamma = 0`、10 機会ごとに嗜好が切替 | 98.3% | **5.0%** |
| `gamma = 0`、ランダムウォーク `sd = 0.10` | 10.0% | **0.0%** |
| `gamma = 0.8`、α 一定 | 91.7% | **90.0%** |
| `gamma = 0.5`、α 一定 | 81.7% | **75.0%** |

棄却されたら、`p_window` が何を言おうと判定は `:nonstationarity` になります。
α が動いている状況こそ窓内 null が扱えない場面だからです。この判定は
**「この世帯では `gamma` が識別されていない」**と読んでください。
**「状態依存がない」ではありません** ── 事前検定は結論を守るだけで、推定値を
救ってはくれません。

同じレポートを、**状態依存が一切ない**のに嗜好が動いている世帯に当てたものです。
上半分だけを古典的に読めば「慣性あり」になります（γ = 0.33、LR の p = 0.009、
区間は 0 を跨がない）。それを止めるのが下半分です。

```
gamma                        0.330   se 0.123
  profile 95% CI          [  0.085,   0.567] *   <- prefer this one
  LR vs gamma=0               6.90   p = 0.0086  (chi2, 1 df)
--------------------------------------------------------------------
window shuffle (W = 25)   p = 0.4520   null mean +0.305 [+0.076, +0.519]
global shuffle            p = 0.0080   null mean -0.015 [-0.285, +0.237]
--------------------------------------------------------------------
stationarity of alpha     p = 0.0020 *
   3 blocks of 133          p = 0.0020   chi2    60.6
   6 blocks of  66          p = 0.0020   chi2    79.5
  12 blocks of  33          p = 0.0020   chi2   106.0
  24 blocks of  16          p = 0.0020   chi2   155.2
  every scale rejects, so the drift is not confined to one time scale
====================================================================
verdict: nonstationarity
```

窓内シャッフルの帰無分布の平均が **+0.305** まで上がっている点に注目してください。嗜好変化による
かたまりは窓内シャッフルでも壊れないので、推定値が judged される基準そのものが
持ち上がり、`p_window = 0.45` になります。事前検定はその理由を言い当てます。

```julia
r = DHRTests(y; mode = :single)
r.p_stationarity                 # 小さい => 嗜好が動いている。γ は識別されていない
r.n_blocks, r.p_blocks           # どの時間尺度で崩れたか

stationarity_test(y)             # 単独でも呼べる
stationarity_test(y; n_blocks = 12)      # 尺度を固定
stationarity_test(y; n_blocks = [4, 8])  # グリッドを指定
```

引数はすべて省略可能です。

| 引数 | 既定 | 意味 |
|---|---|---|
| `n_blocks` | `[3, 6, 12, 24]`（15 機会以上のブロックになるものだけ） | 走査するブロック尺度 |
| `nperm`（`DHRTests`） | `499` | ブートストラップ回数。並べ替えの帰無分布と共通 |
| `nboot`（`stationarity_test`） | `499` | ブートストラップ回数 |
| `seed` | `20260826` | 乱数シード |

### 出力の読み方

**読む順番は上からではありません。**「γ がそもそも識別できる状況か」を先に確かめ、
そのあとで大きさを読みます。実装した二段構え（Bass et al. 1984）がそのまま読み順です。

```
--------------------------------------------------------------------
gamma                        0.625   se 0.104          ③
  Wald    95% CI          [  0.422,   0.828] *
  profile 95% CI          [  0.420,   0.827] *   <- prefer this one
  odds ratio exp(gamma)      1.868   [  1.522,   2.286]
  LR vs gamma=0              34.24   p = 0.0000  (chi2, 1 df)
--------------------------------------------------------------------
window shuffle (W = 25)   p = 0.0020   null mean +0.066 ...  *      ②
global shuffle            p = 0.0020   null mean -0.016 ...
--------------------------------------------------------------------
stationarity of alpha     p = 0.4540                            ①
   3 blocks of 133          p = 0.3540   chi2     9.5
   6 blocks of  66          p = 0.2420   chi2    26.5
  12 blocks of  33          p = 0.3760   chi2    49.6
  24 blocks of  16          p = 0.6920   chi2    87.3
====================================================================
verdict: state_dependence
```

#### ① 一番下から: `stationarity of alpha`

**γ が識別できる状況かどうか**の確認です。棄却されていない（0.45 > 0.05）＝ ブランドシェアは
系列のどこを切っても同じ ＝ **α は動いていない**。窓内シャッフルが置いている
「嗜好は局所的に一定」という仮定が成り立っているので、この先を読む資格があります。
**ここが小さければ γ の値は読まずに捨てます。**

4 行のブロックは時間尺度の走査です。遅い変化は粗いブロックでしか、速い変化は細かい
ブロックでしか見えないので、両端を押さえています。

> **chi2 の列を行間で比べないでください。** 9.5 → 87.3 と増えるのは自由度が 6 → 69 に
> 増えるからで、機械的なものです（chi2/df はどれも 1.3〜1.8）。比較できるのは p の列だけ。
> 見出しの 0.4540 は 4 つの最小 p をブートストラップで較正した値で、単純な最小値ではありません。

#### ② 次に真ん中: 状態依存か、局所的な構造か

**`p = 0.0020` は「ちょうど 0.002」ではなく下限です。** `nperm = 499` のとき
`(0 + 1) / (499 + 1) = 0.002`、つまり **一度も観測値 0.625 に届かなかった**という意味です。
実際この例では 499 個の最大が `+0.380` で、0.625 との間に大きな隙間があります。
もっと細かく見たければ `nperm` を上げてください。

ここで動いているのは 0.625 ではなく比較相手のほうです。系列を窓内でシャッフルして
同じ推定を当てる、を `nperm` 回くり返し、**シャッフルしても偶然これほど大きな γ が
出てしまうことがあるか**を数えています（片側）。

**2 つの `null mean` の差が診断になります。**

| | 平均 | 何を残しているか |
|---|---:|---|
| 窓内シャッフル | **+0.066** | 窓ごとのブランド構成の違い |
| 全体シャッフル | −0.016 | 何も残さない（ほぼ 0） |

差の 0.08 が「窓の尺度に残っている局所構造の量」です。**基準は 0 ではなく 0.066 付近**で、
そこから 0.625 までの距離を見ていることになります。嗜好が動いている世帯では窓内側の
平均が +0.3 まで持ち上がり、同じ γ でも通らなくなります。

`W = 25` の既定は `clamp(T ÷ 8, 8, 25)`。窓を長くすると嗜好変化まで保存してしまい、
短くすると状態依存まで壊れて検出力が落ちます。

#### ③ 最後に上: 大きさ

前回そのブランドを買っていたことが **対数オッズを +0.625 押し上げる**。`*` は区間が
0 を跨がない印です。

**Wald と profile。** Wald は `0.625 ± 1.96 × 0.104 = [0.421, 0.829]` と定義上左右対称です。
profile は尤度比から作るので非対称になり得ます。T = 400 の今回はほぼ一致していますが、
**機会数が少ないと乖離し、そのとき正しいのは profile** です。

**オッズ比 1.868 は「確率が 1.87 倍」ではありません。**

| ベースシェア | 前回買った後 | 上昇幅 | 倍率 |
|---:|---:|---:|---:|
| 10% | 17.2% | +7.2pt | 1.72 |
| 25% | 38.4% | **+13.4pt** | 1.54 |
| 50% | 65.1% | +15.1pt | 1.30 |
| 70% | 81.3% | +11.3pt | 1.16 |

4 ブランドならベースは 25% 前後です。区間 `[1.522, 2.286]` は γ の区間を指数変換した
だけなので、**1 を跨がないことだけが情報**です。

> **`LR vs gamma=0` は単独で読まないでください。** χ²(1) の 5% 臨界値が 3.84 なので
> 34.24 は圧倒的に見えますが、この検定は嗜好が変化しているだけの世帯を 67% 誤検出します
> （上の表）。①が通っているからこそ意味を持つ数字です。

#### 判定

`verdict: state_dependence` は **3 条件が揃ったときだけ**出ます。

1. 定常性が棄却されていない（①）
2. `p_window < 1 - level`（②）
3. profile 区間が 0 を跨がない（③）

ひとつでも欠ければ `:nonstationarity` / `:inconclusive` / `:no_evidence` になります。

なお、この例の**真の γ は 0.8**（`simulate_panel(gamma = 0.8)`）で推定値は 0.625 です。
固定効果 ＋ ラグ従属変数は γ を**下方に**偏らせます（Nickell バイアス、`1/T` で消える）。
se が 0.104 なので今回の差が偏りかノイズかは断定できませんが、**方向は常に下向き**です。
パネル版とは偏りの向きが逆である点に注意してください。

**どれだけの履歴が要るか。** B = 4、5% 水準で:

| 機会数 | `gamma = 0` のサイズ | `gamma = 0.5` の検出力 | `gamma = 1.0` の検出力 | γ の偏り |
|---:|---:|---:|---:|---:|
| 25 | 6.6% | 10.6% | 36.0% | −1.15 |
| 50 | 5.3% | 25.1% | 72.8% | −0.38 |
| 100 | 5.8% | 49.5% | 90.5% | −0.31 |
| 200 | 5.0% | 79.7% | 98.8% | −0.02 |
| 400 | 3.2% | 95.0% | 99.5% | −0.07 |
| 1000 | 5.5% | 99.0% | 99.5% | −0.00 |

実用ラインは **約 200 機会**です。偏りの向きがパネル版と**逆**である点に注意してください。
固定効果＋ラグ従属変数は γ を下方に偏らせ（Nickell）、`1/T` で消えます。50 機会を切ると
点推定は読む価値がありません（レポートにも警告が出ます）。

数字で埋められない問題が 2 つ。1 カテゴリで 200 機会を超える世帯は極端なヘビーバイヤー
なので **結果には選択バイアス**がかかります。そして答えるのは「**この世帯**は慣性的か」
であって「消費者は慣性的か」ではありません。

```julia
r = DHRTests(one.X[1]; mode = :single, window = 20, nperm = 999)
r.ci_profile                      # r.ci_wald よりこちらを読む
r.p_window, r.p_global
sampling_distribution(r)          # Normal(gamma, se)。LiftBijector と合成できる
null_distribution(r, :window)     # 帰無分布そのもの
```

## 区間ではなく分布として取り出す

報告される量はすべて `Distribution` として取り出せます。点推定と区間から組み直す
のではなく、そのまま `rand` して下流に不確実性を流せます。ただし
**どちらのモードの結果かで呼ぶ関数が変わります**。2 つのモードは別のものを推定して
いるので、返ってくるオブジェクトも別です。

| 結果 | 関数 | 中身 |
|---|---|---|
| `DHRTestResult`（パネル） | `posterior(res, :gamma)` | 本物の事後分布（MCMC draw） |
| `DHRSingleResult`（`mode = :single`） | `sampling_distribution(r)` | `Normal(γ̂, se)`。漸近的な標本分布 |
| `DHRSingleResult` | `null_distribution(r, :window)` | 並べ替えの帰無分布（draw） |
| `DHRSingleResult` | `null_distribution(r, :stationarity)` | 定常性検定のブートストラップ帰無分布 |

`posterior` は `DHRSingleResult` には定義していません（単一世帯側は最尤法で、
そこに事後分布は存在しないため）。逆に `sampling_distribution` は
`DHRTestResult` には使えません。

### パネルの結果から

```julia
using Distributions

d = posterior(res, :gamma)      # PosteriorSample <: ContinuousUnivariateDistribution
rand(d, 10_000)                 # 事後draw の再抽出
quantile(d, 0.975)              # res.gamma_ci[2] と一致
cdf(d, 0.0)                     # 0 以下の事後質量
```

`PosteriorSample` は draw そのものを保持します（近似なし）。`pdf` / `logpdf` は
ガウスカーネル密度推定なので 1 回あたり `O(ndraws)` かかります。作図用と考えてください。
**サンプラーの中で使うならパラメトリックに当てはめてから**どうぞ。`fit` は
`Distributions.fit` に委譲するので任意の一変量分布族が使えます:

```julia
n  = fit(Normal, posterior(res, :gamma))          # Turing の prior に刺せる
ln = fit(LogNormal, posterior(res, :odds_ratio))
```

正規近似は実用上よく当たります。350 世帯パネルでの実測で γ の事後は歪度 `+0.03`、
超過尖度 `+0.01`、当てはめた正規と経験分位点の差は 0.5% 点・99.5% 点でも
標準偏差の 0.15 倍以内でした。

| `posterior(res, …)` | 量 |
|---|---|
| `:gamma` | 状態依存係数 |
| `:placebo` | 順序シャッフル後のパネル上の同じ係数 |
| `:excess` | `gamma − gamma_placebo` — 判定の根拠 |
| `:odds_ratio` | `exp(gamma)` |
| `:lift` | 選択確率の変化（シェア加重、%pt） |
| `:beta, l` | 共変量係数 `l` |

`gamma` と `gamma_placebo` は独立した事後なので、`:excess` は draw をランダムに
組み合わせています。これが差の事後からの draw そのものです。

**有効サンプルサイズに注意。** MCMC の draw は自己相関しています。`effective_size(d)`
で ESS が取れます。既定設定だと 3200 draw に対して ESS はおよそ 570 です。そこから
10,000 個 `rand` しても独立サンプル 10,000 個にはなりません（不確実性の伝播には十分ですが、
サンプルサイズとして引用しないでください）。

### 単一世帯の結果から

`mode = :single` は MCMC ではなく最尤法なので、返るのは推定量の**漸近的な標本分布**です。

```julia
r = DHRTests(y; mode = :single, nperm = 999)

d = sampling_distribution(r)             # Normal(0.1098, 0.2850)
rand(d, 10_000)
quantile(d, 0.025), quantile(d, 0.975)   # (-0.449, 0.668)
ccdf(d, 0.0)                             # P(γ > 0) = 0.65
logpdf(d, 0.0)                           # 解析的。サンプラーの中でも使える
```

**事後分布ではなく標本分布**で、定義上左右対称です。尾を使う前に `r.ci_profile` と
突き合わせてください。上の例ではプロファイル区間 `[-0.461, 0.664]` に対して Wald が
`[-0.449, 0.668]` なので実害はありませんが、短い履歴では一致する保証はありません。

並べ替えの帰無分布は `PosteriorSample`（パネル側と同じ経験分布の型）で返ります。

```julia
nw = null_distribution(r, :window)   # PosteriorSample(:null_window, n=999, mean=-0.173, 95%=[-0.713, 0.350])
ng = null_distribution(r, :global)   # PosteriorSample(:null_global, n=999, mean=-0.051, 95%=[-0.713, 0.508])

rand(nw, 1000)
r.null_window                        # 生の Vector{Float64} が欲しいならこちら
```

派生量は draw を変換するのが簡単です。

```julia
or = exp.(rand(sampling_distribution(r), 100_000))     # オッズ比
mean(or), quantile(or, [0.025, 0.975])                 # 1.162, [0.638, 1.944]
```

### Bijectors.jl を使う

Bijectors は **weak dependency** です。コアは `Distributions` だけのまま、
`using Bijectors` したときだけ変換系が有効になります。

```julia
using Bijectors

transformed(posterior(res, :gamma), exp)                  # 経験分布のままでも動く
transformed(fit(Normal, posterior(res, :gamma)), exp)     # 解析的な logpdf 付き
```

`exp` だけなら実は Bijectors は不要です（正規の exp は厳密に `LogNormal`）。
Bijectors が効くのは **名前の付いた分布族にならない変換**で、本パッケージは
その例を 1 つ同梱しています。`LiftBijector(share)` は γ を「ベースシェア `share` の
ブランドの選択確率の変化」に写す単調変換で、

$$g \;\longmapsto\; \mathrm{logistic}\!\left(g + \mathrm{logit}(s)\right) - s$$

逆関数も log ヤコビアンも閉じた形で持っているので、結果は本物の `Distribution` です:

```julia
b = LiftBijector(0.25)                    # シェア 25% のブランド
d = lift_distribution(res, 0.25)          # == transformed(fit(Normal, γ), b)
rand(d, 1000)                             # 確率単位のリフト
logpdf(d, 0.10)
```

単一世帯の結果でも同じです。`sampling_distribution(r)` はただの `Normal` なので、
`transformed(sampling_distribution(r), LiftBijector(0.30))` がそのまま通ります。

`LiftBijector` はコア側では単なる callable で、`InverseFunctions.inverse` を実装
しているので Bijectors の外でも合成できます。全ブランドのシェア加重合計には閉じた
逆関数が無いので、そちらは厳密な draw（`posterior(res, :lift)`）を使ってください。
シェアは `lift_share(res)` で取れます。

## テスト

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

真の γ = 1.0 のダミーで区間が 1.0 を覆い判定が `:state_dependence` になること、
真の γ = 0.0（異質性のみ）のダミーで excess の区間が 0 を覆い判定が
`:no_evidence` になることまで確認しています。

## ライセンス

MIT

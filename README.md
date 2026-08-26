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

A hierarchical Bayes multinomial logit with a flexible heterogeneity distribution:

$$v_{hjt} = \alpha_{hj} + \gamma \cdot \mathbb{1}\{j = y_{h,t-1}\} + x_{hjt}'\beta,
\qquad \alpha_h \sim \sum_{k=1}^{K} \pi_k \, N(\mu_k, \Sigma_k)$$

with $\alpha_{hB} \equiv 0$. Sampling is Gibbs with random-walk Metropolis blocks
(household intercepts, $\gamma$, $\beta$) and a Normal-Inverse-Wishart /
Dirichlet update for the mixture. Priors follow DHR: $\text{Dir}(0.5/K)$,
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

## Posteriors, not just intervals

Every reported quantity is also available as a `Distribution`, so you can sample
from it and carry the uncertainty downstream instead of re-deriving it from a
point estimate and an interval.

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
Julia パッケージです。Dubé, Hitsch & Rossi (2010) に依拠しています。

入力は世帯ごとの **ブランド（行）× 購買機会（列）の数量マトリクス** 1 枚だけ。
`DHRTests(X)` を呼ぶと、状態依存係数 γ と信用区間、順序シャッフルのプラセボ、
そして判定が返ります。

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

異質性を柔軟にした階層ベイズ多項ロジット

$$v_{hjt} = \alpha_{hj} + \gamma \cdot 1\{j = y_{h,t-1}\} + x_{hjt}'\beta, \qquad
\alpha_h \sim \sum_k \pi_k N(\mu_k, \Sigma_k)$$

を Gibbs ＋ ランダムウォーク Metropolis で推定します。事前分布は DHR と同じ
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

## パネルではなく 1 世帯の長い履歴を使う: `mode = :single`

```julia
one = simulate_panel(H = 1, B = 4, T = 400, gamma = 0.8, seed = 2)
DHRTests(one.X[1]; mode = :single)
```

世帯が 1 つなら、世帯間の観測されない異質性が隠れる場所はありません（α は定数）。
階層モデルは不要になるので、`mode = :single` は **固定効果条件付きロジット**を
Newton 法で当て、尤度比で `gamma = 0` を検定し、**プロファイル尤度区間**を返します
（戻り値は `DHRSingleResult`）。

**交絡は消えるのではなく形を変えます。** 異質性の代わりに来るのが、その世帯自身の
**嗜好ドリフト**（数か月単位の好みの変化、配架の変化、季節性）です。時期によって
好みが動く世帯は、慣性とまったく同じ見た目の連続を作ります。パネル版で使っている
全体シャッフルではこれを分離できません（ドリフトも状態依存もまとめて壊すので）。
そこで `mode = :single` は **短い窓の中だけでシャッフル**します。20 機会程度の窓の
中ならドリフトはほぼ一定なので、これが「状態依存なし・嗜好は局所的に一定」の null に
なります。

ダミーの単一世帯での実測（B = 4、T = 300、120 反復、5% 水準、棄却率）:

| シナリオ | LR 検定 | 全体シャッフル | **窓内シャッフル** |
|---|---:|---:|---:|
| SD なし・ドリフトなし | 6.7% | 5.8% | **4.2%** |
| SD なし・弱ドリフト | 13.3% | 12.5% | **3.3%** |
| SD なし・**強ドリフト** | 67.2% | 73.1% | **10.1%** |
| SD `gamma = 0.8`・ドリフトなし | 96.7% | 97.5% | **95.8%** |
| SD `gamma = 0.8`・弱ドリフト | 94.1% | 94.1% | **92.4%** |

古典的な 2 つの検定は、ドリフトしているだけの世帯の 3 分の 2 を「状態依存あり」と
判定します。窓内 null はサイズをほぼ名目通りに保ちつつ、検出力をほとんど落としません。
両方の p 値を出力しており、**その対比自体が診断になります**:

| `p_global` | `p_window` | 判定 |
|---|---|---|
| 小 | 小 | `:state_dependence` |
| 小 | 大 | `:nonstationarity` — 系列構造はドリフトの尺度にしかない |
| 大 | 大 | `:no_evidence` |

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
null_distribution(r, :window)     # 並べ替え分布そのもの
```

## 区間ではなく分布として取り出す

報告される量はすべて `Distribution` として取り出せます。点推定と区間から組み直す
のではなく、そのまま `rand` して下流に不確実性を流せます。

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

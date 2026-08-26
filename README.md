# StatesDependency.jl

[![CI](https://github.com/kimseok1973/StatesDependency.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/kimseok1973/StatesDependency.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Statistical tests for **state dependence in consecutive brand purchases**, following
Dubé, Hitsch & Rossi (2010), *State dependence and alternative explanations for
consumer inertia*, Quantitative Marketing and Economics 8(4), 417–445.

Input is the simplest thing a panel gives you: one **brand (row) × purchase-occasion
(column)** quantity matrix per household. One call returns the state-dependence
coefficient with a credible interval, an order-shuffled placebo, and a verdict.

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

Julia 1.9+. The only non-stdlib dependency is `Distributions`.

## Input format

`X` is a `Vector` of `B × T_h` matrices — one per household, ragged `T_h` allowed —
or a `B × T × H` array.

* **Rows are brands.** Same order in every household.
* **Columns are purchase occasions, in chronological order.** The whole test is about
  what the *previous* column contained, so the column order is the data.
* **Cells are quantities.** An all-zero column is a no-purchase occasion and is
  dropped; a column with several positive rows is resolved by `tie_rule`
  (`:argmax` by default, or `:error` / `:drop`).

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

## API

| function | purpose |
|---|---|
| `DHRTests(X; ...)` | the test; returns a `DHRTestResult` |
| `simulate_panel(; ...)` | dummy panel with a known `gamma` |
| `dummy_data(; ...)` | just the input matrices |
| `build_panel(X; ...)` | matrices → `PurchasePanel` (useful to inspect the parsing) |
| `shuffle_panel(p, rng)` | the order-shuffled placebo panel |
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

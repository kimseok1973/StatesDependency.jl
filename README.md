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

## テスト

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

真の γ = 1.0 のダミーで区間が 1.0 を覆い判定が `:state_dependence` になること、
真の γ = 0.0（異質性のみ）のダミーで excess の区間が 0 を覆い判定が
`:no_evidence` になることまで確認しています。

## ライセンス

MIT

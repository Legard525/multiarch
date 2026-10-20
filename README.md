# multiarch

[![tests](https://github.com/Legard525/multiarch/actions/workflows/tests.yml/badge.svg)](https://github.com/Legard525/multiarch/actions/workflows/tests.yml)

**Multivariate GARCH for Python.** The [`arch`](https://github.com/bashtage/arch) package
only fits univariate models, so Python has had no maintained multivariate GARCH library.
Courses and research therefore send Python users to R (`rmgarch`, `tsmarch`).
multiarch fills that gap:

- **Models:** CCC, DCC and asymmetric DCC, optionally as Gaussian or Student-t copulas, plus GO-GARCH.
- **Margins:** GARCH(1,1) or GJR-GARCH(1,1) with normal or Student-t errors.
- **Inference:** two-step standard errors that correct for the first estimation stage.
- **Large cross-sections:** a composite-likelihood estimator for many assets.
- **Outputs:** forecasts, simulation, portfolio VaR and ES, rolling backtests and specification tests.
- **Validation:** it reproduces `rmgarch`'s likelihood exactly. It runs 2–9× faster than rmgarch and 7–30× faster than tsmarch.

```bash
pip install git+https://github.com/Legard525/multiarch
```

Requires Python ≥ 3.10, NumPy, SciPy, pandas and numba.

## Quick start

```python
import multiarch as ma

res = ma.DCC(returns).fit()          # returns: (T, N) DataFrame, any units
print(res.summary())

res.conditional_correlation          # (T, N, N) R_t
res.conditional_covariance           # (T, N, N) H_t = D_t R_t D_t
res.correlation("JPM", "INTC")       # one rho_t path as a Series
res.forecast(10)                     # mean, variance, correlation, covariance
res.value_at_risk([0.5, 0.5], level=0.01, value=1e6)   # {'VaR': ..., 'ES': ...}
```

[`examples/course_workflow.py`](examples/course_workflow.py) walks through the standard
course sequence: EWMA → CCC → DCC → constant-correlation test → portfolio VaR → rolling VaR backtest.

## What's in it

| | |
|---|---|
| `DCC(returns, model=...)` | `"dcc"` (Engle 2002), `"adcc"` (Cappiello, Engle & Sheppard 2006), `"ccc"` (Bollerslev 1990) |
| `vol=` | `"garch"` or `"gjr"` margins, each (1,1) |
| `mean=` | `"constant"` or `"zero"` |
| `dist=` | margin distribution, `"normal"` or `"t"` |
| `corr_dist=` | correlation-stage distribution, `"normal"` or multivariate `"t"` (defaults to `dist`) |
| `copula=` | `None`, `"normal"` or `"t"`: copula-DCC with parametric margins (as `rmgarch::cgarchfit`) |
| `estimator=` | `"full"` likelihood or `"composite"` pairwise likelihood (Engle, Shephard & Sheppard 2008) for large N |
| `convention=` | `"arch"` (default), `"rmgarch"` or `"tsmarch"`: start-up conventions that reproduce each package |
| `.fit(se=...)` | `"two-step"` (Engle & Sheppard 2001 corrected), `"naive"` or `None` |
| `.fix(params)` / `.apply(new_returns)` | evaluate fixed parameters, or filter new data with fitted ones |
| `.forecast(h)` | closed-form variances; Engle–Sheppard correlation forecast as in rmgarch |
| `.simulate(n, n_paths, start="end")` | Monte Carlo paths, including copula models |
| `.value_at_risk(w, level, horizon)` | analytic (normal or multivariate t) or simulated VaR and ES |
| `.value_at_risk_path(w, level)` | in-sample one-step VaR/ES path |
| `GOGARCH(returns)` | GO-GARCH (van der Weide 2002) with a built-in FastICA rotation |
| `rolling_forecast(returns, window, refit_every)` | rolling or expanding re-estimation with no look-ahead |
| `var_backtest(r, VaR, level)` | violation ratio, Kupiec and Christoffersen tests |
| `constant_correlation_test(returns)` | Engle & Sheppard (2001) test of CCC against DCC |
| `hosking_test(x, lags)` | multivariate portmanteau test |
| `ewma_covariance(returns, lam)` | RiskMetrics EWMA |
| `simulate_dcc(...)` | simulate from given parameters |

Estimation is scale-free: percent and fractional returns give the same parameters,
with ω and μ rescaled.

## Validation

All numbers come from [`validation/`](validation) and are reproducible: `Rscript rmgarch_fit.R .`
then `python compare.py`, and `Rscript tsmarch_fit.R .` then `python compare_tsmarch.py`.
The data are daily returns of DJI stocks from `rmgarch`'s `dji30ret` (T = 5,521).

### Against rmgarch 1.3 (R 4.6)

`ll(ours @ rm)` is multiarch's log-likelihood evaluated at rmgarch's own estimates.
If both packages implement the same model, it equals rmgarch's reported value.

| case | ll (rmgarch) | ll (ours @ rm) | max \|ρ_t diff\| @ rm | ll (ours, own fit) | time rmgarch → ours |
|---|---|---|---|---|---|
| DCC, normal, N=5 | −53514.64292 | −53514.64292 | 0 | −53513.99601 | 2.8 s → 0.32 s |
| aDCC, normal | −53513.86843 | −53513.86843 | 0 | −53513.13231 | 3.4 s → 0.45 s |
| DCC, Student-t | −52235.91344 | −52235.91344 | 0 | −52237.53475 | 3.7 s → 0.60 s |
| DCC, GJR margins | −53481.97322 | −53481.97322 | 0 | −53481.59132 | 3.3 s → 0.39 s |
| DCC, zero mean | −53534.39786 | −53534.39786 | 0 | −53533.64643 | 2.1 s → 0.26 s |
| DCC, N=10 | −101267.48845 | −101267.48845 | 0 | −101266.80559 | 7.6 s → 1.2 s |
| DCC, raw returns, N=2 | 28163.98498 | 28163.98498 | 0 | 28163.97845 | 0.84 s → 0.09 s |
| t-copula DCC, t margins | −52079.16469 | −52079.16469 | 0 | −52080.47842 | 6.3 s → 3.1 s |
| Gaussian-copula DCC, t margins | −52196.46089 | −52196.46089 | 0 | −52197.96371 | 3.6 s → 1.1 s |

Timings for multiarch include the two-step standard errors.

- The likelihood functions are identical, and so are the correlation paths at the same parameters.
- The 10-step DCC covariance forecasts agree to 2e-14.
- Estimates differ only through the optimiser, and multiarch reaches higher margin likelihoods.
  Because estimation is two-step, a better first stage can still leave the joint
  log-likelihood slightly lower (the t and copula rows). That is a property of the two-step
  estimator, not a different model.
- Standard errors for the DCC parameters *a* and *b* are within 5% of rmgarch's. For the shape ν and the aDCC asymmetry *g* they are 17–31% larger.
  A 300-sample Monte Carlo (`mc_se.py`) shows that multiarch's two-step SEs are calibrated:
  the median SE is 0.0059 against a sampling SD of 0.0059 for *a*, and 0.0124 against 0.0139 for *b*.
- **GO-GARCH:** given rmgarch's ICA rotation and factor parameters, the likelihood agrees to 0.06,
  a difference that comes from how the constant mean is estimated. multiarch's own FastICA finds a rotation
  with a likelihood 277 points higher on this data, in 0.04 s against 0.47 s.

### Against tsmarch (the package used in *Financial Risk Forecasting* notes)

| case | ll (tsmarch) | ll (ours @ ts) | max \|ρ_t diff\| @ ts | time tsmarch → ours |
|---|---|---|---|---|
| DCC, N=2 | 28163.94124 | 28163.94306 | 4.5e-4 | 0.73 s → 0.04 s |
| DCC, N=5 | 73611.21223 | 73611.23406 | 4.2e-4 | 2.7 s → 0.10 s |
| aDCC, N=5 | 73612.01170 | 73612.03276 | 4.2e-4 | 3.8 s → 0.11 s |
| DCC, multivariate t | 74873.58661 | 74873.58785 | 3.8e-4 | 3.9 s → 0.13 s |

The margins match tsgarch exactly (σ_t to 1e-14). A residual gap of under 0.03 log-likelihood
points remains in tsmarch's compiled DCC recursion; it fades to 1e-7 by the end of the sample.

### Notes on rmgarch's `DCCtest`

`rmgarch::DCCtest` departs from Engle & Sheppard (2001) in two ways. It calls
`qr.solve(regressand, regressors)`, which regresses the lags on the target instead of the
reverse, and it divides by √σ² instead of σ². `validation/compare.py` reproduces its statistic
exactly (65.34) from its own residuals. On the same residuals, the textbook statistic
returned by `constant_correlation_test` is 41.08.

### Scaling

From `validation/bench.py` (T = 2,500, no SEs):

| N | full likelihood | composite likelihood |
|---|---|---|
| 5 | 0.05 s | 0.02 s |
| 25 | 0.34 s | 0.09 s |
| 50 | 0.35 s | 0.19 s |
| 100 | n/a | 0.41 s |

The full-likelihood DCC estimate of *b* is biased downward as N grows: 0.930 against a true 0.95
at N = 50. The composite estimator does not have this bias.

## Conventions

The default (`"arch"`) follows the `arch` package for the margins (exponential backcast)
and uses Q̄ = Z′Z/T with Q₁ = Q̄. `convention="rmgarch"` reproduces rmgarch's start-up:
- σ²₁ is the sample mean of ε².
- Q̄ = cov(Z).
- The first Q step uses a pre-sample shock of ones in DCC and of zeros in copula-DCC.
- Copula uniforms are clipped at 0.99999.

`convention="tsmarch"` reproduces tsgarch's σ²₁ = ω + persistence · mean(ε²) and uses Q₁ = Q̄ = cov(Z).
These choices only affect the first few dozen observations.

## Limitations

- Margins are GARCH(1,1) or GJR(1,1) with a constant or zero mean. There are no ARMA means or higher orders.
- DCC is of order (1,1).
- Estimation is two-step. The two-step SEs treat Q̄ and N̄ as known, as rmgarch does.
- Composite-likelihood SEs are Hessian-only.
- GO-GARCH SEs ignore estimation error in the rotation.
- Copula models use parametric margins only. Empirical and semi-parametric margins are not implemented.
- BEKK and DCC-MIDAS are not implemented.

## References

- Bollerslev (1990), *Modelling the coherence in short-run nominal exchange rates*, REStat.
- Engle (2002), *Dynamic conditional correlation*, JBES.
- Engle & Sheppard (2001), *Theoretical and empirical properties of dynamic conditional correlation multivariate GARCH*, NBER WP 8554.
- Cappiello, Engle & Sheppard (2006), *Asymmetric dynamics in the correlations of global equity and bond returns*, J. Fin. Econometrics.
- Engle, Shephard & Sheppard (2008), *Fitting vast dimensional time-varying covariance models*, working paper.
- van der Weide (2002), *GO-GARCH: a multivariate generalized orthogonal GARCH model*, J. Applied Econometrics.
- Patton (2006), *Modelling asymmetric exchange rate dependence*, IER.
- Kupiec (1995); Christoffersen (1998); Hosking (1980).

## License

MIT

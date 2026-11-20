# Portfolio research and risk report

Universe: 25 NIFTY 50 stocks, daily adjusted closes. Backtest 2016-01-15 to 2026-10-30, monthly rebalance, 252-day estimation window, decisions at the close and execution at the next close, Rs 500 crore AUM with statutory charges and square-root impact. Constituents are today's, so absolute returns carry survivorship bias; comparisons between strategies are fairer than comparisons with the index.

## Strategies (net of costs)

| strategy | CAGR net | CAGR gross | vol | Sharpe | max DD | beta | turnover/yr | cost/yr |
|---|---|---|---|---|---|---|---|---|
| equal_weight | 17.56% | 17.82% | 15.45% | 0.72 | -34.43% | 0.93 | 0.33 | 0.22% |
| inverse_vol | 16.95% | 17.24% | 14.97% | 0.70 | -33.58% | 0.90 | 0.37 | 0.24% |
| min_var_sample | 14.99% | 16.06% | 13.48% | 0.64 | -26.59% | 0.73 | 1.08 | 0.92% |
| min_var_shrunk | 14.99% | 15.93% | 13.51% | 0.64 | -26.99% | 0.74 | 0.99 | 0.81% |
| risk_parity | 16.93% | 17.24% | 14.73% | 0.71 | -32.42% | 0.88 | 0.39 | 0.26% |
| momentum_12_1 | 12.07% | 17.14% | 17.90% | 0.37 | -34.77% | 0.97 | 2.74 | 4.42% |
| NIFTY 50 (price index) | 11.18% | 11.18% | 16.13% | 0.35 | -38.44% | 1.00 | 0.00 | - |

## Capacity: net CAGR as AUM grows

| strategy | Rs 50 cr | Rs 500 cr | Rs 5,000 cr |
|---|---|---|---|
| min_var_shrunk | 15.42% | 14.99% | 13.63% |
| risk_parity | 17.06% | 16.93% | 16.54% |
| momentum_12_1 | 14.93% | 12.07% | 3.41% |

## 1-day 99% VaR backtest on the min-variance book

Rolling 500-day estimation window, forecasts made before each day's return is seen.

| model | days | exceptions | expected | Kupiec p | Christoffersen p | cond. coverage p | zone | avg VaR | avg ES |
|---|---|---|---|---|---|---|---|---|---|
| historical | 2159 | 25 | 21.600 | 0.472 | 0.294 | 0.446 | green | 2.50% | 3.16% |
| parametric | 2159 | 40 | 21.600 | 0.000 | 0.042 | 0.000 | yellow | 1.88% | 2.17% |
| cornish_fisher | 2159 | 22 | 21.600 | 0.930 | 0.222 | 0.472 | green | 3.16% | 4.71% |
| filtered_hs | 2159 | 24 | 21.600 | 0.609 | 0.269 | 0.476 | green | 2.13% | 2.61% |

## Current book: risk by sector (Ledoit-Wolf shrinkage intensity 0.16)

Market beta 0.71; 75% of variance is NIFTY, 25% residual.

| sector | weight | component_var | pct_of_var | risk_to_weight |
|---|---|---|---|---|
| Consumer Staples | 0.230 | 0.004 | 0.233 | 1.011 |
| Utilities | 0.200 | 0.003 | 0.196 | 0.981 |
| Financials | 0.167 | 0.003 | 0.161 | 0.963 |
| IT | 0.123 | 0.002 | 0.137 | 1.114 |
| Communication | 0.100 | 0.002 | 0.103 | 1.028 |
| Health Care | 0.100 | 0.001 | 0.087 | 0.874 |
| Energy | 0.055 | 0.001 | 0.058 | 1.045 |
| Consumer Discretionary | 0.024 | 0.000 | 0.024 | 1.029 |

## Historical stress replay of the current book

| scenario | start | end | days | portfolio | nifty | worst_point | worst_name |
|---|---|---|---|---|---|---|---|
| Demonetisation | 2016-11-08 | 2016-12-26 | 33 | -4.12% | -7.44% | -4.12% | KOTAKBANK |
| IL&FS default / NBFC liquidity | 2018-08-28 | 2018-10-26 | 39 | -10.30% | -14.55% | -10.46% | MARUTI |
| COVID-19 crash | 2020-02-19 | 2020-03-23 | 21 | -29.21% | -37.24% | -29.21% | ICICIBANK |
| 2022 rate-hike drawdown | 2022-01-14 | 2022-06-17 | 104 | -6.30% | -16.23% | -7.96% | WIPRO |
| Hindenburg-Adani report | 2023-01-24 | 2023-02-28 | 24 | -1.02% | -4.49% | -1.34% | SBIN |
| 2024 election result day | 2024-06-03 | 2024-06-04 | 1 | -5.34% | -5.93% | -5.34% | NTPC |

Beta-mapped NIFTY -10% shock: -7.86%.


## Brinson-Fachler attribution vs equal weight (Carino-linked, gross)

Min-variance 375.6% vs equal weight 464.3%, excess -88.7%; linked effects sum to -88.7%.

| sector | allocation | selection | interaction | total |
|---|---|---|---|---|
| Consumer Staples | -0.3763 | -0.0148 | -0.0416 | -0.4327 |
| Materials | 0.0182 | -0.2686 | -0.0662 | -0.3167 |
| Utilities | -0.3466 | 0.0971 | -0.0395 | -0.2890 |
| Consumer Discretionary | -0.1087 | -0.3023 | 0.1501 | -0.2609 |
| Financials | -0.1071 | -0.2189 | 0.2751 | -0.0508 |
| Industrials | -0.0008 | 0.0000 | 0.0000 | -0.0008 |
| Energy | 0.0803 | 0.0000 | -0.0000 | 0.0803 |
| Communication | 0.0959 | 0.0000 | -0.0000 | 0.0959 |
| IT | 0.1105 | -0.0416 | 0.0342 | 0.1032 |
| Health Care | 0.1849 | 0.0000 | 0.0000 | 0.1849 |

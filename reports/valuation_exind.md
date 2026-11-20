# Example Industrials Ltd (EXIND): valuation memo

All figures in Rs crore unless stated. Fiscal year ends 31 March.

## Historical ratios

| metric | FY24 | FY25 | FY26 |
|---|---|---|---|
| revenue_growth | - | 0.109 | 0.096 |
| gross_margin | 0.370 | 0.376 | 0.379 |
| ebitda_margin | 0.162 | 0.165 | 0.168 |
| dso | 56.367 | 55.667 | 54.750 |
| dio | 76.225 | 75.536 | 74.715 |
| dpo | 63.032 | 63.792 | 63.691 |
| capex_pct | 0.073 | 0.075 | 0.075 |
| roce | 0.131 | 0.138 | 0.144 |
| net_debt_to_ebitda | 1.391 | 1.011 | 0.683 |

## Projection (base case)

| line | FY27 | FY28 | FY29 | FY30 | FY31 |
|---|---|---|---|---|---|
| revenue | 5,328.0 | 5,860.8 | 6,388.3 | 6,899.3 | 7,382.3 |
| ebitda | 905.8 | 1,019.8 | 1,137.1 | 1,241.9 | 1,328.8 |
| ebit | 679.6 | 780.1 | 884.1 | 976.2 | 1,051.6 |
| net_income | 469.4 | 551.6 | 639.8 | 722.8 | 796.7 |
| cfo | 603.2 | 701.4 | 803.9 | 902.2 | 992.2 |
| capex | 399.6 | 410.3 | 415.2 | 414.0 | 406.0 |
| cash | 262.8 | 250.0 | 285.2 | 406.5 | 603.8 |
| term_debt | 750.0 | 600.0 | 450.0 | 300.0 | 150.0 |
| revolver | 0.0 | 11.5 | 0.0 | 0.0 | 0.0 |
| equity | 3,488.6 | 3,874.7 | 4,322.5 | 4,828.5 | 5,386.2 |

The balance sheet balances in every year (checked in code); interest on average debt and cash is solved iteratively and converged in 8 iterations or fewer.

## Cost of capital

Cost of equity 12.83% = 6.65% + 0.95 x 6.50%; after-tax cost of debt 6.55%; target debt weight 15%; WACC 11.88%.

## DCF (mid-year convention)

| line | FY27 | FY28 | FY29 | FY30 | FY31 |
|---|---|---|---|---|---|
| FCFF | 242.71 | 323.30 | 410.43 | 495.92 | 576.38 |
| discount factor | 0.95 | 0.84 | 0.76 | 0.68 | 0.60 |
| PV | 229.46 | 273.18 | 309.97 | 334.76 | 347.75 |

Terminal value 9,031.3 (value-driver, RONIC 18%, g = 5.5%), implying 6.8x terminal EBITDA. Enterprise value 6,944.0, less net debt 550.0, equity 6,394.0, **152.2 per share**. Terminal value is 78% of EV.

## Scenarios

| scenario | EV | equity | per share | TV share of EV |
|---|---|---|---|---|
| bear | 4,593.1 | 4,043.1 | 96.3 | 80% |
| base | 6,944.0 | 6,394.0 | 152.2 | 78% |
| bull | 8,597.7 | 8,047.7 | 191.6 | 80% |

## Sensitivity: value per share, WACC x terminal growth

| WACC | g=4.5% | g=5.5% | g=6.5% |
|---|---|---|---|
| 9.88% | 202.8 | 229.2 | 270.8 |
| 10.88% | 167.9 | 183.6 | 206.1 |
| 11.88% | 142.5 | 152.2 | 165.4 |
| 12.88% | 123.2 | 129.4 | 137.4 |
| 13.88% | 108.0 | 112.1 | 117.0 |

## Trading comparables (year-1 metrics)

| method | stat | multiple | per_share |
|---|---|---|---|
| ev_ebitda | p25 | 8.7 | 175.1 |
| ev_ebitda | median | 9.4 | 190.7 |
| ev_ebitda | p75 | 10.2 | 208.0 |
| pe | p25 | 15.3 | 171.3 |
| pe | median | 16.6 | 185.5 |
| pe | p75 | 18.2 | 204.0 |

Peers at the median imply about 188.1 per share. Reverse DCF: matching that needs perpetual growth of 7.66% at the base WACC. That is the growth the peer multiples embed; whether it is credible for this business, given its return on new capital, is the judgement the memo has to make, and the scenario table is where to argue it.

"""Research universe: 25 NIFTY 50 large caps with their NSE sector.

These are the constituents as of late 2026, so any backtest over earlier years
carries survivorship bias (names that dropped out of the index are missing).
The README reports results with that caveat rather than hiding it.
"""

BENCHMARK = "^NSEI"

UNIVERSE: dict[str, str] = {
    "HDFCBANK": "Financials",
    "ICICIBANK": "Financials",
    "SBIN": "Financials",
    "KOTAKBANK": "Financials",
    "AXISBANK": "Financials",
    "BAJFINANCE": "Financials",
    "TCS": "IT",
    "INFY": "IT",
    "WIPRO": "IT",
    "HCLTECH": "IT",
    "RELIANCE": "Energy",
    "NTPC": "Utilities",
    "POWERGRID": "Utilities",
    "HINDUNILVR": "Consumer Staples",
    "ITC": "Consumer Staples",
    "NESTLEIND": "Consumer Staples",
    "MARUTI": "Consumer Discretionary",
    "M&M": "Consumer Discretionary",
    "TITAN": "Consumer Discretionary",
    "ASIANPAINT": "Materials",
    "ULTRACEMCO": "Materials",
    "TATASTEEL": "Materials",
    "LT": "Industrials",
    "SUNPHARMA": "Health Care",
    "BHARTIARTL": "Communication",
}

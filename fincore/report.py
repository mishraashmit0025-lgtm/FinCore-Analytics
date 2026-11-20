"""Small formatting helpers shared by the CLI reports."""

from __future__ import annotations

import pandas as pd


def md_table(df: pd.DataFrame, index: bool = False, floatfmt: str = "{:,.4f}") -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]

    def fmt(v):
        if isinstance(v, float):
            return "-" if v != v else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def pct(v: float, digits: int = 2) -> str:
    return f"{v * 100:.{digits}f}%"

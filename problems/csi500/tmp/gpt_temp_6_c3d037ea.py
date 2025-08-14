import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive momentum considering both short and long-term trends with exponential smoothing
    short_term_momentum = df['close'].pct_change(periods=5).ewm(span=5, adjust=False).mean()
    long_term_momentum = df['close'].pct_change(periods=30).ewm(span=30, adjust=False).mean()
    adaptive_momentum = 0.6 * short_term_momentum + 0.4 * long_term_momentum

    # Dynamic volatility using the standard deviation of logarithmic returns over the last 14 days

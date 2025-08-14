import pandas as pd

def heuristics_v4(df):
    close_ema = df['close'].ewm(span=50, adjust=False).mean()
    close_momentum_ema = (df['close'] / close_ema) - 1
    daily_returns = df['close'].pct_change()
    std_daily_returns = daily_returns.rolling(window=10).std()
    hl_range = df['high'] - df['low']
    hl_ma = hl_range.rolling(window=20).mean()

    # Compute historical correlation for weighting (simplified using static values here for example)
    corr_close_return = 0.7  # Example value, to be computed dynamically
    corr_std_return = 0.2    # Example value, to be computed dynamically
    corr_hl_range = 0.1      # Example value, to be computed dynamically

    heuristics_matrix = (corr_close_return * close_momentum_ema + 
                         corr_std_return * std_daily_returns + 
                         corr_hl_range * hl_ma)
    return heuristics_matrix

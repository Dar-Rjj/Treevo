import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Adaptive window size for rolling sum
    window_size = 5 + (df['close'].pct_change().abs().rolling(window=10).mean() * 10).round().astype(int)

    # Sum of positive and absolute negative parts with adaptive window
    pos_sum = df['positive_amount_vol'].rolling(window=window_size, min_periods=1).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=window_size, min_periods=1).sum()

    # Factor: ratio of positive sum to absolute negative sum
    factor = pos_sum / (neg_sum_abs + 1e-7)

    # Volatility factor
    volatility = df['close'].pct_change().rolling(window=20).std()

    # VWAP (Volume Weighted Average Price)
    vwap = (df['amount'] / df['volume']).rolling(window=20).mean()

    # Exponential smoothing for close price
    smoothed_close = df['close'].ewm(span=20, adjust=False).mean()

    # Momentum factor
    momentum = df['close'].pct_change(periods=20)

    # Mean reversion factor
    mean_reversion = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()

    # Combine factors
    combined_factor = (factor + volatility + vwap + smoothed_close + momentum + mean_reversion) / 6

    return combined_factor

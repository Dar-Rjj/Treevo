import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:

    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Sum of positive and absolute negative parts
    pos_sum = df['positive_amount_vol'].rolling(window=5).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=5).sum()

    # Factor: ratio of positive sum to absolute negative sum
    sentiment_factor = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate the volatility using the close price
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    volatility = df['log_returns'].rolling(window=20).std() * np.sqrt(252)

    # Calculate the VWAP
    df['vwap'] = (df['amount'] / df['volume']).rolling(window=20).mean()

    # Exponential smoothing on the VWAP
    df['vwap_smoothed'] = df['vwap'].ewm(span=20, adjust=False).mean()

    # Momentum factor
    momentum = df['close'].pct_change(periods=20)

    # Mean reversion factor
    mean_reversion = -df['close'].pct_change(periods=5)

    # Trend strength factor
    trend_strength = df['close'].rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

    # Volatility-adjusted momentum
    v_adj_momentum = momentum / (volatility + 1e-7)

    # Combine factors
    alpha_factor = (sentiment_factor +
                    (1 / (volatility + 1e-7)) +
                    (df['vwap'] / df['vwap_smoothed']) +
                    v_adj_momentum +
                    mean_reversion +
                    trend_strength)

    return alpha_factor

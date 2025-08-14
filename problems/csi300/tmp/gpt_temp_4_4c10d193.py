import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Sum of positive and absolute negative parts with adaptive window based on recent volatility
    window_size = 5 + 15 * (df['close'].rolling(window=50).std() / df['close'].rolling(window=100).std())
    pos_sum = df['positive_amount_vol'].rolling(window=window_size.astype(int)).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=window_size.astype(int)).sum()

    # Factor: ratio of positive sum to absolute negative sum
    sentiment_factor = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate the volatility using the close price
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

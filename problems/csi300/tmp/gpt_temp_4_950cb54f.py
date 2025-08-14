import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adjust window sizes dynamically
    dynamic_window = 30 + 10 * (df['volume'].pct_change().abs().rolling(window=30).mean() / 0.05)
    dynamic_window = dynamic_window.fillna(30).astype(int)

    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Sum of positive and absolute negative parts with dynamic window
    pos_sum = df['positive_amount_vol'].rolling(window=dynamic_window, min_periods=1).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=dynamic_window, min_periods=1).sum()

    # Factor: ratio of positive sum to absolute negative sum
    sentiment_factor = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate the volatility using the close price
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

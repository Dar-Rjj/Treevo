import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Returns
    df['daily_return'] = df['close'].pct_change()

    # Assign Weights Based on Volume and Amount
    total_volume = df['volume'].sum()
    total_amount = df['amount'].sum()
    df['weight'] = (df['volume'] / total_volume) * (df['amount'] / total_amount)

    # Multiply Weights by Daily Returns
    df['weighted_return'] = df['weight'] * df['daily_return']

    # Compute Momentum Indicator
    window_size = 20
    smoothing_factor = 0.2

    # Rolling Standard Deviation of Returns
    df['std_dev'] = df['daily_return'].rolling(window=window_size).std()

    # Adjust Smoothing Factor Dynamically
    df['adjusted_smoothing'] = np.where(df['std_dev'] > df['std_dev'].mean(), smoothing_factor * 1.5, smoothing_factor * 0.5)

    # Exponential Moving Average
    df['momentum_indicator'] = df['weighted_return'].ewm(span=window_size, adjust=False, alpha=df['adjusted_smoothing']).mean()

    return df['momentum_indicator']

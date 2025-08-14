import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Daily Returns (Close price)
    df['daily_return'] = df['close'].pct_change()

    # Weigh by Daily Volume
    df['volume_weighted_return'] = df['daily_return'] * df['volume']

    # Calculate 14-Day Volume-Weighted Price Change
    df['14_day_volume_weighted_return'] = df['volume_weighted_return'].rolling(window=14).sum()

    # Identify Consecutive Up/Down Days
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['down_day'] = (df['open'] > df['close']).astype(int)

    # Adjust for Extreme Movement (High-Low Difference)
    df['high_low_diff'] = df['high'] - df['low']
    df['reversal_component'] = np.where(df['up_day'] == 1, -df['high_low_diff'], df['high_low_diff'])

    # Calculate the True Range for each day
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = df[['high' - 'low', 'high' - df['prev_close'], df['prev_close'] - 'low']].max(axis=1)

    # Calculate the 14-day Simple Moving Average of the True Range
    df['14_day_sma_true_range'] = df['true_range'].rolling(window=14).mean()

    # Calculate the 14-day Standard Deviation of the Close Prices
    df['14_day_std_close'] = df['close'].rolling(window=14).std()

    # Construct the Volatility Adjusted Momentum Factor
    df['volatility_adjusted_momentum'] = (df['close'] - df['14_day_sma_true_range']) / df['14_day_sma_true_range'] * df['14_day_std_close']

    # Final Adjustment
    df['alpha_factor'] = np.where(df['volatility_adjusted_momentum'] > 0, 1,
                                   np.where(df['volatility_adjusted_momentum'] < 0, -1, 0))

    return df['alpha_factor']

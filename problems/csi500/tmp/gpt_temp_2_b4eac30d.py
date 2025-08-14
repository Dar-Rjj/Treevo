import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=20):
    # Calculate Volume-Weighted Cumulative Return
    df['volume_weighted_return'] = df['close'].pct_change() * df['volume']
    volume_weighted_cumulative_return = df['volume_weighted_return'].rolling(window=N).sum()

    # Adjust for Volume Variance
    volume_variance = df['volume'].rolling(window=N).var()
    volume_variance_inverse = 1 / volume_variance
    adjusted_volume_weighted_cumulative_return = volume_weighted_cumulative_return * volume_variance_inverse

    # Calculate Volume-Weighted Price Changes
    df['volume_weighted_close_change'] = (df['close'] - df['close'].shift(1)) * df['volume']
    df['volume_weighted_open_change'] = (df['open'] - df['open'].shift(1)) * df['volume']
    volume_weighted_price_changes = (df['volume_weighted_close_change'] + df['volume_weighted_open_change']).rolling(window=N).sum()
    volume_weighted_price_changes = volume_weighted_price_changes - volume_weighted_price_changes.shift(1)

    # Calculate Volume-Weighted High-Low Range
    df['volume_weighted_high_low_range'] = (df['high'] - df['low']) * df['volume']
    volume_weighted_high_low_range = df['volume_weighted_high_low_range'].rolling(window=N).mean() / N

    # Introduce Volume-Weighted Average True Range
    df['true_range'] = df[['high' - 'low', abs('high' - 'close').shift(1), abs('low' - 'close').shift(1)]].max(axis=1)
    df['volume_weighted_true_range'] = df['true_range'] * df['volume']
    volume_weighted_average_true_range = df['volume_weighted_true_range'].rolling(window=N).mean()

    # Combine Momentum and Volatility Components
    momentum_and_volatility = (
        0.5 * adjusted_volume_weighted_cumulative_return +
        0.3 * volume_weighted_price_changes +
        0.1 * volume_weighted_high_low_range +
        0.1 * volume_weighted_average_true_range
    )

    # Introduce Volume-Weighted Open-Close Range
    df['open_close_range'] = abs(df['open'] - df['close'])
    df['volume_weighted_open_close_range'] = df['open_close_range'] * df['volume']
    volume_weighted_open_close_range = df['volume_weighted_open_close_range'].rolling(window=N).mean()
    factor = momentum_and_volatility + 0.05 * volume_weighted_open_close_range

    return factor.dropna()

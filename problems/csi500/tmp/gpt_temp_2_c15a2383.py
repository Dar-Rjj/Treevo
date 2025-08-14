import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=20):
    # Calculate Volume-Weighted Cumulative Return
    df['volume_weighted_return'] = df['close'] * df['volume']
    volume_weighted_cumulative_return = df['volume_weighted_return'].rolling(window=N).sum() / df['volume'].rolling(window=N).sum()

    # Adjust for Volume Variance
    volume_variance = df['volume'].rolling(window=N).var()
    adjusted_volume_weighted_cumulative_return = volume_weighted_cumulative_return / (volume_variance + 1e-6)

    # Calculate Volume-Weighted Price Changes
    df['volume_weighted_price_change'] = (df['close'] - df['open']) * df['volume']
    volume_weighted_price_changes = df['volume_weighted_price_change'].rolling(window=N).sum()
    volume_weighted_price_changes_diff = volume_weighted_price_changes.diff()

    # Calculate Volume-Weighted High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    df['volume_weighted_high_low_range'] = df['high_low_range'] * df['volume']
    volume_weighted_high_low_range = df['volume_weighted_high_low_range'].rolling(window=N).mean()

    # Introduce Volume-Weighted Average True Range
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x['high'], x['close'].shift(1)) - min(x['low'], x['close'].shift(1)), axis=1)
    df['volume_weighted_true_range'] = df['true_range'] * df['volume']
    volume_weighted_average_true_range = df['volume_weighted_true_range'].rolling(window=N).mean()

    # Combine Components
    combined_factor = (
        0.5 * adjusted_volume_weighted_cumulative_return +
        0.3 * volume_weighted_price_changes_diff +
        0.1 * volume_weighted_high_low_range +
        0.1 * volume_weighted_average_true_range
    )

    # Introduce Volume-Weighted Open-Close Range
    df['open_close_range'] = df['close'] - df['open']
    df['volume_weighted_open_close_range'] = df['open_close_range'] * df['volume']
    volume_weighted_open_close_range = df['volume_weighted_open_close_range'].rolling(window=N).mean()
    combined_factor += 0.05 * volume_weighted_open_close_range

    return combined_factor

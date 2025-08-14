import pandas as pd
import pandas as pd

def heuristics_v2(df, N=20):
    # Calculate Volume-Weighted Cumulative Return
    df['vw_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']
    df['vw_cumulative_return'] = df['vw_return'].rolling(window=N).sum()

    # Adjust for Volume Variance
    df['volume_variance'] = df['volume'].rolling(window=N).var()
    df['adjusted_vw_cumulative_return'] = df['vw_cumulative_return'] / df['volume_variance']

    # Calculate Volume-Weighted Price Changes
    df['vw_close_change'] = (df['close'] - df['close'].shift(1)) * df['volume']
    df['vw_open_change'] = (df['open'] - df['open'].shift(1)) * df['volume']
    df['vw_price_change_sum'] = df['vw_close_change'] + df['vw_open_change']
    df['vw_price_change_diff'] = df['vw_price_change_sum'] - df['vw_price_change_sum'].shift(1)

    # Calculate Volume-Weighted High-Low Range
    df['vw_high_low_range'] = (df['high'] - df['low']) * df['volume']
    df['avg_vw_high_low_range'] = df['vw_high_low_range'].rolling(window=N).mean() / N

    # Introduce Volume-Weighted Average True Range
    df['true_range'] = df[['high' - 'low', 'high' - df['close'].shift(1), df['close'].shift(1) - 'low']].max(axis=1)
    df['vw_true_range'] = df['true_range'] * df['volume']
    df['avg_vw_true_range'] = df['vw_true_range'].rolling(window=N).mean()

    # Introduce Volume-Weighted Open-Close Range
    df['open_close_range'] = abs(df['open'] - df['close'])
    df['vw_open_close_range'] = df['open_close_range'] * df['volume']
    df['avg_vw_open_close_range'] = df['vw_open_close_range'].rolling(window=N).mean()

    # Introduce Volume-Weighted Open-High and Open-Low Ranges
    df['open_high_range'] = df['high'] - df['open']
    df['open_low_range'] = df['open'] - df['low']
    df['vw_open_high_range'] = df['open_high_range'] * df['volume']
    df['vw_open_low_range'] = df['open_low_range'] * df['volume']
    df['avg_vw_open_high_low_range'] = (df['vw_open_high_range'] + df['vw_open_low_range']).rolling(window=N).mean() / 2

    # Combine Momentum and Volatility Components
    momentum_volatility_factor = (
        0.5 * df['adjusted_vw_cumulative_return'] +
        0.3 * df['vw_price_change_diff'] +
        0.1 * df['avg_vw_high_low_range'] +
        0.1 * df['avg_vw_true_range'] +
        0.05 * df['avg_vw_open_close_range'] +
        0.025 * df['avg_vw_open_high_low_range']
    )

    return momentum_volatility_factor

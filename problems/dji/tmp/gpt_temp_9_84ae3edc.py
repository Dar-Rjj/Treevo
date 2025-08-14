import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Averaged Price
    df['volume_averaged_price'] = (df['close'] * df['volume']).rolling(window=10).sum() / df['volume'].rolling(window=10).sum()

    # Calculate Smoothed Price Momentum
    df['daily_returns'] = df['close'] - df['close'].shift(1)
    df['smoothed_momentum'] = df['daily_returns'].rolling(window=10).mean()

    # Calculate True Range
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'], x['close'].shift(1)) - min(x['low'], x['close'].shift(1)), axis=1)

    # Adjust Momentum by True Range
    df['adjusted_momentum'] = df['smoothed_momentum'] / df['true_range']

    # Calculate Volume Change and Normalize by t-1 day volume
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['normalized_volume_change'] = df['volume_change'] / df['volume'].shift(1)
    df['normalized_volume_change'] = df['normalized_volume_change'].fillna(0)  # Ensure no division by zero

    # Define Positive Volume Change Threshold
    volume_change_threshold = 0.5

    # Combine Price Momentum and Volume Surge
    df['combined_momentum'] = df['adjusted_momentum'] * df['normalized_volume_change']
    df['combined_momentum'] = df['combined_momentum'].where(df['normalized_volume_change'] > volume_change_threshold, 0)

    # Calculate Volume-Weighted True Volatility Over Period
    lookback_period = 10
    df['sum_true_range'] = df['true_range'].rolling(window=lookback_period).sum()
    df['average_true_range'] = df['sum_true_range'] / lookback_period

    # Calculate Daily Volume Trend
    df['volume_return'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['volume_trend'] = df['volume_return'].rolling(window=lookback_period).mean()

    # Final Factor
    df['final_factor'] = df['combined_momentum'] * df['average_true_range'] * df['volume_trend']

    return df['final_factor']

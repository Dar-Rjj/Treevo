import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Adjusted Price Change
    df['price_change'] = df['close'].diff()
    df['adjusted_price_change'] = df['price_change'] * (2 if df['volume'] > df['volume'].shift(1) else 1)
    df['agg_adjusted_price_change'] = df['adjusted_price_change'].rolling(window=5).sum()

    # Calculate Raw Returns
    df['raw_return'] = df['close'].pct_change()

    # Compute 14-Day Sum of Upward and Downward Returns
    df['up_return'] = df['raw_return'].apply(lambda x: x if x > 0 else 0)
    df['down_return'] = df['raw_return'].apply(lambda x: -x if x < 0 else 0)
    df['sum_up_14'] = df['up_return'].rolling(window=14).sum()
    df['sum_down_14'] = df['down_return'].rolling(window=14).sum()

    # Calculate Relative Strength
    df['relative_strength'] = df['sum_up_14'] / df['sum_down_14']

    # Smooth with Exponential Moving Average on Volume
    df['ema_volume'] = df['volume'].ewm(span=14, adjust=False).mean()
    df['smoothed_relative_strength'] = df['relative_strength'] * df['ema_volume']

    # Adjust Relative Strength with Price Trend
    df['sma_close_21'] = df['close'].rolling(window=21).mean()
    df['price_trend_adjustment'] = df['close'] / df['sma_close_21']
    df['adjusted_relative_strength'] = df['smoothed_relative_strength'] * df['price_trend_adjustment']

    # Calculate Volume Surge
    df['avg_volume_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['avg_volume_5']
    df['volume_surge'] = df['volume_ratio'].apply(lambda x: 1 if x > 1.5 else 0)

    # Calculate Price Surge
    df['price_ratio'] = df['high'] / df['low']
    df['price_surge'] = df['price_ratio'].apply(lambda x: 1 if x > 1.2 else 0)

    # Combine Adjusted Momentum, Volume, and Price Surge
    df['intermediate_alpha_factor'] = df['agg_adjusted_price_change'] * df['volume_surge'] * df['price_surge']

    # Calculate High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']

    # Calculate High-Low Mean Reversion
    df['avg_high_low_spread_14'] = df['high_low_spread'].rolling(window=14).mean()
    df['high_low_mean_reversion'] = df['high_low_spread'] - df['avg_high_low_spread_14']

    # Calculate Volume-Weighted High-Low Spread
    df['volume_weighted_high_low_spread'] = df['volume'] * df['high_low_spread']

    # Calculate Close Price Momentum
    df['close_price_momentum'] = df['close'].pct_change(periods=14)

    # Combine Momentum and High-Low Factors
    df['combined_factors'] = (df['high_low_mean_reversion'] + df['close_price_momentum']) / df['volume_weighted_high_low_spread']

    # Final Alpha Factor
    df['final_alpha_factor'] = df['adjusted_relative_strength'] * df['combined_factors']

    return df['final_alpha_factor']

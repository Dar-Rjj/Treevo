import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Adjusted Price Change
    df['price_change'] = df['close'].diff()
    df['volume_trend'] = (df['volume'] > df['volume'].shift(1)).astype(int)
    df['adjusted_price_change'] = df['price_change'] * (2 * df['volume_trend'] + 1 - df['volume_trend'])
    df['aggregated_adjusted_price_change'] = df['adjusted_price_change'].rolling(window=5).sum()

    # Calculate Raw Returns
    df['raw_returns'] = df['close'].pct_change()

    # Compute 14-Day Sum of Upward Returns and Downward Returns
    df['upward_returns'] = df['raw_returns'].where(df['raw_returns'] > 0, 0)
    df['downward_returns'] = df['raw_returns'].where(df['raw_returns'] < 0, 0).abs()
    df['14_day_sum_upward'] = df['upward_returns'].rolling(window=14).sum()
    df['14_day_sum_downward'] = df['downward_returns'].rolling(window=14).sum()

    # Calculate Relative Strength
    df['relative_strength'] = df['14_day_sum_upward'] / df['14_day_sum_downward']

    # Smooth with Exponential Moving Average on Volume
    df['ema_volume'] = df['volume'].ewm(span=14, alpha=0.2).mean()
    df['smoothed_relative_strength'] = df['relative_strength'] * df['ema_volume']

    # Adjust Relative Strength with Price Trend
    df['21_day_sma_close'] = df['close'].rolling(window=21).mean()
    df['price_trend_adjustment'] = df['close'] / df['21_day_sma_close']
    df['adjusted_relative_strength'] = df['smoothed_relative_strength'] * df['price_trend_adjustment']

    # Calculate Volume Surge
    df['avg_volume_5_days'] = df['volume'].rolling(window=5).mean().shift(1)
    df['volume_ratio'] = df['volume'] / df['avg_volume_5_days']
    df['volume_surge'] = (df['volume_ratio'] > 1.5).astype(int)

    # Calculate Price Surge
    df['price_ratio'] = df['high'] / df['low']
    df['price_surge'] = (df['price_ratio'] > 1.2).astype(int)

    # Combine Adjusted Momentum, Volume, and Price Surge
    df['intermediate_alpha_factor'] = df['aggregated_adjusted_price_change'] * df['volume_surge'] * df['price_surge']

    # Calculate High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']

    # Calculate High-Low Mean Reversion
    df['avg_high_low_spread'] = df['high_low_spread'].rolling(window=21).mean()
    df['high_low_mean_reversion'] = df['high_low_spread'] - df['avg_high_low_spread']

    # Calculate Volume-Weighted High-Low Spread
    df['volume_weighted_high_low_spread'] = df['volume'] * df['high_low_spread']

    # Calculate Close Price Momentum
    df['close_price_momentum'] = df['close'].pct_change(periods=21)

    # Introduce Price Range Momentum
    df['price_range'] = df['high'] - df['low']
    df['price_range_momentum'] = df['price_range'].pct_change(periods=21)

    # Combine Momentum and High-Low Factors
    df['combined_factors'] = (df['high_low_mean_reversion'] + df['close_price_momentum'] + df['price_range_momentum']) / df['volume_weighted_high_low_spread']

    # Final Alpha Factor
    df['final_alpha_factor'] = df['adjusted_relative_strength'] * df['combined_factors']

    return df['final_alpha_factor']

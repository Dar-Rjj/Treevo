import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Directionally Adjusted High-Low Spread
    high_low_diff = df['high'] - df['low']
    close_open_diff = df['close'] - df['open']
    adjusted_high_low_spread = high_low_diff * (1.5 if close_open_diff > 0 else 0.5)

    # Compute Volume-Adjusted Momentum
    momentum = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).mean()
    avg_price_range_10day = (df['high'] - df['low']).rolling(window=10).mean()
    volume_adjusted_momentum = momentum * df['volume'] / avg_price_range_10day

    # Calculate Volume-Weighted Return with Spike Adjustment
    high_low_return = (df['high'] - df['low']) / df['low']
    volume_weighted_return = high_low_return * df['volume']
    volume_ma_20day = df['volume'].rolling(window=20).mean()
    spike_days = df['volume'] > 1.5 * volume_ma_20day
    volume_weighted_return[spike_days] *= 2.5

    # Integrate Adjusted High-Low Spread, Volume-Weighted Return, and Volume-Adjusted Momentum
    integrated_factor = adjusted_high_low_spread * volume_weighted_return + volume_adjusted_momentum

    # Apply Directional Bias to Integrated Factor
    directional_bias = 1.7 if close_open_diff > 0 else 0.3
    integrated_factor *= directional_bias

    # Incorporate Open-Close Trend
    open_close_trend = df['close'] - df['open']
    open_close_trend_multiplier = 1.3 if open_close_trend > 0 else 0.7
    integrated_factor *= open_close_trend_multiplier

    # Incorporate Volume Trend Analysis
    volume_change = df['volume'] - df['volume'].shift(1)
    volume_trend = (volume_change > 0) & (volume_change.shift(1) > 0)
    integrated_factor[volume_trend] *= 1.1
    integrated_factor[~volume_trend] *= 0.9

    # Generate Alpha Factor
    alpha_factor = pd.Series(index=df.index, data=0.0)
    alpha_factor[integrated_factor > 0] = 1.2
    alpha_factor[integrated_factor <= 0] = -1.2

    return alpha_factor

import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Directionally Adjusted High-Low Spread
    high_low_diff = df['high'] - df['low']
    directional_bias = (df['close'] > df['open']).astype(float) * 2 - 1
    adjusted_high_low_spread = high_low_diff * directional_bias

    # Compute Volume-Adjusted Momentum with Price Range
    momentum = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    avg_price_range = (df['high'] - df['low']).rolling(window=10).mean() * directional_bias
    volume_adjusted_momentum = (momentum * df['volume']) / avg_price_range

    # Calculate Volume-Weighted Return with Spike Adjustment
    high_low_return = (df['high'] - df['low']) / df['low']
    volume_weighted_return = high_low_return * df['volume']
    volume_spike_days = df['volume'] > 1.5 * df['volume'].rolling(window=20).mean()
    volume_weighted_return[volume_spike_days] *= 2.5

    # Calculate Intraday Momentum and Volume Spike
    intraday_range = df['high'] - df['low']
    volume_spike = df['volume'] > 1.5 * df['volume'].shift(1)
    intraday_range[volume_spike] *= 2.5

    # Integrate Adjusted High-Low Spread, Volume-Weighted Return, and Volume-Adjusted Momentum
    integrated_factor = (adjusted_high_low_spread * volume_weighted_return) + volume_adjusted_momentum

    # Apply Directional Bias to Integrated Factor
    integrated_factor *= ((df['close'] > df['open']).astype(float) * 1.4) + 0.3

    # Incorporate Open-Close Trend
    open_close_trend = df['close'] - df['open']
    integrated_factor *= (open_close_trend > 0).astype(float) * 1.3 + (open_close_trend < 0).astype(float) * 0.7

    # Calculate Moving Averages
    short_term_ma = df['close'].rolling(window=5).mean()
    long_term_ma = df['close'].rolling(window=20).mean()

    # Compute Crossover Signal
    crossover_signal = short_term_ma - long_term_ma

    # Generate Alpha Factor
    alpha_factor = crossover_signal.apply(lambda x: 1.2 * integrated_factor if x > 0 else -1.2 * integrated_factor)

    return alpha_factor

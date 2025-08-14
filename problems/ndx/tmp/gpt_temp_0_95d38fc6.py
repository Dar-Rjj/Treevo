import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Average Momentum
    one_day_return = df['close'].pct_change(1)
    five_day_return = df['close'].pct_change(5)
    twenty_day_return = df['close'].pct_change(20)
    average_momentum = (one_day_return + five_day_return + twenty_day_return) / 3
    
    # Weighted Average of Price Gaps
    price_gap = (df['open'] - df['close']).abs()
    volume_trend = df['volume'].rolling(window=5).mean() * (df['volume'] / df['volume'].shift(1))
    weighted_price_gaps = (price_gap * volume_trend).sum(axis=1)
    
    # Adjust for Volume
    daily_volume_percentage_change = df['volume'].pct_change(1)
    adjusted_weighted_price_gaps = weighted_price_gaps * daily_volume_percentage_change
    
    # High-to-Low Range and Momentum
    high_low_range = (df['high'] - df['low'])
    combined_value = high_low_range * 100 + weighted_price_gaps
    smoothed_combined_value = combined_value.rolling(window=5).mean()
    high_low_momentum = smoothed_combined_value - smoothed_combined_value.shift(5)
    
    # Calculate Rolling High-Low Differential
    rolling_high = df['high'].rolling(window=20).max()
    rolling_low = df['low'].rolling(window=20).min()
    rolling_high_low_diff = rolling_high - rolling_low
    
    # Weight by Volume Volatility
    volume_volatility = df['volume'].rolling(window=10).std()
    weighted_rolling_high_low_diff = rolling_high_low_diff * volume_volatility
    
    # Aggregate the Signals
    aggregated_signals = (
        weighted_rolling_high_low_diff
        - adjusted_weighted_price_gaps
    )
    
    # Generate Alpha Factor
    alpha_factor = (
        adjusted_weighted_price_gaps
        + high_low_momentum
        + weighted_rolling_high_low_diff
        - average_momentum
    )
    
    return alpha_factor

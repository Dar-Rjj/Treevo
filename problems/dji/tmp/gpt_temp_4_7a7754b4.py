import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Relative Strength
    n = 5
    relative_strength = df['close'] / df['open'].shift(n)
    
    # Measure Volume Activity Change
    m = 10
    average_volume = df['volume'].rolling(window=m).mean().shift(1)
    volume_change = (df['volume'] - average_volume)
    
    # Combine Relative Strength and Volume Change
    combined_factor_1 = relative_strength * volume_change
    
    # Calculate Daily Price Momentum
    daily_momentum = df['close'] - df['close'].shift(1)
    
    # Calculate Short-Term Trend
    short_term_trend = daily_momentum.ewm(span=5).mean()
    
    # Introduce Short-Term Volatility
    short_term_volatility = daily_momentum.rolling(window=5).std()
    short_term_trend_adjusted = short_term_trend / short_term_volatility
    
    # Generate Volume Synchronized Oscillator
    long_term_trend = daily_momentum.ewm(span=20).mean()
    volume_synchronized_oscillator = (long_term_trend - short_term_trend) * df['volume']
    
    # Calculate Daily High-Low Difference
    high_low_diff = df['high'] - df['low']
    
    # Cumulate the Moving Difference
    cum_high_low_diff = high_low_diff.rolling(window=5).sum()
    
    # Calculate Volume Trend
    volume_trend = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    
    # Adjust Cumulative Moving Difference by Volume Trend
    adjusted_cum_high_low_diff = cum_high_low_diff * volume_trend
    
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Apply Weighted Volume Adjustment
    weighted_avg_volume = df['volume'].rolling(window=5).mean()
    volume_anomaly = df['volume'] - weighted_avg_volume
    adjusted_intraday_range = intraday_range * (1 + volume_anomaly / df['volume'])
    
    # Incorporate Price Oscillation
    price_oscillation = high_low_diff + (df['close'] - df['open']).abs()
    
    # Adjust Momentum by Inverse of True Range
    true_range = df[['high' - 'low', 'high' - 'close'].shift(1), 'close'].shift(1) - 'low']].max(axis=1)
    adjusted_momentum = daily_momentum / true_range
    
    # Integrate Combined Factors
    integrated_factors = combined_factor_1 * adjusted_cum_high_low_diff * adjusted_intraday_range + price_oscillation
    final_factor = (adjusted_momentum * integrated_factors) * volume_synchronized_oscillator
    
    return final_factor

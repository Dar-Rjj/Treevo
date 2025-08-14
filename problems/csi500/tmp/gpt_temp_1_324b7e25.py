import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Adjust for Opening Gap
    opening_gap = df['open'] - df['close'].shift(1)
    adjusted_range = high_low_range - opening_gap.abs()
    
    # Incorporate Volume Momentum
    volume_change = df['volume'] / df['volume'].shift(1)
    volume_weighted_range = adjusted_range * volume_change
    
    # Apply Time Series Momentum
    lookback_period = 10
    moving_average = volume_weighted_range.rolling(window=lookback_period).mean()
    momentum_difference = volume_weighted_range - moving_average
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Combine Close-to-Open Return with Volume
    daily_volatility = (df['high'] - df['low']).abs()
    close_to_open_adjusted = (close_to_open_return * df['volume']) / daily_volatility
    
    # Calculate Volume-Adjusted Momentum
    last_5_days_close = df['close'].rolling(window=5).sum()
    last_5_days_volume = df['volume'].rolling(window=5).mean()
    volume_adjusted_momentum = last_5_days_close * last_5_days_volume
    
    # Trend Analysis
    def calculate_trend(data, window):
        trend = []
        for i in range(len(data)):
            if i < window:
                trend.append(np.nan)
            else:
                y = data.iloc[i-window+1:i+1]
                x = np.arange(window)
                slope, _, _, _, _ = linregress(x, y)
                trend.append(slope)
        return pd.Series(trend, index=data.index)
    
    five_day_trend = calculate_trend(df['close'], 5)
    twenty_day_trend = calculate_trend(df['close'], 20)
    combined_trends = five_day_trend - twenty_day_trend
    
    # Final Alpha Factor
    alpha_factor = (
        momentum_difference +
        volume_adjusted_momentum +
        close_to_open_adjusted +
        combined_trends
    )
    
    return alpha_factor

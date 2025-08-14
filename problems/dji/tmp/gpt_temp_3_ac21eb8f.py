import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close Price Momentum
    n = 5  # Number of days for momentum calculation
    close_momentum = df['close'] / df['close'].shift(n)
    
    # Measure Volume Activity Change
    m = 5  # Number of days for volume activity change
    avg_volume = df['volume'].rolling(window=m, min_periods=1).mean()
    volume_change = df['volume'] - avg_volume
    
    # Combine Relative Strength and Volume Change
    relative_strength_vol_change = close_momentum * volume_change
    
    # Calculate Daily High-Low Difference
    daily_high_low_diff = df['high'] - df['low']
    
    # Cumulate the Moving Difference
    window = 20  # Window for cumulative moving difference
    cum_high_low_diff = daily_high_low_diff.rolling(window=window, min_periods=1).sum()
    
    # Calculate Volume Trend
    volume_trend = df['volume'].rolling(window=window, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    
    # Adjust Cumulative Moving Difference by Volume Trend
    adjusted_cum_diff = cum_high_low_diff * volume_trend
    
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Apply Weighted Volume Adjustment
    recent_volume_period = 21  # Recent fixed period for volume adjustment
    recent_avg_volume = df['volume'].rolling(window=recent_volume_period, min_periods=1).mean()
    volume_anomaly = df['volume'] - recent_avg_volume
    adjusted_intraday_range = intraday_range * (1 + volume_anomaly / recent_avg_volume)
    
    # Synthesize Final Alpha Factor
    final_alpha_factor = relative_strength_vol_change * adjusted_cum_diff * adjusted_intraday_range
    
    return final_alpha_factor

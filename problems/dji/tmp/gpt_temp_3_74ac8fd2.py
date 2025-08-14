import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Adjusted High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Weight by Volume
    volume_weighted_spread = high_low_spread * df['volume']
    
    # Condition on Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) / df['open']
    positive_return_weight = 1.5  # Example weight for positive return
    negative_return_weight = 0.5  # Example weight for negative return
    weighted_spread = volume_weighted_spread * (positive_return_weight if close_to_open_return > 0 else negative_return_weight)
    
    # Calculate Intraday Percent Change
    intraday_percent_change = (df['close'] - df['open']) / df['open']
    
    # Incorporate Volume-Weighted Average Price
    vwap = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
    vwap = vwap.cumsum() / df['volume'].cumsum()
    
    # Combine with Intraday Indicators
    intraday_indicator = weighted_spread + intraday_percent_change
    
    # Enhance with Recent Momentum and Trend
    recent_price_trend = df['close'].rolling(window=5).mean()
    recent_volume_trend = df['volume'].rolling(window=5).mean()
    
    # Finalize Indicator
    final_indicator = (vwap + intraday_indicator + recent_price_trend + recent_volume_trend) / 4
    
    return final_indicator

import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Adjusted Intraday High-Low Spread
    intraday_range = df['high'] - df['low']
    volume_adjusted_intraday_range = intraday_range * df['volume']
    adjusted_intraday_high_low_spread = volume_adjusted_intraday_range * (df['close'] > df['open']).astype(int) - volume_adjusted_intraday_range * (df['close'] < df['open']).astype(int)
    
    # Calculate Volume-Adjusted Opening Gap
    opening_gap = df['open'].shift(1) - df['close'].shift(1)
    volume_adjusted_opening_gap = opening_gap * df['volume']
    
    # Combine Weighted Intraday High-Low Spread and Opening Gap
    combined_value = adjusted_intraday_high_low_spread + volume_adjusted_opening_gap
    
    # Calculate 5-day and 20-day Moving Averages of Close
    ma_5 = df['close'].rolling(window=5).mean()
    ma_20 = df['close'].rolling(window=20).mean()
    
    # Compute Crossover Signal
    crossover_signal = ma_5 - ma_20
    
    # Generate Alpha Factor Based on Crossover
    alpha_factor = (crossover_signal > 0).astype(int) * 2 - 1
    
    # Adjust for Price Direction
    alpha_factor += (combined_value * (df['close'] > df['open']).astype(int)) - (combined_value * (df['close'] <= df['open']).astype(int))
    
    # Integrate High-Low Spread and Volume-Weighted Momentum
    daily_price_range = df['high'] - df['low']
    volume_weighted_momentum = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * df['volume']
    integrated_value = volume_weighted_momentum * daily_price_range
    
    # Consider Directional Bias
    directional_bias = (df['close'] > df['open']).astype(int) * 2 - 1
    integrated_value *= directional_bias
    
    # Combine Final Values
    alpha_factor += integrated_value
    
    # Incorporate Volume-Weighted Price Changes and Trading Activity
    average_price = (df['high'] + df['low']) / 2
    volume_weighted_price_changes = average_price * df['volume']
    
    # Calculate Exponential Moving Average (EMA) of Weighed Spreads and Gaps
    ema_window = 10
    ema_weighed_spreads_gaps = df[['adjusted_intraday_high_low_spread', 'volume_adjusted_opening_gap']].sum(axis=1).ewm(span=ema_window, adjust=False).mean()
    
    # Calculate Momentum using EMA
    momentum = ema_weighed_spreads_gaps - ema_weighed_spreads_gaps.shift(ema_window)
    
    # Integrate Volume-Weighted Price Changes into Momentum
    momentum += volume_weighted_price_changes
    
    # Incorporate Trading Volume Trend
    volume_change = df['volume'] - df['volume'].shift(ema_window)
    momentum += volume_change
    
    # Incorporate Trade Activity Intensity
    trade_activity_intensity = df['volume'] / df['amount']
    momentum += trade_activity_intensity
    
    # Integrate Trade Activity Intensity into Momentum
    alpha_factor += momentum
    
    return alpha_factor

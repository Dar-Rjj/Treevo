import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Intraday Close-Open Change
    intraday_close_open_change = df['close'] - df['open']
    
    # Sum Intraday Movements
    sum_intraday_movements = intraday_high_low_spread + intraday_close_open_change
    
    # Weight by Volume
    volume_weighted_intraday_movement = sum_intraday_movements * df['volume']
    
    # High-Low Range Momentum
    current_high_low_range = df['high'] - df['low']
    previous_high_low_range = df['high'].shift(1) - df['low'].shift(1)
    high_low_range_momentum = current_high_low_range - previous_high_low_range
    
    # Evaluate Momentum and Volume Interaction
    close_to_close_return = (df['close'] / df['close'].shift(1)) - 1
    high_low_difference = current_high_low_range - previous_high_low_range
    adjusted_high_low_momentum = high_low_difference * close_to_close_return
    
    volume_change = df['volume'] - df['volume'].shift(1)
    volume_interaction_factor = adjusted_high_low_momentum * (1 if volume_change > 0 else -1)
    
    # Combine Intraday, High-Low, and Volume Components
    combined_factor = (volume_weighted_intraday_movement + high_low_range_momentum) * volume_interaction_factor
    
    # Evaluate Recent Trend
    five_day_avg_intraday_movement = volume_weighted_intraday_movement.rolling(window=5).mean()
    single_day_intraday_movement = volume_weighted_intraday_movement
    trend_component = single_day_intraday_movement - five_day_avg_intraday_movement
    
    # Introduce a Short-Term Reversal Component
    two_day_close_open_change = (df['close'].shift(1) - df['open'].shift(2))
    final_factor = combined_factor + two_day_close_open_change + trend_component
    
    return final_factor

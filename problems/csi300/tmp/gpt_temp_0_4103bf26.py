import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close to Midpoint Deviation
    close_to_midpoint_deviation = df['close'] - (df['high'] + df['low']) / 2
    
    # Calculate Previous Day Return
    previous_day_return = df['open'] / df['close'].shift(1) - 1
    
    # Generate Intermediate Alpha Factor
    intermediate_alpha_factor = (intraday_range * close_to_midpoint_deviation + previous_day_return)
    
    # Calculate High-Low Price Difference
    high_low_price_difference = df['high'] - df['low']
    
    # Compute Volume Influence Ratio
    upward_volume = (df[df['close'] > df['open']]['volume']).rolling(window=20).sum()
    downward_volume = (df[df['close'] < df['open']]['volume']).rolling(window=20).sum()
    volume_influence_ratio = (upward_volume / downward_volume).fillna(1)
    
    # Adjust Momentum by Volume and Amount
    weighted_high_low_return = high_low_price_difference * df['volume'] * volume_influence_ratio
    volume_adjusted_momentum_factor = weighted_high_low_return.rolling(window=20).sum()
    
    # Introduce Time Decay
    decay_rate = 0.95
    decayed_volume_adjusted_momentum = volume_adjusted_momentum_factor * (decay_rate ** (len(volume_adjusted_momentum_factor) - volume_adjusted_momentum_factor.index))
    
    # Generate Final Alpha Factor
    final_alpha_factor = intermediate_alpha_factor + decayed_volume_adjusted_momentum
    
    return final_alpha_factor

import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close to Midpoint Deviation
    midpoint = (df['high'] + df['low']) / 2
    close_to_midpoint_deviation = df['close'] - midpoint
    
    # Calculate Previous Day Return
    prev_close = df['close'].shift(1)
    previous_day_return = (df['open'] - prev_close) / prev_close
    
    # Generate Intermediate Alpha Factor
    intermediate_alpha = (intraday_range * close_to_midpoint_deviation) + previous_day_return
    
    # Calculate High-Low Price Difference
    high_low_diff = df['high'] - df['low']
    
    # Compute Volume Influence Ratio
    upward_volume = df.loc[df['close'] > df['open'], 'volume'].sum()
    downward_volume = df.loc[df['close'] < df['open'], 'volume'].sum()
    volume_influence_ratio = upward_volume / (downward_volume + 1e-6)  # Add small epsilon to avoid division by zero
    
    # Adjust Momentum by Volume and Amount
    weighted_high_low_return = high_low_diff * df['volume'] * volume_influence_ratio
    volume_adjusted_momentum = weighted_high_low_return.rolling(window=10, min_periods=1).sum()  # 10-day rolling sum
    
    # Introduce Time Decay
    decay_rate = 0.95
    decayed_volume_momentum = volume_adjusted_momentum * (decay_rate ** (df.index.to_series().diff().dt.days - 1))
    
    # Incorporate Price Reversal Indicator
    price_reversal = (df['high'].shift(1) - df['low']) / df['high'].shift(1)
    adjusted_intermediate_alpha = intermediate_alpha * price_reversal
    
    # Include Volume Spike Indicator
    volume_spike = (df['volume'] > df['volume'].shift(1) * 2).astype(int)
    modified_final_alpha = adjusted_intermediate_alpha * volume_spike
    
    # Generate Final Alpha Factor
    final_alpha_factor = modified_final_alpha + decayed_volume_momentum
    
    return final_alpha_factor

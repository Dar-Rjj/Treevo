import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Price Momentum
    close_prices = df['close']
    
    # Compute 10-Day Average Return
    ten_day_avg_return = close_prices.pct_change().rolling(window=10).mean()
    
    # Compute 5-Day Average Return
    five_day_avg_return = close_prices.pct_change().rolling(window=5).mean()
    
    # Calculate Volume Reversal
    volume = df['volume']
    
    # Compute 5-Day Moving Average of Volume
    five_day_vol_ma = volume.rolling(window=5).mean()
    
    # Compute 1-Day Volume Change
    one_day_vol_change = volume - five_day_vol_ma
    
    # Determine Volume Reversal
    volume_reversal = one_day_vol_change.apply(lambda x: 1 if x < 0 else -1)
    
    # Adjust Price Momentum Based on Volume Reversal
    adjusted_momentum = ten_day_avg_return * volume_reversal
    
    # Calculate Close-to-Low Spread
    close_to_low_spread = (close_prices - df['low']).apply(lambda x: max(x, 0))
    
    # Sum Volume over Period
    sum_volume = volume.rolling(window=10).sum()
    
    # Cumulative Spread Over Period
    cum_spread = close_to_low_spread.rolling(window=10).sum()
    
    # Divide by Accumulated Volume to get Spread per Unit Volume
    spread_per_unit_volume = cum_spread / sum_volume
    spread_per_unit_volume = spread_per_unit_volume.replace([pd.np.inf, -pd.np.inf], 0).fillna(0)
    
    # Combine Adjusted Price Momentum and Spread per Unit Volume
    final_alpha_factor = adjusted_momentum * spread_per_unit_volume
    
    return final_alpha_factor

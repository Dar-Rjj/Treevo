import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    intraday_high_low_spread = df['High'] - df['Low']
    
    # Compute Previous Day's Close-to-Open Return
    previous_close_to_open_return = df['Close'].shift(1) - df['Open']
    
    # Calculate Intraday Momentum
    intraday_momentum = intraday_high_low_spread + previous_close_to_open_return
    
    # Calculate Volume Weighted Average Price (VWAP)
    total_volume = df['Volume']
    price_volume_sum = (df['Open'] * df['Volume'] + df['High'] * df['Volume'] + 
                        df['Low'] * df['Volume'] + df['Close'] * df['Volume']) / 4
    vwap = price_volume_sum / total_volume
    
    # Combine Intraday Momentum and VWAP
    combined_value = vwap - intraday_high_low_spread
    volume_weighted_combined_value = combined_value * df['Volume']
    
    # Smooth the Factor
    alpha_factor = volume_weighted_combined_value.ewm(span=5, adjust=False).mean()
    
    return alpha_factor

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    intraday_high_low_spread = df['High'] - df['Low']
    
    # Compute Previous Day's Open-to-Close Return
    prev_open_to_close_return = df['Close'] - df['Open'].shift(1)
    
    # Calculate Volume Weighted Average Price (VWAP)
    price_sum = (df['High'] + df['Low'] + df['Close'] + df['Open']) * df['Volume']
    total_volume = df['Volume']
    vwap = price_sum / total_volume
    
    # Combine Intraday Momentum and VWAP
    combined_value = vwap - intraday_high_low_spread
    weighted_combined_value = combined_value * df['Volume']
    
    # Smooth the Factor using Exponential Moving Average (EMA)
    smoothed_factor = weighted_combined_value.ewm(span=7, adjust=False).mean()
    
    return smoothed_factor

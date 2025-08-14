import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    intraday_high_low_spread = df['High'] - df['Low']
    
    # Compute Previous Day's Close-to-Open Return
    prev_day_close_to_open_return = df['Close'].shift(1) - df['Open']
    
    # Calculate Intraday Momentum
    intraday_momentum = (intraday_high_low_spread + prev_day_close_to_open_return)
    
    # Calculate Volume Weighted Average Price (VWAP)
    total_volume = df['Volume']
    vwap_numerator = (df['Open'] * df['Volume'] + df['High'] * df['Volume'] + df['Low'] * df['Volume'] + df['Close'] * df['Volume'])
    vwap = vwap_numerator / (4 * total_volume)
    
    # Combine Intraday Momentum and VWAP
    combined_factor = vwap - intraday_high_low_spread
    combined_factor_weighted = combined_factor * df['Volume']
    
    # Smooth the Factor using Exponential Moving Average (EMA)
    smoothed_factor = combined_factor_weighted.ewm(span=7, adjust=False).mean()
    
    return smoothed_factor

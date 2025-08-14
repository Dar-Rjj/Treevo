import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    intraday_high_low_spread = df['High'] - df['Low']
    
    # Compute Previous Day's Close-to-Open Return
    prev_close_to_open_return = df['Close'].shift(1) - df['Open']
    
    # Calculate Intraday Momentum
    intraday_momentum = (intraday_high_low_spread + prev_close_to_open_return)
    
    # Calculate Volume Weighted Average Price (VWAP)
    total_volume = df['Volume']
    vwap = (df['Open'] * df['Volume'] + df['High'] * df['Volume'] + df['Low'] * df['Volume'] + df['Close'] * df['Volume']) / (4 * total_volume)
    
    # Combine Intraday Momentum and VWAP
    combined_factor = vwap - intraday_high_low_spread
    
    # Weight by Intraday Volume
    volume_weighted_factor = combined_factor * df['Volume']
    
    # Smooth the Factor using Exponential Moving Average (EMA)
    ema_smoothed_factor = volume_weighted_factor.ewm(span=10, adjust=False).mean()
    
    return ema_smoothed_factor

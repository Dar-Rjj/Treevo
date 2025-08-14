import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    intraday_high_low_spread = df['High'] - df['Low']
    
    # Compute Previous Day's Close-to-Open Return
    prev_close_to_open_return = df['Close'].shift(1) - df['Open']
    
    # Calculate Volume Weighted Average Price (VWAP)
    vwap = ((df['Open'] + df['High'] + df['Low'] + df['Close']) / 4) * df['Volume']
    total_volume = df['Volume']
    vwap = vwap.cumsum() / total_volume.cumsum()
    
    # Combine Intraday Momentum and VWAP
    combined_value = vwap - intraday_high_low_spread
    weighted_value = combined_value * df['Volume']
    
    # Smooth the Factor with Exponential Moving Average (EMA)
    factor = weighted_value.ewm(span=5, adjust=False).mean()
    
    return factor

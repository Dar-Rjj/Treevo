import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    intraday_high_low_spread = df['High'] - df['Low']
    
    # Compute Previous Day's Close-to-Open Return
    df['Prev_Close'] = df['Close'].shift(1)
    close_to_open_return = df['Prev_Close'] - df['Open']

    # Calculate Volume Weighted Average Price (VWAP)
    total_price_volume = (df['Open'] + df['High'] + df['Low'] + df['Close']) * df['Volume']
    daily_vwap = total_price_volume / (4 * df['Volume'])
    
    # Combine Intraday Momentum and VWAP
    combined_value = daily_vwap - intraday_high_low_spread
    weighted_value = combined_value * df['Volume']
    
    # Smooth the Factor using Exponential Moving Average (EMA) over the last 20 days
    smoothed_factor = weighted_value.ewm(span=20, adjust=False).mean()
    
    return smoothed_factor

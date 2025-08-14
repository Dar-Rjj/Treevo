import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['High'] - df['Low']
    
    # Compute Previous Day's Close-to-Open Return
    df['prev_close_to_open_return'] = df['Close'].shift(1) - df['Open']
    
    # Calculate Volume Weighted Average Price (VWAP)
    total_volume = df['Volume']
    weighted_prices = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4 * df['Volume']
    df['VWAP'] = weighted_prices.cumsum() / total_volume.cumsum()
    
    # Combine Intraday Momentum and VWAP
    df['combined_value'] = df['VWAP'] - df['intraday_high_low_spread']
    df['weighted_combined_value'] = df['combined_value'] * df['Volume']
    
    # Smooth the Factor using Exponential Moving Average (EMA) over the last 5 days
    df['alpha_factor'] = df['weighted_combined_value'].ewm(span=5, adjust=False).mean()
    
    return df['alpha_factor']

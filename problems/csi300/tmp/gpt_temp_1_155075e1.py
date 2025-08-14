import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['High'] - df['Low']
    
    # Compute Previous Day's Close-to-Open Return
    df['prev_close_open_return'] = df['Close'].shift(1) - df['Open']
    
    # Compute Volume Weighted Average Price (VWAP)
    df['daily_vwap'] = ((df['Open'] + df['High'] + df['Low'] + df['Close']) / 4 * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Combine Intraday Momentum and VWAP
    df['combined_value'] = df['daily_vwap'] - df['intraday_high_low_spread']
    
    # Integrate More Price Dynamics: Intraday Open-Close Return
    df['intraday_open_close_return'] = df['Close'] - df['Open']
    df['combined_value'] += df['intraday_open_close_return']
    
    # Refine Volume Weighting: Volume Weighted Momentum
    df['volume_weighted_momentum'] = (df['intraday_high_low_spread'] + df['prev_close_open_return']) * df['Volume']
    df['adjusted_vwap'] = df['daily_vwap'] + df['volume_weighted_momentum']
    
    # Smooth the Factor: Apply Exponential Moving Average (EMA) over the last 10 days
    df['alpha_factor'] = df['adjusted_vwap'].ewm(span=10).mean()
    
    return df['alpha_factor']

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
    prices = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['VWAP'] = (prices * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Combine Intraday Momentum and VWAP
    df['combined_value'] = df['VWAP'] - df['intraday_high_low_spread']
    df['volume_weighted_factor'] = df['combined_value'] * df['Volume']
    
    # Smooth the Factor using Exponential Moving Average (EMA)
    df['factor_ema_short'] = df['volume_weighted_factor'].ewm(span=3).mean()
    df['factor_ema_long'] = df['volume_weighted_factor'].ewm(span=10).mean()
    
    # Final factor value can be either the short or long EMA, depending on the desired reactivity
    df['alpha_factor'] = df['factor_ema_long']
    
    return df['alpha_factor']

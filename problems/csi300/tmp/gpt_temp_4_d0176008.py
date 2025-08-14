import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['High'] - df['Low']
    
    # Compute Previous Day's Close-to-Open Return
    prev_close_to_open_return = df['Close'].shift(1) - df['Open']
    
    # Calculate Volume Weighted Average Price (VWAP)
    vwap = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Combine Intraday Momentum and VWAP
    combined_momentum_vwap = (vwap - intraday_high_low_spread) * df['Volume']
    
    # Apply Exponential Moving Averages (EMA)
    ema_5_days = combined_momentum_vwap.ewm(span=5, adjust=False).mean()
    ema_20_days = combined_momentum_vwap.ewm(span=20, adjust=False).mean()
    
    # Final Alpha Factor
    alpha_factor = ema_5_days - ema_20_days
    
    return alpha_factor

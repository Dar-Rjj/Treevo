import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    intraday_high_low_spread = df['High'] - df['Low']
    
    # Compute Previous Day's Close-to-Open Return
    close_to_open_return = df['Close'].shift(1) - df['Open']
    
    # Calculate Intraday Momentum
    intraday_momentum = intraday_high_low_spread + close_to_open_return
    
    # Calculate Volume Weighted Average Price (VWAP)
    total_volume = df['Volume']
    vwap = (df['Open'] * df['Volume'] + df['High'] * df['Volume'] + df['Low'] * df['Volume'] + df['Close'] * df['Volume']) / (4 * total_volume)
    
    # Combine Intraday Momentum and VWAP
    combined_value = vwap - intraday_high_low_spread
    volume_weighted_combined_value = combined_value * df['Volume']
    
    # Calculate Relative Strength
    sma_5_day = df['Close'].rolling(window=5).mean()
    relative_strength = df['Close'] / sma_5_day
    
    # Adaptive Exponential Moving Average (EMA)
    alpha = 2 / (5 + 1) * relative_strength
    ema = df['Close'].ewm(span=5, adjust=False).mean()
    adaptive_ema = (1 - alpha) * ema.shift(1) + alpha * df['Close']
    adaptive_ema = adaptive_ema.fillna(method='bfill')
    
    # Include Trade Direction
    trade_direction = np.where(df['Close'] > df['Open'], 1, -1)
    
    # Final Alpha Factor
    final_alpha_factor = adaptive_ema * trade_direction
    
    return final_alpha_factor

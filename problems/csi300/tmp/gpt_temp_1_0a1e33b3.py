import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Compute Previous Day's Open-to-Close Return
    open_close_return = df['close'] - df['open'].shift(1)
    
    # Calculate Volume Weighted Average Price (VWAP)
    total_volume = df['volume']
    daily_vwap = ((df['high'] + df['low'] + df['close'] + df['open']) / 4 * df['volume']).cumsum() / total_volume.cumsum()
    
    # Combine Intraday Momentum and VWAP
    combined_value = daily_vwap - high_low_spread
    weighted_value = combined_value * df['volume']
    
    # Integrate Additional Market Signals
    ma_5 = df['close'].rolling(window=5).mean()
    ma_20 = df['close'].rolling(window=20).mean()
    
    # Adjust Weights Based on Market Conditions
    condition = ma_5 > ma_20
    adjusted_weights = np.where(condition, weighted_value * 1.2, weighted_value * 0.8)
    
    # Smooth the Factor
    ema_7 = adjusted_weights.ewm(span=7, adjust=False).mean()
    ema_21 = adjusted_weights.ewm(span=21, adjust=False).mean()
    
    final_factor = (ema_7 + ema_21) / 2
    
    return final_factor

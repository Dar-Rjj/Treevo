import pandas as pd
import numpy as pd

def heuristics_v2(df):
    # On-Balance Volume (OBV)
    obv = (df['close'] > df['close'].shift()).astype(int) * df['volume'] - (df['close'] < df['close'].shift()).astype(int) * df['volume']
    obv = obv.cumsum()
    obv_ema = obv.ewm(span=20, adjust=False).mean()
    
    # Bollinger Bands (BB) Width
    sma = df['close'].rolling(window=10).mean()
    std = df['close'].rolling(window=10).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    bb_width = upper_band - lower_band
    
    # Natural logarithm of Closing Price to BB Width Ratio
    close_to_bb_width_ratio_ln = np.log(df['close'] / bb_width)
    
    # Composite heuristic matrix
    heuristics_matrix = obv_ema + close_to_bb_width_ratio_ln
    
    return heuristics_matrix

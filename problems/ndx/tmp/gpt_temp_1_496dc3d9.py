import pandas as pd
import numpy as pd

def heuristics_v2(df):
    # Moving Average Convergence Divergence (MACD)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    upper_band = sma_20 + (std_20 * 2)
    lower_band = sma_20 - (std_20 * 2)
    bollinger_band_width = upper_band - lower_band
    
    # Natural logarithm of Closing Price to Bollinger Band Width Ratio
    close_to_bb_ratio_log = np.log(df['close'] / bollinger_band_width)
    
    # Composite heuristic matrix
    heuristics_matrix = 0.7 * macd + 0.3 * close_to_bb_ratio_log
    
    return heuristics_matrix

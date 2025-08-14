import pandas as pd
import numpy as pd

def heuristics_v2(df):
    # Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Average True Range (ATR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Natural Logarithm of Closing Price to ATR Ratio
    close_to_atr_ratio_ln = np.log(df['close'] / atr)
    
    # Composite heuristic matrix
    heuristics_matrix = 0.7 * rsi + 0.3 * close_to_atr_ratio_ln
    
    return heuristics_matrix

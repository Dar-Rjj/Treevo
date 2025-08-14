import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = (df['high'] - df['low']) / df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) / df['open']
    
    # Combine Intraday and Close-to-Open Returns
    weight_intraday = np.abs(intraday_return)
    weight_close_to_open = np.abs(close_to_open_return)
    total_weight = weight_intraday + weight_close_to_open
    
    preliminary_factor = (weight_intraday * intraday_return + weight_close_to_open * close_to_open_return) / total_weight
    
    # Calculate True Range (TR)
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate Positive Directional Movement (+DM)
    pos_dm = np.where((df['high'] > df['high'].shift(1)) & 
                      ((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low'])), 
                      df['high'] - df['high'].shift(1), 0)
    
    # Calculate Negative Directional Movement (-DM)
    neg_dm = np.where((df['low'].shift(1) > df['low']) & 
                      ((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1))), 
                      df['low'].shift(1) - df['low'], 0)
    
    # Smooth +DM and -DM over 14 periods
    pos_dm_smoothed = pos_dm.rolling(window=14).sum()
    neg_dm_smoothed = neg_dm.rolling(window=14).sum()
    
    # Calculate ATR (Average True Range) over 14 periods
    atr = true_range.rolling(window=14).mean()
    
    # Calculate +DI and -DI
    di_pos = pos_dm_smoothed / atr
    di_neg = neg_dm_smoothed / atr
    
    # Calculate ADMI
    admi = (di_pos - di_neg) / (di_pos + di_neg)
    
    # Synthesize Final Alpha Factor
    final_alpha_factor = preliminary_factor * admi
    
    return final_alpha_factor

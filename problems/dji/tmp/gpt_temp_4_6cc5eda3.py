import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume Weighted Reversal Signal
    conditions = [df['intraday_return'] > 0, df['intraday_return'] <= 0]
    choices = [-1 * df['volume'], 1 * df['volume']]
    df['volume_weighted_reversal'] = pd.np.select(conditions, choices, default=0)
    
    # Calculate ATR Component
    prev_close = df['close'].shift(1)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - prev_close).abs()
    low_close = (df['low'] - prev_close).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()  # Using a 14-day period for ATR
    
    # Calculate Final Momentum Indicator
    df['momentum_indicator'] = df['volume_weighted_reversal'] * atr
    final_momentum = df['momentum_indicator'].ewm(span=5, adjust=False).mean()  # Short EMA period of 5 days
    
    return final_momentum

import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate True Range
    df['true_range'] = df[['high', 'low']].apply(lambda x: x['high'] - x['low'], axis=1)
    df['true_range'] = df.apply(lambda x: max(x['true_range'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    
    # Compute 14-Day Average True Range
    df['avg_true_range'] = df['true_range'].rolling(window=14).mean()
    
    # Calculate ATR Based Momentum
    df['atr_based_momentum'] = (df['close'] - df['close'].shift(14)) / df['avg_true_range']
    
    return df['atr_based_momentum']

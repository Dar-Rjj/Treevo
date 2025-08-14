import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Compute Moving Averages of High-Low Range
    df['sma_10'] = df['high_low_range'].rolling(window=10).mean()
    df['ema_02'] = df['high_low_range'].ewm(alpha=0.2, adjust=False).mean()
    
    # Weighted Moving Average (WMA) for the last 5 days
    weights = np.array([1, 2, 3, 4, 5])
    df['wma_5'] = df['high_low_range'].rolling(window=5).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    
    # Determine Momentum
    def assign_signal(row):
        if row['high_low_range'] > row['sma_10'] and row['high_low_range'] > row['ema_02'] and row['high_low_range'] > row['wma_5']:
            return 'Strong Buy'
        elif row['high_low_range'] > row['sma_10'] and row['high_low_range'] > row['ema_02'] and row['high_low_range'] < row['wma_5']:
            return 'Weak Buy'
        elif row['high_low_range'] > row['sma_10'] and row['high_low_range'] < row['ema_02'] and row['high_low_range'] < row['wma_5']:
            return 'Neutral'
        elif row['high_low_range'] < row['sma_10'] and row['high_low_range'] > row['ema_02'] and row['high_low_range'] > row['wma_5']:
            return 'Weak Sell'
        else:
            return 'Strong Sell'
    
    df['signal'] = df.apply(assign_signal, axis=1)
    
    # Return the signal as a pandas Series
    return df['signal']

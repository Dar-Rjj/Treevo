import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Compute Moving Average of High-Low Range
    # Simple Moving Average (SMA) with a lookback period of 10 days
    df['sma_high_low_range'] = df['high_low_range'].rolling(window=10).mean()
    
    # Exponential Moving Average (EMA) with a smoothing factor of 0.2
    df['ema_high_low_range'] = df['high_low_range'].ewm(alpha=0.2, adjust=False).mean()
    
    # Determine Momentum
    conditions = [
        (df['high_low_range'] > df['sma_high_low_range']) & (df['high_low_range'] > df['ema_high_low_range']),
        (df['high_low_range'] > df['sma_high_low_range']) & (df['high_low_range'] < df['ema_high_low_range']),
        (df['high_low_range'] < df['sma_high_low_range']) & (df['high_low_range'] > df['ema_high_low_range']),
        (df['high_low_range'] < df['sma_high_low_range']) & (df['high_low_range'] < df['ema_high_low_range'])
    ]
    choices = ['Strong Buy', 'Weak Buy', 'Weak Sell', 'Strong Sell']
    
    df['signal'] = pd.Series(pd.np.select(conditions, choices, default='Neutral'), index=df.index)
    
    # Convert the signal to a numerical value for further use in quantitative analysis
    signal_mapping = {'Strong Buy': 2, 'Weak Buy': 1, 'Neutral': 0, 'Weak Sell': -1, 'Strong Sell': -2}
    df['factor_value'] = df['signal'].map(signal_mapping)
    
    return df['factor_value']

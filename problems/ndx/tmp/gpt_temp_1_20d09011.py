import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Momentum
    df['daily_momentum'] = df['close'] - df['close'].shift(1)
    
    # Calculate 20-Day Moving Average of Volume
    df['vol_20dma'] = df['volume'].rolling(window=20).mean()
    
    # Identify Volume Spike
    df['vol_spike'] = df['volume'] > 1.5 * df['vol_20dma']
    
    # Calculate 20-Day Moving Average of Amount
    df['amt_20dma'] = df['amount'].rolling(window=20).mean()
    
    # Identify Amount Spike
    df['amt_spike'] = df['amount'] > 1.5 * df['amt_20dma']
    
    # Determine Multiplier Based on Spike Indicators
    conditions = [
        (df['vol_spike'] & df['amt_spike']),
        (df['vol_spike'] | df['amt_spike'])
    ]
    choices = [3, 2]
    df['multiplier'] = pd.np.select(conditions, choices, default=1)
    
    # Final Factor Value
    df['factor'] = df['daily_momentum'] * df['multiplier']
    
    return df['factor']

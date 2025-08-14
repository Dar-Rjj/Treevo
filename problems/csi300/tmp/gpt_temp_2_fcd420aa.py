import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Returns
    df['daily_return'] = df['close'] - df['close'].shift(1)
    
    # Identify Volume Spike
    df['volume_ema_20'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['is_volume_spike'] = (df['volume'] > 3 * df['volume_ema_20'])
    
    # Identify Amount Spike
    df['amount_ema_20'] = df['amount'].ewm(span=20, adjust=False).mean()
    df['is_amount_spike'] = (df['amount'] > 3 * df['amount_ema_20'])
    
    # Adjust Daily Price Returns
    conditions = [
        (df['volume'] > 3 * df['volume_ema_20']) & (df['amount'] > 3 * df['amount_ema_20']),
        (df['volume'] > 2 * df['volume_ema_20']) | (df['amount'] > 2 * df['amount_ema_20'])
    ]
    choices = [5, 4]
    
    df['multiplier'] = pd.np.select(conditions, choices, default=1)
    df['adjusted_daily_return'] = df['daily_return'] * df['multiplier']
    
    return df['adjusted_daily_return']

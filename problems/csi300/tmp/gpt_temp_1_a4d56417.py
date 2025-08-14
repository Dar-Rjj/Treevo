import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate daily price returns using close price
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate 10-day exponential moving average of volume and amount
    df['ema_volume'] = df['volume'].ewm(span=10, adjust=False).mean()
    df['ema_amount'] = df['amount'].ewm(span=10, adjust=False).mean()
    
    # Identify volume and amount spike days
    df['volume_spike'] = df['volume'] > 2 * df['ema_volume']
    df['amount_spike'] = df['amount'] > 2 * df['ema_amount']
    
    # Combine price returns with volume and amount spikes
    conditions = [
        (df['volume_spike'] & df['amount_spike']),
        (df['volume_spike'] | df['amount_spike'])
    ]
    
    choices = [
        4 * df['daily_return'],
        3 * df['daily_return']
    ]
    
    df['heuristic_factor'] = pd.np.select(conditions, choices, default=df['daily_return'])
    
    return df['heuristic_factor'].dropna()

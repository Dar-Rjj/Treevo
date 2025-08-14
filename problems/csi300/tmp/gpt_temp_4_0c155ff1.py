import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Returns
    df['price_return'] = df['close'].pct_change()
    
    # Identify Volume Spike Days
    df['vol_moving_avg'] = df['volume'].rolling(window=10).mean()
    df['volume_spike'] = (df['volume'] > 2 * df['vol_moving_avg'])
    
    # Identify Amount Spike Days
    df['amount_moving_avg'] = df['amount'].rolling(window=10).mean()
    df['amount_spike'] = (df['amount'] > 2 * df['amount_moving_avg'])
    
    # Combine Price Returns with Volume and Amount Spikes
    conditions = [
        (df['volume_spike'] & df['amount_spike']),
        (df['volume_spike'] | df['amount_spike'])
    ]
    choices = [3, 2]
    
    df['multiplier'] = pd.np.select(conditions, choices, default=1)
    df['enhanced_price_return'] = df['price_return'] * df['multiplier']
    
    return df['enhanced_price_return']

import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Calculate daily return
    data['daily_return'] = data['close'].pct_change()
    
    # Calculate 20-day cumulative return, excluding the most recent return for stability
    data['20d_cumulative_return'] = data['daily_return'].shift(1).rolling(window=20).sum()
    
    # Identify synchronized volume patterns
    data['volume_change'] = data['volume'].diff()
    data['volume_sign'] = (data['volume_change'] > 0).astype(int)  # 1 if positive, 0 if negative or zero
    
    # Track consecutive positive and negative volume days
    data['consecutive_positive'] = data.groupby((data['volume_sign'] != data['volume_sign'].shift()).cumsum())['volume_sign'].cumsum()
    data['consecutive_negative'] = data.groupby((data['volume_sign'] != data['volume_sign'].shift()).cumsum())['volume_sign'].apply(lambda x: (~x.astype(bool)).cumsum())
    
    # Combine cumulative return with synchronized volume patterns
    data['alpha_factor'] = (data['20d_cumulative_return'] * data['consecutive_positive']) - (data['20d_cumulative_return'] * data['consecutive_negative'])
    
    return data['alpha_factor']

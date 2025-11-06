import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'] / data['close'].shift(1) - 1
    
    # Volatility Acceleration Analysis
    # Short-term volatility (5-day window)
    data['STV'] = data['returns'].rolling(window=5).std()
    # Medium-term volatility (10-day window)
    data['MTV'] = data['returns'].rolling(window=10).std()
    # Volatility acceleration
    data['vol_accel'] = (data['STV'] / data['STV'].shift(5)) - (data['MTV'] / data['MTV'].shift(5))
    # Volume-weighted acceleration
    data['vol_weighted_accel'] = data['vol_accel'] * (data['volume'] / (data['volume'].shift(5) + 0.001))
    
    # Price-Volume Divergence Detection
    # Price momentum (5-day)
    data['price_momentum'] = data['close'] / data['close'].shift(5) - 1
    # Volume momentum (5-day)
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    # Divergence strength
    data['div_strength'] = data['price_momentum'] / (abs(data['volume_momentum']) + 0.001)
    
    # Calculate consecutive divergence count
    data['divergence_flag'] = ((data['price_momentum'] > 0) & (data['volume_momentum'] < 0)).astype(int)
    data['consec_div'] = 0
    for i in range(1, len(data)):
        if data['divergence_flag'].iloc[i] == 1:
            data['consec_div'].iloc[i] = data['consec_div'].iloc[i-1] + 1
    
    # Intraday Reversal Dynamics
    # Intraday strength
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    # Reversal persistence
    data['reversal_flag'] = ((data['intraday_strength'] > 0.7) | (data['intraday_strength'] < 0.3)).astype(int)
    data['reversal_persistence'] = 0
    for i in range(1, len(data)):
        if data['reversal_flag'].iloc[i] == 1:
            data['reversal_persistence'].iloc[i] = data['reversal_persistence'].iloc[i-1] + 1
    # Volume-confirmed reversal
    data['vol_reversal'] = data['reversal_persistence'] * (data['volume'] / (data['volume'].shift(1) + 0.001))
    
    # True Range Efficiency
    data['true_range_eff'] = 0.0
    for i in range(5, len(data)):
        if i >= 5:
            price_change_5d = data['close'].iloc[i] / data['close'].iloc[i-5] - 1
            vol_sum = sum(abs(data['returns'].iloc[i-j] if i-j >= 0 else 0) for j in range(5))
            data['true_range_eff'].iloc[i] = price_change_5d / (vol_sum + 0.001)
    
    # Adaptive Alpha Signal Generation
    # Combine all components
    data['alpha_signal'] = (
        data['vol_weighted_accel'] * 
        data['div_strength'] * 
        (data['consec_div'] + 1) * 
        data['vol_reversal']
    ) / (abs(data['true_range_eff']) + 0.001)
    
    # Return the alpha signal series
    return data['alpha_signal']

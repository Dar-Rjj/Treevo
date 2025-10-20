import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate momentum components
    data['short_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_momentum'] = data['close'] / data['close'].shift(13) - 1
    
    # Detect momentum regime changes
    data['short_momentum_sign'] = np.sign(data['short_momentum'])
    data['medium_momentum_sign'] = np.sign(data['medium_momentum'])
    
    # Identify transitions when momentum signs change
    data['momentum_transition'] = (
        (data['short_momentum_sign'] != data['short_momentum_sign'].shift(1)) | 
        (data['medium_momentum_sign'] != data['medium_momentum_sign'].shift(1))
    ).astype(int)
    
    # Create transition state that persists for 5 days after change
    transition_state = np.zeros(len(data))
    for i in range(len(data)):
        if data['momentum_transition'].iloc[i] == 1:
            for j in range(min(5, len(data) - i)):
                transition_state[i + j] = max(transition_state[i + j], 1 - j/5)
    
    data['transition_weight'] = transition_state
    
    # Calculate volume pressure components
    # Up-day and down-day volume over 5-day window
    up_volume = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[data['close'].iloc[x.index].values > data['close'].iloc[x.index].shift(1).values]) if len(x) == 5 else np.nan, 
        raw=False
    )
    
    down_volume = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[data['close'].iloc[x.index].values <= data['close'].iloc[x.index].shift(1).values]) if len(x) == 5 else np.nan, 
        raw=False
    )
    
    data['volume_pressure_ratio'] = up_volume / (down_volume + 1e-8)
    
    # Volume momentum
    data['volume_momentum'] = data['volume'] / data['volume'].rolling(window=10).mean()
    
    # Combine into directional volume pressure signal
    data['volume_pressure'] = np.sign(data['short_momentum']) * data['volume_pressure_ratio'] * data['volume_momentum']
    
    # Assess price efficiency
    data['daily_range'] = data['high'] - data['low']
    data['range_utilization'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    
    # Track consecutive days with extreme range utilization
    extreme_high = (data['range_utilization'] > 0.7).astype(int)
    extreme_low = (data['range_utilization'] < 0.3).astype(int)
    
    # Calculate consecutive days count
    data['consecutive_extreme'] = 0
    for i in range(1, len(data)):
        if extreme_high.iloc[i] == 1 or extreme_low.iloc[i] == 1:
            data['consecutive_extreme'].iloc[i] = data['consecutive_extreme'].iloc[i-1] + 1
    
    # Efficiency persistence factor (decays with consecutive extremes)
    data['efficiency_persistence'] = np.exp(-data['consecutive_extreme'] / 3)
    
    # Generate final alpha factor
    data['momentum_transition_score'] = (
        data['transition_weight'] * 
        data['volume_pressure'] * 
        data['efficiency_persistence']
    )
    
    return data['momentum_transition_score']

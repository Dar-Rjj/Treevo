import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday return
    data['intraday_return'] = data['close'] / data['open'] - 1
    
    # Calculate reversal strength (current intraday return vs previous)
    data['prev_intraday_return'] = data['intraday_return'].shift(1)
    data['reversal_strength'] = -data['intraday_return'] * data['prev_intraday_return']
    
    # Calculate daily volatility proxy (High-Low range normalized by average price)
    data['avg_price'] = (data['high'] + data['low']) / 2
    data['volatility_proxy'] = (data['high'] - data['low']) / data['avg_price']
    
    # Apply volatility weighting to reversal strength
    data['vol_weighted_reversal'] = data['reversal_strength'] / (data['volatility_proxy'] + 1e-8)
    
    # Calculate volume persistence
    data['volume_change'] = data['volume'].pct_change()
    data['volume_direction'] = np.sign(data['volume_change'])
    
    # Initialize volume persistence streak
    data['volume_persistence'] = 0
    
    # Calculate consecutive volume persistence streak
    for i in range(1, len(data)):
        if data['volume_direction'].iloc[i] == data['volume_direction'].iloc[i-1]:
            data.loc[data.index[i], 'volume_persistence'] = data['volume_persistence'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'volume_persistence'] = 1
    
    # Combine volatility-weighted reversal with volume persistence
    data['alpha_factor'] = data['vol_weighted_reversal'] * (data['volume_persistence'] + 1)
    
    # Return the alpha factor series
    return data['alpha_factor']

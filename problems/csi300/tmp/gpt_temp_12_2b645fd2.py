import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Acceleration with Volume-Price Divergence factor
    Combines price momentum acceleration with volume-price relationship analysis
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Analysis
    # Calculate 3-day return
    ret_3d = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    
    # Calculate 10-day return  
    ret_10d = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Calculate momentum acceleration (3-day minus 10-day return)
    momentum_acceleration = ret_3d - ret_10d
    
    # Volume-Price Divergence Detection
    # Calculate 5-day volume SMA
    volume_sma_5d = data['volume'].rolling(window=5, min_periods=5).mean()
    
    # Compute volume ratio (current volume vs 5-day average)
    volume_ratio = data['volume'] / volume_sma_5d
    
    # Calculate 3-day volume-weighted price change
    def calc_vwap_change(window):
        if len(window) < 3:
            return np.nan
        volume_sum = window['volume'].sum()
        if volume_sum == 0:
            return np.nan
        price_changes = window['close'].diff().fillna(0)
        weighted_changes = (window['volume'] * price_changes).sum()
        return weighted_changes / volume_sum
    
    # Apply rolling calculation for volume-weighted price change
    vwap_change_3d = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        if i >= 2:
            window_data = data.iloc[i-2:i+1][['close', 'volume']]
            vwap_change_3d.iloc[i] = calc_vwap_change(window_data)
    
    # Compute volume-price divergence
    volume_price_divergence = vwap_change_3d - ret_3d
    
    # Factor Combination
    # Multiply momentum acceleration by volume-price divergence
    factor = momentum_acceleration * volume_price_divergence
    
    return factor

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Divergence
    # Compute Multiple Period Returns
    data['short_return'] = data['close'] / data['close'].shift(5) - 1
    data['medium_return'] = data['close'] / data['close'].shift(20) - 1
    data['long_return'] = data['close'] / data['close'].shift(60) - 1
    
    # Calculate Divergence Pattern
    data['short_medium_div'] = np.abs(data['short_return'] - data['medium_return'])
    data['medium_long_div'] = np.abs(data['medium_return'] - data['long_return'])
    
    # Identify directional alignment and calculate divergence composite score
    data['directional_alignment'] = np.sign(data['short_return']) * np.sign(data['medium_return']) * np.sign(data['long_return'])
    data['divergence_composite'] = (data['short_medium_div'] + data['medium_long_div']) * data['directional_alignment']
    
    # Volatility Regime Analysis with Asymmetry Weighting
    # Calculate Volatility Dynamics
    data['vol_10d'] = data['close'].rolling(window=10).std()
    data['vol_20d'] = data['close'].rolling(window=20).std()
    data['vol_ratio'] = data['vol_10d'] / data['vol_20d']
    
    # Incorporate Intraday Volatility Asymmetry
    data['daily_range'] = data['high'] - data['low']
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['volatility_weight'] = data['vol_ratio'] * np.sign(data['overnight_gap']) * data['daily_range']
    
    # Volume-Persistence Filter with Price Confirmation
    # Calculate Volume Acceleration Profile
    data['vol_growth_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['vol_growth_20d'] = data['volume'] / data['volume'].shift(20) - 1
    data['volume_momentum'] = (data['vol_growth_5d'] + data['vol_growth_20d']) / 2
    
    # Volume-Price Divergence Analysis
    data['price_range_5d'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / data['close']
    data['volume_to_range'] = data['volume'] / (data['high'] - data['low'])
    data['volume_sma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_surge'] = (data['volume'] > 1.2 * data['volume_sma_5d']).astype(int)
    
    # Track Persistence of Signal Alignment
    # Create alignment signal (divergence composite and volume momentum have same sign)
    data['alignment_signal'] = (np.sign(data['divergence_composite']) == np.sign(data['volume_momentum'])).astype(int)
    
    # Calculate persistence duration
    data['persistence_count'] = 0
    for i in range(1, len(data)):
        if data['alignment_signal'].iloc[i] == 1:
            data['persistence_count'].iloc[i] = data['persistence_count'].iloc[i-1] + 1
        else:
            data['persistence_count'].iloc[i] = 0
    
    data['persistence_weight'] = np.minimum(data['persistence_count'] / 5, 1.0)
    
    # Generate Final Factor with Efficiency Adjustment
    # Calculate Trading Efficiency Component
    data['range_efficiency'] = (data['high'] - data['low']) / data['amount']
    data['efficiency_component'] = 1 / (1 + np.abs(data['range_efficiency']))
    
    # Combine All Components
    data['base_signal'] = data['divergence_composite'] * data['volatility_weight']
    data['filtered_signal'] = data['base_signal'] * data['persistence_weight'] * data['volume_surge']
    data['final_factor'] = data['filtered_signal'] * data['efficiency_component'] * data['daily_range']
    
    return data['final_factor']

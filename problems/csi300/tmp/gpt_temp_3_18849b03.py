import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-timeframe Momentum Ratios
    data['mom_2d'] = data['close'] / data['close'].shift(2)
    data['mom_5d'] = data['close'] / data['close'].shift(5)
    data['short_to_medium_mom'] = data['mom_2d'] / data['mom_5d']
    
    # Acceleration Slope
    data['roc_2d'] = (data['close'] / data['close'].shift(2)) - 1
    data['roc_5d'] = (data['close'] / data['close'].shift(5)) - 1
    data['accel_slope'] = data['roc_2d'] - data['roc_5d']
    
    # Acceleration Dynamics
    data['roc_2d_1d'] = (data['close'].shift(1) / data['close'].shift(3)) - 1
    data['roc_5d_1d'] = (data['close'].shift(1) / data['close'].shift(6)) - 1
    data['accel_slope_1d'] = data['roc_2d_1d'] - data['roc_5d_1d']
    data['accel_change'] = data['accel_slope'] - data['accel_slope_1d']
    data['mom_persistence'] = np.sign(data['roc_2d']) * np.sign(data['roc_5d'])
    
    # Directional Volume Strength
    up_days = data['close'] > data['close'].shift(1)
    down_days = data['close'] < data['close'].shift(1)
    
    # Calculate rolling bull volume ratio (5-day window)
    bull_volume_ratio = []
    for i in range(len(data)):
        if i < 5:
            bull_volume_ratio.append(0.5)
            continue
            
        window_data = data.iloc[i-4:i+1]
        up_volume = window_data.loc[window_data['close'] > window_data['close'].shift(1), 'volume'].sum()
        down_volume = window_data.loc[window_data['close'] < window_data['close'].shift(1), 'volume'].sum()
        
        total_volume = up_volume + down_volume
        if total_volume > 0:
            ratio = up_volume / total_volume
        else:
            ratio = 0.5
        bull_volume_ratio.append(ratio)
    
    data['bull_volume_ratio'] = bull_volume_ratio
    data['volume_asymmetry'] = (data['bull_volume_ratio'] - 0.5) * 2
    
    # Volume-Confirmation Dynamics
    data['volume_expansion'] = (data['volume'] / data['volume'].shift(5)) * np.sign(data['accel_slope'])
    data['price_volume_convergence'] = np.sign(data['roc_2d']) * np.sign(data['volume'] - data['volume'].shift(2))
    data['volume_preceding_mom'] = np.sign(data['volume'].shift(1) - data['volume'].shift(2)) * np.sign(data['roc_2d'])
    
    # Gap-Enhanced Momentum Signals
    data['gap_momentum'] = (data['open'] / data['close'].shift(1)) - 1
    data['gap_strength'] = np.abs(data['gap_momentum']) * np.abs(data['accel_slope'])
    
    # Gap-Momentum Integration
    data['gap_enhanced_accel'] = data['accel_slope'] * (1 + data['gap_strength'] * np.sign(data['gap_momentum']))
    data['gap_volume_interaction'] = data['gap_momentum'] * data['volume_asymmetry']
    
    # Composite Signal Synthesis
    # Core Acceleration Factor
    data['volume_weighted_accel'] = data['accel_slope'] * data['volume_asymmetry']
    data['persistence_enhanced'] = data['volume_weighted_accel'] * data['mom_persistence']
    
    # Convergence Confirmation
    data['volume_price_alignment'] = data['volume_weighted_accel'] * data['price_volume_convergence']
    data['multi_timeframe_confirmation'] = data['volume_weighted_accel'] * data['volume_preceding_mom']
    
    # Final Factor Integration
    data['gap_enhanced_core'] = data['volume_weighted_accel'] * (1 + data['gap_strength'] * np.sign(data['gap_momentum']))
    data['final_factor'] = (data['gap_enhanced_core'] * 
                           np.sign(data['volume_asymmetry']) * 
                           np.sign(data['accel_change']))
    
    return data['final_factor']

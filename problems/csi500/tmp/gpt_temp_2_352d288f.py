import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Multi-Scale Momentum Fractals
    # Short-Term Momentum Persistence
    data['ret_1'] = data['close'].pct_change(1)
    data['ret_2'] = data['close'].pct_change(2)
    data['momentum_sign'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['close'].shift(1) - data['close'].shift(2))
    data['momentum_persistence'] = data['momentum_sign'] * np.abs(data['ret_1'] * data['ret_2'])
    
    # Detect Momentum Fractal Breakpoints
    data['is_local_max'] = (data['close'] > data['close'].shift(1)) & (data['close'] > data['close'].shift(-1))
    data['is_local_min'] = (data['close'] < data['close'].shift(1)) & (data['close'] < data['close'].shift(-1))
    
    # Calculate Fractal Momentum Intensity
    data['fractal_intensity'] = 0.0
    for i in range(2, len(data)-1):
        if data['is_local_max'].iloc[i] or data['is_local_min'].iloc[i]:
            # Find previous breakpoint
            for j in range(i-1, 1, -1):
                if data['is_local_max'].iloc[j] or data['is_local_min'].iloc[j]:
                    distance = np.abs(data['close'].iloc[i] - data['close'].iloc[j])
                    data['fractal_intensity'].iloc[i] = distance * data['volume'].iloc[i]
                    break
    
    # Volume-Weighted Momentum Acceleration
    data['vol_weighted_momentum'] = (data['close'] - data['close'].shift(1)) * np.log(1 + data['volume'])
    data['vol_weighted_momentum_prev'] = data['vol_weighted_momentum'].shift(1)
    data['momentum_acceleration'] = data['vol_weighted_momentum'] - data['vol_weighted_momentum_prev']
    
    # Acceleration divergence with sign consistency
    data['sign_consistency'] = np.sign(data['vol_weighted_momentum']) * np.sign(data['vol_weighted_momentum_prev'])
    data['acceleration_divergence'] = data['momentum_acceleration'] * data['sign_consistency']
    
    # Fractal Volume Distribution Analysis
    # Volume Clustering Asymmetry
    data['intraday_range'] = data['high'] - data['low']
    data['volume_concentration'] = data['intraday_range'] / (data['volume'] + 1e-8)
    data['volume_concentration_median'] = data['volume_concentration'].rolling(window=3, min_periods=1).median()
    
    # Volume Distribution Fractals
    data['volume_spike_intensity'] = data['volume'] / (data['intraday_range'] + 1e-8)
    data['volume_spike_median'] = data['volume_spike_intensity'].rolling(window=5, min_periods=1).median()
    data['volume_fractal_momentum'] = (data['volume_spike_intensity'] - data['volume_spike_median']) * data['ret_1']
    
    # Multi-Timeframe Fractal Alignment
    data['momentum_3d'] = data['close'].pct_change(3)
    data['momentum_7d'] = data['close'].pct_change(7)
    data['fractal_alignment'] = np.sign(data['momentum_3d']) * np.sign(data['momentum_7d'])
    data['fractal_convergence'] = data['momentum_3d'] * data['momentum_7d'] * data['volume_concentration']
    
    # Fractal Regime Transitions
    data['fractal_alignment_change'] = data['fractal_alignment'].diff()
    data['transition_momentum'] = np.abs(data['fractal_alignment_change']) * data['volume'] * np.sign(data['ret_1'])
    
    # Combine Fractal Components with Adaptive Thresholds
    # Multiply Momentum Fractals by Volume Distribution
    data['combined_fractal'] = data['fractal_intensity'] * data['volume_fractal_momentum']
    
    # Volatility-Dependent Thresholds
    data['range_volatility'] = data['intraday_range'].rolling(window=5, min_periods=1).std()
    data['adaptive_threshold'] = data['range_volatility'] * 0.1  # Scale factor
    
    # Dynamic Component Weighting
    data['regime_persistence'] = 0
    current_regime = 0
    persistence_count = 0
    for i in range(1, len(data)):
        if np.sign(data['combined_fractal'].iloc[i]) == np.sign(data['combined_fractal'].iloc[i-1]):
            persistence_count += 1
        else:
            persistence_count = 1
        data['regime_persistence'].iloc[i] = persistence_count
    
    # Apply Threshold-Based Signal Filtering
    data['filtered_signal'] = data['combined_fractal'].copy()
    threshold_mask = np.abs(data['combined_fractal']) < data['adaptive_threshold']
    data.loc[threshold_mask, 'filtered_signal'] = data.loc[threshold_mask, 'filtered_signal'] * 0.1  # Reduce weak signals
    
    # Weight by regime persistence
    data['weighted_signal'] = data['filtered_signal'] * np.sqrt(data['regime_persistence'])
    
    # Generate Final Alpha Factor
    # Combine with other fractal components
    data['final_alpha'] = (data['weighted_signal'] + 
                          data['acceleration_divergence'] * 0.3 + 
                          data['fractal_convergence'] * 0.2 + 
                          data['transition_momentum'] * 0.1)
    
    return data['final_alpha']

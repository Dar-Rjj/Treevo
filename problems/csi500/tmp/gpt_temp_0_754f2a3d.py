import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(3, len(data)):
        current_data = data.iloc[:i+1].copy()
        
        # Price Path Asymmetry
        # Recovery Speed: (close_t - low_t) / (high_t - low_t) after declines
        high_low_range = current_data.iloc[-1]['high'] - current_data.iloc[-1]['low']
        if high_low_range > 0:
            recovery_speed = (current_data.iloc[-1]['close'] - current_data.iloc[-1]['low']) / high_low_range
        else:
            recovery_speed = 0
        
        # Downward Persistence: count(close_t < close_{t-i} for i=1..3)
        downward_persistence = 0
        if i >= 3:
            for j in range(1, 4):
                if current_data.iloc[-1]['close'] < current_data.iloc[-1-j]['close']:
                    downward_persistence += 1
        
        # Volume-Weighted Price Extremes
        # Volume Concentration: volume_t / (high_t - low_t)
        if high_low_range > 0:
            volume_concentration = current_data.iloc[-1]['volume'] / high_low_range
        else:
            volume_concentration = 0
        
        # Thin-Volume Moves: (close_t - close_{t-1}) / volume_t
        if i >= 1 and current_data.iloc[-1]['volume'] > 0:
            thin_volume_moves = (current_data.iloc[-1]['close'] - current_data.iloc[-2]['close']) / current_data.iloc[-1]['volume']
        else:
            thin_volume_moves = 0
        
        # Opening vs Closing Dynamics
        # Session Momentum: (close_t - open_t) / (high_t - low_t)
        if high_low_range > 0:
            session_momentum = (current_data.iloc[-1]['close'] - current_data.iloc[-1]['open']) / high_low_range
        else:
            session_momentum = 0
        
        # Gap Preservation: (open_t - close_{t-1}) / (high_t - low_t)
        if i >= 1 and high_low_range > 0:
            gap_preservation = (current_data.iloc[-1]['open'] - current_data.iloc[-2]['close']) / high_low_range
        else:
            gap_preservation = 0
        
        # Price-Volume Divergence
        # Directional Mismatch: sign(close_t - close_{t-1}) * sign(volume_t - volume_{t-1})
        if i >= 1:
            price_change_sign = np.sign(current_data.iloc[-1]['close'] - current_data.iloc[-2]['close'])
            volume_change_sign = np.sign(current_data.iloc[-1]['volume'] - current_data.iloc[-2]['volume'])
            directional_mismatch = price_change_sign * volume_change_sign
        else:
            directional_mismatch = 0
        
        # Magnitude Discrepancy: |close_t - close_{t-1}| / volume_t
        if i >= 1 and current_data.iloc[-1]['volume'] > 0:
            magnitude_discrepancy = abs(current_data.iloc[-1]['close'] - current_data.iloc[-2]['close']) / current_data.iloc[-1]['volume']
        else:
            magnitude_discrepancy = 0
        
        # Multi-timeframe Momentum
        # Acceleration: (close_t - close_{t-1}) - (close_{t-1} - close_{t-2})
        if i >= 2:
            current_momentum = current_data.iloc[-1]['close'] - current_data.iloc[-2]['close']
            previous_momentum = current_data.iloc[-2]['close'] - current_data.iloc[-3]['close']
            acceleration = current_momentum - previous_momentum
        else:
            acceleration = 0
        
        # Consistency: count(consistent_sign(close_t - close_{t-i}) for i=1..3)
        consistency = 0
        if i >= 3:
            current_sign = np.sign(current_data.iloc[-1]['close'] - current_data.iloc[-2]['close'])
            for j in range(2, 4):
                if current_sign == np.sign(current_data.iloc[-1]['close'] - current_data.iloc[-1-j]['close']):
                    consistency += 1
        
        # Combine all components with appropriate weights
        factor_value = (
            recovery_speed * 0.15 +
            (downward_persistence / 3.0) * 0.12 +
            np.log1p(volume_concentration) * 0.10 +
            thin_volume_moves * 1000 * 0.08 +
            session_momentum * 0.15 +
            gap_preservation * 0.10 +
            directional_mismatch * 0.08 +
            magnitude_discrepancy * 1000 * 0.07 +
            acceleration * 10 * 0.08 +
            (consistency / 2.0) * 0.07
        )
        
        factor.iloc[i] = factor_value
    
    # Fill initial NaN values with 0
    factor = factor.fillna(0)
    
    return factor

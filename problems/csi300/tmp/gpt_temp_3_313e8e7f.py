import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap-Based Reversal Acceleration Component
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_momentum'] = (data['overnight_gap'] - data['overnight_gap'].shift(5)) / np.abs(data['overnight_gap'].shift(5)).replace(0, np.nan)
    data['recovery_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_reversal_acceleration'] = data['gap_momentum'] * data['recovery_efficiency']
    
    # Volatility-Adjusted Gap Factor
    data['gap_volatility'] = np.abs(data['overnight_gap']) * (data['high'] - data['low'])
    data['avg_gap_volatility'] = data['gap_volatility'].rolling(window=5, min_periods=3).mean()
    data['volatility_adjusted_factor'] = data['gap_reversal_acceleration'] / data['avg_gap_volatility'].replace(0, np.nan)
    
    # Asymmetric Volume Confirmation
    data['gap_volume_intensity'] = data['volume'] * np.abs(data['overnight_gap'])
    data['volume_gap_intensity_3d'] = data['gap_volume_intensity'].rolling(window=3, min_periods=2).mean()
    data['volume_gap_intensity_8d'] = data['gap_volume_intensity'].rolling(window=8, min_periods=5).mean()
    data['volume_gap_ratio'] = data['volume_gap_intensity_3d'] / data['volume_gap_intensity_8d'].replace(0, np.nan)
    
    data['recovery_volume_efficiency'] = data['volume'] * data['recovery_efficiency']
    data['recovery_volume_3d'] = data['recovery_volume_efficiency'].rolling(window=3, min_periods=2).mean()
    data['recovery_volume_8d'] = data['recovery_volume_efficiency'].rolling(window=8, min_periods=5).mean()
    data['recovery_volume_ratio'] = data['recovery_volume_3d'] / data['recovery_volume_8d'].replace(0, np.nan)
    
    # Asymmetric Gap Direction Enhancement
    direction_multiplier = np.ones(len(data))
    gap_up = data['overnight_gap'] > 0
    gap_down = data['overnight_gap'] < 0
    positive_recovery = data['recovery_efficiency'] > 0
    negative_recovery = data['recovery_efficiency'] < 0
    
    direction_multiplier[gap_up & positive_recovery] = 2.0
    direction_multiplier[gap_down & negative_recovery] = -2.0
    direction_multiplier[gap_up & negative_recovery] = -1.0
    direction_multiplier[gap_down & positive_recovery] = 1.0
    
    # Scale by absolute gap magnitude
    gap_magnitude_scaling = 1 + np.abs(data['overnight_gap'])
    direction_enhanced_factor = data['volatility_adjusted_factor'] * direction_multiplier * gap_magnitude_scaling
    
    # Volume-Enhanced Confirmation
    volume_confirmation = data['volume_gap_ratio'] * data['recovery_volume_ratio']
    
    # Gap direction consistency
    gap_direction_3d = data['overnight_gap'].rolling(window=3, min_periods=2).apply(
        lambda x: 1 if all(x > 0) or all(x < 0) else 0.5 if (x > 0).sum() >= 2 or (x < 0).sum() >= 2 else 0.2
    )
    
    # Primary Gap Reversal Component
    primary_component = direction_enhanced_factor * volume_confirmation * gap_direction_3d
    
    # Intraday Strength Refinement
    data['intraday_strength'] = (
        (data['close'] - np.minimum(data['low'], data['open'])) - 
        (np.maximum(data['high'], data['open']) - data['close'])
    ) / (data['high'] - data['low']).replace(0, np.nan)
    
    refined_signal = primary_component * (1 + 0.3 * data['intraday_strength'])
    
    # Volume Intensity Finalization
    data['avg_volume_20d'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_ratio'] = data['volume'] / data['avg_volume_20d'].replace(0, np.nan)
    
    # Final alpha factor
    alpha_factor = refined_signal * data['volume_ratio']
    
    return alpha_factor

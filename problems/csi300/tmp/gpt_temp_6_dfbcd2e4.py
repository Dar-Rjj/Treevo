import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adaptive Momentum Acceleration with Multi-Dimensional Confirmation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration Framework
    # Multi-Timeframe Momentum Calculation
    data['ultra_short_momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['short_momentum'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['medium_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['long_momentum'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Acceleration Hierarchy
    data['primary_acceleration'] = data['short_momentum'] - data['ultra_short_momentum']
    data['secondary_acceleration'] = data['medium_momentum'] - data['short_momentum']
    data['tertiary_acceleration'] = data['long_momentum'] - data['medium_momentum']
    
    # Acceleration consistency
    data['acceleration_consistency'] = (
        (data['primary_acceleration'] > 0).astype(int) +
        (data['secondary_acceleration'] > 0).astype(int) +
        (data['tertiary_acceleration'] > 0).astype(int)
    )
    
    # Momentum Quality Assessment
    # Direction persistence
    data['momentum_sign'] = np.sign(data['medium_momentum'])
    data['direction_persistence'] = 0
    for i in range(1, len(data)):
        if data['momentum_sign'].iloc[i] == data['momentum_sign'].iloc[i-1]:
            data['direction_persistence'].iloc[i] = data['direction_persistence'].iloc[i-1] + 1
    
    # Acceleration persistence
    data['acceleration_persistence'] = 0
    for i in range(1, len(data)):
        if data['acceleration_consistency'].iloc[i] >= 2:
            data['acceleration_persistence'].iloc[i] = data['acceleration_persistence'].iloc[i-1] + 1
    
    # Momentum strength
    data['momentum_strength'] = data['medium_momentum'] * (1 + data['direction_persistence'] / 15)
    
    # Volume Confirmation Engine
    # Volume Dynamics Analysis
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_trend'] = data['volume'] / data['volume'].shift(3)
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    data['volume_acceleration'] = data['volume_momentum'] - data['volume_ratio']
    
    # Multi-Dimensional Price-Volume Alignment
    data['direction_alignment'] = np.sign(data['volume_ratio'] - 1) * np.sign(data['ultra_short_momentum'])
    data['trend_alignment'] = np.sign(data['volume_trend'] - 1) * np.sign(data['medium_momentum'])
    data['acceleration_alignment'] = np.sign(data['volume_acceleration']) * np.sign(data['primary_acceleration'])
    data['momentum_alignment'] = np.sign(data['volume_momentum']) * np.sign(data['secondary_acceleration'])
    
    # Volume Confidence Matrix
    alignment_scores = (
        (data['direction_alignment'] > 0).astype(int) +
        (data['trend_alignment'] > 0).astype(int) +
        (data['acceleration_alignment'] > 0).astype(int) +
        (data['momentum_alignment'] > 0).astype(int)
    )
    
    data['volume_confidence_level'] = np.where(alignment_scores >= 3, 3, 
                                              np.where(alignment_scores == 2, 2, 
                                                      np.where(alignment_scores >= 1, 1, 0)))
    
    data['volume_multiplier'] = 0.8 + (data['volume_confidence_level'] * 0.2)
    
    # Volatility Adaptive Framework
    # Intraday Volatility Metrics
    data['normalized_range'] = (data['high'] - data['low']) / data['close']
    data['range_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 0.001)
    data['volatility_momentum'] = data['normalized_range'] / data['normalized_range'].shift(1).replace(0, 0.001)
    data['efficiency_trend'] = data['range_efficiency'] - data['range_efficiency'].shift(1)
    
    # Volatility Context Assessment
    data['recent_volatility'] = data['normalized_range'].rolling(window=3, min_periods=1).mean()
    data['volatility_stability'] = 1 / (np.abs(data['normalized_range'] - data['normalized_range'].shift(1)) + 0.001)
    
    # Efficiency stability
    data['efficiency_stability'] = 0
    for i in range(1, len(data)):
        if data['range_efficiency'].iloc[i] > 0.5:
            data['efficiency_stability'].iloc[i] = data['efficiency_stability'].iloc[i-1] + 1
    
    data['volatility_regime'] = np.where(data['recent_volatility'] > 0.02, "high", "normal")
    
    # Volatility Scaling Components
    data['range_adjustment'] = 1 / data['normalized_range'].replace(0, 0.001)
    data['stability_multiplier'] = data['volatility_stability'] * (1 + data['efficiency_stability'] / 10)
    data['regime_adaptation'] = np.where(data['volatility_regime'] == "high", 1.2, 1.0)
    data['volatility_confidence'] = data['stability_multiplier'] * data['regime_adaptation']
    
    # Multi-Dimensional Signal Construction
    # Base Acceleration Signal
    data['core_momentum'] = data['momentum_strength'] * (1 + data['primary_acceleration'])
    data['hierarchy_enhanced'] = data['core_momentum'] * (1 + data['acceleration_consistency'] / 10)
    data['persistence_boosted'] = data['hierarchy_enhanced'] * (1 + data['acceleration_persistence'] / 20)
    
    # Confidence-Based Enhancement
    data['volume_confirmed'] = data['persistence_boosted'] * data['volume_multiplier']
    data['volatility_scaled'] = data['volume_confirmed'] * data['range_adjustment']
    data['stability_applied'] = data['volatility_scaled'] * data['volatility_confidence']
    
    # Final Factor Assembly
    factor = data['stability_applied'] * 100
    
    return factor

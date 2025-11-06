import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Confirmed Momentum Acceleration with Volatility Scaling factor
    Combines momentum acceleration, volume confirmation, and volatility scaling
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Core Momentum Acceleration
    # Price Momentum Components
    data['short_momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['medium_momentum'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['long_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Acceleration Signals
    data['short_acceleration'] = data['medium_momentum'] - data['short_momentum']
    data['long_acceleration'] = data['long_momentum'] - data['medium_momentum']
    
    # Momentum Quality Assessment
    # Direction persistence: count consecutive days with same medium_momentum sign
    data['momentum_sign'] = np.sign(data['medium_momentum'])
    data['direction_persistence'] = 0
    for i in range(1, len(data)):
        if data['momentum_sign'].iloc[i] == data['momentum_sign'].iloc[i-1]:
            data['direction_persistence'].iloc[i] = data['direction_persistence'].iloc[i-1] + 1
        else:
            data['direction_persistence'].iloc[i] = 0
    
    # Acceleration persistence: count consecutive days where short_acceleration and long_acceleration have same sign
    data['acceleration_aligned'] = (np.sign(data['short_acceleration']) == np.sign(data['long_acceleration'])).astype(int)
    data['acceleration_persistence'] = 0
    for i in range(1, len(data)):
        if data['acceleration_aligned'].iloc[i] == 1:
            data['acceleration_persistence'].iloc[i] = data['acceleration_persistence'].iloc[i-1] + 1
        else:
            data['acceleration_persistence'].iloc[i] = 0
    
    # Volume Confirmation Engine
    # Volume Dynamics
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_trend'] = data['volume'] / data['volume'].shift(3)
    
    # Price-Volume Alignment
    data['short_alignment'] = np.sign(data['volume_ratio'] - 1) * np.sign(data['short_momentum'])
    data['medium_alignment'] = np.sign(data['volume_trend'] - 1) * np.sign(data['medium_momentum'])
    data['acceleration_alignment'] = np.sign(data['volume_trend'] - data['volume_ratio']) * np.sign(data['short_acceleration'])
    
    # Count positive alignments (volume confidence)
    data['positive_alignments'] = ((data['short_alignment'] > 0).astype(int) + 
                                  (data['medium_alignment'] > 0).astype(int) + 
                                  (data['acceleration_alignment'] > 0).astype(int))
    
    # Volatility Context Framework
    # Price Range Analysis
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    
    # Range persistence: count consecutive days where daily_range within ±20% of 5-day average
    data['range_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['range_within_threshold'] = (abs(data['daily_range'] - data['range_5d_avg']) / data['range_5d_avg'] <= 0.2).astype(int)
    data['range_persistence'] = 0
    for i in range(1, len(data)):
        if data['range_within_threshold'].iloc[i] == 1:
            data['range_persistence'].iloc[i] = data['range_persistence'].iloc[i-1] + 1
        else:
            data['range_persistence'].iloc[i] = 0
    
    # Volatility Scaling
    data['recent_volatility'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['volatility_stability'] = 1 / (data['daily_range'].rolling(window=5, min_periods=1).std() + 0.001)
    
    # Signal Integration & Confidence
    # Base Signal Construction
    data['accelerated_momentum'] = (data['medium_momentum'] * 
                                   (1 + data['direction_persistence'] / 10) * 
                                   (1 + data['short_acceleration']))
    
    data['volume_enhanced'] = data['accelerated_momentum'] * (1 + data['positive_alignments'] * 0.2)
    
    # Confidence Assessment
    data['volatility_confidence'] = ((data['range_persistence'] >= 2) & 
                                    (data['volatility_stability'] > 1.0)).astype(int)
    
    # Final Factor Calculation
    data['volatility_scaled'] = data['volume_enhanced'] / (data['recent_volatility'] + 0.001)
    data['confidence_adjusted'] = data['volatility_scaled'] * data['volatility_stability']
    data['final_factor'] = data['confidence_adjusted'] * 1000
    
    # Return the final factor series
    return data['final_factor']

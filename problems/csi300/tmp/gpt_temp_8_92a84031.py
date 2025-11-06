import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Price Acceleration
    # High Price Acceleration
    data['short_term_high_momentum'] = (data['high'] - data['high'].shift(2)) / data['high'].shift(2)
    data['medium_term_high_momentum'] = (data['high'] - data['high'].shift(5)) / data['high'].shift(5)
    data['high_acceleration_ratio'] = data['short_term_high_momentum'] / data['medium_term_high_momentum']
    
    # Low Price Acceleration
    data['short_term_low_momentum'] = (data['low'] - data['low'].shift(2)) / data['low'].shift(2)
    data['medium_term_low_momentum'] = (data['low'] - data['low'].shift(5)) / data['low'].shift(5)
    data['low_acceleration_ratio'] = data['short_term_low_momentum'] / data['medium_term_low_momentum']
    
    # Volume-Confirmed Acceleration Signals
    # Volume Acceleration Component
    data['short_term_volume_change'] = data['volume'] / data['volume'].shift(2) - 1
    data['medium_term_volume_change'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_acceleration'] = data['short_term_volume_change'] / data['medium_term_volume_change']
    
    # Price-Volume Synchronization
    data['high_low_acceleration_spread'] = data['high_acceleration_ratio'] - data['low_acceleration_ratio']
    data['volume_weighted_spread'] = data['high_low_acceleration_spread'] * data['volume_acceleration']
    data['synchronization_score'] = np.sign(data['high_low_acceleration_spread']) * data['volume_weighted_spread']
    
    # Intraday Pressure Dynamics Integration
    # Opening Pressure Assessment
    high_low_range = data['high'] - data['low']
    high_low_range = high_low_range.replace(0, np.nan)  # Avoid division by zero
    data['opening_position'] = (data['open'] - data['low']) / high_low_range
    data['opening_momentum'] = data['opening_position'] * (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Closing Pressure Assessment
    data['closing_position'] = (data['close'] - data['low']) / high_low_range
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    data['pressure_divergence'] = data['opening_position'] - data['closing_position']
    
    # Volatility-Adjusted Signal Combination
    # Price Range Volatility Component
    data['daily_range'] = data['high'] - data['low']
    data['avg_range'] = data['daily_range'].rolling(window=10, min_periods=1).mean().shift(1)
    data['range_volatility_ratio'] = data['daily_range'] / data['avg_range']
    
    # Volatility-Confidence Weighting
    data['volatility_confidence'] = 1 / (1 + data['range_volatility_ratio'])
    data['core_acceleration_signal'] = data['synchronization_score'] * data['pressure_divergence']
    data['volatility_adjusted_signal'] = data['core_acceleration_signal'] * data['volatility_confidence']
    
    # Pattern Persistence Validation
    # Historical Pattern Consistency
    feature_cols = ['high_acceleration_ratio', 'low_acceleration_ratio', 'volume_acceleration']
    
    def calculate_pattern_distance(row_idx, data, feature_cols, window=5):
        if row_idx < window:
            return np.nan
        
        current_features = data[feature_cols].iloc[row_idx].values
        past_features = data[feature_cols].iloc[row_idx-window:row_idx].values
        
        distances = []
        for i in range(len(past_features)):
            dist = np.sqrt(np.sum((current_features - past_features[i])**2))
            distances.append(dist)
        
        return np.mean(distances)
    
    pattern_distances = []
    for i in range(len(data)):
        pattern_distances.append(calculate_pattern_distance(i, data, feature_cols))
    
    data['pattern_distance'] = pattern_distances
    
    # Gap Momentum Integration
    data['overnight_gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_adjusted_signal'] = data['volatility_adjusted_signal'] * data['overnight_gap_momentum']
    
    # Final Alpha Factor
    data['pattern_confidence'] = 1 / (1 + data['pattern_distance'])
    data['volume_enhanced_acceleration_divergence'] = data['gap_adjusted_signal'] * data['pattern_confidence']
    
    # Return the final factor series
    return data['volume_enhanced_acceleration_divergence']

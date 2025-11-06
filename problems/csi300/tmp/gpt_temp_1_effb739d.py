import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Acceleration with Volume Alignment and Volatility Scaling
    """
    data = df.copy()
    
    # Momentum Acceleration Framework
    # Multi-Timeframe Momentum Calculation
    data['ultra_short_momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['short_momentum'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['medium_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['long_momentum'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Acceleration Signal Generation
    data['immediate_acceleration'] = data['short_momentum'] - data['ultra_short_momentum']
    data['medium_acceleration'] = data['medium_momentum'] - data['short_momentum']
    data['long_acceleration'] = data['long_momentum'] - data['medium_momentum']
    
    # Acceleration consistency: count positive signs among three acceleration signals
    data['acceleration_consistency'] = (
        (data['immediate_acceleration'] > 0).astype(int) +
        (data['medium_acceleration'] > 0).astype(int) +
        (data['long_acceleration'] > 0).astype(int)
    )
    
    # Acceleration Quality Assessment
    data['direction_coherence'] = np.sign(data['medium_momentum']) * (data['acceleration_consistency'] - 1.5)
    
    # Acceleration persistence: count consecutive days with acceleration_consistency ≥ 2
    data['acceleration_persistence'] = 0
    persistence_count = 0
    for i in range(len(data)):
        if data['acceleration_consistency'].iloc[i] >= 2:
            persistence_count += 1
        else:
            persistence_count = 0
        data['acceleration_persistence'].iloc[i] = persistence_count
    
    data['momentum_strength'] = data['medium_momentum'] * (1 + data['acceleration_persistence'] / 5)
    
    # Multi-Timeframe Volume Alignment Engine
    # Volume Momentum Calculation
    data['volume_change_1day'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 1e-8)
    data['volume_change_3day'] = (data['volume'] - data['volume'].shift(3)) / (data['volume'].shift(3) + 1e-8)
    data['volume_change_5day'] = (data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 1e-8)
    data['volume_acceleration'] = data['volume_change_3day'] - data['volume_change_1day']
    
    # Price-Volume Alignment Scoring
    data['short_term_alignment'] = np.sign(data['volume_change_1day']) * np.sign(data['ultra_short_momentum'])
    data['medium_term_alignment'] = np.sign(data['volume_change_3day']) * np.sign(data['short_momentum'])
    data['long_term_alignment'] = np.sign(data['volume_change_5day']) * np.sign(data['medium_momentum'])
    data['acceleration_alignment'] = np.sign(data['volume_acceleration']) * np.sign(data['immediate_acceleration'])
    
    # Volume Confidence Assessment
    alignment_indicators = [
        (data['short_term_alignment'] > 0).astype(int),
        (data['medium_term_alignment'] > 0).astype(int),
        (data['long_term_alignment'] > 0).astype(int),
        (data['acceleration_alignment'] > 0).astype(int)
    ]
    data['alignment_score'] = sum(alignment_indicators)
    data['volume_strength'] = np.abs(data['volume_change_3day']) * data['alignment_score']
    data['volume_multiplier'] = 1 + (data['volume_strength'] * 0.1)
    
    # Volatility-Aware Scaling Framework
    # Dynamic Volatility Measurement
    data['daily_range_volatility'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    data['close_to_close_volatility'] = np.abs(data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['volatility_ratio'] = data['daily_range_volatility'] / (data['close_to_close_volatility'] + 1e-8)
    data['volatility_trend'] = data['daily_range_volatility'] / data['daily_range_volatility'].rolling(window=5).mean()
    
    # Volatility Context Classification
    data['volatility_regime'] = 'normal'
    data.loc[data['volatility_trend'] > 1.2, 'volatility_regime'] = 'high'
    data.loc[data['volatility_trend'] < 0.8, 'volatility_regime'] = 'low'
    
    # Volatility stability
    data['volatility_stability'] = 1 / (data['daily_range_volatility'].rolling(window=5).std() + 0.001)
    
    # Volatility-Adjusted Scaling
    data['base_volatility_adjustment'] = 1 / (data['daily_range_volatility'] + 1e-8)
    
    # Regime multiplier
    regime_multiplier = {
        'high': 1.5,
        'normal': 1.0,
        'low': 0.7
    }
    data['regime_multiplier'] = data['volatility_regime'].map(regime_multiplier)
    
    data['stability_enhancement'] = data['regime_multiplier'] * data['volatility_stability']
    data['final_volatility_scaling'] = data['base_volatility_adjustment'] * data['stability_enhancement']
    
    # Integrated Signal Construction
    # Core Signal Generation
    data['accelerated_momentum'] = data['momentum_strength'] * data['direction_coherence']
    data['volume_enhanced'] = data['accelerated_momentum'] * data['volume_multiplier']
    data['persistence_boosted'] = data['volume_enhanced'] * (1 + data['acceleration_persistence'] / 10)
    
    # Confidence-Based Filtering
    data['confidence_level'] = 'low'
    data.loc[(data['alignment_score'] >= 2) | (data['acceleration_consistency'] >= 2), 'confidence_level'] = 'medium'
    data.loc[(data['alignment_score'] >= 3) & (data['acceleration_consistency'] >= 2), 'confidence_level'] = 'high'
    
    confidence_multiplier = {
        'high': 1.2,
        'medium': 1.0,
        'low': 0.8
    }
    data['confidence_multiplier'] = data['confidence_level'].map(confidence_multiplier)
    
    # Volatility-Scaled Final Factor
    data['confidence_weighted'] = data['persistence_boosted'] * data['confidence_multiplier']
    data['volatility_scaled'] = data['confidence_weighted'] * data['final_volatility_scaling']
    data['final_alpha'] = data['volatility_scaled'] * 100
    
    return data['final_alpha']

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Momentum-Volume Persistence Factor
    Combines multi-timeframe momentum with volume confirmation and volatility adaptation
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Persistence Framework
    # Multi-Timeframe Momentum
    data['momentum_1d'] = data['close'] - data['close'].shift(1)
    data['momentum_3d'] = data['close'] - data['close'].shift(2)
    data['momentum_5d'] = data['close'] - data['close'].shift(4)
    
    # Momentum Quality Assessment
    data['momentum_sign_1'] = np.sign(data['momentum_1d'])
    data['momentum_sign_2'] = np.sign(data['momentum_1d'].shift(1))
    data['momentum_sign_3'] = np.sign(data['momentum_1d'].shift(2))
    
    # Direction Consistency: count of sign matches in last 3 daily momentum
    data['direction_consistency'] = (
        (data['momentum_sign_1'] == data['momentum_sign_2']).astype(int) +
        (data['momentum_sign_1'] == data['momentum_sign_3']).astype(int) +
        (data['momentum_sign_2'] == data['momentum_sign_3']).astype(int)
    ) / 3.0
    
    # Momentum Acceleration
    data['momentum_acceleration'] = data['momentum_1d'] - data['momentum_1d'].shift(1)
    
    # Gap Persistence
    data['gap_persistence'] = (data['open'] - data['close'].shift(1)) * np.sign(data['momentum_1d'])
    
    # Momentum Strength Indicators
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['momentum_magnitude'] = np.abs(data['momentum_1d']) / (data['high'] - data['low']).replace(0, np.nan)
    data['persistence_score'] = data['direction_consistency'] * np.sign(data['momentum_acceleration'])
    
    # Volume Alignment System
    # Volume Trend Analysis
    data['volume_direction'] = np.sign(data['volume'] - data['volume'].shift(1))
    
    # Volume Streak calculation
    data['volume_streak'] = 0
    for i in range(1, len(data)):
        if data['volume_direction'].iloc[i] == data['volume_direction'].iloc[i-1]:
            data['volume_streak'].iloc[i] = data['volume_streak'].iloc[i-1] + 1
        else:
            data['volume_streak'].iloc[i] = 1
    
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Volume-Momentum Integration
    data['direction_alignment'] = np.sign(data['momentum_1d']) * data['volume_direction']
    data['strength_correlation'] = np.abs(data['momentum_1d']) * np.abs(data['volume_momentum'])
    data['persistence_alignment'] = data['volume_streak'] * data['direction_alignment']
    
    # Volume Regime Detection
    data['volume_spike'] = (data['volume'] > 1.8 * data['volume'].shift(1)).astype(float)
    data['volume_decline'] = (data['volume'] < 0.6 * data['volume'].shift(1)).astype(float)
    data['volume_stability'] = 1.0 + 0.4 * data['volume_spike'] - 0.3 * data['volume_decline']
    
    # Volatility Context Adaptation
    # Range-Based Volatility Measures
    data['daily_range'] = data['high'] - data['low']
    data['avg_3d_range'] = (data['daily_range'] + data['daily_range'].shift(1) + data['daily_range'].shift(2)) / 3
    data['volatility_ratio'] = data['daily_range'] / data['avg_3d_range'].replace(0, np.nan)
    
    # Volatility Regime Classification
    conditions = [
        data['volatility_ratio'] > 1.25,
        (data['volatility_ratio'] >= 0.75) & (data['volatility_ratio'] <= 1.25),
        data['volatility_ratio'] < 0.75
    ]
    
    # Momentum Timeframe Weights
    weights_high = [0.7, 0.2, 0.1]  # High volatility
    weights_normal = [0.5, 0.3, 0.2]  # Normal volatility
    weights_low = [0.3, 0.4, 0.3]  # Low volatility
    
    data['momentum_weights_1d'] = np.select(conditions, [weights_high[0], weights_normal[0], weights_low[0]])
    data['momentum_weights_3d'] = np.select(conditions, [weights_high[1], weights_normal[1], weights_low[1]])
    data['momentum_weights_5d'] = np.select(conditions, [weights_high[2], weights_normal[2], weights_low[2]])
    
    # Volatility Scaling Factors
    data['volatility_scaling'] = np.select(conditions, [0.6, 1.0, 1.4])
    
    # Factor Construction & Integration
    # Core Momentum Signal
    data['weighted_momentum'] = (
        data['momentum_1d'] * data['momentum_weights_1d'] +
        data['momentum_3d'] * data['momentum_weights_3d'] +
        data['momentum_5d'] * data['momentum_weights_5d']
    )
    
    data['quality_adjusted_momentum'] = data['weighted_momentum'] * (1 + 0.1 * data['persistence_score'])
    data['range_normalized_momentum'] = data['quality_adjusted_momentum'] / data['daily_range'].replace(0, np.nan)
    
    # Volume Confirmation Layer
    data['volume_enhanced_momentum'] = data['range_normalized_momentum'] * data['volume_stability']
    data['alignment_boost'] = data['volume_enhanced_momentum'] * (1 + 0.15 * data['persistence_alignment'])
    data['strength_confirmation'] = data['alignment_boost'] * (1 + 0.1 * data['strength_correlation'])
    
    # Volatility Adaptation
    data['regime_adapted_signal'] = data['strength_confirmation'] * data['volatility_scaling']
    
    # Final Factor Value
    factor = data['regime_adapted_signal']
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor.dropna()

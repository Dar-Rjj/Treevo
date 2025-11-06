import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Volume-Aligned Momentum Acceleration Factor
    """
    df = data.copy()
    
    # Multi-Timeframe Price Momentum
    df['very_short_momentum'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['short_term_momentum'] = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    df['medium_term_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['long_term_momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Acceleration Hierarchy
    df['primary_acceleration'] = df['short_term_momentum'] - df['very_short_momentum']
    df['secondary_acceleration'] = df['medium_term_momentum'] - df['short_term_momentum']
    df['tertiary_acceleration'] = df['long_term_momentum'] - df['medium_term_momentum']
    
    # Acceleration Consistency Score
    acceleration_signals = pd.DataFrame({
        'primary': (df['primary_acceleration'] > 0).astype(int),
        'secondary': (df['secondary_acceleration'] > 0).astype(int),
        'tertiary': (df['tertiary_acceleration'] > 0).astype(int)
    })
    weights = [3, 2, 1]
    df['acceleration_consistency_score'] = (
        acceleration_signals['primary'] * weights[0] + 
        acceleration_signals['secondary'] * weights[1] + 
        acceleration_signals['tertiary'] * weights[2]
    ) / 6
    
    # Momentum Persistence Assessment
    df['medium_persistence'] = (df['medium_term_momentum'] > 0).astype(int)
    df['long_persistence'] = (df['long_term_momentum'] > 0).astype(int)
    
    # Calculate consecutive positive days for medium and long term momentum
    for window in [5, 10]:
        df[f'persistence_{window}'] = 0
        for i in range(len(df)):
            if i >= window:
                current_momentum = df[f'{"medium" if window == 5 else "long"}_term_momentum'].iloc[i]
                if current_momentum > 0:
                    count = 1
                    for j in range(1, window):
                        if df[f'{"medium" if window == 5 else "long"}_term_momentum'].iloc[i-j] > 0:
                            count += 1
                        else:
                            break
                    df.loc[df.index[i], f'persistence_{window}'] = count
    
    df['persistence_score'] = (df['persistence_5'] + df['persistence_10']) / 2
    
    # Acceleration Persistence
    df['acceleration_persistence'] = 0
    for i in range(len(df)):
        if i > 0 and df['acceleration_consistency_score'].iloc[i] > 0:
            if df['acceleration_consistency_score'].iloc[i-1] > 0:
                df.loc[df.index[i], 'acceleration_persistence'] = df['acceleration_persistence'].iloc[i-1] + 1
            else:
                df.loc[df.index[i], 'acceleration_persistence'] = 1
    
    df['max_acceleration_persistence'] = df['acceleration_persistence'].expanding().max()
    df['relative_persistence'] = df['acceleration_persistence'] / (df['max_acceleration_persistence'] + 0.001)
    
    # Volume Alignment Engine
    df['daily_volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['short_term_volume_trend'] = df['volume'] / df['volume'].shift(3)
    df['medium_term_volume_trend'] = df['volume'] / df['volume'].shift(7)
    df['volume_acceleration'] = df['short_term_volume_trend'] - df['daily_volume_ratio']
    
    # Cross-Timeframe Volume-Price Alignment
    # Short-term alignment
    df['short_dir_match'] = np.sign(df['daily_volume_ratio'] - 1) * np.sign(df['very_short_momentum'])
    df['short_strength_match'] = abs(df['daily_volume_ratio'] - 1) * abs(df['very_short_momentum'])
    
    # Medium-term alignment
    df['medium_dir_match'] = np.sign(df['short_term_volume_trend'] - 1) * np.sign(df['short_term_momentum'])
    df['medium_strength_match'] = abs(df['short_term_volume_trend'] - 1) * abs(df['short_term_momentum'])
    
    # Long-term alignment
    df['long_dir_match'] = np.sign(df['medium_term_volume_trend'] - 1) * np.sign(df['medium_term_momentum'])
    df['long_strength_match'] = abs(df['medium_term_volume_trend'] - 1) * abs(df['medium_term_momentum'])
    
    # Volume Confidence Scoring
    direction_matches = pd.DataFrame({
        'short': (df['short_dir_match'] > 0).astype(int),
        'medium': (df['medium_dir_match'] > 0).astype(int),
        'long': (df['long_dir_match'] > 0).astype(int)
    })
    df['direction_count'] = direction_matches.sum(axis=1)
    df['average_strength'] = (df['short_strength_match'] + df['medium_strength_match'] + df['long_strength_match']) / 3
    df['volume_alignment_score'] = (df['direction_count'] / 3) * df['average_strength']
    
    # Volume Multiplier Construction
    df['base_multiplier'] = 1 + df['volume_alignment_score']
    df['acceleration_bonus'] = 1 + (df['volume_acceleration'] * 0.5)
    df['volume_multiplier'] = df['base_multiplier'] * df['acceleration_bonus']
    
    # Volatility Context Assessment
    df['normalized_daily_range'] = (df['high'] - df['low']) / df['close']
    df['range_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
    
    # Range persistence calculation
    df['range_persistence'] = 0
    for i in range(len(df)):
        if i >= 5:
            current_range = df['normalized_daily_range'].iloc[i]
            similar_count = 0
            for j in range(1, 6):
                prev_range = df['normalized_daily_range'].iloc[i-j]
                if abs(current_range - prev_range) / prev_range <= 0.15:
                    similar_count += 1
            df.loc[df.index[i], 'range_persistence'] = similar_count
    
    # Volatility Stability Metrics
    df['recent_volatility'] = df['normalized_daily_range'].rolling(window=5, min_periods=1).mean()
    df['volatility_consistency'] = 1 / (df['normalized_daily_range'].rolling(window=5, min_periods=1).std() + 0.001)
    df['range_adjusted_momentum'] = df['medium_term_momentum'] / (df['normalized_daily_range'] + 0.001)
    
    # Volatility Confidence Framework
    conditions = [
        (df['range_persistence'] >= 3) & (df['volatility_consistency'] > 2.0),
        (df['range_persistence'] >= 2) | (df['volatility_consistency'] > 1.0)
    ]
    choices = [2, 1]  # High: 2, Medium: 1, Low: 0
    df['volatility_confidence_level'] = np.select(conditions, choices, default=0)
    df['volatility_adjustment_factor'] = df['volatility_confidence_level'] * df['volatility_consistency']
    
    # Multi-Dimensional Signal Integration
    df['accelerated_momentum'] = df['medium_term_momentum'] * (1 + df['acceleration_consistency_score'])
    df['persistence_enhanced'] = df['accelerated_momentum'] * (1 + df['relative_persistence'])
    df['volume_aligned'] = df['persistence_enhanced'] * df['volume_multiplier']
    
    # Confidence-Based Signal Filtering
    conditions_valid = [
        (df['volume_alignment_score'] > 0.6) & (df['volatility_confidence_level'] == 2),
        (df['volume_alignment_score'] > 0.3) | (df['volatility_confidence_level'] >= 1)
    ]
    choices_valid = [df['volume_aligned'], df['volume_aligned'] * 0.7]  # Full signal for high confidence, reduced for medium
    df['valid_signal'] = np.select(conditions_valid, choices_valid, default=0)
    
    # Volatility-Scaled Final Factor
    df['range_normalized'] = df['valid_signal'] / (df['recent_volatility'] + 0.001)
    df['stability_adjusted'] = df['range_normalized'] * df['volatility_adjustment_factor']
    df['final_alpha_factor'] = df['stability_adjusted'] * 100
    
    return df['final_alpha_factor']

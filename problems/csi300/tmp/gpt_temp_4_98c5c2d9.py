import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Aligned Momentum Acceleration with Volatility Scaling factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate momentum components
    df['short_momentum'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['medium_momentum'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['long_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Momentum acceleration framework
    df['short_acceleration'] = df['medium_momentum'] - df['short_momentum']
    df['long_acceleration'] = df['long_momentum'] - df['medium_momentum']
    df['cross_timeframe_validation'] = np.sign(df['short_acceleration']) * np.sign(df['long_acceleration'])
    
    # Trend persistence assessment
    df['medium_momentum_sign'] = np.sign(df['medium_momentum'])
    df['direction_consistency'] = df['medium_momentum_sign'].groupby(
        (df['medium_momentum_sign'] != df['medium_momentum_sign'].shift(1)).cumsum()
    ).cumcount() + 1
    
    df['acceleration_persistence'] = (df['cross_timeframe_validation'] > 0).groupby(
        (df['cross_timeframe_validation'] != df['cross_timeframe_validation'].shift(1)).cumsum()
    ).cumcount() + 1
    
    df['momentum_quality'] = df['medium_momentum'] * (1 + df['direction_consistency'] / 10)
    
    # Volume alignment engine
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['volume_trend'] = df['volume'] / df['volume'].shift(3)
    df['volume_acceleration'] = df['volume_trend'] - df['volume_ratio']
    
    df['short_alignment'] = np.sign(df['volume_ratio'] - 1) * np.sign(df['short_momentum'])
    df['medium_alignment'] = np.sign(df['volume_trend'] - 1) * np.sign(df['medium_momentum'])
    df['acceleration_alignment'] = np.sign(df['volume_acceleration']) * np.sign(df['short_acceleration'])
    
    # Volume confidence scoring
    alignment_scores = (df[['short_alignment', 'medium_alignment', 'acceleration_alignment']] > 0).sum(axis=1)
    df['volume_confidence_level'] = np.select(
        [alignment_scores == 3, alignment_scores == 2, alignment_scores <= 1],
        ['high', 'medium', 'low'],
        default='low'
    )
    df['volume_multiplier'] = np.select(
        [df['volume_confidence_level'] == 'high', df['volume_confidence_level'] == 'medium'],
        [1.4, 1.2],
        default=1.0
    )
    
    # Volatility-aware signal processing
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['range_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Range stability calculation
    df['range_stability'] = 0
    for i in range(5, len(df)):
        recent_ranges = df['daily_range'].iloc[i-4:i+1]
        if len(recent_ranges) >= 2:
            stability_count = 0
            for j in range(1, len(recent_ranges)):
                if abs(recent_ranges.iloc[j] - recent_ranges.iloc[j-1]) / recent_ranges.iloc[j-1] <= 0.2:
                    stability_count += 1
            df.iloc[i, df.columns.get_loc('range_stability')] = stability_count
    
    # Volatility scaling framework
    df['recent_volatility'] = df['daily_range'].rolling(window=5, min_periods=1).mean()
    df['volatility_consistency'] = 1 / (df['daily_range'].rolling(window=5, min_periods=1).std() + 0.001)
    df['range_adjusted_momentum'] = df['momentum_quality'] / df['daily_range']
    
    # Volatility confidence assessment
    df['volatility_confidence_level'] = np.select(
        [
            (df['range_stability'] >= 3) & (df['volatility_consistency'] > 1.5),
            (df['range_stability'] >= 2) | (df['volatility_consistency'] > 1.0)
        ],
        ['high', 'medium'],
        default='low'
    )
    df['volatility_adjustment'] = np.select(
        [df['volatility_confidence_level'] == 'high', df['volatility_confidence_level'] == 'medium'],
        [df['volatility_consistency'] * 1.0, df['volatility_consistency'] * 0.7],
        default=df['volatility_consistency'] * 0.3
    )
    
    # Signal integration & enhancement
    df['accelerated_momentum'] = df['momentum_quality'] * (1 + df['short_acceleration'])
    df['volume_aligned'] = df['accelerated_momentum'] * df['volume_multiplier']
    df['persistence_enhanced'] = df['volume_aligned'] * (1 + df['acceleration_persistence'] / 10)
    
    # Multi-dimensional confidence filtering
    confidence_conditions = [
        (df['volume_confidence_level'] == 'high') & (df['volatility_confidence_level'] == 'high'),
        (df['volume_confidence_level'].isin(['high', 'medium'])) | (df['volatility_confidence_level'].isin(['high', 'medium']))
    ]
    df['signal_validity'] = np.select(confidence_conditions, [1.0, 0.5], default=0.1)
    
    df['valid_signal'] = df['persistence_enhanced'] * df['signal_validity']
    
    # Volatility-scaled final factor
    df['range_normalized'] = df['valid_signal'] / df['daily_range']
    df['stability_adjusted'] = df['range_normalized'] * df['volatility_adjustment']
    
    # Final alpha factor
    result = df['stability_adjusted'] * 1000
    
    return result

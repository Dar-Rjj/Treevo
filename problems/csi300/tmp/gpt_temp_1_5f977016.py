import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Acceleration with Volume-Volatility Confirmation
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Core Momentum Framework
    df['momentum_1d'] = df['close'].pct_change(1)
    df['momentum_3d'] = df['close'].pct_change(3)
    df['momentum_5d'] = df['close'].pct_change(5)
    
    # Momentum Acceleration Analysis
    df['acceleration_1'] = df['momentum_3d'] - df['momentum_1d']
    df['acceleration_2'] = df['momentum_5d'] - df['momentum_3d']
    df['acceleration_consistency'] = np.sign(df['acceleration_1']) * np.sign(df['acceleration_2'])
    
    # Momentum Persistence Scoring
    df['momentum_3d_sign'] = np.sign(df['momentum_3d'])
    df['direction_persistence'] = df['momentum_3d_sign'].groupby(df.index).transform(
        lambda x: x.expanding().apply(lambda y: (y == y.iloc[-1]).sum() if len(y) > 0 else 1)
    )
    
    df['acceleration_consistency_positive'] = (df['acceleration_consistency'] > 0).astype(int)
    df['acceleration_persistence'] = df['acceleration_consistency_positive'].groupby(df.index).transform(
        lambda x: x.expanding().apply(lambda y: (y == 1).sum() if len(y) > 0 else 1)
    )
    
    df['combined_persistence_score'] = df['direction_persistence'] * df['acceleration_persistence']
    
    # Volume Confirmation Engine
    df['volume_momentum'] = df['volume'].pct_change(1)
    df['volume_trend'] = df['volume'].pct_change(3)
    df['volume_acceleration'] = df['volume_trend'] - df['volume_momentum']
    
    # Price-Volume Alignment
    df['direction_alignment'] = np.sign(df['volume_momentum']) * np.sign(df['momentum_1d'])
    df['trend_alignment'] = np.sign(df['volume_trend']) * np.sign(df['momentum_3d'])
    df['acceleration_alignment'] = np.sign(df['volume_acceleration']) * np.sign(df['acceleration_consistency'])
    
    # Volume Confidence Scoring
    alignment_scores = (df[['direction_alignment', 'trend_alignment', 'acceleration_alignment']] > 0).sum(axis=1)
    df['volume_confidence_multiplier'] = np.where(
        alignment_scores == 3, 1.2,
        np.where(alignment_scores == 2, 1.0, 0.8)
    )
    
    # Volatility Context Framework
    df['daily_range_ratio'] = (df['high'] - df['low']) / df['close']
    df['range_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['range_persistence'] = abs(df['range_efficiency'])
    
    # Volatility Stability Assessment
    df['recent_volatility'] = df['daily_range_ratio'].rolling(window=5, min_periods=1).mean()
    df['volatility_consistency'] = 1 / (df['daily_range_ratio'].rolling(window=5, min_periods=1).std() + 0.0001)
    df['volatility_adjusted_momentum'] = df['momentum_3d'] / df['daily_range_ratio'].replace(0, np.nan)
    
    # Volatility Confidence Scoring
    volatility_confidence_conditions = [
        (df['range_persistence'] > 0.7) & (df['volatility_consistency'] > 2),
        (df['range_persistence'] > 0.5) & (df['volatility_consistency'] > 1)
    ]
    volatility_confidence_choices = [2, 1]  # High=2, Medium=1, Low=0
    df['volatility_confidence_score'] = np.select(volatility_confidence_conditions, volatility_confidence_choices, default=0)
    df['volatility_adjustment_factor'] = df['volatility_confidence_score'] * df['volatility_consistency']
    
    # Multi-Timeframe Signal Integration
    df['accelerated_momentum'] = df['momentum_3d'] * (1 + df['acceleration_1'])
    df['volume_enhanced'] = df['accelerated_momentum'] * df['volume_confidence_multiplier']
    df['persistence_boosted'] = df['volume_enhanced'] * (1 + df['combined_persistence_score'] / 10)
    
    # Confidence-Based Filtering
    volume_confidence_high = (df['volume_confidence_multiplier'] == 1.2)
    volume_confidence_medium = (df['volume_confidence_multiplier'] >= 1.0)
    volatility_confidence_high = (df['volatility_confidence_score'] == 2)
    volatility_confidence_medium = (df['volatility_confidence_score'] >= 1)
    
    df['signal_validity'] = np.where(
        volume_confidence_high & volatility_confidence_high, 1.0,  # High confidence
        np.where(
            volume_confidence_medium | volatility_confidence_medium, 0.7,  # Medium confidence
            0.3  # Low confidence
        )
    )
    
    df['valid_signal'] = df['persistence_boosted'] * df['signal_validity']
    
    # Volatility Scaling
    df['range_normalized'] = df['valid_signal'] / df['daily_range_ratio'].replace(0, np.nan)
    df['stability_adjusted'] = df['range_normalized'] * df['volatility_adjustment_factor']
    
    # Final factor
    factor = df['stability_adjusted'] * 100
    
    # Clean up intermediate columns
    columns_to_drop = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'amount', 'volume']]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    return factor

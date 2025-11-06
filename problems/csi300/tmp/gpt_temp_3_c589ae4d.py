import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volume-Confirmed Momentum Acceleration factor that combines price momentum acceleration
    with volume confirmation and volatility scaling for enhanced predictive power.
    """
    df = data.copy()
    
    # Momentum Acceleration Engine
    # Price Momentum Calculation
    df['momentum_1d'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['momentum_3d'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['momentum_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Acceleration Detection
    df['short_acceleration'] = df['momentum_3d'] - df['momentum_1d']
    df['long_acceleration'] = df['momentum_5d'] - df['momentum_3d']
    df['acceleration_consistency'] = np.sign(df['short_acceleration']) * np.sign(df['long_acceleration'])
    
    # Momentum Quality
    # Direction persistence
    df['momentum_3d_sign'] = np.sign(df['momentum_3d'])
    df['direction_persistence'] = 0
    for i in range(1, len(df)):
        if df['momentum_3d_sign'].iloc[i] == df['momentum_3d_sign'].iloc[i-1]:
            df['direction_persistence'].iloc[i] = df['direction_persistence'].iloc[i-1] + 1
    
    # Acceleration persistence
    df['acceleration_persistence'] = 0
    for i in range(1, len(df)):
        if df['acceleration_consistency'].iloc[i] > 0:
            df['acceleration_persistence'].iloc[i] = df['acceleration_persistence'].iloc[i-1] + 1
    
    # Momentum strength
    df['momentum_strength'] = df['momentum_3d'] * (1 + df['direction_persistence'] / 10)
    
    # Volume Confirmation System
    # Volume Dynamics
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['volume_trend'] = df['volume'] / df['volume'].shift(3)
    df['volume_acceleration'] = df['volume_trend'] - df['volume_ratio']
    
    # Volume-Momentum Alignment
    df['direction_alignment'] = np.sign(df['volume_ratio'] - 1) * np.sign(df['momentum_1d'])
    df['trend_alignment'] = np.sign(df['volume_trend'] - 1) * np.sign(df['momentum_3d'])
    df['acceleration_alignment'] = np.sign(df['volume_acceleration']) * np.sign(df['short_acceleration'])
    
    # Confidence Scoring
    alignment_scores = (df[['direction_alignment', 'trend_alignment', 'acceleration_alignment']] > 0).sum(axis=1)
    df['volume_multiplier'] = 1 + (alignment_scores * 0.2)
    
    # Volatility Scaling Framework
    # Range Analysis
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['range_stability'] = 1 / (df['daily_range'].rolling(window=5).std() + 0.001)
    
    # Range persistence
    df['range_persistence'] = 0
    range_avg = df['daily_range'].rolling(window=5).mean()
    for i in range(1, len(df)):
        if abs(df['daily_range'].iloc[i] - range_avg.iloc[i]) / range_avg.iloc[i] <= 0.2:
            df['range_persistence'].iloc[i] = df['range_persistence'].iloc[i-1] + 1
    
    # Volatility Confidence
    volatility_confidence = np.where(
        (df['range_persistence'] >= 3) & (df['range_stability'] > 1.5), 2,
        np.where(
            (df['range_persistence'] >= 2) | (df['range_stability'] > 1.0), 1, 0
        )
    )
    df['volatility_adjustment'] = volatility_confidence * df['range_stability']
    
    # Final Factor Construction
    # Base Signal
    df['accelerated_momentum'] = df['momentum_strength'] * (1 + df['short_acceleration'])
    df['volume_confirmed'] = df['accelerated_momentum'] * df['volume_multiplier']
    df['persistence_enhanced'] = df['volume_confirmed'] * (1 + df['acceleration_persistence'] / 10)
    
    # Volatility Scaling
    df['range_normalized'] = df['persistence_enhanced'] / df['daily_range']
    df['final_factor'] = df['range_normalized'] * df['volatility_adjustment']
    
    return df['final_factor']

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Scaled Multi-Timeframe Momentum with Volume Confirmation
    """
    # Initialize epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Multi-Timeframe Momentum Calculation
    # Short-Term Momentum (1-day)
    short_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + epsilon)
    
    # Medium-Term Momentum (3-day)
    high_3d = df['high'].rolling(window=3).max()
    low_3d = df['low'].rolling(window=3).min()
    medium_momentum = (df['close'] - df['close'].shift(3)) / (high_3d - low_3d + epsilon)
    
    # Long-Term Momentum (5-day)
    high_5d = df['high'].rolling(window=5).max()
    low_5d = df['low'].rolling(window=5).min()
    long_momentum = (df['close'] - df['close'].shift(5)) / (high_5d - low_5d + epsilon)
    
    # Momentum Persistence with Exponential Decay
    lambda_decay = 0.9
    
    # Short-term direction consistency (3-day rolling)
    short_direction = np.sign(short_momentum)
    short_persistence = short_direction.rolling(window=3).apply(
        lambda x: np.sum([lambda_decay**i * x.iloc[-i-1] for i in range(len(x)) if not pd.isna(x.iloc[-i-1])]), 
        raw=False
    )
    
    # Medium-term trend confirmation (5-day rolling)
    medium_direction = np.sign(medium_momentum)
    medium_persistence = medium_direction.rolling(window=5).apply(
        lambda x: np.sum([lambda_decay**i * x.iloc[-i-1] for i in range(len(x)) if not pd.isna(x.iloc[-i-1])]), 
        raw=False
    )
    
    # Long-term trend alignment (10-day rolling)
    long_direction = np.sign(long_momentum)
    long_persistence = long_direction.rolling(window=10).apply(
        lambda x: np.sum([lambda_decay**i * x.iloc[-i-1] for i in range(len(x)) if not pd.isna(x.iloc[-i-1])]), 
        raw=False
    )
    
    # Weighted Persistence Score
    persistence_score = (
        0.5 * short_persistence + 
        0.3 * medium_persistence + 
        0.2 * long_persistence
    )
    
    # Volume-Based Confirmation Signals
    # Volume Momentum Persistence
    volume_ratio_1d = df['volume'] / (df['volume'].shift(1) + epsilon)
    volume_ma_3d = df['volume'].rolling(window=3).mean()
    volume_ratio_3d = df['volume'] / (volume_ma_3d.shift(1) + epsilon)
    volume_acceleration = df['volume'] / (df['volume'].shift(3) + epsilon)
    
    # Price-Volume Alignment
    volume_direction_1d = np.sign(volume_ratio_1d - 1)
    volume_direction_3d = np.sign(volume_ratio_3d - 1)
    
    price_volume_alignment_short = short_direction * volume_direction_1d
    price_volume_alignment_medium = medium_direction * volume_direction_3d
    
    # Volume alignment persistence
    volume_alignment_persistence = (
        price_volume_alignment_short.rolling(window=3).mean() +
        price_volume_alignment_medium.rolling(window=5).mean()
    ) / 2
    
    # Volatility-Scaled Factor Construction
    # Dynamic Volatility Adjustment
    recent_volatility = (df['high'] - df['low']).rolling(window=5).mean()
    volatility_scaling = 1 / (recent_volatility + epsilon)
    
    # Multi-Timeframe Signal Blending
    short_term_signal = short_momentum * persistence_score * volume_alignment_persistence
    medium_term_signal = medium_momentum * persistence_score * volume_alignment_persistence
    long_term_signal = long_momentum * persistence_score * volume_alignment_persistence
    
    # Final composite factor
    composite_factor = (
        0.4 * short_term_signal +
        0.35 * medium_term_signal +
        0.25 * long_term_signal
    ) * volatility_scaling
    
    # Clean and return the factor
    factor = composite_factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return factor

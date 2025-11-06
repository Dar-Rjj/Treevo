import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Acceleration with Volume Confirmation alpha factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate momentum across different timeframes
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    # Multi-Timeframe Momentum Calculation
    momentum_1d = (close - close.shift(1)) / close.shift(1)
    momentum_3d = (close - close.shift(3)) / close.shift(3)
    momentum_5d = (close - close.shift(5)) / close.shift(5)
    
    # Acceleration Gradient Analysis
    primary_gradient = momentum_3d - momentum_1d
    secondary_gradient = momentum_5d - momentum_3d
    
    # Acceleration Persistence
    persistence_scores = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 5:
            persistence_scores.iloc[i] = 0
            continue
            
        current_persistence = 0
        weight = 1.0
        for j in range(5):  # Look back 5 days
            if i - j - 1 < 0:
                break
            if primary_gradient.iloc[i - j - 1] > 0 and secondary_gradient.iloc[i - j - 1] > 0:
                current_persistence += weight
                weight *= 0.8  # Exponential decay
            else:
                break
        persistence_scores.iloc[i] = current_persistence
    
    # Volume Confirmation Engine
    volume_ratio = volume / volume.shift(1)
    volume_momentum = (volume - volume.shift(3)) / volume.shift(3)
    
    # Volume trend - consecutive days with volume increase
    volume_trend = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if i == 0:
            volume_trend.iloc[i] = 0
            continue
        if volume_ratio.iloc[i] > 1:
            volume_trend.iloc[i] = volume_trend.iloc[i-1] + 1 if volume_trend.iloc[i-1] > 0 else 1
        else:
            volume_trend.iloc[i] = 0
    
    # Price-Volume Alignment
    direction_alignment = np.sign(momentum_3d) * np.sign(volume_momentum)
    strength_alignment = np.minimum(np.abs(momentum_3d), np.abs(volume_momentum))
    alignment_score = direction_alignment * strength_alignment
    
    # Volume Confidence Assessment
    volume_confidence = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if volume_trend.iloc[i] >= 2 and alignment_score.iloc[i] > 0:
            volume_confidence.iloc[i] = 1.2  # High confidence
        elif alignment_score.iloc[i] > 0:
            volume_confidence.iloc[i] = 1.0  # Medium confidence
        else:
            volume_confidence.iloc[i] = 0.8  # Low confidence
    
    # Multi-Timeframe Signal Integration
    # Core Momentum Signal
    acceleration_boost = primary_gradient * persistence_scores
    enhanced_momentum = momentum_3d * (1 + acceleration_boost)
    
    # Volume-Enhanced Signal
    volume_aligned = enhanced_momentum * volume_confidence
    
    # Cross-Timeframe Validation
    consistency_multiplier = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        signs = [np.sign(momentum_1d.iloc[i]), np.sign(momentum_3d.iloc[i]), np.sign(momentum_5d.iloc[i])]
        if len(set(signs)) == 1:  # All same sign
            consistency_multiplier.iloc[i] = 1.5
        elif len(set(signs)) == 2:  # Mixed signs
            consistency_multiplier.iloc[i] = 1.0
        else:  # Conflicting signs
            consistency_multiplier.iloc[i] = 0.7
    
    final_signal = volume_aligned * consistency_multiplier
    
    # Volatility-Adjusted Output
    daily_range = (high - low) / close
    range_stability = 1 / (daily_range.rolling(window=5).std() + 0.0001)
    
    # Signal Scaling
    scaled_signal = final_signal / (daily_range + 0.0001)
    
    # Stability-Weighted Factor
    stability_confidence = range_stability * 0.8
    final_alpha = scaled_signal * stability_confidence
    
    return final_alpha.fillna(0)

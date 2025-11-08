import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Core Momentum Strength Assessment
    # Intraday Momentum Component
    raw_momentum = (data['close'] - data['open']) / data['open']
    range_adjusted_momentum = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    prior_day_carryover = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Multi-Timeframe Confirmation
    short_term_persistence = np.sign(data['close'] - data['open']) * np.sign(data['close'].shift(1) - data['open'].shift(1))
    medium_term_trend = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    momentum_quality = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume-Price Timing Dynamics
    # Volume Confirmation Signals
    volume_trend_strength = data['volume'] / data['volume'].shift(1)
    volume_price_alignment = np.sign(data['close'] - data['open']) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Volume Persistence Patterns
    volume_surge_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 3:
            count = sum(data['volume'].iloc[i-j] > data['volume'].iloc[i-j-1] for j in range(3))
            volume_surge_count.iloc[i] = count
    
    volume_climax = data['volume'] / data['volume'].rolling(window=10).mean()
    
    price_change = np.abs(data['close'] - data['open']) / data['open']
    volume_exhaustion = np.where(price_change < 0.01, data['volume'] / data['volume'].shift(1), 0)
    
    # Volatility Regime Transition Detection
    # Range-Based Regime Classification
    current_range_intensity = (data['high'] - data['low']) / data['close'].shift(1)
    range_5d_avg = (data['high'] - data['low']).rolling(window=5).mean()
    range_expansion_ratio = (data['high'] - data['low']) / range_5d_avg
    range_trend = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    
    # Regime Transition Signals
    compression_breakout = range_expansion_ratio * np.sign(data['close'] - data['open'])
    expansion_continuation = range_trend * np.sign(data['close'] - data['open'])
    regime_stability = 1 / (1 + np.abs(range_expansion_ratio - 1))
    
    # Adaptive Alpha Construction
    # Momentum Persistence Score
    base_momentum = raw_momentum * range_adjusted_momentum
    persistence_multiplier = 1 + 0.5 * short_term_persistence
    quality_adjustment = momentum_quality * prior_day_carryover
    momentum_persistence_score = base_momentum * persistence_multiplier * quality_adjustment
    
    # Volume Timing Enhancement
    volume_confirmation = volume_price_alignment * volume_trend_strength
    volume_persistence = volume_surge_count / 3.0  # Normalize to [0,1]
    pattern_recognition = volume_price_alignment * volume_persistence
    volume_filter = 1 + 0.3 * (volume_climax - volume_exhaustion)
    
    # Regime-Adaptive Weighting
    compression_regime = momentum_persistence_score * volume_filter * regime_stability
    expansion_regime = momentum_persistence_score * volume_confirmation * range_expansion_ratio
    transition_phase = momentum_persistence_score * pattern_recognition * compression_breakout
    
    # Determine regime based on range expansion ratio
    regime_adaptive_weighting = np.where(
        range_expansion_ratio < 0.8, compression_regime,
        np.where(range_expansion_ratio > 1.2, expansion_regime, transition_phase)
    )
    
    # Final Alpha Output
    raw_signal = regime_adaptive_weighting * medium_term_trend
    confidence_score = np.abs(raw_signal) * momentum_quality
    directional_prediction = raw_signal * np.sign(medium_term_trend)
    
    # Return the final alpha factor
    return directional_prediction * confidence_score

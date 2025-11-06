import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Decay Asymmetry with Volume Anchoring alpha factor
    """
    close = df['close']
    volume = df['volume']
    
    # Price Momentum Decay Patterns
    # Short-term momentum decay (3-day, 5-day)
    mom_3 = close / close.shift(3) - 1
    mom_5 = close / close.shift(5) - 1
    mom_1 = close / close.shift(1) - 1
    
    momentum_acceleration = mom_3 - mom_5
    decay_rate = mom_1 / (mom_3 + 0.001)
    
    # Momentum persistence
    mom_sign_1 = np.sign(mom_1)
    mom_sign_3 = np.sign(mom_3)
    momentum_persistence = pd.Series([
        (mom_sign_1.iloc[i-4:i+1] == mom_sign_3.iloc[i-4:i+1]).sum() 
        if i >= 4 else np.nan 
        for i in range(len(close))
    ], index=close.index) / 5
    
    # Medium-term momentum reversal (8-day, 13-day)
    mom_8 = close / close.shift(8) - 1
    mom_13 = close / close.shift(13) - 1
    
    reversal_strength = mom_8 * mom_13
    momentum_exhaustion = np.abs(mom_8) / (np.abs(mom_13) + 0.001)
    
    # Trend consistency
    trend_consistency = mom_sign_1.rolling(window=8, min_periods=1).std()
    
    # Cross-timeframe momentum alignment
    short_medium_divergence = np.sign(mom_3) * np.sign(mom_8)
    mom_accel_8_13 = mom_8 - mom_13
    acceleration_consistency = momentum_acceleration * mom_accel_8_13
    decay_rate_medium = mom_8 / (mom_13 + 0.001)
    decay_pattern_matching = decay_rate / (decay_rate_medium + 0.001)
    
    # Volume Anchoring Dynamics
    # Volume concentration patterns
    volume_median_20 = volume.rolling(window=20, min_periods=1).median()
    volume_spike = volume / (volume_median_20 + 0.001)
    
    # Volume persistence
    volume_persistence = (volume > volume.shift(1)).rolling(window=5, min_periods=1).sum() / 5
    volume_decay = (volume / volume.shift(3)) - (volume / volume.shift(8))
    
    # Price-volume anchoring relationships
    anchored_momentum = mom_3 * (volume / volume.shift(3) - 1)
    volume_led_moves = np.sign(volume / volume.shift(1) - 1) * np.sign(mom_1)
    anchoring_strength = np.abs(close - close.shift(5)) / (close.rolling(window=5, min_periods=1).std() + 0.001)
    
    # Asymmetric volume impact
    high_volume_anchoring = volume_spike * momentum_persistence
    low_volume_drift = (1 - volume_persistence) * decay_rate
    volume_confirmation_divergence = volume_led_moves - anchored_momentum
    
    # Decay-Asymmetry Signal Generation
    # Momentum decay asymmetry
    positive_momentum = (mom_3 > 0).astype(float)
    decay_rate_change = decay_rate - decay_rate.shift(1)
    decreasing_decay_rate = (decay_rate_change < 0).astype(float)
    increasing_decay_rate = (decay_rate_change > 0).astype(float)
    
    bullish_decay = positive_momentum * decreasing_decay_rate
    bearish_decay = (mom_3 < 0).astype(float) * increasing_decay_rate
    decay_reversal = np.sign(decay_rate_change)
    
    # Volume anchoring asymmetry
    strong_anchor = (volume_spike > volume_spike.quantile(0.7)).astype(float) * (anchoring_strength > anchoring_strength.quantile(0.7)).astype(float)
    weak_anchor = (volume_persistence < volume_persistence.quantile(0.3)).astype(float) * (anchoring_strength < anchoring_strength.quantile(0.3)).astype(float)
    volume_concentration = volume / volume.rolling(window=5, min_periods=1).mean()
    anchor_shift = volume_concentration / (volume_concentration.shift(3) + 0.001)
    
    # Cross-pattern alignment
    decay_anchor_convergence = decay_rate * anchoring_strength
    pattern_signs = np.sign(momentum_acceleration) * np.sign(volume_led_moves) * np.sign(decay_rate)
    pattern_persistence = pd.Series([
        (pattern_signs.iloc[i-2:i+1] == pattern_signs.iloc[i]).sum() 
        if i >= 2 else np.nan 
        for i in range(len(close))
    ], index=close.index) / 3
    
    decay_asymmetry = bullish_decay - bearish_decay
    anchor_asymmetry = strong_anchor - weak_anchor
    asymmetry_strength = np.abs(decay_asymmetry) * np.abs(anchor_asymmetry)
    
    # Multi-Scale Pattern Integration
    # Short-term pattern dominance (3-day)
    volume_led_moves_3day = volume_led_moves.rolling(window=3, min_periods=1).mean()
    momentum_acceleration_3 = momentum_acceleration.rolling(window=3, min_periods=1).mean()
    volume_led_momentum = volume_led_moves_3day * momentum_acceleration_3
    
    decay_rate_3day = decay_rate.rolling(window=3, min_periods=1).mean()
    volume_concentration_3day = volume_concentration.rolling(window=3, min_periods=1).mean()
    decay_concentration = decay_rate_3day * volume_concentration_3day
    
    pattern_signals = momentum_acceleration + volume_led_moves + decay_rate
    pattern_consistency = pattern_signals.rolling(window=3, min_periods=1).std()
    
    # Medium-term pattern stability (8-day)
    anchoring_strength_8day = anchoring_strength.rolling(window=8, min_periods=1).mean()
    trend_consistency_8day = trend_consistency.rolling(window=8, min_periods=1).mean()
    anchored_trend = anchoring_strength_8day * trend_consistency_8day
    
    decay_persistence_8day = (decay_rate > decay_rate.shift(1)).rolling(window=8, min_periods=1).mean()
    volume_persistence_8day = volume_persistence.rolling(window=8, min_periods=1).mean()
    decay_persistence = decay_persistence_8day * volume_persistence_8day
    
    cross_pattern_alignment_8day = (momentum_acceleration * volume_led_moves * decay_rate).rolling(window=8, min_periods=1).mean()
    
    # Pattern transition detection
    pattern_dominance_3day = (volume_led_momentum + decay_concentration - pattern_consistency).rolling(window=3, min_periods=1).mean()
    pattern_stability_8day = (anchored_trend + decay_persistence + cross_pattern_alignment_8day).rolling(window=8, min_periods=1).mean()
    
    short_to_medium_shift = pattern_dominance_3day / (pattern_stability_8day + 0.001)
    anchor_decay_transition = anchor_shift * decay_reversal
    pattern_regime_change = np.sign(pattern_consistency - pattern_consistency.shift(5))
    
    # Asymmetry-Regime Adaptive Signals
    # Decay regime classification
    decay_rate_increasing = (decay_rate > decay_rate.shift(1)).astype(float)
    decay_rate_decreasing = (decay_rate < decay_rate.shift(1)).astype(float)
    decay_variance = decay_rate.rolling(window=5, min_periods=1).std()
    low_decay_variance = (decay_variance < decay_variance.quantile(0.3)).astype(float)
    consistent_momentum = (momentum_persistence > 0.6).astype(float)
    high_decay_reversal = (np.abs(decay_reversal) > 0.5).astype(float)
    
    accelerating_decay = decay_rate_increasing * positive_momentum
    decelerating_decay = decay_rate_decreasing * (mom_3 < 0).astype(float)
    stable_decay = low_decay_variance * consistent_momentum
    reversing_decay = high_decay_reversal * (momentum_exhaustion > 1).astype(float)
    
    # Anchoring regime detection
    high_volume_spike = (volume_spike > volume_spike.quantile(0.7)).astype(float)
    low_volume_persistence = (volume_persistence < volume_persistence.quantile(0.3)).astype(float)
    medium_volume_concentration = ((volume_concentration > volume_concentration.quantile(0.3)) & 
                                  (volume_concentration < volume_concentration.quantile(0.7))).astype(float)
    low_volume_impact = (volume_spike < volume_spike.quantile(0.3)).astype(float)
    random_moves = (pattern_consistency > pattern_consistency.quantile(0.7)).astype(float)
    
    strong_anchoring = high_volume_spike * (anchoring_strength > anchoring_strength.quantile(0.7)).astype(float)
    weak_anchoring = low_volume_persistence * (anchor_shift > 1.2).astype(float)
    shifting_anchor = medium_volume_concentration * (anchor_decay_transition > 0).astype(float)
    no_anchor = low_volume_impact * random_moves
    
    # Regime-specific signal weighting
    accelerating_decay_weight = momentum_persistence * 2.0
    strong_anchoring_weight = volume_confirmation_divergence * 1.5
    weak_anchoring_weight = pattern_regime_change * 1.8
    transition_weight = (short_to_medium_shift + anchor_decay_transition + pattern_regime_change) / 3
    
    # Composite Alpha Construction
    # Core asymmetry score
    decay_asymmetry_component = bullish_decay - bearish_decay
    anchoring_asymmetry_component = strong_anchor - weak_anchor
    pattern_alignment = decay_anchor_convergence * pattern_persistence
    
    # Multi-scale integration
    short_term_weight = pattern_dominance_3day
    medium_term_weight = pattern_stability_8day
    transition_adjustment = pattern_regime_change
    
    # Regime-adaptive finalization
    regime_multiplier = (
        accelerating_decay * accelerating_decay_weight +
        strong_anchoring * strong_anchoring_weight +
        weak_anchoring * weak_anchoring_weight +
        (shifting_anchor + no_anchor) * transition_weight
    ) / 4
    
    # Final alpha output
    alpha = (
        decay_asymmetry_component * 0.4 +
        anchoring_asymmetry_component * 0.3 +
        pattern_alignment * 0.3
    ) * (
        short_term_weight * 0.4 +
        medium_term_weight * 0.4 +
        transition_adjustment * 0.2
    ) * regime_multiplier * asymmetry_strength
    
    return alpha.fillna(0)

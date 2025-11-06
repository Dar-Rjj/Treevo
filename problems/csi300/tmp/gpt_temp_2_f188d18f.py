import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor using momentum, volatility, and volume regimes
    with non-linear interactions and adaptive scaling.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Momentum Structure Analysis
    # Multi-timeframe returns
    ret_3d = close.pct_change(3)
    ret_8d = close.pct_change(8)
    ret_15d = close.pct_change(15)
    
    # Acceleration profile
    primary_accel = ret_3d - ret_8d
    secondary_accel = ret_8d - ret_15d
    momentum_curvature = primary_accel - secondary_accel
    
    # Momentum quality metrics
    direction_consistency = ret_3d.rolling(3).apply(lambda x: len(set(np.sign(x.dropna()))) == 1 if len(x.dropna()) == 3 else 0)
    strength_ratio = abs(ret_3d) / (abs(ret_8d) + 0.001)
    smoothness = 1 - (abs(ret_3d - ret_8d) / (abs(ret_3d) + abs(ret_8d) + 0.001))
    
    # Momentum regime classification
    strong_trending = ((ret_3d > 0) & (ret_8d > 0) & (primary_accel > 0)).astype(float)
    weak_trending = ((ret_3d > 0) & (ret_8d > 0) & (primary_accel < 0)).astype(float)
    strong_reversing = ((ret_3d * ret_8d < 0) & (abs(momentum_curvature) > abs(momentum_curvature).quantile(0.7))).astype(float)
    momentum_exhaustion = ((abs(ret_3d) < abs(ret_3d).quantile(0.3)) & (abs(primary_accel) < abs(primary_accel).quantile(0.3))).astype(float)
    
    # Adaptive momentum signals
    trending_momentum = ret_3d * direction_consistency
    reversing_momentum = momentum_curvature * strength_ratio
    exhaustion_momentum = primary_accel * smoothness
    
    # Multi-scale momentum blend
    momentum_blend = (
        strong_trending * trending_momentum +
        strong_reversing * reversing_momentum +
        momentum_exhaustion * exhaustion_momentum +
        weak_trending * ret_3d * 0.5
    )
    
    # Volatility Regime Framework
    # Multi-scale volatility
    short_vol = ((high - low) / close).rolling(3).std()
    medium_vol = ((high - low) / close).rolling(8).std()
    volatility_ratio = short_vol / (medium_vol + 0.001)
    
    # Volatility state machine
    expanding_vol = ((volatility_ratio > 1.2) & (volatility_ratio > volatility_ratio.shift(1))).astype(float)
    contracting_vol = ((volatility_ratio < 0.8) & (volatility_ratio < volatility_ratio.shift(1))).astype(float)
    high_stable_vol = ((volatility_ratio > 1.1) & (abs(volatility_ratio - volatility_ratio.shift(1)) < 0.05)).astype(float)
    low_stable_vol = ((volatility_ratio < 0.9) & (abs(volatility_ratio - volatility_ratio.shift(1)) < 0.05)).astype(float)
    
    # Volatility quality
    vol_regime_persistence = expanding_vol.rolling(3).sum() + contracting_vol.rolling(3).sum() + high_stable_vol.rolling(3).sum() + low_stable_vol.rolling(3).sum()
    transition_smoothness = 1 - abs(volatility_ratio - volatility_ratio.shift(1))
    
    # Volatility-adaptive signals
    low_vol_momentum = ret_3d / (medium_vol + 0.001)
    high_vol_momentum = primary_accel * volatility_ratio
    stable_vol_momentum = smoothness * vol_regime_persistence
    
    # Volume Confirmation System
    # Volume dynamics
    volume_momentum = volume.pct_change()
    volume_acceleration = volume_momentum - volume_momentum.shift(1)
    volume_trend_strength = volume.pct_change(3)
    
    # Volume quality metrics
    volume_persistence = (volume_momentum > 0).rolling(3).sum()
    volume_stability = 1 - abs(volume_acceleration)
    volume_confirmation_strength = abs(volume_momentum) * volume_persistence
    
    # Price-volume integration
    strong_confirmation = ((np.sign(ret_3d) == np.sign(volume_momentum)) & (volume_confirmation_strength > volume_confirmation_strength.quantile(0.7))).astype(float)
    bullish_divergence = ((ret_3d < 0) & (volume_momentum > 0)).astype(float)
    bearish_divergence = ((ret_3d > 0) & (volume_momentum < 0)).astype(float)
    divergence_strength = abs(ret_3d) * abs(volume_momentum)
    
    # Volume-weighted adjustments
    volume_confirmed_momentum = ret_3d * volume_confirmation_strength
    divergence_momentum = momentum_curvature * divergence_strength
    volume_regime_momentum = primary_accel * volume_persistence
    
    # Non-linear Interaction Framework
    # Momentum-volatility interactions
    low_vol_strong_trend = trending_momentum * (1 / (volatility_ratio + 0.001))
    high_vol_acceleration = primary_accel * volatility_ratio
    stable_vol_quality = smoothness * vol_regime_persistence
    
    # Volume-momentum interactions
    high_volume_acceleration = primary_accel * volume_confirmation_strength
    low_volume_reversal = momentum_curvature * divergence_strength
    volume_persistence_momentum = momentum_blend * volume_persistence
    
    # Three-way regime integration
    perfect_alignment = strong_trending * low_stable_vol * strong_confirmation
    partial_alignment = smoothness * high_stable_vol * volume_confirmation_strength
    conflict_resolution = momentum_curvature * expanding_vol * divergence_strength
    
    # Composite Alpha Construction
    # Base signal generation
    core_momentum = (
        strong_trending * trending_momentum +
        strong_reversing * reversing_momentum +
        momentum_exhaustion * exhaustion_momentum
    )
    
    # Volatility adjustment
    volatility_adjusted = (
        low_stable_vol * core_momentum * (1 + 1/(volatility_ratio + 0.001)) +
        high_stable_vol * core_momentum * vol_regime_persistence +
        expanding_vol * core_momentum * (2 - volatility_ratio) +
        contracting_vol * core_momentum * transition_smoothness
    )
    
    # Volume confirmation overlay
    volume_enhanced = (
        strong_confirmation * volatility_adjusted * volume_confirmation_strength +
        bullish_divergence * volatility_adjusted * divergence_strength +
        bearish_divergence * volatility_adjusted * (-divergence_strength)
    )
    
    # Interaction enhancement
    interaction_terms = (
        perfect_alignment * core_momentum * 2.0 +
        partial_alignment * core_momentum * 1.5 +
        conflict_resolution * core_momentum * 0.5 +
        low_vol_strong_trend * volume_confirmation_strength +
        high_vol_acceleration * volume_persistence +
        stable_vol_quality * volume_stability
    )
    
    # Final composite alpha
    alpha = (
        0.4 * volume_enhanced +
        0.3 * interaction_terms +
        0.2 * momentum_blend +
        0.1 * volatility_adjusted
    )
    
    # Normalize and clean
    alpha = (alpha - alpha.mean()) / (alpha.std() + 0.001)
    alpha = alpha.fillna(0)
    
    return alpha

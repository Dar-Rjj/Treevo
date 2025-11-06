import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a regime-aligned, volume-confirmed, volatility-adjusted momentum factor
    with multi-timeframe quality enhancement.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Multi-Timeframe Returns
    short_term = close.pct_change(periods=3)
    medium_term = close.pct_change(periods=8)
    long_term = close.pct_change(periods=15)
    
    # Momentum Quality Assessment
    momentum_acceleration = short_term - medium_term
    momentum_curvature = short_term - 2 * medium_term + long_term
    
    # Momentum consistency: count of same sign returns over 3 days
    returns_3d = close.pct_change()
    sign_consistency = returns_3d.rolling(window=3).apply(
        lambda x: len(set(np.sign(x.dropna()))) == 1 if len(x.dropna()) == 3 else 0
    )
    
    # Volatility-Adjusted Momentum
    hl_range = (high - low) / close
    short_vol = hl_range.rolling(window=3).std()
    risk_adjusted_momentum = short_term / (short_vol + 1e-8)
    volatility_normalized_acceleration = momentum_acceleration / (short_vol + 1e-8)
    
    # Volume Dynamics
    volume_momentum = volume.pct_change()
    volume_trend = volume.pct_change(periods=3)
    volume_acceleration = volume_momentum - volume_momentum.shift(1)
    
    # Price-Volume Alignment
    volume_confirmation = (np.sign(short_term) == np.sign(volume_momentum)).astype(float)
    confirmation_strength = np.abs(short_term) * np.abs(volume_momentum)
    volume_divergence = (np.sign(short_term) != np.sign(volume_momentum)).astype(float)
    
    # Volume Regime Quality
    high_volume_regime = (volume > 1.5 * volume.shift(1)).astype(float)
    volume_persistence = volume_momentum.rolling(window=3).apply(
        lambda x: (x > 0).sum() if len(x.dropna()) == 3 else 0
    ) / 3.0
    volume_trend_alignment = (np.sign(volume_trend) == np.sign(volume_momentum)).astype(float)
    
    # Momentum Regime Classification
    returns_signs = returns_3d.rolling(window=3).apply(
        lambda x: len(set(np.sign(x.dropna()))) if len(x.dropna()) == 3 else 2
    )
    trending_regime = (returns_signs == 1).astype(float)
    reversing_regime = (returns_signs > 1).astype(float)
    accelerating_regime = ((momentum_acceleration > 0) & (momentum_curvature > 0)).astype(float)
    
    # Volatility Regime Classification
    medium_vol = hl_range.rolling(window=8).std()
    volatility_expansion = short_vol / (medium_vol + 1e-8)
    high_volatility = (volatility_expansion > 1.3).astype(float)
    low_volatility = (volatility_expansion < 0.7).astype(float)
    
    # Volatility regime persistence
    vol_regime_sign = np.sign(volatility_expansion - 1.0)
    volatility_regime_persistence = vol_regime_sign.rolling(window=3).apply(
        lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) == 3 else 0
    ).astype(float)
    
    # Volume Regime Classification
    high_volume_class = (volume_momentum > 0.5).astype(float)
    low_volume_class = (volume_momentum < -0.3).astype(float)
    
    # Volume regime persistence
    vol_class_sign = np.where(volume_momentum > 0.5, 1, np.where(volume_momentum < -0.3, -1, 0))
    volume_regime_persistence = pd.Series(vol_class_sign, index=volume.index).rolling(window=3).apply(
        lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) == 3 else 0
    ).astype(float)
    
    # Cross-Regime Alignment Signals
    # Momentum-Volume Alignment
    confirmed_trending = trending_regime * volume_confirmation
    volume_accelerated_momentum = momentum_acceleration * volume_acceleration
    high_quality_momentum = risk_adjusted_momentum * volume_persistence
    
    # Momentum-Volatility Alignment
    low_vol_momentum_premium = risk_adjusted_momentum * low_volatility
    high_vol_reversal = reversing_regime * high_volatility
    volatility_stable_acceleration = momentum_acceleration * volatility_regime_persistence
    
    # Three-Way Regime Alignment
    perfect_alignment = trending_regime * low_volatility * high_volume_regime
    strong_alignment = accelerating_regime * volume_confirmation * volatility_regime_persistence
    regime_conflict = reversing_regime * high_volatility * volume_divergence
    
    # Composite Alpha Construction
    # Base Signal Components
    core_momentum = risk_adjusted_momentum
    volume_enhanced = core_momentum * (1 + confirmation_strength)
    three_way_alignment = perfect_alignment + strong_alignment - regime_conflict
    regime_aligned = volume_enhanced * (1 + three_way_alignment)
    
    # Quality Overlay
    momentum_quality_boost = regime_aligned * sign_consistency
    volume_quality_enhanced = momentum_quality_boost * volume_trend_alignment
    volatility_stability_enhanced = volume_quality_enhanced * volatility_regime_persistence
    
    # Final Alpha Output
    alpha_factor = volatility_stability_enhanced
    
    return alpha_factor

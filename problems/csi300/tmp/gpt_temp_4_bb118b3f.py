import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Regime Momentum-Volume Convergence Factor
    Combines momentum, volatility, and volume signals across multiple timeframes
    with regime-dependent weighting and non-linear interactions.
    """
    # Calculate returns for momentum analysis
    returns = df['close'].pct_change()
    
    # Multi-Timeframe Momentum Analysis
    # Short-term momentum (3-day)
    short_momentum = df['close'] / df['close'].shift(3) - 1
    short_momentum_deriv = (short_momentum - short_momentum.shift(1)) * short_momentum.shift(1)
    
    # Medium-term momentum (8-day)
    medium_momentum = df['close'] / df['close'].shift(8) - 1
    medium_acceleration = medium_momentum - medium_momentum.shift(2)
    
    # Momentum Regime Classification
    trend_aligned = np.sign(short_momentum) == np.sign(medium_momentum)
    acceleration_regime = np.abs(short_momentum) > np.abs(medium_momentum)
    convergence_score = short_momentum_deriv * medium_acceleration
    
    # Volatility-Adjusted Components
    # Adaptive Volatility Measurement
    short_vol = returns.rolling(window=5).std()  # t-4 to t
    medium_vol = returns.rolling(window=10).std()  # t-9 to t
    vol_ratio = short_vol / medium_vol
    
    # Volatility Regime Detection
    high_vol_regime = vol_ratio > 1.2
    low_vol_regime = vol_ratio < 0.8
    transition_regime = (vol_ratio >= 0.8) & (vol_ratio <= 1.2)
    
    # Volatility-Scaled Signals
    momentum_stability = short_momentum / (short_vol + 1e-8)
    vol_adjusted_returns = returns / (short_vol + 1e-8)
    
    # Volume Divergence Analysis
    # Volume Momentum Signals
    volume_change = df['volume'] / df['volume'].shift(1) - 1
    volume_acceleration = volume_change - volume_change.shift(1)
    
    # Volume persistence (count of positive volume changes over 3 days)
    volume_persistence = pd.Series([
        sum(volume_change.iloc[max(0, i-2):i+1] > 0) if i >= 2 else np.nan 
        for i in range(len(volume_change))
    ], index=volume_change.index)
    
    # Price-Volume Divergence
    price_up = returns > 0
    price_down = returns < 0
    volume_down = volume_change < 0
    volume_up = volume_change > 0
    
    positive_divergence = price_up & volume_down
    negative_divergence = price_down & volume_up
    
    # 5-day price-volume correlation
    price_volume_corr = pd.Series([
        df['close'].iloc[max(0, i-4):i+1].pct_change().corr(
            df['volume'].iloc[max(0, i-4):i+1].pct_change()
        ) if i >= 4 else np.nan
        for i in range(len(df))
    ], index=df.index)
    
    # Multi-Timeframe Volume Patterns
    avg_volume_3d = df['volume'].rolling(window=3).mean()
    volume_breakout = df['volume'] > 1.5 * avg_volume_3d
    volume_drying = df['volume'] < 0.7 * avg_volume_3d
    
    # Non-Linear Interaction Modeling
    # Second-Order Momentum Effects
    momentum_curvature = short_momentum_deriv.diff()
    
    # Volume-Momentum Coupling
    volume_weighted_momentum = short_momentum * (1 + volume_change)
    momentum_volume_confirmation = short_momentum * np.sign(volume_change)
    
    # Regime-Dependent Interactions
    high_vol_momentum = short_momentum * high_vol_regime.astype(float)
    low_vol_trend = medium_momentum * low_vol_regime.astype(float)
    
    # Composite Factor Generation
    # Multi-Regime Weighting
    momentum_regime_strength = (
        trend_aligned.astype(float) * 0.4 + 
        acceleration_regime.astype(float) * 0.3 + 
        np.tanh(convergence_score * 10) * 0.3
    )
    
    volatility_confidence = pd.Series(np.where(
        transition_regime, 0.6,
        np.where(high_vol_regime, 0.3, 0.8)
    ), index=df.index)
    
    volume_confirmation = (
        (~positive_divergence & ~negative_divergence).astype(float) * 0.7 +
        positive_divergence.astype(float) * 0.3 +
        (1 - negative_divergence.astype(float)) * 0.5
    )
    
    # Signal Convergence Detection
    cross_timeframe_alignment = (
        np.sign(short_momentum) == np.sign(medium_momentum)
    ).astype(float)
    
    multi_signal_confirmation = (
        (np.sign(short_momentum) == np.sign(volume_weighted_momentum)).astype(float) * 0.6 +
        (np.sign(short_momentum) == np.sign(momentum_volume_confirmation)).astype(float) * 0.4
    )
    
    divergence_resolution = (
        (positive_divergence & (returns.shift(-1) > 0)).astype(float) * 0.8 +
        (negative_divergence & (returns.shift(-1) < 0)).astype(float) * 0.2
    ).shift(1).fillna(0)
    
    # Enhanced Alpha Output
    # Base momentum component
    base_momentum = (
        short_momentum * 0.6 + 
        medium_momentum * 0.4
    ) * momentum_stability
    
    # Volume validation component
    volume_component = (
        volume_weighted_momentum * 0.5 +
        momentum_volume_confirmation * 0.3 +
        volume_persistence.fillna(0) * 0.2
    )
    
    # Regime-adaptive scaling
    regime_scaling = (
        momentum_regime_strength * volatility_confidence * volume_confirmation
    )
    
    # Final composite factor
    alpha_factor = (
        base_momentum * 0.5 +
        volume_component * 0.3 +
        (cross_timeframe_alignment + multi_signal_confirmation) * 0.2
    ) * regime_scaling
    
    # Apply non-linear transformation to enhance signal quality
    alpha_factor = np.tanh(alpha_factor * 5)
    
    return alpha_factor

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Regime Momentum-Volume Convergence Factor
    Combines momentum derivatives, volatility-adjusted volume divergence, and regime-based signal integration
    """
    close = df['close']
    volume = df['volume']
    
    # Momentum Regime Analysis
    # First-Order Momentum (5-day) with triple smoothing
    momentum_5d = close / close.shift(5) - 1
    momentum_5d_smooth = (momentum_5d + momentum_5d.shift(1) * 0.8 + momentum_5d.shift(2) * 0.6) / 2.4
    
    # Second-Order Momentum (10-day) with double smoothing
    momentum_10d = close / close.shift(10) - 1
    momentum_10d_smooth = (momentum_10d + momentum_10d.shift(1) * 0.7) / 1.7
    
    # Momentum Acceleration
    momentum_accel = momentum_5d_smooth - momentum_10d_smooth
    
    # Regime Detection & Weighting
    momentum_convergence = np.where(
        (momentum_5d_smooth * momentum_10d_smooth > 0) & (np.abs(momentum_accel) < 0.02),
        np.sign(momentum_5d_smooth) * (np.abs(momentum_5d_smooth) + np.abs(momentum_10d_smooth)) / 2,
        0
    )
    
    # Non-Linear Momentum Effects
    momentum_saturation = np.tanh(momentum_5d_smooth * 5)  # Saturation detection
    reversal_prob = 1 - np.exp(-np.abs(momentum_5d_smooth) * 10)  # Reversal probability
    
    # Volatility-Adjusted Volume Divergence
    # Volume Pattern Analysis
    volume_ratio = volume / volume.shift(1)
    volume_momentum_3d = volume_ratio.rolling(window=3).sum()
    
    # Volume Persistence
    high_volume_days = (volume > volume.rolling(window=20).mean() * 1.2).rolling(window=3).sum()
    volume_trend = volume.rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Volume Breakout Detection
    volume_spike = (volume > volume.rolling(window=20).mean() * 1.5).astype(int)
    volume_consolidation = (volume.rolling(window=10).std() / volume.rolling(window=10).mean() < 0.15).astype(int)
    
    # Volatility Context
    returns = close.pct_change()
    volatility_5d = returns.rolling(window=5).std()
    
    # Volume-Volatility Relationship
    volume_efficiency = returns.rolling(window=5).std() / (volume.rolling(window=5).mean() + 1e-8)
    volatility_adj_volume = volume_momentum_3d / (volatility_5d + 1e-8)
    
    # Volume divergence from volatility
    volume_vol_divergence = volume_momentum_3d - volatility_5d.rolling(window=3).mean()
    
    # Signal Integration & Enhancement
    # Volatility-Adjusted Momentum
    risk_adj_momentum = momentum_5d_smooth / (volatility_5d + 1e-8)
    
    # Volume-Confirmed Signals
    volume_weighted_momentum = momentum_5d_smooth * volume_momentum_3d
    volume_confirmation = np.where(
        (momentum_5d_smooth * volume_momentum_3d > 0) & (np.abs(volume_momentum_3d) > 0.1),
        momentum_5d_smooth * 1.5,
        momentum_5d_smooth * 0.5
    )
    
    # Second-Order Interactions
    triple_interaction = momentum_5d_smooth * volatility_5d * volume_momentum_3d
    
    # Robustness Enhancement
    regime_strength = (
        np.abs(momentum_convergence) + 
        np.abs(volume_confirmation) + 
        np.abs(risk_adj_momentum)
    ) / 3
    
    # Composite Alpha Construction
    # Factor Blending with regime-adaptive weighting
    momentum_component = (
        momentum_5d_smooth * 0.4 + 
        momentum_convergence * 0.3 + 
        risk_adj_momentum * 0.3
    )
    
    volume_component = (
        volatility_adj_volume * 0.5 + 
        volume_confirmation * 0.3 + 
        volume_vol_divergence * 0.2
    )
    
    # Non-Linear Transformation
    momentum_saturated = momentum_component * (1 - reversal_prob)
    volume_enhanced = volume_component * np.tanh(regime_strength * 2)
    
    # Final Alpha Score
    alpha = (
        momentum_saturated * 0.6 + 
        volume_enhanced * 0.4 + 
        triple_interaction * 0.1
    )
    
    # Apply regime-based filters
    alpha = np.where(
        regime_strength > 0.1,
        alpha * regime_strength,
        alpha * 0.3
    )
    
    # Volume confirmation filter
    alpha = np.where(
        volume_momentum_3d.abs() > 0.05,
        alpha,
        alpha * 0.5
    )
    
    return pd.Series(alpha, index=df.index)

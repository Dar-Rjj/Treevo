import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence via percentile-based regime weights.
    
    Interpretation:
    - Momentum acceleration hierarchy across intraday, overnight, and daily timeframes
    - Volume divergence detection using percentile-based regime classification
    - Smooth regime transitions using exponential weighting for persistence
    - Multiplicative combinations enhance signal robustness across market conditions
    - Regime-specific emphasis adapts to changing market environments
    - Positive values indicate synchronized momentum acceleration with volume confirmation
    - Negative values suggest momentum deceleration with volume divergence
    """
    
    # Core momentum components (normalization-free)
    intraday_return = (df['close'] - df['open'])
    overnight_return = (df['open'] - df['close'].shift(1))
    daily_return = (df['close'] - df['close'].shift(1))
    
    # Momentum acceleration hierarchy
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    daily_accel = daily_return - daily_return.shift(1)
    
    # Multi-timeframe momentum convergence
    momentum_convergence = (
        intraday_accel.rolling(window=3).mean() * 
        overnight_accel.rolling(window=3).mean() * 
        daily_accel.rolling(window=3).mean()
    )
    
    # Volume divergence detection
    volume_5d_mean = df['volume'].rolling(window=5).mean()
    volume_20d_mean = df['volume'].rolling(window=20).mean()
    volume_divergence = (volume_5d_mean - volume_20d_mean) / (volume_20d_mean + 1e-7)
    
    # Percentile-based regime classification
    vol_regime_20d = df['volume'].rolling(window=20).apply(
        lambda x: np.percentile(x, 70) if len(x) == 20 else np.nan
    )
    vol_regime_50d = df['volume'].rolling(window=50).apply(
        lambda x: np.percentile(x, 30) if len(x) == 50 else np.nan
    )
    
    # Volume regime weights using percentile thresholds
    high_vol_regime = (df['volume'] > vol_regime_20d).astype(float)
    low_vol_regime = (df['volume'] < vol_regime_50d).astype(float)
    medium_vol_regime = 1 - high_vol_regime - low_vol_regime
    
    # Smooth regime transitions with exponential persistence
    regime_persistence = 0.7
    high_vol_smooth = high_vol_regime.ewm(alpha=1-regime_persistence).mean()
    low_vol_smooth = low_vol_regime.ewm(alpha=1-regime_persistence).mean()
    medium_vol_smooth = medium_vol_regime.ewm(alpha=1-regime_persistence).mean()
    
    # Regime-specific momentum emphasis
    high_vol_momentum = (
        high_vol_smooth * intraday_accel * 0.6 +
        high_vol_smooth * daily_accel * 0.4
    )
    
    low_vol_momentum = (
        low_vol_smooth * overnight_accel * 0.7 +
        low_vol_smooth * momentum_convergence * 0.3
    )
    
    medium_vol_momentum = (
        medium_vol_smooth * intraday_accel * 0.4 +
        medium_vol_smooth * overnight_accel * 0.3 +
        medium_vol_smooth * daily_accel * 0.3
    )
    
    # Volume-momentum synchronization via multiplicative combination
    volume_momentum_sync = (
        volume_divergence * momentum_convergence * 
        np.sign(volume_divergence * momentum_convergence)
    )
    
    # Combined alpha factor with regime-adaptive weights
    alpha_factor = (
        high_vol_momentum * 0.4 +
        low_vol_momentum * 0.3 +
        medium_vol_momentum * 0.3 +
        volume_momentum_sync * 0.15
    )
    
    return alpha_factor

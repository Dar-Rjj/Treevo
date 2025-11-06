import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced regime-adaptive momentum factor with volatility scaling and volume confirmation.
    Economic intuition: Stocks with volatility-normalized momentum signals that are confirmed
    by volume divergence and adaptively weighted across market regimes tend to predict returns better.
    """
    
    # Volatility-scaled momentum with multiple time horizons
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_8d = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    momentum_15d = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    
    # Multi-period volatility estimation
    volatility_10d = df['close'].pct_change().rolling(window=10, min_periods=8).std()
    volatility_25d = df['close'].pct_change().rolling(window=25, min_periods=20).std()
    
    # Volatility-scaled momentum composites
    vol_scaled_momentum_short = momentum_3d / (volatility_10d + 1e-7)
    vol_scaled_momentum_medium = momentum_8d / (volatility_25d + 1e-7)
    
    # Momentum acceleration component
    momentum_acceleration = momentum_3d - momentum_15d
    
    # Multi-timeframe volume divergence
    volume_3d_avg = df['volume'].rolling(window=3, min_periods=2).mean()
    volume_15d_avg = df['volume'].rolling(window=15, min_periods=12).mean()
    volume_25d_avg = df['volume'].rolling(window=25, min_periods=20).mean()
    
    volume_divergence_short = (df['volume'] - volume_3d_avg) / (volume_15d_avg + 1e-7)
    volume_divergence_medium = (df['volume'] - volume_15d_avg) / (volume_25d_avg + 1e-7)
    
    # Enhanced regime detection using volatility regime classification
    volatility_regime = volatility_25d.rolling(window=60, min_periods=45).apply(
        lambda x: 2 if x.iloc[-1] > x.quantile(0.75) else (1 if x.iloc[-1] > x.quantile(0.25) else 0), 
        raw=False
    ).fillna(1)
    
    # Adaptive regime weighting with smooth transitions
    high_vol_weight = 0.6 * (volatility_regime == 2)
    medium_vol_weight = 0.3 * (volatility_regime == 1)
    low_vol_weight = 0.1 * (volatility_regime == 0)
    regime_weight = high_vol_weight + medium_vol_weight + low_vol_weight
    
    # Volume confirmation strength varies by regime
    volume_confirmation_strength = 0.4 + 0.3 * regime_weight
    
    # Composite factor construction with regime-adaptive blending
    momentum_component = (vol_scaled_momentum_short + 0.7 * vol_scaled_momentum_medium + 
                         0.3 * momentum_acceleration)
    
    volume_confirmation = (volume_divergence_short + 0.6 * volume_divergence_medium)
    
    # Final factor with regime-adaptive volume confirmation
    alpha_factor = momentum_component * (1 + volume_confirmation_strength * volume_confirmation)
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)

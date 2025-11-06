import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-scaled momentum components
    # Short-term momentum (3-day) normalized by recent volatility (5-day)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    vol_5d = df['close'].pct_change().rolling(window=5).std()
    vol_scaled_momentum = momentum_3d / (vol_5d + 1e-7)
    
    # Volume acceleration with regime detection
    # Volume trend acceleration (difference between 3-day and 6-day volume momentum)
    vol_momentum_3d = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3)
    vol_momentum_6d = (df['volume'] - df['volume'].shift(6)) / df['volume'].shift(6)
    volume_acceleration = vol_momentum_3d - vol_momentum_6d
    
    # Regime shift detection using price and volume breakouts
    # Price regime: break from 10-day range
    price_regime = (df['close'] - df['close'].rolling(window=10).mean()) / df['close'].rolling(window=10).std()
    # Volume regime: break from 10-day volume pattern
    volume_regime = (df['volume'] - df['volume'].rolling(window=10).mean()) / df['volume'].rolling(window=10).std()
    
    # Adaptive bounds using rolling percentiles for robust scaling
    # Momentum bounds (20-day rolling percentiles)
    momentum_bounds = momentum_3d.rolling(window=20).apply(lambda x: (x.iloc[-1] - x.quantile(0.1)) / (x.quantile(0.9) - x.quantile(0.1) + 1e-7))
    
    # Volume acceleration bounds
    vol_accel_bounds = volume_acceleration.rolling(window=20).apply(lambda x: (x.iloc[-1] - x.quantile(0.1)) / (x.quantile(0.9) - x.quantile(0.1) + 1e-7))
    
    # Smooth transitions using sigmoid-like functions for regime components
    def smooth_transition(x, center=0, scale=2):
        return 1 / (1 + np.exp(-scale * (x - center)))
    
    price_regime_smooth = smooth_transition(price_regime)
    volume_regime_smooth = smooth_transition(volume_regime)
    
    # Multiplicative combination with regime-dependent scaling
    factor = (
        vol_scaled_momentum * 
        (1 + volume_acceleration) * 
        price_regime_smooth * 
        volume_regime_smooth * 
        momentum_bounds * 
        vol_accel_bounds
    )
    
    return factor

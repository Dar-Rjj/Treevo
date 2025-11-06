import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-scaled momentum components
    # Short-term momentum (3-day) scaled by rolling volatility
    returns_3d = df['close'].pct_change(periods=3)
    vol_10d = df['close'].pct_change().rolling(window=10, min_periods=5).std()
    vol_scaled_momentum = returns_3d / (vol_10d + 1e-7)
    
    # Medium-term momentum (8-day) with adaptive volatility scaling
    returns_8d = df['close'].pct_change(periods=8)
    vol_20d = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    adaptive_momentum = returns_8d / (vol_20d + 1e-7)
    
    # Volume acceleration with regime detection
    volume_ma_5 = df['volume'].rolling(window=5, min_periods=3).mean()
    volume_ma_20 = df['volume'].rolling(window=20, min_periods=10).mean()
    volume_acceleration = (volume_ma_5 - volume_ma_20) / (volume_ma_20 + 1e-7)
    
    # Regime shift detection using price and volume confluence
    price_regime = df['close'].rolling(window=10, min_periods=5).apply(
        lambda x: 1 if (x[-1] > x.mean() + x.std()) else (-1 if (x[-1] < x.mean() - x.std()) else 0)
    )
    volume_regime = df['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: 1 if (x[-1] > x.mean() + x.std()) else (-1 if (x[-1] < x.mean() - x.std()) else 0)
    )
    regime_shift = price_regime * volume_regime
    
    # Adaptive bounds for momentum components using rolling percentiles
    momentum_bounds = vol_scaled_momentum.rolling(window=50, min_periods=25).apply(
        lambda x: np.clip(x.iloc[-1], x.quantile(0.1), x.quantile(0.9))
    )
    adaptive_bounds = adaptive_momentum.rolling(window=50, min_periods=25).apply(
        lambda x: np.clip(x.iloc[-1], x.quantile(0.1), x.quantile(0.9))
    )
    
    # Smooth transitions using exponential weighting
    smooth_momentum = vol_scaled_momentum.ewm(span=5, adjust=False).mean()
    smooth_adaptive = adaptive_momentum.ewm(span=5, adjust=False).mean()
    smooth_volume = volume_acceleration.ewm(span=5, adjust=False).mean()
    
    # Multiplicative combination with regime-aware scaling
    factor = (
        (1 + momentum_bounds) * 
        (1 + adaptive_bounds) * 
        (1 + smooth_volume) * 
        (1 + regime_shift * 0.1) * 
        np.exp(smooth_momentum + smooth_adaptive)
    )
    
    return factor

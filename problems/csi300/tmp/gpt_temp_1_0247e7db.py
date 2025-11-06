import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-scaled momentum components
    # Short-term momentum (3-day) normalized by 10-day volatility
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    volatility_10d = df['close'].pct_change().rolling(window=10, min_periods=5).std()
    scaled_momentum = momentum_3d / (volatility_10d + 1e-7)
    
    # Volume acceleration with regime detection
    # Volume momentum (3-day) with adaptive smoothing
    volume_momentum_3d = (df['volume'] - df['volume'].shift(3)) / (df['volume'].shift(3) + 1e-7)
    volume_volatility = df['volume'].pct_change().rolling(window=10, min_periods=5).std()
    volume_regime = volume_momentum_3d / (volume_volatility + 1e-7)
    
    # Regime shift detection using price and volume confluence
    # Price-volume trend consistency
    price_trend = df['close'].rolling(window=5, min_periods=3).mean()
    volume_trend = df['volume'].rolling(window=5, min_periods=3).mean()
    regime_shift = ((df['close'] > price_trend) & (df['volume'] > volume_trend)).astype(float) - \
                   ((df['close'] < price_trend) & (df['volume'] < volume_trend)).astype(float)
    
    # Adaptive bounds using rolling percentiles
    momentum_bounds = scaled_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.1)) / (x.quantile(0.9) - x.quantile(0.1) + 1e-7)
    )
    volume_bounds = volume_regime.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.1)) / (x.quantile(0.9) - x.quantile(0.1) + 1e-7)
    )
    
    # Smooth transitions using sigmoid-like functions
    def smooth_transition(x, center=0, scale=2):
        return 1 / (1 + np.exp(-scale * (x - center)))
    
    momentum_signal = smooth_transition(momentum_bounds, center=0.5, scale=3)
    volume_signal = smooth_transition(volume_bounds, center=0.5, scale=3)
    regime_signal = smooth_transition(regime_shift, center=0, scale=2)
    
    # Multiplicative combination with synergy enhancement
    factor = momentum_signal * volume_signal * regime_signal * \
             (1 + 0.5 * momentum_signal * volume_signal) * \
             (1 + 0.3 * momentum_signal * regime_signal) * \
             (1 + 0.2 * volume_signal * regime_signal)
    
    return factor

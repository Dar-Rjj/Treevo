import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adjusted Price-Volume Divergence with Efficiency Filter
    """
    # Calculate Multi-Period Momentum with Decay
    close = df['close']
    volume = df['volume']
    
    # 1-day momentum
    mom_1d = close.pct_change(1)
    
    # 5-day momentum with exponential decay weighting
    weights_5d = np.exp(-np.arange(5) / 2.5)  # Decay factor
    weights_5d /= weights_5d.sum()
    mom_5d = close.pct_change(periods=5).rolling(window=5, min_periods=5).apply(
        lambda x: np.sum(x * weights_5d[:len(x)]), raw=False
    )
    
    # 10-day momentum with exponential decay weighting
    weights_10d = np.exp(-np.arange(10) / 5)  # Decay factor
    weights_10d /= weights_10d.sum()
    mom_10d = close.pct_change(periods=10).rolling(window=10, min_periods=10).apply(
        lambda x: np.sum(x * weights_10d[:len(x)]), raw=False
    )
    
    # Calculate Volume Momentum Patterns
    # Short-term volume changes (3-day momentum)
    vol_mom_3d = volume.pct_change(3)
    
    # Volume acceleration (change in volume momentum)
    vol_accel = vol_mom_3d.diff(2)
    
    # Compute Price-Volume Divergence
    # Weighted price momentum composite
    price_momentum = 0.4 * mom_1d + 0.4 * mom_5d + 0.2 * mom_10d
    
    # Volume momentum composite
    volume_momentum = 0.7 * vol_mom_3d + 0.3 * vol_accel
    
    # Divergence strength (positive when price and volume move in opposite directions)
    divergence_strength = -price_momentum * volume_momentum
    
    # Assess Price Movement Efficiency
    high = df['high']
    low = df['low']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Absolute price movement
    abs_movement = abs(close - close.shift(1))
    
    # Efficiency ratio (movement/range)
    efficiency_ratio = abs_movement / true_range
    efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    efficiency_ratio = efficiency_ratio.clip(0, 1)  # Bound between 0 and 1
    
    # Identify Volatility Regime
    # 20-day average true range
    atr_20d = true_range.rolling(window=20, min_periods=10).mean()
    
    # Current volatility relative to average
    vol_ratio = true_range / atr_20d
    
    # Volatility regime classification (0=low, 1=normal, 2=high)
    vol_regime = pd.cut(vol_ratio, 
                       bins=[0, 0.7, 1.3, np.inf], 
                       labels=[0, 1, 2], 
                       include_lowest=True).astype(float)
    
    # Generate Regime-Adjusted Signal with Efficiency Filter
    # Regime adjustment factors
    regime_weights = vol_regime.map({0: 1.5, 1: 1.0, 2: 0.5})
    
    # Efficiency-weighted divergence
    efficiency_weighted_divergence = divergence_strength * efficiency_ratio
    
    # Final regime-adjusted signal
    alpha_signal = efficiency_weighted_divergence * regime_weights
    
    return alpha_signal

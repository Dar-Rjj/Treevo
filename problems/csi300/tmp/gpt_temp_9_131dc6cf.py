import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Regime-aware volatility-scaled momentum with volume divergence.
    Combines momentum strength, volume confirmation, and volatility normalization
    with regime-dependent weighting for robustness across market conditions.
    """
    
    # Volatility-scaled momentum - 5-day return normalized by 20-day volatility
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    volatility_20d = df['close'].pct_change().rolling(window=20, min_periods=15).std()
    vol_scaled_momentum = momentum_5d / (volatility_20d + 1e-7)
    
    # Volume divergence - current volume vs 10-day trend
    volume_ma_10 = df['volume'].rolling(window=10, min_periods=7).mean()
    volume_trend = df['volume'].rolling(window=5, min_periods=3).mean() / volume_ma_10
    volume_divergence = (df['volume'] / volume_ma_10) / (volume_trend + 1e-7)
    
    # Regime detection - market state based on 20-day volatility percentile
    vol_percentile = volatility_20d.rolling(window=60, min_periods=40).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 70)), raw=False
    )
    
    # Regime-aware weights - adjust factor sensitivity based on market volatility
    high_vol_weight = 0.7  # More conservative in high volatility
    low_vol_weight = 1.3   # More aggressive in low volatility
    
    regime_weight = np.where(vol_percentile == 1, high_vol_weight, low_vol_weight)
    
    # Combine components with regime-aware scaling
    alpha_factor = vol_scaled_momentum * volume_divergence * regime_weight
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)

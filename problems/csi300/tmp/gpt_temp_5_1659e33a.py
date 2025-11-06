import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-scaled momentum with volume divergence and regime-aware weights
    # Clear interpretation: regime-adjusted momentum strength × volume divergence signal
    
    # Volatility-scaled momentum - 10-day return scaled by 20-day volatility
    momentum_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    volatility_20d = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    vol_scaled_momentum = momentum_10d / (volatility_20d + 1e-7)
    
    # Volume divergence - current volume vs 15-day average, adjusted for price move direction
    volume_avg_15d = df['volume'].rolling(window=15, min_periods=8).mean()
    volume_divergence = (df['volume'] - volume_avg_15d) / volume_avg_15d
    price_direction = np.sign(df['close'] - df['close'].shift(1))
    directional_volume_div = volume_divergence * price_direction
    
    # Regime detection - market state based on 20-day volatility percentile
    vol_percentile = volatility_20d.rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 70)), raw=False
    )
    
    # Regime-aware weights - higher momentum weight in low vol regimes
    momentum_weight = 0.7 - 0.3 * vol_percentile
    volume_weight = 0.3 + 0.3 * vol_percentile
    
    # Combine components with regime-aware weights
    alpha_factor = (
        momentum_weight * vol_scaled_momentum + 
        volume_weight * directional_volume_div
    )
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)

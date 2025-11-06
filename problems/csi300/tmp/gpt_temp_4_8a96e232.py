import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and dynamic regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration terms
    - Volume divergence detection across different time horizons for confirmation
    - Dynamic regime classification based on volatility and volume characteristics
    - Percentile-based normalization preserves cross-sectional ranking properties
    - Multiplicative combinations enhance signal strength during confirmed regimes
    - Positive values indicate strong momentum with volume confirmation across timeframes
    - Negative values suggest momentum breakdown with volume distribution patterns
    """
    
    # Hierarchical momentum components
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    intraday_accel = intraday_momentum - intraday_momentum.shift(1)
    overnight_accel = overnight_momentum - overnight_momentum.shift(1)
    weekly_accel = weekly_momentum - weekly_momentum.shift(1)
    
    # Volume divergence across timeframes
    volume_short = df['volume'] / (df['volume'].rolling(window=3).mean() + 1e-7)
    volume_medium = df['volume'] / (df['volume'].rolling(window=8).mean() + 1e-7)
    volume_long = df['volume'] / (df['volume'].rolling(window=21).mean() + 1e-7)
    
    volume_divergence = (
        (volume_short - volume_medium) * np.sign(volume_short) +
        (volume_medium - volume_long) * np.sign(volume_medium) +
        (volume_short - volume_long) * np.sign(volume_short)
    )
    
    # Dynamic regime classification
    daily_range = df['high'] - df['low']
    vol_regime = daily_range.rolling(window=5).std() / (daily_range.rolling(window=21).std() + 1e-7)
    volume_regime = df['volume'].rolling(window=5).std() / (df['volume'].rolling(window=21).std() + 1e-7)
    
    # Regime weights based on volatility and volume characteristics
    high_vol_weight = np.where(vol_regime > 1.2, 1.8, 1.0)
    low_vol_weight = np.where(vol_regime < 0.8, 0.6, 1.0)
    high_volume_weight = np.where(volume_regime > 1.1, 1.4, 1.0)
    low_volume_weight = np.where(volume_regime < 0.9, 0.7, 1.0)
    
    # Multiplicative regime combination
    regime_weight = high_vol_weight * low_vol_weight * high_volume_weight * low_volume_weight
    
    # Hierarchical momentum with volume confirmation
    momentum_hierarchy = (
        intraday_momentum.rolling(window=3).apply(lambda x: x.rank(pct=True).iloc[-1]) *
        overnight_momentum.rolling(window=3).apply(lambda x: x.rank(pct=True).iloc[-1]) *
        weekly_momentum.rolling(window=3).apply(lambda x: x.rank(pct=True).iloc[-1])
    )
    
    # Acceleration hierarchy with volume divergence
    accel_hierarchy = (
        intraday_accel.rolling(window=3).apply(lambda x: x.rank(pct=True).iloc[-1]) *
        overnight_accel.rolling(window=3).apply(lambda x: x.rank(pct=True).iloc[-1]) *
        weekly_accel.rolling(window=3).apply(lambda x: x.rank(pct=True).iloc[-1])
    )
    
    # Volume-confirmed momentum convergence
    volume_momentum_alignment = (
        volume_divergence.rolling(window=3).apply(lambda x: x.rank(pct=True).iloc[-1]) *
        momentum_hierarchy *
        np.sign(volume_divergence * momentum_hierarchy)
    )
    
    # Final alpha factor with regime adaptation
    alpha_factor = (
        momentum_hierarchy * 0.4 +
        accel_hierarchy * 0.3 +
        volume_momentum_alignment * 0.3
    ) * regime_weight
    
    return alpha_factor

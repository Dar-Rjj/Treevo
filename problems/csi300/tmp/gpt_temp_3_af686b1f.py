import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and dynamic regime weighting.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (ultra-short, short, medium) with acceleration signals
    - Volume divergence detection across different momentum regimes
    - Dynamic regime weighting based on volatility and volume percentiles
    - Multiplicative combinations enhance signal strength and interpretability
    - Percentile-based regime classification for robust adaptation to market conditions
    - Positive values indicate strong momentum acceleration with volume confirmation
    - Negative values suggest momentum deceleration with volume divergence
    """
    
    # Hierarchical momentum components
    ultra_short_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    short_momentum = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(3).max() - df['low'].rolling(3).min() + 1e-7)
    medium_momentum = (df['close'] - df['close'].shift(8)) / (df['high'].rolling(8).max() - df['low'].rolling(8).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    ultra_short_accel = ultra_short_momentum - ultra_short_momentum.shift(2)
    short_accel = short_momentum - short_momentum.shift(3)
    medium_accel = medium_momentum - medium_momentum.shift(5)
    
    # Volume divergence components
    volume_ma_3 = df['volume'].rolling(window=3).mean()
    volume_ma_8 = df['volume'].rolling(window=8).mean()
    
    volume_divergence_short = (df['volume'] - volume_ma_3) * np.sign(ultra_short_momentum)
    volume_divergence_medium = (df['volume'] - volume_ma_8) * np.sign(short_momentum)
    
    # Percentile-based regime classification
    daily_range = df['high'] - df['low']
    vol_5d = daily_range.rolling(window=5).std()
    vol_percentile = vol_5d.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1)
    
    volume_percentile = df['volume'].rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1)
    
    # Dynamic regime weights based on percentile combinations
    high_vol_regime = (vol_percentile >= 2)
    medium_vol_regime = (vol_percentile == 1)
    low_vol_regime = (vol_percentile == 0)
    
    high_volume_regime = (volume_percentile >= 2)
    medium_volume_regime = (volume_percentile == 1)
    low_volume_regime = (volume_percentile == 0)
    
    # Regime-specific momentum weights
    ultra_short_weight = np.where(high_vol_regime, 0.4, 
                                 np.where(low_vol_regime, 0.2, 0.3))
    short_weight = np.where(high_vol_regime, 0.3,
                           np.where(low_vol_regime, 0.3, 0.35))
    medium_weight = np.where(high_vol_regime, 0.2,
                            np.where(low_vol_regime, 0.4, 0.25))
    accel_weight = np.where(high_vol_regime, 0.1,
                           np.where(low_vol_regime, 0.1, 0.1))
    
    # Volume regime multipliers
    volume_multiplier = np.where(high_volume_regime, 1.5,
                                np.where(low_volume_regime, 0.7, 1.0))
    
    # Multiplicative combinations for enhanced signals
    momentum_accel_combo = ultra_short_accel * short_accel * np.sign(ultra_short_accel + short_accel)
    volume_momentum_combo = volume_divergence_short * volume_divergence_medium * np.sign(volume_divergence_short + volume_divergence_medium)
    
    # Hierarchical alpha factor construction
    alpha_factor = (
        ultra_short_weight * ultra_short_momentum +
        short_weight * short_momentum +
        medium_weight * medium_momentum +
        accel_weight * (ultra_short_accel + short_accel + medium_accel) +
        0.15 * momentum_accel_combo +
        0.12 * volume_momentum_combo
    ) * volume_multiplier
    
    return alpha_factor

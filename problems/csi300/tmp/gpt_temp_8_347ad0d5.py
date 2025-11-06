import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and percentile-based regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration signals
    - Volume divergence detection across different time horizons for confirmation
    - Percentile-based regime classification for robust adaptation to market conditions
    - Multiplicative combination of momentum and volume components enhances signal strength
    - Dynamic emphasis on different momentum components based on volatility percentiles
    - Positive values indicate strong bullish momentum with volume confirmation across timeframes
    - Negative values suggest bearish pressure with volume distribution patterns
    """
    
    # Hierarchical momentum components
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    intraday_accel = intraday_momentum - intraday_momentum.shift(1)
    overnight_accel = overnight_momentum - overnight_momentum.shift(1)
    weekly_accel = weekly_momentum - weekly_momentum.shift(1)
    
    # Volume divergence components
    volume_ratio_3d = df['volume'] / (df['volume'].rolling(window=3).mean() + 1e-7)
    volume_ratio_10d = df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-7)
    volume_divergence = volume_ratio_3d - volume_ratio_10d
    
    # Percentile-based regime classification
    daily_range = df['high'] - df['low']
    vol_5d_std = daily_range.rolling(window=5).std()
    vol_percentile = vol_5d_std.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1)
    
    # Volume pressure percentiles
    volume_pressure = df['volume'] / (df['volume'].rolling(window=5).mean() + 1e-7)
    volume_percentile = volume_pressure.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1)
    
    # Multiplicative momentum combinations
    momentum_product = intraday_momentum * overnight_momentum * weekly_momentum
    accel_product = intraday_accel * overnight_accel * weekly_accel
    
    # Volume-confirmed momentum with divergence
    volume_momentum_sync = (intraday_momentum * volume_ratio_3d + 
                           overnight_momentum * volume_ratio_10d + 
                           weekly_momentum * volume_divergence)
    
    # Regime-adaptive weights using percentile ranks
    intraday_weight = np.where(vol_percentile == 3, 0.4, 
                              np.where(vol_percentile == 2, 0.3, 0.2))
    overnight_weight = np.where(vol_percentile == 3, 0.2,
                               np.where(vol_percentile == 2, 0.25, 0.3))
    weekly_weight = np.where(vol_percentile == 3, 0.15,
                            np.where(vol_percentile == 2, 0.2, 0.25))
    accel_weight = np.where(vol_percentile == 3, 0.25,
                           np.where(vol_percentile == 2, 0.25, 0.25))
    
    # Volume regime emphasis
    volume_emphasis = np.where(volume_percentile == 3, 1.5,
                              np.where(volume_percentile == 2, 1.2, 0.8))
    
    # Combined alpha factor with hierarchical structure
    alpha_factor = (
        intraday_weight * intraday_momentum * volume_emphasis +
        overnight_weight * overnight_momentum * volume_emphasis +
        weekly_weight * weekly_momentum * volume_emphasis +
        accel_weight * (intraday_accel + overnight_accel + weekly_accel) * np.sign(momentum_product) +
        volume_momentum_sync * 0.1 * volume_emphasis +
        momentum_product * 0.05 * np.sign(accel_product)
    )
    
    return alpha_factor

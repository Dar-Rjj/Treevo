import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with regime-aware volume divergence and percentile normalization.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, multi-day) with regime-specific emphasis
    - Volume divergence detection across different momentum regimes using percentile ranking
    - Regime-aware dynamic weighting based on volatility states and volume pressure
    - Multiplicative combinations enhance signal strength during confirmed regimes
    - Percentile normalization maintains cross-sectional comparability without traditional scaling
    - Positive values indicate bullish momentum with volume confirmation across hierarchical timeframes
    - Negative values suggest bearish pressure with volume distribution patterns
    """
    
    # Hierarchical momentum components
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    short_term_momentum = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(3).max() - df['low'].rolling(3).min() + 1e-7)
    
    # Volatility regime detection using percentile ranking
    daily_range = df['high'] - df['low']
    vol_5d = daily_range.rolling(window=5).std()
    vol_15d_median = vol_5d.rolling(window=15).median()
    vol_ratio = vol_5d / (vol_15d_median + 1e-7)
    
    # Percentile-based regime classification
    vol_percentile = vol_ratio.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)), raw=False)
    
    # Volume pressure with hierarchical percentile ranking
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_pressure = df['volume'] / (volume_5d_avg + 1e-7)
    volume_percentile = volume_pressure.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.8)) * 3 + (x.iloc[-1] > x.quantile(0.6)) * 2 + (x.iloc[-1] > x.quantile(0.4)), raw=False)
    
    # Momentum acceleration hierarchy with multiplicative combinations
    ultra_short_accel = intraday_momentum * overnight_momentum * np.sign(intraday_momentum + overnight_momentum)
    multi_timeframe_accel = (intraday_momentum + short_term_momentum) * np.sign(intraday_momentum * short_term_momentum)
    
    # Volume divergence detection using percentile combinations
    volume_intraday_divergence = volume_percentile * intraday_momentum * (volume_percentile > 2)
    volume_overnight_divergence = volume_percentile * overnight_momentum * (volume_percentile > 1)
    volume_short_term_divergence = volume_percentile * short_term_momentum * (volume_percentile > 0)
    
    # Regime-aware dynamic weighting using percentile-based regimes
    intraday_weight = np.where(vol_percentile == 2, 0.4, 
                              np.where(vol_percentile == 1, 0.3, 0.2))
    overnight_weight = np.where(vol_percentile == 2, 0.2,
                               np.where(vol_percentile == 1, 0.25, 0.3))
    short_term_weight = np.where(vol_percentile == 2, 0.2,
                                np.where(vol_percentile == 1, 0.25, 0.3))
    accel_weight = np.where(vol_percentile == 2, 0.2,
                           np.where(vol_percentile == 1, 0.2, 0.2))
    
    # Volume divergence weights based on volume percentile regimes
    volume_div_weight = np.where(volume_percentile >= 3, 0.15,
                                np.where(volume_percentile >= 2, 0.1, 0.05))
    
    # Combined alpha factor with hierarchical structure
    momentum_component = (
        intraday_weight * intraday_momentum +
        overnight_weight * overnight_momentum +
        short_term_weight * short_term_momentum +
        accel_weight * (ultra_short_accel + multi_timeframe_accel)
    )
    
    volume_divergence_component = (
        volume_div_weight * (volume_intraday_divergence + 
                           volume_overnight_divergence + 
                           volume_short_term_divergence)
    )
    
    # Final alpha factor with regime-aware combination
    alpha_factor = momentum_component * (1 + volume_divergence_component)
    
    return alpha_factor

import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with regime-aware volume divergence and percentile normalization.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration signals
    - Dynamic regime classification based on volatility and volume pressure percentiles
    - Volume divergence detection using percentile-based outlier identification
    - Multiplicative combination of momentum acceleration and volume confirmation
    - Regime-specific emphasis weights adapt to market conditions
    - Positive values indicate strong momentum acceleration with volume confirmation
    - Negative values suggest momentum deceleration with volume divergence
    """
    
    # Hierarchical momentum components
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    ultra_short_accel = intraday_momentum * overnight_momentum * np.sign(intraday_momentum + overnight_momentum)
    short_term_accel = (intraday_momentum + overnight_momentum) * weekly_momentum * np.sign(intraday_momentum + overnight_momentum + weekly_momentum)
    combined_accel = ultra_short_accel * short_term_accel * np.sign(ultra_short_accel + short_term_accel)
    
    # Percentile-based regime detection
    daily_range = df['high'] - df['low']
    vol_5d = daily_range.rolling(window=5).std()
    vol_percentile = vol_5d.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1, raw=False)
    
    # Volume pressure with percentile regimes
    volume_ratio = df['volume'] / (df['volume'].rolling(window=5).mean() + 1e-7)
    volume_regime = volume_ratio.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.8)) * 3 + (x.iloc[-1] > x.quantile(0.6)) * 2 + (x.iloc[-1] > x.quantile(0.4)) * 1, raw=False)
    
    # Volume divergence detection
    price_change = df['close'] / df['close'].shift(1) - 1
    volume_divergence = volume_ratio * price_change * np.sign(volume_ratio - 1) * np.sign(price_change)
    
    # Regime-aware dynamic weights
    intraday_weight = np.where(vol_percentile == 3, 0.4, 
                              np.where(vol_percentile == 2, 0.3, 0.2))
    overnight_weight = np.where(vol_percentile == 3, 0.2,
                               np.where(vol_percentile == 2, 0.25, 0.3))
    weekly_weight = np.where(vol_percentile == 3, 0.1,
                            np.where(vol_percentile == 2, 0.15, 0.2))
    accel_weight = np.where(vol_percentile == 3, 0.3,
                           np.where(vol_percentile == 2, 0.3, 0.3))
    
    # Volume regime multipliers
    volume_multiplier = np.where(volume_regime == 3, 1.5,
                                np.where(volume_regime == 2, 1.2,
                                        np.where(volume_regime == 1, 0.8, 0.6)))
    
    # Multiplicative combination with hierarchical structure
    momentum_component = (
        intraday_weight * intraday_momentum +
        overnight_weight * overnight_momentum +
        weekly_weight * weekly_momentum +
        accel_weight * combined_accel
    )
    
    volume_component = volume_multiplier * volume_divergence * np.sign(momentum_component)
    
    # Final alpha factor with regime-aware blending
    alpha_factor = momentum_component * (1 + 0.3 * volume_component) * np.sign(momentum_component + volume_component)
    
    return alpha_factor

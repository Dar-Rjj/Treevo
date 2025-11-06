import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with regime-aware volume divergence and percentile normalization.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration signals
    - Dynamic regime detection using volatility and volume pressure percentiles
    - Volume divergence measures alignment between price momentum and trading intensity
    - Multiplicative combinations enhance signal strength during confirmed regimes
    - Percentile-based regime classification provides robust adaptation to market conditions
    - Positive values indicate strong momentum with volume confirmation across hierarchical timeframes
    - Negative values suggest momentum breakdown with volume-pressure misalignment
    """
    
    # Hierarchical momentum components with acceleration
    intraday_return = (df['close'] - df['open']) / df['open']
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Momentum acceleration signals
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    weekly_accel = weekly_momentum - weekly_momentum.shift(5)
    
    # Volatility regime using percentile-based classification
    daily_range_pct = (df['high'] - df['low']) / df['open']
    vol_5d_std = daily_range_pct.rolling(window=5).std()
    vol_percentile = vol_5d_std.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1)
    
    # Volume pressure regimes using multi-tier percentiles
    volume_ma_ratio = df['volume'] / df['volume'].rolling(window=10).mean()
    volume_pressure = volume_ma_ratio.rolling(window=15).apply(lambda x: (x.iloc[-1] > x.quantile(0.8)) * 3 + (x.iloc[-1] > x.quantile(0.6)) * 2 + (x.iloc[-1] > x.quantile(0.4)) * 1)
    
    # Volume divergence - alignment between momentum and trading intensity
    intraday_volume_div = intraday_return * volume_ma_ratio * np.sign(intraday_return * volume_ma_ratio)
    overnight_volume_div = overnight_return * volume_ma_ratio * np.sign(overnight_return * volume_ma_ratio)
    weekly_volume_div = weekly_momentum * volume_ma_ratio * np.sign(weekly_momentum * volume_ma_ratio)
    
    # Multiplicative momentum combinations
    ultra_short_combo = intraday_return * overnight_return * np.sign(intraday_return * overnight_return)
    short_term_combo = overnight_return * weekly_momentum * np.sign(overnight_return * weekly_momentum)
    hierarchical_combo = intraday_return * weekly_momentum * np.sign(intraday_return * weekly_momentum)
    
    # Regime-aware dynamic weights using percentile classifications
    intraday_weight = np.where(vol_percentile == 3, 0.4, 
                              np.where(vol_percentile == 2, 0.3, 0.2))
    overnight_weight = np.where(vol_percentile == 3, 0.2,
                               np.where(vol_percentile == 2, 0.25, 0.3))
    weekly_weight = np.where(vol_percentile == 3, 0.15,
                            np.where(vol_percentile == 2, 0.2, 0.25))
    accel_weight = np.where(vol_percentile == 3, 0.25,
                           np.where(vol_percentile == 2, 0.25, 0.25))
    
    # Volume regime multipliers
    volume_multiplier = np.where(volume_pressure == 3, 1.8,
                                np.where(volume_pressure == 2, 1.4,
                                        np.where(volume_pressure == 1, 1.1, 0.8)))
    
    # Combined alpha factor with hierarchical structure
    momentum_core = (
        intraday_weight * intraday_return +
        overnight_weight * overnight_return +
        weekly_weight * weekly_momentum +
        accel_weight * (intraday_accel + overnight_accel + weekly_accel)
    )
    
    volume_confirmation = (
        intraday_volume_div * 0.3 +
        overnight_volume_div * 0.3 +
        weekly_volume_div * 0.4
    )
    
    multiplicative_enhancement = (
        ultra_short_combo * 0.4 +
        short_term_combo * 0.3 +
        hierarchical_combo * 0.3
    )
    
    alpha_factor = (
        momentum_core * volume_multiplier +
        volume_confirmation * 0.6 +
        multiplicative_enhancement * 0.4
    )
    
    return alpha_factor

import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum convergence with volatility-adaptive scaling and volume-pressure regime alignment.
    
    Interpretation:
    - Dual-timeframe momentum (short-term 3-day vs medium-term 8-day) captures different market rhythms
    - Volatility-adaptive scaling adjusts signal intensity based on market turbulence levels
    - Volume-pressure percentiles identify unusual trading activity that confirms momentum signals
    - Momentum convergence enhances signal reliability when short and medium-term trends align
    - Volume-regime weighting amplifies signals during high-conviction trading periods
    - Positive values indicate bullish momentum convergence with volume confirmation
    - Negative values suggest bearish momentum alignment with distribution pressure
    """
    
    # Multi-timeframe momentum components
    short_term_momentum = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min() + 1e-7)
    medium_term_momentum = (df['close'] - df['close'].shift(8)) / (df['high'].rolling(window=8).max() - df['low'].rolling(window=8).min() + 1e-7)
    
    # Volatility regime detection using rolling percentiles
    daily_range = df['high'] - df['low']
    vol_5d_std = daily_range.rolling(window=5).std()
    vol_20d_median = vol_5d_std.rolling(window=20).median()
    vol_regime_ratio = vol_5d_std / (vol_20d_median + 1e-7)
    
    # Dynamic volatility scaling based on regime intensity
    vol_scale = np.where(vol_regime_ratio > 1.8, 0.4,
                        np.where(vol_regime_ratio > 1.3, 0.7,
                                np.where(vol_regime_ratio < 0.6, 1.6,
                                        np.where(vol_regime_ratio < 0.9, 1.3, 1.0))))
    
    # Volume-pressure regime using rolling percentiles
    volume_5d_mean = df['volume'].rolling(window=5).mean()
    volume_pressure = df['volume'] / (volume_5d_mean + 1e-7)
    volume_20d_percentile = volume_pressure.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 1.5 + (x.iloc[-1] > x.quantile(0.9)) * 1.0, raw=False)
    
    # Momentum convergence factor
    momentum_alignment = np.sign(short_term_momentum * medium_term_momentum)
    momentum_magnitude_diff = abs(short_term_momentum) - abs(medium_term_momentum)
    convergence_strength = momentum_alignment * (abs(short_term_momentum) + abs(medium_term_momentum)) / 2
    
    # Volume-regime weighted momentum
    volume_weighted_short = short_term_momentum * volume_20d_percentile
    volume_weighted_medium = medium_term_momentum * volume_20d_percentile
    
    # Combined alpha factor with volatility adaptation
    alpha_factor = (
        convergence_strength * 0.6 * vol_scale +
        volume_weighted_short * 0.25 * vol_scale +
        volume_weighted_medium * 0.15 * vol_scale +
        momentum_magnitude_diff * 0.1 * np.where(volume_20d_percentile > 2.0, 1.2, 0.8)
    )
    
    return alpha_factor

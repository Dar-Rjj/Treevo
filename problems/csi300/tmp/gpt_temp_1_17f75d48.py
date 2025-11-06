import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum with volatility-normalized regimes and adaptive volume thresholds.
    Uses quantile-based triggers for robust signals without data normalization.
    """
    
    # Multi-timeframe momentum (1-day and 3-day)
    mom_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Volatility regimes using high-low range (3-day window)
    vol_3d = (df['high'] - df['low']).rolling(window=3).mean()
    
    # Volatility-normalized momentum
    vol_norm_mom_1d = mom_1d / (vol_3d + 1e-7)
    vol_norm_mom_3d = mom_3d / (vol_3d + 1e-7)
    
    # Volume momentum with adaptive threshold (5-day median)
    vol_median_5d = df['volume'].rolling(window=5).median()
    vol_mom = (df['volume'] - vol_median_5d) / (vol_median_5d + 1e-7)
    
    # Price position within daily range (volatility context)
    daily_range = df['high'] - df['low']
    price_position = (df['close'] - df['low']) / (daily_range + 1e-7)
    
    # Quantile-based regime detection (20-day rolling quantiles)
    vol_quantile_20d = vol_3d.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    price_quantile_20d = df['close'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Regime classification
    high_vol_regime = vol_quantile_20d > 0.7
    low_vol_regime = vol_quantile_20d < 0.3
    high_price_regime = price_quantile_20d > 0.7
    low_price_regime = price_quantile_20d < 0.3
    
    # Core factor: blended momentum with volume confirmation
    core_factor = (0.6 * vol_norm_mom_1d + 0.4 * vol_norm_mom_3d) * (1 + vol_mom.clip(lower=-0.3, upper=0.8))
    
    # Regime-based adjustments
    factor = core_factor.copy()
    
    # High volatility regime: emphasize short-term momentum
    factor = factor.mask(high_vol_regime, core_factor * 1.3)
    
    # Low volatility regime: emphasize mean reversion
    factor = factor.mask(low_vol_regime, 
                        core_factor * np.where(price_position > 0.6, -0.8, 
                                              np.where(price_position < 0.4, 0.8, 1.0)))
    
    # Extreme price levels: contrarian signals
    factor = factor.mask(high_price_regime & (mom_1d < 0), core_factor * 1.4)
    factor = factor.mask(low_price_regime & (mom_1d > 0), core_factor * 1.4)
    
    # Volume breakout confirmation (adaptive threshold)
    vol_breakout = df['volume'] > (vol_median_5d * 2.0)
    factor = factor.mask(vol_breakout, factor * 1.2)
    
    # Price breakout detection with volume confirmation
    price_range_3d = df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min()
    upside_breakout = (df['close'] > df['high'].shift(1)) & (df['volume'] > vol_median_5d)
    downside_breakout = (df['close'] < df['low'].shift(1)) & (df['volume'] > vol_median_5d)
    
    factor = factor.mask(upside_breakout, factor * 1.5)
    factor = factor.mask(downside_breakout, factor * 1.5)
    
    return factor

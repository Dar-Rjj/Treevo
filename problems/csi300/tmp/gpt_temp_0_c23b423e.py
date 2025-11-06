import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and regime-aware dynamic weighting.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with percentile normalization
    - Volume divergence detection using multiplicative combinations of volume pressure and momentum alignment
    - Triple-regime volatility adaptation with dynamic weight allocation based on market conditions
    - Percentile rank normalization preserves signal distribution while enhancing robustness
    - Multiplicative volume-momentum interaction captures non-linear relationships
    - Positive values indicate strong momentum acceleration with volume confirmation across timeframes
    - Negative values suggest momentum deceleration with volume divergence patterns
    """
    
    # Hierarchical momentum components with percentile normalization
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Percentile rank normalization for momentum components
    intraday_rank = intraday_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x.dropna()) >= 10 else np.nan), raw=False)
    overnight_rank = overnight_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x.dropna()) >= 10 else np.nan), raw=False)
    weekly_rank = weekly_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x.dropna()) >= 10 else np.nan), raw=False)
    
    # Volume divergence detection with multiplicative combinations
    volume_pressure = df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-7)
    volume_momentum_alignment = volume_pressure * np.sign(intraday_momentum + overnight_momentum)
    
    # Volume divergence factor - multiplicative combination of pressure and momentum alignment
    volume_divergence = volume_pressure * volume_momentum_alignment * np.sign(weekly_momentum)
    
    # Triple-regime volatility classification
    daily_range = df['high'] - df['low']
    vol_short = daily_range.rolling(window=5).std()
    vol_medium = daily_range.rolling(window=15).std()
    vol_regime_ratio = vol_short / (vol_medium + 1e-7)
    
    # Dynamic regime classification with percentile thresholds
    vol_regime = np.where(vol_regime_ratio > vol_regime_ratio.rolling(20).quantile(0.8), 'high',
                         np.where(vol_regime_ratio < vol_regime_ratio.rolling(20).quantile(0.2), 'low', 'medium'))
    
    # Regime-aware dynamic weight allocation
    intraday_weight = np.where(vol_regime == 'high', 0.35,
                              np.where(vol_regime == 'low', 0.15, 0.25))
    overnight_weight = np.where(vol_regime == 'high', 0.20,
                               np.where(vol_regime == 'low', 0.25, 0.20))
    weekly_weight = np.where(vol_regime == 'high', 0.25,
                            np.where(vol_regime == 'low', 0.40, 0.35))
    volume_weight = np.where(vol_regime == 'high', 0.20,
                            np.where(vol_regime == 'low', 0.20, 0.20))
    
    # Momentum acceleration hierarchy with multiplicative combinations
    short_term_accel = intraday_rank * overnight_rank * np.sign(intraday_rank + overnight_rank)
    medium_term_accel = weekly_rank * (intraday_rank + overnight_rank) * np.sign(weekly_rank)
    
    # Combined alpha factor with hierarchical structure
    alpha_factor = (
        intraday_weight * intraday_rank +
        overnight_weight * overnight_rank +
        weekly_weight * weekly_rank +
        volume_weight * volume_divergence +
        0.15 * short_term_accel +
        0.10 * medium_term_accel
    )
    
    return alpha_factor

import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum convergence with volume-divergence confirmation and dynamic regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with percentile normalization
    - Volume divergence detection identifies unusual trading activity across different time horizons
    - Dynamic regime classification based on volatility and volume characteristics
    - Multiplicative combination of momentum acceleration and volume-pressure synchronization
    - Adaptive weighting scheme that responds to market regime changes
    - Positive values indicate strong momentum with volume confirmation across multiple timeframes
    - Negative values suggest momentum breakdown with volume distribution patterns
    """
    
    # Hierarchical momentum components with percentile normalization
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Percentile rank normalization for momentum components
    intraday_rank = intraday_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) if len(x.dropna()) >= 10 else np.nan, raw=False
    )
    overnight_rank = overnight_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) if len(x.dropna()) >= 10 else np.nan, raw=False
    )
    weekly_rank = weekly_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) if len(x.dropna()) >= 10 else np.nan, raw=False
    )
    
    # Volume divergence detection across multiple timeframes
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    volume_divergence_short = df['volume'] / (volume_5d_avg + 1e-7)
    volume_divergence_long = df['volume'] / (volume_20d_avg + 1e-7)
    
    # Multiplicative volume-pressure synchronization
    volume_pressure = volume_divergence_short * volume_divergence_long * np.sign(volume_divergence_short * volume_divergence_long)
    
    # Dynamic regime classification using volatility and volume characteristics
    daily_range = df['high'] - df['low']
    vol_5d_std = daily_range.rolling(window=5).std()
    vol_20d_median = vol_5d_std.rolling(window=20).median()
    vol_regime_ratio = vol_5d_std / (vol_20d_median + 1e-7)
    
    volume_regime_ratio = volume_divergence_long.rolling(window=10).mean()
    
    # Triple regime classification
    volatility_regime = np.where(vol_regime_ratio > 1.3, 'high',
                               np.where(vol_regime_ratio < 0.8, 'low', 'medium'))
    volume_regime = np.where(volume_regime_ratio > 1.2, 'high',
                           np.where(volume_regime_ratio < 0.9, 'low', 'medium'))
    
    # Momentum acceleration hierarchy with multiplicative combinations
    ultra_short_accel = intraday_rank * overnight_rank * np.sign(intraday_rank * overnight_rank)
    medium_term_accel = overnight_rank * weekly_rank * np.sign(overnight_rank * weekly_rank)
    hierarchical_accel = ultra_short_accel * medium_term_accel * np.sign(ultra_short_accel * medium_term_accel)
    
    # Volume-confirmed momentum convergence
    volume_confirmed_intraday = volume_pressure * intraday_rank * np.sign(volume_pressure * intraday_rank)
    volume_confirmed_weekly = volume_pressure * weekly_rank * np.sign(volume_pressure * weekly_rank)
    volume_synchronized_momentum = volume_confirmed_intraday * volume_confirmed_weekly * np.sign(volume_confirmed_intraday * volume_confirmed_weekly)
    
    # Dynamic regime-aware weights
    # High volatility regime emphasizes weekly momentum and volume confirmation
    intraday_weight = np.where((volatility_regime == 'high') & (volume_regime == 'high'), 0.15,
                             np.where((volatility_regime == 'low') & (volume_regime == 'low'), 0.35, 0.25))
    
    overnight_weight = np.where((volatility_regime == 'high') & (volume_regime == 'high'), 0.20,
                              np.where((volatility_regime == 'low') & (volume_regime == 'low'), 0.25, 0.30))
    
    weekly_weight = np.where((volatility_regime == 'high') & (volume_regime == 'high'), 0.35,
                           np.where((volatility_regime == 'low') & (volume_regime == 'low'), 0.15, 0.25))
    
    accel_weight = np.where((volatility_regime == 'high') & (volume_regime == 'high'), 0.20,
                          np.where((volatility_regime == 'low') & (volume_regime == 'low'), 0.15, 0.15))
    
    volume_weight = np.where((volatility_regime == 'high') & (volume_regime == 'high'), 0.10,
                           np.where((volatility_regime == 'low') & (volume_regime == 'low'), 0.10, 0.05))
    
    # Combined alpha factor with hierarchical momentum and volume divergence
    alpha_factor = (
        intraday_weight * intraday_rank +
        overnight_weight * overnight_rank +
        weekly_weight * weekly_rank +
        accel_weight * hierarchical_accel +
        volume_weight * volume_synchronized_momentum
    )
    
    return alpha_factor

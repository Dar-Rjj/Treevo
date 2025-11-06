import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and dynamic regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration signals
    - Volume divergence detection identifies momentum-velocity mismatches across timeframes
    - Dynamic regime classification based on volatility and volume pressure percentiles
    - Multiplicative combination of ranked components enhances signal robustness
    - Regime-adaptive weights optimize signal extraction across market environments
    - Positive values indicate strong momentum acceleration with volume confirmation
    - Negative values suggest momentum deceleration with volume divergence
    """
    
    # Hierarchical momentum components with acceleration
    intraday_return = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_return = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Momentum acceleration signals
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    weekly_accel = weekly_momentum - weekly_momentum.shift(3)
    
    # Volume divergence detection using percentile ranks
    volume_rank = df['volume'].rolling(window=20).apply(lambda x: (x[-1] > x[:-1]).mean())
    amount_rank = df['amount'].rolling(window=20).apply(lambda x: (x[-1] > x[:-1]).mean())
    volume_divergence = volume_rank - amount_rank
    
    # Multi-timeframe volume-pressure regimes
    short_volume_pressure = df['volume'] / (df['volume'].rolling(window=5).mean() + 1e-7)
    medium_volume_pressure = df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-7)
    long_volume_pressure = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-7)
    
    # Volatility regime classification using percentile-based approach
    daily_range = df['high'] - df['low']
    vol_short = daily_range.rolling(window=5).std()
    vol_medium = daily_range.rolling(window=10).std()
    vol_regime_ratio = vol_short / (vol_medium + 1e-7)
    
    # Dynamic regime classification with percentile thresholds
    vol_regime = np.where(vol_regime_ratio > vol_regime_ratio.rolling(20).quantile(0.8), 'high',
                         np.where(vol_regime_ratio < vol_regime_ratio.rolling(20).quantile(0.2), 'low', 'medium'))
    
    # Volume regime classification
    volume_regime = np.where(short_volume_pressure > short_volume_pressure.rolling(20).quantile(0.8), 'high',
                           np.where(short_volume_pressure < short_volume_pressure.rolling(20).quantile(0.2), 'low', 'medium'))
    
    # Multiplicative combination of ranked components
    intraday_ranked = intraday_return.rolling(20).apply(lambda x: (x[-1] > x[:-1]).mean())
    overnight_ranked = overnight_return.rolling(20).apply(lambda x: (x[-1] > x[:-1]).mean())
    weekly_ranked = weekly_momentum.rolling(20).apply(lambda x: (x[-1] > x[:-1]).mean())
    
    # Acceleration ranks
    intraday_accel_rank = intraday_accel.rolling(15).apply(lambda x: (x[-1] > x[:-1]).mean())
    weekly_accel_rank = weekly_accel.rolling(15).apply(lambda x: (x[-1] > x[:-1]).mean())
    
    # Regime-adaptive weights using multiplicative scaling
    intraday_weight = np.where(vol_regime == 'high', 0.4 * intraday_ranked,
                              np.where(vol_regime == 'low', 0.2 * intraday_ranked, 0.3 * intraday_ranked))
    
    overnight_weight = np.where(volume_regime == 'high', 0.3 * overnight_ranked,
                               np.where(volume_regime == 'low', 0.1 * overnight_ranked, 0.2 * overnight_ranked))
    
    weekly_weight = np.where(vol_regime == 'high', 0.2 * weekly_ranked,
                            np.where(vol_regime == 'low', 0.4 * weekly_ranked, 0.3 * weekly_ranked))
    
    acceleration_weight = np.where(volume_regime == 'high', 0.3 * intraday_accel_rank,
                                  np.where(volume_regime == 'low', 0.2 * intraday_accel_rank, 0.25 * intraday_accel_rank))
    
    # Hierarchical alpha factor with volume divergence adjustment
    momentum_component = (
        intraday_weight * intraday_return +
        overnight_weight * overnight_return +
        weekly_weight * weekly_momentum +
        acceleration_weight * (intraday_accel + weekly_accel)
    )
    
    # Volume divergence adjustment with regime sensitivity
    divergence_multiplier = np.where(volume_divergence > 0, 1 + volume_divergence * 0.5,
                                    np.where(volume_divergence < 0, 1 + volume_divergence * 0.8, 1.0))
    
    # Final alpha factor with multiplicative enhancement
    alpha_factor = momentum_component * divergence_multiplier
    
    return alpha_factor

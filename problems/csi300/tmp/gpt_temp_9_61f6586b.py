import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence using percentile-based regime adaptation.
    
    Interpretation:
    - Triple-timeframe momentum hierarchy (intraday, overnight, multi-day) with acceleration signals
    - Volume divergence detection using percentile ranks across multiple windows
    - Dynamic regime classification based on volatility and volume percentile thresholds
    - Multiplicative combination of momentum acceleration and volume confirmation
    - Percentile-based weighting adapts to relative market conditions
    - Positive values indicate strong momentum acceleration with volume confirmation
    - Negative values suggest momentum deceleration with volume divergence
    """
    
    # Hierarchical momentum components with acceleration
    intraday_return = (df['close'] - df['open']) / df['open']
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    daily_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Momentum acceleration signals
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    daily_accel = daily_return - daily_return.shift(2)
    
    # Volume divergence using percentile ranks
    volume_5d_rank = df['volume'].rolling(window=5).apply(lambda x: (x[-1] > x[:-1]).mean())
    volume_10d_rank = df['volume'].rolling(window=10).apply(lambda x: (x[-1] > x[:-1]).mean())
    volume_20d_rank = df['volume'].rolling(window=20).apply(lambda x: (x[-1] > x[:-1]).mean())
    
    # Combined volume divergence score
    volume_divergence = (volume_5d_rank * 0.4 + volume_10d_rank * 0.35 + volume_20d_rank * 0.25)
    
    # Volatility regime using percentile-based classification
    daily_range = (df['high'] - df['low']) / df['close']
    vol_5d_percentile = daily_range.rolling(window=20).apply(lambda x: (x[-5:].mean() > x[:-5]).mean())
    vol_regime = np.where(vol_5d_percentile > 0.7, 'high',
                         np.where(vol_5d_percentile < 0.3, 'low', 'medium'))
    
    # Volume-pressure regime using percentile thresholds
    volume_pressure = df['volume'] / df['volume'].rolling(window=10).mean()
    vol_pressure_percentile = volume_pressure.rolling(window=20).apply(lambda x: (x[-1] > x[:-1]).mean())
    volume_regime = np.where(vol_pressure_percentile > 0.8, 'high',
                            np.where(vol_pressure_percentile < 0.2, 'low', 'medium'))
    
    # Regime-adaptive momentum weights using multiplicative combinations
    intraday_weight = np.where(vol_regime == 'high', 0.35,
                              np.where(vol_regime == 'low', 0.15, 0.25))
    overnight_weight = np.where(vol_regime == 'high', 0.25,
                               np.where(vol_regime == 'low', 0.35, 0.30))
    daily_weight = np.where(vol_regime == 'high', 0.20,
                           np.where(vol_regime == 'low', 0.30, 0.25))
    accel_weight = np.where(vol_regime == 'high', 0.20,
                           np.where(vol_regime == 'low', 0.20, 0.20))
    
    # Volume regime multipliers
    volume_multiplier = np.where(volume_regime == 'high', 1.4,
                                np.where(volume_regime == 'low', 0.6, 1.0))
    
    # Hierarchical momentum combination with acceleration emphasis
    momentum_hierarchy = (
        intraday_weight * intraday_accel * np.sign(intraday_return) +
        overnight_weight * overnight_accel * np.sign(overnight_return) +
        daily_weight * daily_accel * np.sign(daily_return) +
        accel_weight * (intraday_accel + overnight_accel) * np.sign(intraday_accel * overnight_accel)
    )
    
    # Multiplicative alpha factor combining momentum acceleration and volume divergence
    alpha_factor = momentum_hierarchy * volume_divergence * volume_multiplier
    
    return alpha_factor

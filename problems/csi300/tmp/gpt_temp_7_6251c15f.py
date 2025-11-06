import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and regime-aware dynamic weighting.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration signals
    - Volume divergence detection across different momentum regimes
    - Dynamic regime classification based on volatility and volume pressure
    - Percentile-based normalization preserves cross-sectional information
    - Multiplicative combination of momentum acceleration and volume confirmation
    - Regime-adaptive weights optimize signal extraction across market conditions
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
    
    # Volume divergence detection
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_pressure = df['volume'] / (volume_5d_avg + 1e-7)
    volume_momentum_divergence = volume_pressure * (intraday_momentum - intraday_momentum.rolling(3).mean())
    
    # Regime classification using percentile ranks
    daily_range = df['high'] - df['low']
    vol_5d_std = daily_range.rolling(window=5).std()
    vol_rank = vol_5d_std.rolling(window=20).apply(lambda x: (x.iloc[-1] > np.percentile(x, 70)) * 2 + (x.iloc[-1] > np.percentile(x, 30)) * 1)
    
    volume_rank = volume_pressure.rolling(window=20).apply(lambda x: (x.iloc[-1] > np.percentile(x, 70)) * 2 + (x.iloc[-1] > np.percentile(x, 30)) * 1)
    
    # Dynamic regime-aware weights
    momentum_weight = np.where(vol_rank == 3, 0.6, 
                              np.where(vol_rank == 2, 0.8, 1.0))
    
    volume_weight = np.where(volume_rank == 3, 1.2,
                            np.where(volume_rank == 2, 1.0, 0.8))
    
    acceleration_weight = np.where((vol_rank + volume_rank) >= 4, 1.4,
                                  np.where((vol_rank + volume_rank) >= 3, 1.1, 0.9))
    
    # Multiplicative combination with hierarchical structure
    base_momentum = (intraday_momentum * momentum_weight + 
                    overnight_momentum * momentum_weight * 0.7 + 
                    weekly_momentum * momentum_weight * 0.4)
    
    acceleration_component = (ultra_short_accel * acceleration_weight + 
                            short_term_accel * acceleration_weight * 0.6)
    
    volume_component = volume_momentum_divergence * volume_weight
    
    # Final alpha factor with regime adaptation
    alpha_factor = (base_momentum * acceleration_component * 
                   np.sign(base_momentum * acceleration_component) + 
                   volume_component * np.sign(base_momentum))
    
    return alpha_factor

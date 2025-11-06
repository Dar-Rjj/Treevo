import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and dynamic regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration signals
    - Volume divergence detection across different time horizons for confirmation
    - Dynamic regime classification based on volatility and volume characteristics
    - Percentile-based normalization preserves cross-sectional ranking information
    - Multiplicative combinations enhance signal strength during regime alignments
    - Positive values indicate strong momentum acceleration with volume confirmation
    - Negative values suggest momentum deceleration with volume divergence
    """
    
    # Hierarchical momentum components
    intraday_mom = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_mom = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_mom = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    intraday_accel = intraday_mom - intraday_mom.shift(1)
    overnight_accel = overnight_mom - overnight_mom.shift(1)
    weekly_accel = weekly_mom - weekly_mom.shift(3)
    
    # Volume divergence components
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    
    short_volume_div = (df['volume'] - volume_5d_avg) / (volume_5d_avg + 1e-7)
    long_volume_div = (df['volume'] - volume_20d_avg) / (volume_20d_avg + 1e-7)
    volume_momentum_div = short_volume_div - long_volume_div
    
    # Dynamic regime classification
    daily_range = df['high'] - df['low']
    vol_5d_std = daily_range.rolling(window=5).std()
    vol_20d_median = vol_5d_std.rolling(window=20).median()
    vol_regime_ratio = vol_5d_std / (vol_20d_median + 1e-7)
    
    # Volume regime classification
    volume_regime_ratio = volume_5d_avg / (volume_20d_avg + 1e-7)
    
    # Multi-regime classification
    volatility_regime = np.where(vol_regime_ratio > 1.3, 'high',
                                np.where(vol_regime_ratio < 0.8, 'low', 'medium'))
    
    volume_regime = np.where(volume_regime_ratio > 1.2, 'high',
                            np.where(volume_regime_ratio < 0.9, 'low', 'medium'))
    
    # Regime-adaptive weights
    intraday_weight = np.where((volatility_regime == 'high') & (volume_regime == 'high'), 0.4,
                              np.where((volatility_regime == 'low') & (volume_regime == 'low'), 0.1, 0.25))
    
    overnight_weight = np.where((volatility_regime == 'high') & (volume_regime == 'high'), 0.2,
                               np.where((volatility_regime == 'low') & (volume_regime == 'low'), 0.3, 0.25))
    
    weekly_weight = np.where((volatility_regime == 'high') & (volume_regime == 'high'), 0.15,
                            np.where((volatility_regime == 'low') & (volume_regime == 'low'), 0.4, 0.25))
    
    volume_div_weight = np.where((volatility_regime == 'high') & (volume_regime == 'high'), 0.25,
                                np.where((volatility_regime == 'low') & (volume_regime == 'low'), 0.2, 0.25))
    
    # Percentile-based momentum ranks
    intraday_rank = intraday_mom.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x.dropna()) >= 10 else np.nan, raw=False)
    
    overnight_rank = overnight_mom.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x.dropna()) >= 10 else np.nan, raw=False)
    
    weekly_rank = weekly_mom.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x.dropna()) >= 10 else np.nan, raw=False)
    
    # Multiplicative combinations for regime alignment
    momentum_alignment = intraday_rank * overnight_rank * weekly_rank
    acceleration_alignment = np.sign(intraday_accel) * np.sign(overnight_accel) * np.sign(weekly_accel)
    
    # Volume-momentum synchronization
    volume_momentum_sync = volume_momentum_div * np.sign(intraday_mom + overnight_mom + weekly_mom)
    
    # Hierarchical alpha factor construction
    base_momentum = (
        intraday_weight * intraday_mom +
        overnight_weight * overnight_mom +
        weekly_weight * weekly_mom
    )
    
    acceleration_component = (
        intraday_weight * intraday_accel +
        overnight_weight * overnight_accel +
        weekly_weight * weekly_accel
    )
    
    volume_component = volume_div_weight * volume_momentum_sync
    
    # Final alpha factor with multiplicative enhancements
    alpha_factor = (
        base_momentum * (1 + 0.5 * momentum_alignment) +
        acceleration_component * (1 + 0.3 * acceleration_alignment) +
        volume_component * (1 + 0.2 * np.sign(base_momentum * volume_component))
    )
    
    return alpha_factor

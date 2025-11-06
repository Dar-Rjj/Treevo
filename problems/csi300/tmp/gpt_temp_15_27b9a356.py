import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum hierarchy with volume divergence confirmation and percentile-based regime adaptation.
    
    Interpretation:
    - Triple-horizon momentum (intraday, overnight, multi-day) with acceleration persistence
    - Volume divergence detection using rolling percentile ranks to identify abnormal trading activity
    - Multiplicative momentum combinations enhance signal strength through directional alignment
    - Volatility regime classification via range percentile ranks for adaptive market conditioning
    - Volume-momentum synchronization captures confirmation/divergence patterns
    - Positive values indicate accelerating momentum with volume confirmation across timeframes
    - Negative values suggest momentum deceleration with volume divergence or distribution
    """
    
    # Hierarchical momentum components with normalized returns
    intraday_return = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_return = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    multiday_return = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(3).max() - df['low'].rolling(3).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    multiday_accel = multiday_return - multiday_return.shift(2)
    
    # Multiplicative momentum combinations for directional alignment
    ultra_short_momentum = intraday_return * overnight_return * np.sign(intraday_return * overnight_return)
    short_term_momentum = overnight_return * multiday_return * np.sign(overnight_return * multiday_return)
    hierarchical_momentum = ultra_short_momentum * short_term_momentum * np.sign(ultra_short_momentum * short_term_momentum)
    
    # Volume divergence using rolling percentile ranks
    volume_ratio = df['volume'] / (df['volume'].rolling(window=5).mean() + 1e-7)
    volume_pct_rank = volume_ratio.rolling(window=20).apply(lambda x: (x.iloc[-1] > x).mean())
    
    # Volatility regime detection via range percentile ranks
    daily_range = df['high'] - df['low']
    range_pct_rank = daily_range.rolling(window=20).apply(lambda x: (x.iloc[-1] > x).mean())
    
    # Percentile-based regime classification
    vol_regime = np.where(range_pct_rank > 0.75, 'high',
                         np.where(range_pct_rank < 0.25, 'low', 'medium'))
    
    # Volume-momentum synchronization components
    volume_momentum_sync = volume_pct_rank * hierarchical_momentum * np.sign(volume_pct_rank * hierarchical_momentum)
    volume_accel_sync = volume_pct_rank * (intraday_accel + overnight_accel) * np.sign(volume_pct_rank * (intraday_accel + overnight_accel))
    
    # Regime-adaptive weights using percentile-informed scaling
    intraday_weight = np.where(vol_regime == 'high', 0.30,
                              np.where(vol_regime == 'low', 0.12, 0.22))
    overnight_weight = np.where(vol_regime == 'high', 0.22,
                               np.where(vol_regime == 'low', 0.18, 0.20))
    multiday_weight = np.where(vol_regime == 'high', 0.18,
                              np.where(vol_regime == 'low', 0.32, 0.24))
    hierarchical_weight = np.where(vol_regime == 'high', 0.30,
                                  np.where(vol_regime == 'low', 0.38, 0.34))
    
    # Combined alpha factor with multiplicative volume integration
    alpha_factor = (
        intraday_weight * intraday_return * volume_pct_rank +
        overnight_weight * overnight_return * volume_pct_rank +
        multiday_weight * multiday_return * (1 + volume_pct_rank) +
        hierarchical_weight * hierarchical_momentum * volume_momentum_sync +
        0.15 * volume_accel_sync * np.sign(hierarchical_momentum)
    )
    
    return alpha_factor

import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and dynamic regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, daily) with acceleration signals
    - Volume divergence detection across different momentum regimes
    - Dynamic regime classification based on volatility and volume characteristics
    - Percentile rank normalization for cross-sectional comparability
    - Multiplicative combination of momentum and volume components
    - Positive values indicate strong momentum acceleration with volume confirmation
    - Negative values suggest momentum reversal with volume divergence
    """
    
    # Hierarchical momentum components
    intraday_return = (df['close'] - df['open']) / df['open']
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    daily_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Momentum acceleration hierarchy
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    daily_accel = daily_return - daily_return.shift(1)
    
    # Volume divergence detection
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    volume_divergence = (volume_ma_5 / volume_ma_20) - 1
    
    # Dynamic regime classification
    daily_range = (df['high'] - df['low']) / df['close']
    vol_regime = daily_range.rolling(window=10).mean()
    volume_regime = df['volume'].rolling(window=10).mean()
    
    # Regime-based momentum weights
    high_vol_weight = 0.6 * (vol_regime > vol_regime.rolling(window=20).quantile(0.7))
    low_vol_weight = 0.8 * (vol_regime < vol_regime.rolling(window=20).quantile(0.3))
    base_vol_weight = 0.7
    
    # Volume regime adaptation
    high_volume_weight = 1.2 * (volume_regime > volume_regime.rolling(window=20).quantile(0.7))
    low_volume_weight = 0.8 * (volume_regime < volume_regime.rolling(window=20).quantile(0.3))
    base_volume_weight = 1.0
    
    # Combined regime weights
    vol_weight = high_vol_weight + low_vol_weight + base_vol_weight
    volume_weight = high_volume_weight + low_volume_weight + base_volume_weight
    
    # Percentile rank normalization
    def percentile_rank(series):
        return series.rolling(window=20).apply(lambda x: (x[-1] > x[:-1]).mean())
    
    intraday_rank = percentile_rank(intraday_accel)
    overnight_rank = percentile_rank(overnight_accel)
    daily_rank = percentile_rank(daily_accel)
    volume_rank = percentile_rank(volume_divergence)
    
    # Multiplicative combination of components
    momentum_component = (
        intraday_rank * intraday_accel * 0.4 +
        overnight_rank * overnight_accel * 0.3 +
        daily_rank * daily_accel * 0.3
    )
    
    volume_component = volume_rank * volume_divergence * np.sign(momentum_component)
    
    # Hierarchical alpha factor
    alpha_factor = (
        momentum_component * vol_weight +
        volume_component * volume_weight * 0.5
    )
    
    return alpha_factor

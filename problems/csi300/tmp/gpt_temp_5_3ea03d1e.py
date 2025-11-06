import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and dynamic regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration signals
    - Volume divergence detection across different momentum regimes
    - Dynamic regime classification based on volatility and volume characteristics
    - Percentile-based normalization preserves cross-sectional ranking properties
    - Multiplicative combination of momentum and volume components enhances signal strength
    - Positive values indicate strong momentum acceleration with volume confirmation
    - Negative values suggest momentum deceleration with volume divergence
    """
    
    # Hierarchical momentum components
    intraday_return = (df['close'] - df['open']) / (df['open'] + 1e-7)
    overnight_return = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-7)
    
    # Momentum acceleration signals
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    weekly_accel = weekly_momentum - weekly_momentum.shift(3)
    
    # Volume divergence detection
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_divergence = (df['volume'] - volume_5d_avg) / (volume_5d_avg + 1e-7)
    
    # Dynamic regime classification
    daily_range = (df['high'] - df['low']) / df['open']
    volatility_regime = daily_range.rolling(window=10).mean()
    volume_regime = df['volume'].rolling(window=10).mean() / (df['volume'].rolling(window=50).mean() + 1e-7)
    
    # Regime weights based on volatility and volume characteristics
    high_vol_weight = (volatility_regime > volatility_regime.rolling(window=20).quantile(0.7)).astype(float)
    low_vol_weight = (volatility_regime < volatility_regime.rolling(window=20).quantile(0.3)).astype(float)
    high_volume_weight = (volume_regime > 1.2).astype(float)
    
    # Percentile-based momentum ranking
    def rolling_percentile(series, window):
        return series.rolling(window=window).apply(lambda x: (x[-1] - x.mean()) / (x.std() + 1e-7))
    
    intraday_rank = rolling_percentile(intraday_return, 20)
    overnight_rank = rolling_percentile(overnight_return, 20)
    weekly_rank = rolling_percentile(weekly_momentum, 20)
    
    # Multiplicative combination of momentum and volume components
    momentum_volume_sync = (
        intraday_return * volume_divergence * np.sign(intraday_return) +
        overnight_return * volume_divergence * np.sign(overnight_return) +
        weekly_momentum * volume_divergence * np.sign(weekly_momentum)
    )
    
    # Acceleration-volume divergence
    accel_volume_div = (
        intraday_accel * volume_divergence * np.sign(intraday_accel) +
        overnight_accel * volume_divergence * np.sign(overnight_accel) +
        weekly_accel * volume_divergence * np.sign(weekly_accel)
    )
    
    # Hierarchical timeframe combination with regime adaptation
    short_term_component = (
        intraday_rank * (1 - high_vol_weight) * (1 + high_volume_weight) +
        overnight_rank * (1 + low_vol_weight) * (1 + high_volume_weight)
    )
    
    medium_term_component = (
        weekly_rank * (1 + high_vol_weight) * (1 - low_vol_weight) +
        momentum_volume_sync * (1 + volume_regime)
    )
    
    acceleration_component = (
        accel_volume_div * (1 + high_vol_weight) * (1 + high_volume_weight) +
        (intraday_accel + overnight_accel) * volume_divergence * (1 - low_vol_weight)
    )
    
    # Final alpha factor with hierarchical weighting
    alpha_factor = (
        short_term_component * 0.4 +
        medium_term_component * 0.35 +
        acceleration_component * 0.25
    )
    
    return alpha_factor

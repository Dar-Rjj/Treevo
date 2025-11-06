import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multiplicative momentum acceleration with percentile regime detection and volume-pressure synchronization.
    
    Interpretation:
    - Multi-timeframe momentum acceleration (3-day vs 8-day, 5-day vs 13-day, 8-day vs 21-day)
    - Percentile-based regime classification using volatility and volume characteristics
    - Multiplicative combinations enhance signal-to-noise ratio and robustness
    - Volume-pressure synchronization quantifies institutional participation intensity
    - Acceleration divergence detects momentum regime shifts across timeframes
    - Positive values indicate strong momentum acceleration with volume confirmation
    - Negative values suggest momentum deterioration with distribution pressure
    - Cross-sectional percentile ranks preserve relative positioning information
    """
    
    # Momentum acceleration hierarchy with multiplicative combinations
    intraday_return = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_return = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Multiplicative momentum acceleration signals
    ultra_short_accel = intraday_return.rolling(3).mean() * intraday_return.rolling(8).mean()
    short_term_accel = overnight_return.rolling(5).mean() * overnight_return.rolling(13).mean()
    medium_term_accel = weekly_momentum.rolling(8).mean() * weekly_momentum.rolling(21).mean()
    
    # Acceleration divergence across timeframes
    accel_divergence = (ultra_short_accel * short_term_accel * medium_term_accel) ** (1/3)
    
    # Volume-pressure synchronization with multiplicative enhancement
    volume_pressure = (df['volume'] / (df['volume'].rolling(5).mean() + 1e-7)) * (df['amount'] / (df['amount'].rolling(5).mean() + 1e-7))
    volume_momentum_sync = volume_pressure * np.sign(intraday_return + overnight_return)
    
    # Percentile-based regime classification
    daily_range = df['high'] - df['low']
    range_percentile = daily_range.rolling(20).apply(lambda x: (x.rank(pct=True).iloc[-1]))
    volume_percentile = df['volume'].rolling(20).apply(lambda x: (x.rank(pct=True).iloc[-1]))
    
    # Regime detection using percentile thresholds
    high_vol_regime = (range_percentile > 0.7).astype(float)
    low_vol_regime = (range_percentile < 0.3).astype(float)
    high_volume_regime = (volume_percentile > 0.7).astype(float)
    low_volume_regime = (volume_percentile < 0.3).astype(float)
    
    # Multiplicative regime-aware momentum combinations
    intraday_regime_momentum = intraday_return * (1 + 0.5 * high_vol_regime - 0.3 * low_vol_regime)
    overnight_regime_momentum = overnight_return * (1 + 0.3 * high_volume_regime - 0.2 * low_volume_regime)
    
    # Acceleration-volume multiplicative synchronization
    accel_volume_sync = accel_divergence * volume_momentum_sync * np.sign(accel_divergence * volume_momentum_sync)
    
    # Regime-adaptive weighting using percentile information
    momentum_weight = 0.4 - 0.1 * high_vol_regime + 0.05 * low_vol_regime
    acceleration_weight = 0.3 + 0.1 * high_volume_regime - 0.05 * low_volume_regime
    volume_sync_weight = 0.2 + 0.15 * high_vol_regime + 0.1 * high_volume_regime
    regime_momentum_weight = 0.1 + 0.05 * low_vol_regime - 0.1 * high_vol_regime
    
    # Multiplicative alpha factor with percentile regime adaptation
    alpha_factor = (
        momentum_weight * (intraday_regime_momentum * overnight_regime_momentum) +
        acceleration_weight * accel_divergence +
        volume_sync_weight * accel_volume_sync +
        regime_momentum_weight * (intraday_regime_momentum + overnight_regime_momentum)
    )
    
    return alpha_factor

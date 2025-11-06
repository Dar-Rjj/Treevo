import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence using percentile-based regime weights.
    
    Interpretation:
    - Triple-timeframe momentum hierarchy (intraday, short-term, medium-term) with acceleration signals
    - Volume divergence detection across multiple time horizons (1-day, 3-day, 5-day)
    - Percentile-based regime classification for adaptive signal weighting
    - Multiplicative combination of momentum acceleration and volume divergence
    - Smooth regime transitions using percentile boundaries
    - Positive values indicate bullish momentum with confirming volume patterns
    - Negative values suggest bearish pressure with diverging volume characteristics
    """
    
    # Multi-timeframe momentum components
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    short_term_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    medium_term_momentum = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    
    # Momentum acceleration signals
    intraday_accel = intraday_momentum - intraday_momentum.shift(1)
    short_term_accel = short_term_momentum - short_term_momentum.shift(2)
    medium_term_accel = medium_term_momentum - medium_term_momentum.shift(3)
    
    # Volume divergence components
    volume_1d = df['volume']
    volume_3d_avg = df['volume'].rolling(window=3).mean()
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    
    volume_divergence_short = (volume_1d - volume_3d_avg) / (volume_3d_avg + 1e-7)
    volume_divergence_medium = (volume_1d - volume_5d_avg) / (volume_5d_avg + 1e-7)
    volume_momentum = volume_1d / volume_1d.shift(1) - 1
    
    # Percentile-based regime classification
    momentum_strength = (intraday_momentum.abs() + short_term_momentum.abs() + medium_term_momentum.abs()) / 3
    momentum_percentile = momentum_strength.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    volume_strength = (volume_divergence_short.abs() + volume_divergence_medium.abs()) / 2
    volume_percentile = volume_strength.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Regime classification using percentile boundaries
    momentum_regime = np.where(momentum_percentile > 0.7, 'high',
                              np.where(momentum_percentile < 0.3, 'low', 'medium'))
    
    volume_regime = np.where(volume_percentile > 0.7, 'high',
                            np.where(volume_percentile < 0.3, 'low', 'medium'))
    
    # Adaptive regime weights with smooth transitions
    momentum_weights = np.where(momentum_regime == 'high', 0.5,
                               np.where(momentum_regime == 'low', 0.2, 0.3))
    
    volume_weights = np.where(volume_regime == 'high', 0.4,
                             np.where(volume_regime == 'low', 0.1, 0.25))
    
    acceleration_weights = np.where(momentum_regime == 'high', 0.3,
                                   np.where(momentum_regime == 'low', 0.1, 0.2))
    
    # Multiplicative combination of momentum acceleration and volume divergence
    momentum_acceleration = (
        momentum_weights * intraday_accel +
        acceleration_weights * short_term_accel +
        acceleration_weights * medium_term_accel
    )
    
    volume_divergence_signal = (
        volume_weights * volume_divergence_short +
        volume_weights * 0.7 * volume_divergence_medium +
        volume_weights * 0.5 * volume_momentum
    )
    
    # Final alpha factor with multiplicative combination
    alpha_factor = (
        momentum_acceleration * 
        (1 + volume_divergence_signal) * 
        np.sign(momentum_acceleration + volume_divergence_signal)
    )
    
    return alpha_factor

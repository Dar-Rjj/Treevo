import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence using percentile-based regime weights.
    
    Interpretation:
    - Triple-timeframe momentum acceleration (ultra-short, short, medium) with hierarchical confirmation
    - Volume divergence detection across multiple time horizons (1-day, 3-day, 5-day)
    - Percentile-based regime classification for adaptive signal weighting
    - Multiplicative combinations enhance signal robustness while maintaining interpretability
    - Smooth transitions between regimes using percentile thresholds
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest deteriorating momentum with volume divergence
    """
    
    # Multi-timeframe momentum components
    ultra_short_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    short_momentum = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    medium_momentum = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(3).max() - df['low'].rolling(3).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    acceleration_1 = ultra_short_momentum * np.sign(ultra_short_momentum * short_momentum)
    acceleration_2 = short_momentum * np.sign(short_momentum * medium_momentum)
    combined_acceleration = acceleration_1 + acceleration_2 * np.sign(acceleration_1 * acceleration_2)
    
    # Volume divergence detection
    volume_1d = df['volume'] / (df['volume'].shift(1) + 1e-7)
    volume_3d = df['volume'] / (df['volume'].rolling(3).mean() + 1e-7)
    volume_5d = df['volume'] / (df['volume'].rolling(5).mean() + 1e-7)
    
    volume_divergence = (
        volume_1d * np.sign(ultra_short_momentum) +
        volume_3d * np.sign(short_momentum) +
        volume_5d * np.sign(medium_momentum)
    ) * np.sign(volume_1d * volume_3d * volume_5d)
    
    # Percentile-based regime classification
    momentum_strength = ultra_short_momentum.abs() + short_momentum.abs() + medium_momentum.abs()
    momentum_percentile = momentum_strength.rolling(window=20).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70)) * 2 + (x.iloc[-1] > np.percentile(x, 30)) * 1
    )
    
    volume_strength = volume_1d.abs() + volume_3d.abs() + volume_5d.abs()
    volume_percentile = volume_strength.rolling(window=20).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70)) * 2 + (x.iloc[-1] > np.percentile(x, 30)) * 1
    )
    
    # Regime-based adaptive weights
    momentum_regime_weights = np.where(momentum_percentile == 3, 0.4,
                                     np.where(momentum_percentile == 2, 0.3,
                                            np.where(momentum_percentile == 1, 0.2, 0.1)))
    
    volume_regime_weights = np.where(volume_percentile == 3, 0.35,
                                   np.where(volume_percentile == 2, 0.25,
                                          np.where(volume_percentile == 1, 0.15, 0.1)))
    
    # Multiplicative factor combination
    momentum_component = (
        momentum_regime_weights * 
        (ultra_short_momentum + short_momentum + medium_momentum + combined_acceleration)
    )
    
    volume_component = (
        volume_regime_weights * 
        volume_divergence * 
        np.sign(momentum_component)
    )
    
    # Final alpha factor with smooth regime transitions
    alpha_factor = (
        momentum_component * 
        (1 + 0.3 * np.tanh(volume_component)) * 
        np.sign(momentum_component * volume_component)
    )
    
    return alpha_factor

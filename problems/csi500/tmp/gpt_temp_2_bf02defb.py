import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a regime-adaptive alpha factor that combines volatility regimes, 
    momentum signals, volume analysis, and price position analysis.
    """
    # Calculate returns for volatility
    returns = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility Regime Detection
    short_term_vol = returns.rolling(window=5).std()
    long_term_vol = returns.rolling(window=20).std()
    
    # Regime Classification
    high_vol_regime = short_term_vol > (1.5 * long_term_vol)
    low_vol_regime = short_term_vol < (0.67 * long_term_vol)
    normal_vol_regime = ~high_vol_regime & ~low_vol_regime
    
    # Regime-Adaptive Momentum
    short_momentum = np.log(df['close'] / df['close'].shift(5))
    medium_momentum = np.log(df['close'] / df['close'].shift(20))
    long_momentum = np.log(df['close'] / df['close'].shift(60))
    
    # Volume Regime Detection
    volume_baseline = df['volume'].rolling(window=20).median()
    volume_mad = df['volume'].rolling(window=20).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    # Volume Threshold Assignment
    extreme_volume = df['volume'] > (volume_baseline + 3 * volume_mad)
    high_volume = (df['volume'] > (volume_baseline + 2 * volume_mad)) & ~extreme_volume
    moderate_volume = (df['volume'] > (volume_baseline + volume_mad)) & ~high_volume & ~extreme_volume
    
    volume_threshold = np.ones_like(df['volume'])
    volume_threshold = np.where(extreme_volume, 2.0, volume_threshold)
    volume_threshold = np.where(high_volume, 1.5, volume_threshold)
    volume_threshold = np.where(moderate_volume, 1.2, volume_threshold)
    
    # Price Position Analysis
    high_20d = df['high'].rolling(window=20).max()
    low_20d = df['low'].rolling(window=20).min()
    
    near_resistance = df['close'] > (0.9 * high_20d)
    near_support = df['close'] < (1.1 * low_20d)
    
    position_multiplier = np.ones_like(df['close'])
    position_multiplier = np.where(near_resistance, 0.7, position_multiplier)
    position_multiplier = np.where(near_support, 1.3, position_multiplier)
    
    # Composite Alpha Construction
    # Regime-Weighted Momentum
    regime_weighted_momentum = np.zeros_like(df['close'])
    
    # High Volatility regime
    regime_weighted_momentum = np.where(
        high_vol_regime, 
        0.6 * short_momentum + 0.4 * medium_momentum, 
        regime_weighted_momentum
    )
    
    # Normal Volatility regime
    regime_weighted_momentum = np.where(
        normal_vol_regime,
        0.3 * short_momentum + 0.4 * medium_momentum + 0.3 * long_momentum,
        regime_weighted_momentum
    )
    
    # Low Volatility regime
    regime_weighted_momentum = np.where(
        low_vol_regime,
        0.2 * short_momentum + 0.3 * medium_momentum + 0.5 * long_momentum,
        regime_weighted_momentum
    )
    
    # Volume Enhanced
    volume_enhanced = regime_weighted_momentum * volume_threshold
    
    # Final Alpha
    final_alpha = volume_enhanced * position_multiplier
    
    return pd.Series(final_alpha, index=df.index)

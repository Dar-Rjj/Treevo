import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Alpha factor that adjusts signals based on market volatility regimes.
    Combines price pattern strength, volume-volatility dynamics, and regime-adaptive signal construction.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    # Parameters
    vol_lookback = 20
    regime_lookback = 60
    support_resistance_window = 10
    volume_ma_window = 5
    
    # Calculate daily price range (volatility proxy)
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Volatility regime identification
    rolling_vol = daily_range.rolling(window=vol_lookback).std()
    
    # Historical volatility percentiles for regime classification
    vol_percentiles = rolling_vol.rolling(window=regime_lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == regime_lookback else np.nan, 
        raw=False
    )
    
    # Regime classification: low (<0.3), medium (0.3-0.7), high (>0.7)
    volatility_regime = pd.Series(index=df.index, dtype=int)
    volatility_regime[vol_percentiles < 0.3] = 0  # Low volatility
    volatility_regime[(vol_percentiles >= 0.3) & (vol_percentiles <= 0.7)] = 1  # Medium volatility
    volatility_regime[vol_percentiles > 0.7] = 2  # High volatility
    
    # Price pattern strength - Support/resistance detection
    support_levels = df['low'].rolling(window=support_resistance_window, center=True).min()
    resistance_levels = df['high'].rolling(window=support_resistance_window, center=True).max()
    
    # Distance to support/resistance as pattern strength indicator
    distance_to_support = (df['close'] - support_levels) / (resistance_levels - support_levels)
    pattern_strength = 1 - 2 * abs(distance_to_support - 0.5)  # Strongest at extremes
    
    # Volume-confirmed breakout patterns
    volume_ma = df['volume'].rolling(window=volume_ma_window).mean()
    volume_ratio = df['volume'] / volume_ma
    
    # Breakout detection: price near resistance with high volume
    near_resistance = (df['close'] > resistance_levels * 0.95) & (df['close'] < resistance_levels * 1.05)
    breakout_signal = near_resistance & (volume_ratio > 1.5)
    
    # Volume-volatility dynamics
    volume_vol_ratio = df['volume'] / (daily_range + 1e-8)
    volume_vol_zscore = (volume_vol_ratio - volume_vol_ratio.rolling(window=vol_lookback).mean()) / \
                       (volume_vol_ratio.rolling(window=vol_lookback).std() + 1e-8)
    
    # Abnormal volume detection per regime
    regime_volume_threshold = volatility_regime.map({0: 1.2, 1: 1.5, 2: 2.0})
    abnormal_volume = volume_ratio > regime_volume_threshold
    
    # Regime-adaptive signal construction
    # Momentum signal for high volatility regime
    momentum_signal = df['close'] / df['close'].rolling(window=5).mean() - 1
    
    # Mean reversion signal for low volatility regime
    mean_reversion_signal = (df['close'].rolling(window=10).mean() - df['close']) / \
                           (df['close'].rolling(window=10).std() + 1e-8)
    
    # Dynamic weighting by regime persistence
    regime_persistence = volatility_regime.rolling(window=5).apply(
        lambda x: len(set(x)) == 1 if len(x) == 5 else False, raw=False
    ).astype(float)
    
    # Combine signals based on volatility regime
    adaptive_signal = pd.Series(index=df.index, dtype=float)
    
    for regime in [0, 1, 2]:
        regime_mask = volatility_regime == regime
        
        if regime == 0:  # Low volatility - mean reversion
            adaptive_signal[regime_mask] = mean_reversion_signal[regime_mask]
        elif regime == 1:  # Medium volatility - balanced approach
            adaptive_signal[regime_mask] = 0.5 * momentum_signal[regime_mask] + \
                                          0.5 * mean_reversion_signal[regime_mask]
        else:  # High volatility - momentum
            adaptive_signal[regime_mask] = momentum_signal[regime_mask]
    
    # Final factor construction with regime persistence weighting
    base_factor = adaptive_signal * pattern_strength
    
    # Enhance with volume confirmation
    volume_enhanced = base_factor * np.where(abnormal_volume, 1.5, 1.0)
    volume_enhanced = volume_enhanced * np.where(breakout_signal, 2.0, 1.0)
    
    # Apply regime persistence weighting
    final_factor = volume_enhanced * (0.5 + 0.5 * regime_persistence)
    
    # Normalize the factor
    result = (final_factor - final_factor.rolling(window=vol_lookback).mean()) / \
            (final_factor.rolling(window=vol_lookback).std() + 1e-8)
    
    return result

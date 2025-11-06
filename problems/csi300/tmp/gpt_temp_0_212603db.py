import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Momentum acceleration with volume divergence across multiple timeframes using percentile-based regime weights.
    
    Interpretation:
    - Momentum acceleration hierarchy (intraday, short-term, medium-term) with smooth regime transitions
    - Volume divergence detection across different momentum timeframes for signal confirmation
    - Percentile-based regime classification for robust weight assignment
    - Volume-momentum synchronization prioritization for enhanced signal reliability
    - Regime persistence mechanisms to maintain stable factor behavior
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest bearish momentum acceleration with volume distribution
    """
    
    # Momentum acceleration components across timeframes
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    short_term_momentum = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(3).max() - df['low'].rolling(3).min() + 1e-7)
    medium_term_momentum = (df['close'] - df['close'].shift(8)) / (df['high'].rolling(8).max() - df['low'].rolling(8).min() + 1e-7)
    
    # Momentum acceleration signals
    intraday_accel = intraday_momentum - intraday_momentum.shift(2)
    short_term_accel = short_term_momentum - short_term_momentum.shift(3)
    medium_term_accel = medium_term_momentum - medium_term_momentum.shift(5)
    
    # Volume divergence detection
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    
    intraday_volume_div = (df['volume'] / (volume_5d_avg + 1e-7)) * np.sign(intraday_momentum)
    short_term_volume_div = (volume_5d_avg / (volume_20d_avg + 1e-7)) * np.sign(short_term_momentum)
    medium_term_volume_div = (volume_20d_avg / (df['volume'].rolling(window=60).mean() + 1e-7)) * np.sign(medium_term_momentum)
    
    # Percentile-based regime classification
    vol_5d = (df['high'] - df['low']).rolling(window=5).std()
    vol_percentile = vol_5d.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1)
    
    volume_percentile = df['volume'].rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1)
    
    # Smooth regime transitions using exponential weighting
    regime_persistence = 0.85
    vol_regime = vol_percentile.ewm(alpha=1-regime_persistence).mean()
    volume_regime = volume_percentile.ewm(alpha=1-regime_persistence).mean()
    
    # Percentile-based regime weights
    high_vol_weight = np.where(vol_regime > 1.5, 0.6, np.where(vol_regime > 0.8, 0.3, 0.1))
    low_vol_weight = np.where(vol_regime < 0.5, 0.6, np.where(vol_regime < 1.2, 0.3, 0.1))
    medium_vol_weight = 1 - high_vol_weight - low_vol_weight
    
    high_volume_weight = np.where(volume_regime > 1.5, 0.7, np.where(volume_regime > 0.8, 0.4, 0.1))
    normal_volume_weight = 1 - high_volume_weight
    
    # Volume-momentum synchronization factors
    intraday_sync = intraday_momentum * intraday_volume_div * np.sign(intraday_momentum * intraday_volume_div)
    short_term_sync = short_term_momentum * short_term_volume_div * np.sign(short_term_momentum * short_term_volume_div)
    medium_term_sync = medium_term_momentum * medium_term_volume_div * np.sign(medium_term_momentum * medium_term_volume_div)
    
    # Regime-adaptive momentum weights with volume synchronization
    intraday_component = (
        high_vol_weight * intraday_accel * 0.4 +
        medium_vol_weight * intraday_momentum * 0.3 +
        low_vol_weight * intraday_sync * 0.3
    ) * high_volume_weight
    
    short_term_component = (
        high_vol_weight * short_term_momentum * 0.3 +
        medium_vol_weight * short_term_accel * 0.4 +
        low_vol_weight * short_term_sync * 0.3
    ) * normal_volume_weight
    
    medium_term_component = (
        high_vol_weight * medium_term_sync * 0.3 +
        medium_vol_weight * medium_term_momentum * 0.3 +
        low_vol_weight * medium_term_accel * 0.4
    ) * normal_volume_weight
    
    # Combined alpha factor with regime persistence
    alpha_factor = (
        intraday_component * 0.35 +
        short_term_component * 0.40 +
        medium_term_component * 0.25
    )
    
    # Final smoothing for regime persistence
    alpha_factor = alpha_factor.ewm(alpha=0.1).mean()
    
    return alpha_factor

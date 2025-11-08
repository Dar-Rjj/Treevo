import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate regime-adaptive momentum quality factor based on volatility regimes
    """
    df = data.copy()
    
    # Calculate ATR for different periods
    def calculate_atr(period):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    atr_3 = calculate_atr(3)
    atr_5 = calculate_atr(5)
    atr_10 = calculate_atr(10)
    
    # Volatility regime classification
    short_term_vol = atr_3 / df['close'].shift(1)
    medium_term_vol = atr_10 / df['close'].shift(1)
    regime_indicator = atr_3 / atr_10
    
    # Initialize factor components
    high_vol_component = pd.Series(index=df.index, dtype=float)
    normal_vol_component = pd.Series(index=df.index, dtype=float)
    low_vol_component = pd.Series(index=df.index, dtype=float)
    
    # High volatility regime (ratio > 1.2)
    high_vol_mask = regime_indicator > 1.2
    if high_vol_mask.any():
        mean_reversion = (df['close'] - (df['high'] + df['low']) / 2) / atr_3
        volume_confirmation = df['volume'] / df['volume'].shift(1)
        high_vol_component[high_vol_mask] = mean_reversion[high_vol_mask] * volume_confirmation[high_vol_mask]
    
    # Normal volatility regime (0.8 ≤ ratio ≤ 1.2)
    normal_vol_mask = (regime_indicator >= 0.8) & (regime_indicator <= 1.2)
    if normal_vol_mask.any():
        trend_momentum = (df['close'] - df['close'].shift(3)) / atr_5
        volume_ma = df['volume'].rolling(window=5).mean()
        volume_trend = df['volume'] / volume_ma
        normal_vol_component[normal_vol_mask] = trend_momentum[normal_vol_mask] * volume_trend[normal_vol_mask]
    
    # Low volatility regime (ratio < 0.8)
    low_vol_mask = regime_indicator < 0.8
    if low_vol_mask.any():
        breakout_potential = (df['high'] - df['low']) / atr_10
        volume_compression = df['volume'] / df['volume'].shift(1)
        low_vol_component[low_vol_mask] = breakout_potential[low_vol_mask] * volume_compression[low_vol_mask]
    
    # Regime persistence weighting
    def calculate_regime_persistence(regime_mask, window=5):
        persistence = pd.Series(index=df.index, dtype=float)
        for i in range(len(regime_mask)):
            if i >= window:
                persistence.iloc[i] = regime_mask.iloc[i-window:i+1].mean()
        return persistence
    
    high_vol_persistence = calculate_regime_persistence(high_vol_mask)
    normal_vol_persistence = calculate_regime_persistence(normal_vol_mask)
    low_vol_persistence = calculate_regime_persistence(low_vol_mask)
    
    # Final regime-adaptive momentum factor
    factor = (
        high_vol_component * high_vol_persistence +
        normal_vol_component * normal_vol_persistence +
        low_vol_component * low_vol_persistence
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std()
    
    return factor

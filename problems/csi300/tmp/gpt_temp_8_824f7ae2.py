import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Extract price and volume data
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate momentum acceleration components
    # Short-term momentum acceleration
    ret_1d = close.pct_change(1)
    ret_3d = close.pct_change(3)
    short_term_accel = ret_1d - ret_3d
    
    # Medium-term momentum persistence
    ret_5d = close.pct_change(5)
    ret_10d = close.pct_change(10)
    medium_term_persist = ret_5d - ret_10d
    
    # Multi-timeframe acceleration signal
    combined_accel = 0.6 * short_term_accel + 0.4 * medium_term_persist
    
    # Volume persistence confirmation
    # Volume trend strength
    vol_3d_avg = volume.rolling(window=3).mean()
    vol_10d_avg = volume.rolling(window=10).mean()
    vol_trend_ratio = vol_3d_avg / vol_10d_avg
    
    # Volume direction consistency
    vol_increase_count = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window_vol = volume.iloc[i-5:i]
        vol_increase_count.iloc[i] = (window_vol > window_vol.shift(1)).sum()
    vol_persistence_score = vol_increase_count / 5
    vol_momentum = volume / volume.shift(1)
    
    # Volume-momentum integration
    vol_weighted_accel = combined_accel * vol_trend_ratio
    persistence_multiplier = 1 + vol_persistence_score
    enhanced_signal = vol_weighted_accel * persistence_multiplier
    
    # Adaptive regime scaling
    # Volatility regime detection
    true_range = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    atr_5d = true_range.rolling(window=5).mean()
    atr_20d = true_range.rolling(window=20).mean()
    vol_regime = atr_5d / atr_20d
    
    # Volume regime detection
    vol_20d_median = volume.rolling(window=20).median()
    vol_regime_ratio = volume / vol_20d_median
    
    # Dynamic scaling factors
    volatility_scaling = pd.Series(1.0, index=data.index)
    volatility_scaling[vol_regime > 1.2] = 0.7
    
    volume_scaling = pd.Series(1.0, index=data.index)
    volume_scaling[vol_regime_ratio < 0.8] = 0.8
    
    combined_scaling = volatility_scaling * volume_scaling
    
    # Final alpha construction
    raw_alpha = enhanced_signal
    regime_adjusted_alpha = raw_alpha * combined_scaling
    
    alpha = regime_adjusted_alpha
    
    return alpha

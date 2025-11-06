import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Volatility-Normalized Momentum
    # Calculate 5-day momentum
    momentum_5d = (df['close'] / df['close'].shift(5)) - 1
    
    # Calculate 5-day ATR
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr_5d = true_range.rolling(window=5).mean()
    
    # Normalize momentum
    normalized_momentum = momentum_5d / atr_5d
    
    # Volume Divergence
    # Calculate 5-day volume trend using linear regression
    def calc_volume_slope(volume_window):
        if len(volume_window) < 5 or volume_window.isna().any():
            return np.nan
        x = np.arange(len(volume_window))
        slope, _, _, _, _ = linregress(x, volume_window)
        return slope
    
    volume_slope = df['volume'].rolling(window=5).apply(calc_volume_slope, raw=False)
    
    # Determine momentum direction and volume trend direction
    momentum_sign = np.sign(momentum_5d)
    volume_sign = np.sign(volume_slope)
    
    # Flag divergence (1 if signs are opposite, 0 otherwise)
    divergence_flag = (momentum_sign != volume_sign).astype(int)
    
    # Regime Detection
    # Calculate 20-day ATR
    atr_20d = true_range.rolling(window=20).mean()
    
    # Calculate 60-day ATR median
    atr_60d_median = true_range.rolling(window=60).median()
    
    # High volatility regime: 20-day ATR > 60-day ATR median
    high_vol_regime = (atr_20d > atr_60d_median)
    
    # Signal Integration
    # Base signal: volatility-normalized momentum
    base_signal = normalized_momentum
    
    # Volume-adjusted signal: base_signal × (1 + divergence_flag)
    volume_adjusted_signal = base_signal * (1 + divergence_flag)
    
    # Regime weighting
    # High volatility: weight = -1 (mean reversion emphasis)
    # Low volatility: weight = 1 (momentum emphasis)
    regime_weight = np.where(high_vol_regime, -1, 1)
    
    # Final alpha: regime_weight × volume_adjusted_signal
    final_alpha = regime_weight * volume_adjusted_signal
    
    return final_alpha

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Volatility-Normalized Momentum Calculation
    # Compute short-term momentum (close_t / close_{t-5} - 1)
    momentum = df['close'] / df['close'].shift(5) - 1
    
    # Calculate daily volatility proxy (high_t - low_t)
    volatility_proxy = df['high'] - df['low']
    
    # Normalize momentum by volatility (momentum / volatility_proxy)
    volatility_normalized_momentum = momentum / volatility_proxy
    
    # Volume Divergence Detection
    # Calculate volume trend slope (5-day linear regression on volume)
    def calc_volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(5)
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    volume_trend = df['volume'].rolling(window=5).apply(calc_volume_slope, raw=False)
    
    # Compare volume trend direction with price momentum sign
    volume_trend_sign = np.sign(volume_trend)
    momentum_sign = np.sign(momentum)
    
    # Identify divergence when volume_trend_sign ≠ momentum_sign
    divergence_detected = volume_trend_sign != momentum_sign
    
    # Signal Strength Assessment
    # Strong signal: volatility-normalized momentum with volume confirmation
    # Weak signal: volatility-normalized momentum alone
    signal_strength = volatility_normalized_momentum.copy()
    signal_strength[divergence_detected] = signal_strength[divergence_detected] * 0.5  # Reduce strength when divergence
    
    # Regime-Based Adjustment
    # Calculate volatility regime using 20-day ATR
    # ATR = average(high-low over 20 days)
    atr = (df['high'] - df['low']).rolling(window=20).mean()
    atr_median = atr.median()
    
    # High volatility regime (ATR > median): Reduce signal strength (multiply by 0.7)
    # Low volatility regime (ATR ≤ median): Maintain signal strength (multiply by 1.0)
    regime_adjustment = np.where(atr > atr_median, 0.7, 1.0)
    
    # Final factor calculation
    factor = signal_strength * regime_adjustment
    
    return factor

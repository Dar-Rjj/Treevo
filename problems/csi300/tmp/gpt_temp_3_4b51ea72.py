import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Volatility-Normalized Momentum
    # Calculate 5-day price momentum
    momentum = (df['close'] / df['close'].shift(5)) - 1
    
    # Compute 5-day average true range
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    atr_5 = true_range.rolling(window=5).mean()
    
    # Normalize momentum by volatility
    normalized_momentum = momentum / atr_5
    
    # Volume Confirmation Signal
    # Calculate 5-day volume trend using linear regression slope
    def volume_slope(volume_series):
        if len(volume_series) < 5 or volume_series.isna().any():
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = linregress(x, volume_series)
        return slope
    
    volume_trend = df['volume'].rolling(window=5).apply(volume_slope, raw=False)
    
    # Determine direction alignment and generate confirmation strength
    volume_confirmation = volume_trend * normalized_momentum
    
    # Regime Detection
    # Compute 20-day volatility regime
    returns = df['close'].pct_change()
    vol_20 = returns.rolling(window=20).std()
    
    # Calculate historical volatility percentiles
    vol_history = returns.rolling(window=252).std()  # 1-year lookback
    high_vol_threshold = vol_history.rolling(window=252).quantile(0.8)
    low_vol_threshold = vol_history.rolling(window=252).quantile(0.2)
    
    # Adaptive Signal Combination
    # Initialize regime weights
    regime_weight = pd.Series(1.0, index=df.index)
    
    # High volatility regime: emphasize mean reversion (negative weight)
    high_vol_mask = vol_20 > high_vol_threshold
    regime_weight[high_vol_mask] = -1.0
    
    # Low volatility regime: emphasize momentum (positive weight)
    low_vol_mask = vol_20 < low_vol_threshold
    regime_weight[low_vol_mask] = 1.5  # Stronger positive weight for momentum
    
    # Final alpha: regime_weight * (normalized_momentum + volume_confirmation)
    alpha = regime_weight * (normalized_momentum + volume_confirmation)
    
    return alpha

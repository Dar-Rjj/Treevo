import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Volatility-Normalized Momentum
    # Calculate 5-day price momentum
    momentum = (df['close'] / df['close'].shift(5)) - 1
    
    # Compute 5-day average daily range
    daily_range = df['high'] - df['low']
    avg_range = daily_range.rolling(window=5).mean()
    
    # Normalize momentum
    normalized_momentum = momentum / avg_range
    
    # Volume Confirmation Signal
    # Calculate 5-day volume slope using linear regression
    def calc_volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(5)
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    volume_slope = df['volume'].rolling(window=5).apply(calc_volume_slope, raw=True)
    
    # Determine direction alignment and generate confirmation factor
    volume_confirmation = np.where(
        np.sign(momentum) == np.sign(volume_slope), 
        1.0, 
        0.5
    )
    
    # Regime Detection
    # Compute 20-day ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = true_range.rolling(window=20).mean()
    
    # Classify regime and set multiplier
    atr_median = atr_20.rolling(window=252).median()  # Use 1-year rolling median
    regime_multiplier = np.where(atr_20 > atr_median, 1.2, 0.8)
    
    # Final Alpha Factor
    base_factor = normalized_momentum * volume_confirmation
    final_factor = base_factor * regime_multiplier
    
    return final_factor

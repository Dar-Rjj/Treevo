import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Volatility-Normalized Momentum
    # Calculate 5-day price momentum
    momentum_5d = (df['close'] / df['close'].shift(5)) - 1
    
    # Compute 5-day average daily range
    daily_range = df['high'] - df['low']
    avg_daily_range_5d = daily_range.rolling(window=5).mean()
    
    # Normalize momentum
    normalized_momentum = momentum_5d / avg_daily_range_5d
    
    # Volume Confirmation Signal
    # Calculate 5-day volume slope using linear regression
    volume_slope = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        if i >= 5:
            window_volume = df['volume'].iloc[i-4:i+1].values
            if not np.isnan(window_volume).any():
                slope, _, _, _, _ = stats.linregress(range(5), window_volume)
                volume_slope.iloc[i] = slope
    
    # Generate confirmation flag
    volume_confirmation = np.where(
        (normalized_momentum > 0) & (volume_slope > 0), 1,
        np.where((normalized_momentum < 0) & (volume_slope < 0), 1,
                np.where((normalized_momentum > 0) & (volume_slope < 0), -1,
                        np.where((normalized_momentum < 0) & (volume_slope > 0), -1, 0)))
    )
    
    # Regime Detection
    # Compute 20-day average true range (ATR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = true_range.rolling(window=20).mean()
    
    # Classify regime and assign weights
    atr_median = atr_20.rolling(window=20).median()
    regime_weights = np.where(atr_20 > atr_median, 0.7, 1.3)
    
    # Final Alpha Factor
    base_signal = normalized_momentum * volume_confirmation
    weighted_signal = base_signal * regime_weights
    
    return pd.Series(weighted_signal, index=df.index, name='alpha_factor')

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Volatility-Normalized Momentum Calculation
    # Compute 5-day momentum
    momentum = df['close'] / df['close'].shift(5) - 1
    
    # Calculate 5-day volatility (sum of daily ranges)
    volatility = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            vol_sum = 0
            for j in range(5):
                vol_sum += df['high'].iloc[i-j] - df['low'].iloc[i-j]
            volatility.iloc[i] = vol_sum
    
    # Normalize momentum
    normalized_momentum = momentum / volatility
    
    # Volume Divergence Detection
    volume_slope = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            x = np.arange(5)
            y = df['volume'].iloc[i-4:i+1].values
            slope, _, _, _, _ = stats.linregress(x, y)
            volume_slope.iloc[i] = slope
    
    momentum_sign = np.sign(normalized_momentum)
    volume_slope_sign = np.sign(volume_slope)
    divergence_exists = momentum_sign != volume_slope_sign
    
    # Signal Strength Assignment
    signal = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if pd.notna(normalized_momentum.iloc[i]):
            if divergence_exists.iloc[i]:
                signal.iloc[i] = normalized_momentum.iloc[i] * 2
            else:
                signal.iloc[i] = normalized_momentum.iloc[i]
    
    # Regime-Based Adjustment
    # Calculate 20-day ATR
    atr = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 19:
            atr_sum = 0
            for j in range(20):
                idx = i - j
                high_low = df['high'].iloc[idx] - df['low'].iloc[idx]
                high_close = abs(df['high'].iloc[idx] - df['close'].iloc[idx-1]) if idx > 0 else 0
                low_close = abs(df['low'].iloc[idx] - df['close'].iloc[idx-1]) if idx > 0 else 0
                atr_sum += max(high_low, high_close, low_close)
            atr.iloc[i] = atr_sum / 20
    
    # Apply regime-based adjustment
    atr_median = atr.median()
    final_signal = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if pd.notna(signal.iloc[i]) and pd.notna(atr.iloc[i]):
            if atr.iloc[i] > atr_median:
                final_signal.iloc[i] = signal.iloc[i] * -1
            else:
                final_signal.iloc[i] = signal.iloc[i]
    
    return final_signal

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Component
    close = df['close']
    price_momentum = (close / close.shift(10)) - 1
    
    # Momentum Persistence
    momentum_sign = np.sign(price_momentum)
    momentum_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        current_sign = momentum_sign.iloc[i]
        count = 0
        for j in range(1, 6):
            if momentum_sign.iloc[i-j] == current_sign:
                count += 1
            else:
                break
        momentum_persistence.iloc[i] = count
    
    # Volatility-Range Component
    daily_range_ratio = (df['high'] - df['low']) / df['close']
    range_volatility = daily_range_ratio.rolling(window=10).std()
    
    # Volume Component
    volume = df['volume']
    volume_momentum = (volume / volume.shift(5)) - 1
    
    # Volume-Range Correlation
    volume_range_corr = pd.Series(index=df.index, dtype=float)
    for i in range(10, len(df)):
        vol_window = volume.iloc[i-9:i+1]
        range_window = daily_range_ratio.iloc[i-9:i+1]
        if len(vol_window) >= 2 and len(range_window) >= 2:
            volume_range_corr.iloc[i] = vol_window.corr(range_window)
        else:
            volume_range_corr.iloc[i] = 0
    
    # Divergence Detection
    # Price-Volume Divergence
    price_volume_divergence = np.sign(price_momentum) * np.sign(volume_momentum)
    
    # Range-Volume Divergence
    historical_corr_mean = volume_range_corr.rolling(window=20, min_periods=1).mean().shift(1)
    range_volume_divergence = volume_range_corr - historical_corr_mean
    
    # Final Alpha Construction
    base_signal = price_momentum * momentum_persistence
    volatility_adjusted_base = base_signal / range_volatility.replace(0, np.nan)
    volume_multiplier = 1 + abs(price_volume_divergence)
    range_multiplier = 1 + abs(range_volume_divergence)
    
    final_factor = volatility_adjusted_base * volume_multiplier * range_multiplier
    
    return final_factor

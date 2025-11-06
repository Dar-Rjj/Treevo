import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum Component
    price_momentum = df['close'] / df['close'].shift(5) - 1
    
    # Calculate Volume Momentum Component
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    
    # Apply exponential decay with 5-day half-life over 20-day window
    decay_factor = 0.5 ** (1/5)  # 5-day half-life
    
    # Create decay weights for 20-day window
    decay_weights = np.array([decay_factor ** i for i in range(20)])[::-1]
    decay_weights = decay_weights / decay_weights.sum()
    
    # Apply decay to price momentum
    decayed_price_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 19:
            window_data = price_momentum.iloc[i-19:i+1]
            decayed_price_momentum.iloc[i] = (window_data * decay_weights).sum()
        else:
            decayed_price_momentum.iloc[i] = np.nan
    
    # Apply decay to volume momentum
    decayed_volume_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 19:
            window_data = volume_momentum.iloc[i-19:i+1]
            decayed_volume_momentum.iloc[i] = (window_data * decay_weights).sum()
        else:
            decayed_volume_momentum.iloc[i] = np.nan
    
    # Calculate True Range Volatility
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate 10-day Average True Range
    atr_10 = true_range.rolling(window=10, min_periods=10).mean()
    
    # Calculate divergence magnitude
    divergence = decayed_price_momentum - decayed_volume_momentum
    divergence_magnitude = abs(divergence)
    
    # Apply volatility weighting
    volatility_weighted_divergence = divergence / atr_10
    
    # Calculate volume confirmation
    volume_20d_avg = df['volume'].rolling(window=20, min_periods=20).mean()
    volume_ratio = df['volume'] / volume_20d_avg
    
    # Generate final alpha signal with contrarian logic
    # Inverse relationship: negative divergence suggests potential reversal
    alpha_signal = -volatility_weighted_divergence * volume_ratio
    
    return alpha_signal

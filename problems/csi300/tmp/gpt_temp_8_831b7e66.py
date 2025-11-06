import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Component
    # Calculate Price Rate of Change (ROC)
    roc_5 = df['close'] / df['close'].shift(5) - 1
    roc_3_lag = df['close'].shift(3) / df['close'].shift(8) - 1
    
    # Calculate Price Acceleration
    price_acceleration = roc_5 - roc_3_lag
    
    # Volume Divergence Component
    # Calculate Volume Stability
    volume_std = df['volume'].rolling(window=10, min_periods=5).std()
    volume_avg = df['volume'].rolling(window=10, min_periods=5).mean()
    volume_stability = volume_std / volume_avg
    
    # Calculate Volume-Price Correlation
    volume_price_corr = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        if i >= 4:
            volume_window = df['volume'].iloc[i-4:i+1]
            close_window = df['close'].iloc[i-4:i+1]
            if len(volume_window) >= 3 and not (volume_window.std() == 0 or close_window.std() == 0):
                volume_price_corr.iloc[i] = volume_window.corr(close_window)
            else:
                volume_price_corr.iloc[i] = 0
    
    # Combine components to create final factor
    # Price momentum positive with volume divergence (low correlation) indicates strong signal
    factor = price_acceleration * (1 - abs(volume_price_corr)) * (1 / (volume_stability + 1e-6))
    
    return factor

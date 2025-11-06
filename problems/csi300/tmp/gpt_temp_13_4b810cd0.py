import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Decay Acceleration Factor with Volume Confirmation
    """
    # Calculate Short-Term Momentum
    close = df['close']
    short_momentum_5 = close.pct_change(periods=5)
    short_momentum_10 = close.pct_change(periods=10)
    short_momentum = (short_momentum_5 + short_momentum_10) / 2
    
    # Calculate Medium-Term Momentum
    medium_momentum_20 = close.pct_change(periods=20)
    medium_momentum_30 = close.pct_change(periods=30)
    medium_momentum = (medium_momentum_20 + medium_momentum_30) / 2
    
    # Compute Momentum Acceleration with Exponential Decay
    momentum_diff = short_momentum - medium_momentum
    decay_weights = np.exp(-np.arange(len(momentum_diff)) / 30)  # 30-day decay
    momentum_acceleration = momentum_diff.rolling(window=30, min_periods=1).apply(
        lambda x: np.average(x.dropna(), weights=decay_weights[:len(x.dropna())]), 
        raw=False
    )
    
    # Volume Confirmation
    volume = df['volume']
    volume_trend = volume.rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else np.nan,
        raw=False
    )
    volume_normalized = volume_trend / volume.rolling(window=10).mean()
    
    # Final Factor
    factor = momentum_acceleration * volume_normalized
    
    return factor.fillna(0)

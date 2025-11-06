import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum
    close = df['close']
    ret_5d = (close / close.shift(5)) - 1
    ret_10d = (close / close.shift(10)) - 1
    
    # Calculate Momentum Acceleration
    momentum_acceleration = (ret_5d - ret_10d) / (np.abs(ret_5d) + 1e-8)
    
    # Volume Trend Confirmation
    volume = df['volume']
    volume_ma_5d = volume.rolling(window=5).mean()
    volume_trend = np.sign(volume / (volume_ma_5d + 1e-8))
    
    # Combine Signals
    combined_signal = momentum_acceleration * volume_trend
    
    # Apply rolling rank
    factor = combined_signal.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    return factor

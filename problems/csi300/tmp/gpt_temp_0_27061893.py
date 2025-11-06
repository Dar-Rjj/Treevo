import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Short-term momentum acceleration (3-day ROC of 5-day ROC)
    momentum_5d = close.pct_change(5)
    momentum_accel = momentum_5d.pct_change(3)
    
    # Long-term volatility-adjusted momentum (20-day momentum scaled by 20-day volatility)
    vol_20d = close.pct_change().rolling(20).std()
    momentum_20d = close.pct_change(20)
    vol_adjusted_momentum = momentum_20d / (vol_20d + 1e-8)
    
    # Price range efficiency (recent closing position within daily range)
    range_position = (close - low) / (high - low + 1e-8)
    range_efficiency = range_position.rolling(10).mean()
    
    # Volume confirmation (volume trend relative to price movement)
    volume_trend = volume.rolling(10).mean() / volume.rolling(30).mean()
    volume_confirmation = np.sign(momentum_5d) * volume_trend
    
    # Composite factor combining acceleration, momentum, and volume signals
    heuristics_matrix = (momentum_accel * 0.4 + 
                        vol_adjusted_momentum * 0.3 + 
                        range_efficiency * 0.2 + 
                        volume_confirmation * 0.1)
    
    # Non-linear transformation to emphasize extreme signals
    heuristics_matrix = np.tanh(heuristics_matrix * 2)
    
    return heuristics_matrix

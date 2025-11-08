import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Trend Acceleration with Volume Confirmation factor
    Combines price trend acceleration with volume trend alignment
    """
    # Price Acceleration component
    # Short-term trend (5-day close slope)
    close_5d_slope = df['close'].rolling(window=5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0] if len(x) == 5 else np.nan, 
        raw=True
    )
    
    # Medium-term trend (20-day close slope)
    close_20d_slope = df['close'].rolling(window=20).apply(
        lambda x: np.polyfit(range(20), x, 1)[0] if len(x) == 20 else np.nan, 
        raw=True
    )
    
    # Price Acceleration = Short-Term / Medium-Term - 1
    price_acceleration = (close_5d_slope / close_20d_slope) - 1
    
    # Volume Alignment component
    # Volume trend (5-day volume slope)
    volume_5d_slope = df['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0] if len(x) == 5 else np.nan, 
        raw=True
    )
    
    # Direction Match = sign(Price Acceleration) × sign(Volume Trend)
    direction_match = np.sign(price_acceleration) * np.sign(volume_5d_slope)
    
    # Volume normalization: current volume / 20-day average volume
    volume_20d_mean = df['volume'].rolling(window=20).mean()
    volume_normalized = df['volume'] / volume_20d_mean
    
    # Final factor: Price Acceleration × Direction Match × Volume Normalization
    factor = price_acceleration * direction_match * volume_normalized
    
    return factor

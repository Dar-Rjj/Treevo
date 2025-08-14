import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Calculate Volume Change
    volume_change = df['volume'].diff()
    
    # Adjust High-Low Spread with Volume
    avg_volume = df['volume'].rolling(window=5).mean()
    adjusted_high_low_spread = high_low_spread * (volume_change / avg_volume)
    
    # Calculate Price Change
    price_change = df['close'].diff()
    
    # Combine High-Low Spread, Volume, and Price
    combined_factor = np.where((price_change > 0) & (volume_change > 0),
                               adjusted_high_low_spread * (price_change + volume_change),
                               adjusted_high_low_spread)
    
    # Apply Exponential Moving Average (EMA)
    ema_period = 5
    ema_factor = combined_factor.ewm(span=ema_period, adjust=False).mean()
    
    return ema_factor

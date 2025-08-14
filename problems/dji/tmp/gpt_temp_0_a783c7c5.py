import pandas as pd
import pandas as pd

def heuristics_v2(df, n_days=20):
    # Calculate Intraday Price Range
    intraday_price_range = df['high'] - df['low']
    
    # Calculate Intraday Price Movement
    intraday_price_movement = df['close'] - df['open']
    
    # Calculate N-day Volume Average
    volume_rolling_mean = df['volume'].rolling(window=n_days).mean()
    
    # Calculate Volume Surge
    def volume_surge(volume, rolling_mean):
        if volume > rolling_mean:
            return 1
        else:
            return -1
    
    volume_surge_series = (df['volume'] / volume_rolling_mean).apply(lambda x: volume_surge(x, volume_rolling_mean))
    
    # Combine Intraday Price Movement and Volume Surge
    alpha_factor = intraday_price_movement * volume_surge_series
    
    return alpha_factor

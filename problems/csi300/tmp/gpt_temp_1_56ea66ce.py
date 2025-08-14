import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Calculate Daily Volume Trend
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_trend = df['volume'] - volume_ma_10
    volume_trend_sign = np.where(volume_trend > 0, 1, -1)
    
    # Calculate Price Trend
    close_ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    price_trend = df['close'] - close_ema_10
    price_trend_sign = np.where(price_trend > 0, 1, -1)
    
    # Combine Spread, Volume, and Price Trends
    adjusted_high_low_spread = high_low_spread * (volume_trend_sign * 1.5 if volume_trend_sign > 0 else volume_trend_sign * 0.5)
    final_alpha_factor = adjusted_high_low_spread * (price_trend_sign * 1.2 if price_trend_sign > 0 else price_trend_sign * 0.8)
    
    return final_alpha_factor

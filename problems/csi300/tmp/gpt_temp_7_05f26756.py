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
    volume_trend_sign = np.sign(volume_trend)
    
    # Calculate Daily Volatility
    volatility = df['close'].rolling(window=10).std()
    volatility_threshold = volatility.mean()
    
    # Combine Spread, Volume Trend, and Volatility
    adjusted_spread = high_low_spread * volume_trend
    adjusted_spread = adjusted_spread * (1.5 if volume_trend_sign > 0 else 0.5)
    adjusted_spread = adjusted_spread * (1.2 if volatility > volatility_threshold else 0.8)
    
    # Incorporate Additional Price Levels
    open_close_spread = df['open'] - df['close']
    combined_spread = (high_low_spread + open_close_spread) / 2
    combined_spread_adjusted = combined_spread * (1.5 if volume_trend_sign > 0 else 0.5)
    
    # Incorporate Price Momentum
    close_ma_10 = df['close'].rolling(window=10).mean()
    price_momentum = df['close'] - close_ma_10
    price_momentum_sign = np.sign(price_momentum)
    
    # Final Adjustment
    final_factor = combined_spread_adjusted * (1.5 if volume_trend_sign > 0 else 0.5)
    final_factor = final_factor * (1.2 if price_momentum_sign > 0 else 0.8)
    
    return final_factor

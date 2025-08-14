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
    
    # Calculate Price Trend
    close_ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    price_trend = df['close'] - close_ema_5
    
    # Calculate Momentum Factor
    momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Adjust for positive and negative trends in Volume
    volume_trend_factor = 1.5 if volume_trend > 0 else 0.5
    
    # Incorporate Price Trend
    price_trend_factor = 1.2 if price_trend > 0 else 0.8
    
    # Incorporate Momentum
    momentum_factor = 1.3 if momentum > 0 else 0.7
    
    # Combine all factors
    alpha_factor = (high_low_spread * volume_trend_factor * 
                    price_trend_factor * momentum_factor * 
                    volume_trend.fillna(0) * price_trend.fillna(0) * 
                    momentum.fillna(0))
    
    return alpha_factor

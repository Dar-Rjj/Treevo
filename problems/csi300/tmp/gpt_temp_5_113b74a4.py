import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Calculate Daily Volume Trend
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_trend = df['volume'] - volume_ma_10
    volume_trend_sign = (volume_trend > 0).astype(int)  # 1 for positive, 0 for negative
    
    # Calculate Price Momentum
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    price_momentum = df['close'] - ema_12
    price_momentum_sign = (price_momentum > 0).astype(int)  # 1 for positive, 0 for negative
    
    # Combine High-Low Spread, Volume Trend, and Price Momentum
    combined_factor = high_low_spread * volume_trend
    
    # Adjust for Volume Trend
    volume_adjustment = 1.5 if volume_trend_sign == 1 else 0.5
    combined_factor *= volume_adjustment
    
    # Adjust for Price Momentum
    if price_momentum_sign == 1:
        combined_factor += combined_factor * 0.10
    else:
        combined_factor -= combined_factor * 0.10
    
    # Final Adjustment
    if volume_trend_sign == 1 and price_momentum_sign == 1:
        combined_factor += combined_factor * 0.05
    elif volume_trend_sign == 0 and price_momentum_sign == 0:
        combined_factor -= combined_factor * 0.05
    
    return combined_factor

import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Momentum
    high_low_momentum = df['High'] - df['Low']
    high_low_momentum_smoothed = high_low_momentum.rolling(window=5).mean()

    # Volume Surge Indicator
    volume_change = df['Volume'].pct_change()
    volume_surge = (volume_change > 0.10).astype(int)
    
    # Conditionally Apply Volume Surge
    high_low_momentum_with_volume_surge = high_low_momentum_smoothed * volume_surge
    
    # Calculate Smoothed Price Momentum
    price_momentum = df['Close'] - df['Close'].shift(1)
    smoothed_price_momentum = price_momentum.rolling(window=5).mean()
    
    # Combine Factors
    combined_alpha_factor = high_low_momentum_with_volume_surge * smoothed_price_momentum
    
    return combined_alpha_factor

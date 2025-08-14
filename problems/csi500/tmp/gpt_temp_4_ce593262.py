import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Price Spread
    high_low_spread = df['high'] - df['low']
    
    # Compute 20-Day Moving Average of High-Low Spread
    ma_20_high_low = high_low_spread.rolling(window=20).mean()
    
    # Calculate Daily Difference of High-Low Spread
    daily_diff = high_low_spread.diff()
    
    # Cumulative Momentum Over 20 Days
    cumulative_momentum = daily_diff.rolling(window=20).sum()
    
    # Compute 20-Day Average Volume
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    
    # Compute 20-Day Average Amount
    avg_amount_20 = df['amount'].rolling(window=20).mean()
    
    # Filter by Volume Above Average
    volume_filter = df['volume'] > avg_volume_20
    
    # Filter by Amount Above Average
    amount_filter = df['amount'] > avg_amount_20
    
    # Calculate 10-Day Exponential Moving Average (EMA) of High-Low Spread
    ema_10_high_low = high_low_spread.ewm(span=10, adjust=False).mean()
    
    # Adjust Final Factor Value Based on EMA
    adjusted_momentum = cumulative_momentum * (1.5 if ema_10_high_low > ma_20_high_low else 1)
    
    # Final Factor Value
    final_factor = adjusted_momentum.where(volume_filter & amount_filter, 0)
    
    return final_factor

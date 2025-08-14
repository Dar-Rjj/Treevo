import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily High-Low Spread
    daily_high_low_spread = df['high'] - df['low']
    
    # Calculate 10-Day Volume-Weighted Average of High-Low Spread
    ten_day_volume_weighted_avg = (daily_high_low_spread.rolling(window=10) * df['volume']).sum() / df['volume'].rolling(window=10).sum()
    
    # Calculate 50-Day Volume-Weighted Average of High-Low Spread
    fifty_day_volume_weighted_avg = (daily_high_low_spread.rolling(window=50) * df['volume']).sum() / df['volume'].rolling(window=50).sum()
    
    # Calculate Difference Between Averages
    difference_between_averages = ten_day_volume_weighted_avg - fifty_day_volume_weighted_avg
    
    # Calculate Momentum Score
    momentum_score = difference_between_averages * df['volume']
    
    return momentum_score

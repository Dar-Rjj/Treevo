import pandas as pd
import pandas as pd

def heuristics_v2(df, n=14):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate True Ranges
    true_ranges = pd.Series(index=df.index)
    for i in range(1, len(df)):
        true_ranges.iloc[i] = max(
            df['high'].iloc[i] - df['close'].iloc[i-1],
            abs(df['close'].iloc[i-1] - df['low'].iloc[i])
        )
    true_ranges.iloc[0] = 0  # First day has no previous close
    
    # Calculate Average True Range (n-day)
    average_true_range = true_ranges.rolling(window=n).mean()
    
    # Calculate Volume-Weighted Price
    volume_weighted_price = (
        (df['open'] * df['volume']) + 
        (df['high'] * df['volume']) + 
        (df['low'] * df['volume']) + 
        (df['close'] * df['volume'])
    ) / (4 * df['volume'])
    
    # Normalize Volume-Weighted Price by Average True Range
    normalized_volume_weighted_price = volume_weighted_price / average_true_range
    
    # Incorporate High-Low Range
    alpha_factor = high_low_range * normalized_volume_weighted_price
    
    return alpha_factor

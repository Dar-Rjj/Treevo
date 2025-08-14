import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Confirm with Volume
    volume_change = df['volume'] - df['volume'].shift(1)
    
    # Combine High-Low Spread with Volume Change if Volume Change > 0
    combined_high_low = high_low_spread * (volume_change > 0).astype(float) * volume_change
    
    # Incorporate Price Trend
    price_change = df['close'] - df['close'].shift(1)
    
    # Integrate Combined High-Low Spread, Volume Change, and Price Change if Price Change > 0
    integrated_factor = combined_high_low * (price_change > 0).astype(float) * price_change
    
    # Apply Weighted Moving Average (WMA) with weights [1, 2, 3, 4, 5]
    wma_weights = [1, 2, 3, 4, 5]
    wma_sum = integrated_factor.rolling(window=len(wma_weights), min_periods=1).apply(lambda x: (x * wma_weights).sum() / sum(wma_weights), raw=True)
    
    return wma_sum

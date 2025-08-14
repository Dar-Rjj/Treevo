importance of each based on their relative strength.}

```python
import pandas as pd

def heuristics_v2(df):
    # Calculate 30-day moving average of close price
    df['ma_close'] = df['close'].rolling(window=30).mean()
    
    # Calculate the ratio of the current close price to its 30-day moving average
    df['price_ratio'] = df['close'] / df['ma_close']
    
    # Calculate the 30-day change in volume
    df['volume_change_30d'] = df['volume'].pct_change(periods=30).fillna(0)
    
    # Determine the relative strength of price and volume
    df['relative_strength_price'] = (df['price_ratio'] - 1).abs()
    df['relative_strength_volume'] = df['volume_change_30d'].abs()
    
    # Calculate the total relative strength
    df['total_relative_strength'] = df['relative_strength_price'] + df['relative_strength_volume']
    
    # Assign weights based on the relative strength
    df['price_weight'] = df['relative_strength_price'] / df['total_relative_strength']
    df['volume_weight'] = df['relative_strength_volume'] / df['total_relative_strength']
    
    # Calculate the weighted sum of the price ratio and volume change
    df['composite_factor'] = (df['price_ratio'] * df['price_weight']) + (df['volume_change_30d'] * df['volume_weight'])
    
    # Create the heuristics matrix
    heuristics_matrix = df['composite_factor']
    
    return heuristics_matrix

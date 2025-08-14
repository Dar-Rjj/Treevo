import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Average Price
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Calculate Volume Change
    df['volume_change'] = df['volume'] / df['volume'].shift(1).fillna(1)
    
    # Calculate Momentum using the Volume-Weighted Average Price
    momentum_window = 30  # Example window, can be adjusted
    df['momentum'] = df['vwap'] / df['vwap'].shift(momentum_window) - 1
    
    # Adjust Momentum by Price Range
    df['adjusted_momentum'] = df['momentum'] / df['price_range']
    
    # Adjust Momentum by Volume Change
    df['final_factor'] = df['adjusted_momentu'] * df['volume_change']
    
    return df['final_factor']

import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Compute Volume-Adjusted Momentum
    df['momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['volume_adjusted_momentum'] = df['momentum'] * df['volume']
    
    # Integrate High-Low Spread and Volume-Adjusted Momentum
    df['10_day_avg_price_range'] = df['price_range'].rolling(window=10).mean()
    df['combined_factor'] = (df['volume_adjusted_momentum'] * df['price_range']) / df['10_day_avg_price_range']
    
    # Apply Directional Bias
    df['directional_bias'] = 1.5 if df['close'] > df['open'] else 0.5
    df['combined_factor'] *= df['directional_bias']
    
    # Incorporate Open-Close Trend
    df['open_close_diff'] = df['close'] - df['open']
    df['open_close_trend'] = 1.2 if df['open_close_diff'] > 0 else 0.8
    df['combined_factor'] *= df['open_close_trend']
    
    # Construct Final Alpha Factor
    final_alpha_factor = df['combined_factor']
    
    return final_alpha_factor

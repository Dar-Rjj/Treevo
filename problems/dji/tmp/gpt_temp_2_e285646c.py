import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Weighted High-Low Spread
    high_low_spread = (df['high'] - df['low']) * df['volume']
    
    # Apply Conditional Weight to High-Low Spread
    positive_return_weight = 1.5
    negative_return_weight = 0.5
    close_open_return = (df['close'] - df['open']) / df['open']
    high_low_spread_weighted = high_low_spread * (positive_return_weight if close_open_return > 0 else negative_return_weight)
    
    # Calculate Intraday Price Movement
    intraday_price_movement = df['close'] - df['open']
    
    # Determine Volume Increase from Average
    volume_5d_ma = df['volume'].rolling(window=5).mean()
    volume_increase = df['volume'] - volume_5d_ma
    
    # Compute Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) * df['volume']
    
    # Weight Intraday Price Movement by Volume Spike
    weighted_intraday_movement = abs(volume_increase) * intraday_price_movement
    
    # Calculate Close-to-Open Return
    close_open_return = (df['close'] - df['open']) / df['open']
    
    # Weight Close-to-Open Return by Volume
    weighted_close_open_return = close_open_return * df['volume']
    
    # Introduce Dynamic Weighting Based on Recent Performance
    cumulative_return_5d = df['close'].pct_change().rolling(window=5).sum()
    dynamic_weight = (cumulative_return_5d + 1).clip(lower=0.5, upper=1.5)
    
    # Combine All Components with Dynamic Weights
    combined_factor = (
        high_low_spread_weighted * dynamic_weight +
        weighted_intraday_movement * dynamic_weight +
        intraday_volatility * dynamic_weight +
        weighted_close_open_return * dynamic_weight
    )
    
    return combined_factor

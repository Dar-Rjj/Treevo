import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Price Difference
    high_low_diff = df['high'] - df['low']
    
    # Calculate High-Low Return
    high_low_return = high_low_diff / df['low']
    
    # Weight by Volume
    volume_weighted_high_low_return = high_low_return * df['volume']
    
    # Calculate Moving Average of Volume-Weighted High-Low Return (10-day window)
    moving_avg_volume_weighted_high_low_return = volume_weighted_high_low_return.rolling(window=10).mean()
    
    # Calculate Volume Change
    volume_change = df['volume'] - df['volume'].shift(1)
    
    # Weight Moving Average by Volume Change
    weighted_moving_avg = moving_avg_volume_weighted_high_low_return * volume_change
    
    # Calculate Short-Term Moving Average (5-day window on Close price)
    short_term_ma = df['close'].rolling(window=5).mean()
    
    # Calculate Long-Term Moving Average (20-day window on Close price)
    long_term_ma = df['close'].rolling(window=20).mean()
    
    # Compute Crossover Signal
    crossover_signal = short_term_ma - long_term_ma
    
    # Generate Combined Alpha Factor
    combined_alpha_factor = crossover_signal.apply(lambda x: weighted_moving_avg + x if x > 0 else weighted_moving_avg - x)
    
    # Final Alpha Factor
    final_alpha_factor = combined_alpha_factor.apply(lambda x: 1 if x > 0 else -1)
    
    return final_alpha_factor

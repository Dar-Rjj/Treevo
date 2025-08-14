import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Close-to-Open Change
    close_to_open_change = df['close'] - df['open']
    
    # Combine Intraday and Opening Gaps
    combined_change = intraday_high_low_spread - close_to_open_change
    
    # Weight by Volume for Intraday Movement
    weighted_intraday_movement = combined_change * df['volume']
    
    # Calculate Daily Price Movement
    daily_range = df['high'] - df['low']
    midpoint = (df['high'] + df['low']) / 2
    
    # Calculate Multi-Day Average Midpoint
    window = 5
    multi_day_midpoint = df['midpoint'].rolling(window=window).mean()
    
    # Compute Volume Adjusted Momentum
    close_to_close_return = df['close'].pct_change()
    volume_adjusted_momentum = close_to_close_return * df['volume']
    
    # Calculate Price Envelopment
    multi_day_range = daily_range.rolling(window=window).sum()
    multi_day_volume = df['volume'].rolling(window=window).sum()
    price_envelopment = multi_day_range / multi_day_volume
    
    # Final Alpha Factor
    final_alpha_factor = (volume_adjusted_momentum * price_envelopment * weighted_intraday_movement)
    
    return final_alpha_factor

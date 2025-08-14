import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Momentum Component
    short_term_return = df['close'].pct_change(5)
    medium_term_return = df['close'].pct_change(10)
    long_term_return = df['close'].pct_change(20)
    
    # Enhance Momentum Component
    enhanced_momentum = (0.4 * short_term_return + 
                         0.3 * medium_term_return + 
                         0.3 * long_term_return)

    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Intraday Close-Open Change
    intraday_close_open_change = df['close'] - df['open']
    
    # Combine Intraday Movements
    combined_intraday_movement = intraday_high_low_spread + intraday_close_open_change
    
    # Weight by Volume-Amount Ratio
    volume_amount_ratio = df['volume'] / df['amount']
    weighted_intraday_movement = combined_intraday_movement * volume_amount_ratio
    
    # Evaluate Recent Trend
    avg_weighted_intraday_5d = weighted_intraday_movement.rolling(window=5).mean()
    recent_trend_factor = weighted_intraday_movement - avg_weighted_intraday_5d
    
    # Introduce Reversal Indicator
    twenty_day_high = df['high'].rolling(window=20).max()
    twenty_day_low = df['low'].rolling(window=20).min()
    price_position_in_range = (df['close'] - twenty_day_low) / (twenty_day_high - twenty_day_low)
    
    # Calculate High-Low Range Momentum
    high_low_range_diff = (df['high'] - df['low']) - (df['high'].shift(1) - df['low'].shift(1))
    close_to_close_return = df['close'].pct_change(1)
    high_low_range_momentum = high_low_range_diff * close_to_close_return
    
    # Calculate Volume Change
    volume_change = df['volume'].pct_change(1)
    
    # Adjust for Sign
    adjusted_high_low_momentum = high_low_range_momentum * volume_change.where(volume_change > 0, -volume_change)
    
    # Combine Momentum, Intraday, and Reversal Indicators
    combined_factor = (enhanced_momentum + 
                       recent_trend_factor - 
                       price_position_in_range)
    
    # Final Adjustment
    final_factor = (combined_factor * 
                    adjusted_high_low_momentum).apply(lambda x: x ** 0.5) - df['close'].rolling(window=10).mean()
    
    return final_factor

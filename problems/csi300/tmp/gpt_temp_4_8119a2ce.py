import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 20-Day Average Close
    df['20_day_avg_close'] = df['close'].rolling(window=20).mean()
    
    # Subtract 20-Day Average Close from Today's Close
    df['close_minus_20_day_avg'] = df['close'] - df['20_day_avg_close']
    
    # Compute 5-day Price Return
    df['5_day_price_return'] = df['close'].pct_change(5)
    
    # Compute 20-day Price Return
    df['20_day_price_return'] = df['close'].pct_change(20)
    
    # Calculate Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Change
    df['close_to_open_change'] = df['close'] - df['open']
    
    # Combine Intraday and Opening Gaps
    df['combined_change'] = df['intraday_high_low_spread'] - df['close_to_open_change']
    
    # Weight by Volume
    df['weighted_combined_change'] = df['volume'] * df['combined_change']
    
    # Measure Close Location within Intraday Range
    df['close_proportional_distance'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Adjust for Volume Intensity
    df['weighted_intraday_reversal_potential'] = df['volume'] * df['close_proportional_distance']
    
    # Calculate Volume Surge
    df['5_day_avg_volume'] = df['volume'].rolling(window=5).mean()
    df['10_day_avg_volume'] = df['volume'].rolling(window=10).mean()
    
    # Synthesize All Factors
    df['daily_momentum'] = -df['close_minus_20_day_avg']
    df['intraday_volatility'] = df['intraday_high_low_spread']
    df['short_term_momentum_diff'] = df['5_day_price_return'] - df['20_day_price_return']
    df['long_term_momentum_diff'] = df['20_day_price_return'] - df['price_return_50d']
    df['synthesized_factors'] = (
        df['daily_momentum'] + 
        df['intraday_volatility'] + 
        df['weighted_intraday_reversal_potential'] + 
        df['short_term_momentum_diff'] * df['long_term_momentum_diff'] * df['volume_adjusted_momentum']
    )
    
    # Incorporate Intraday High-Low Spread Ratio
    df['5_day_avg_high'] = df['high'].rolling(window=5).mean()
    df['5_day_avg_low'] = df['low'].rolling(window=5).mean()
    df['intraday_high_low_spread_ratio'] = df['5_day_avg_high'] / df['5_day_avg_low']
    df['adjusted_synthesized_factors'] = df['synthesized_factors'] + (df['intraday_high_low_spread_ratio'] * df['intraday_volatility'])
    
    # Final Alpha Factor
    df['final_alpha_factor'] = df['adjusted_synthesized_factors']
    
    return df['final_alpha_factor']

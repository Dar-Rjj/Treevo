import pandas as pd
import pandas as pd

def heuristics_v2(df, market_index_df):
    # Compute 3-Day Rolling Return
    df['3_day_return'] = (df['close'].shift(-2) / df['close']) - 1
    
    # Compute 5-Day Rolling Return
    df['5_day_return'] = (df['close'].shift(-4) / df['close']) - 1
    
    # Compute 10-Day Rolling Return
    df['10_day_return'] = (df['close'].shift(-9) / df['close']) - 1
    
    # Combine 3-Day, 5-Day, and 10-Day Returns
    df['combined_return'] = 0.3 * df['3_day_return'] + 0.4 * df['5_day_return'] + 0.3 * df['10_day_return']
    
    # Multiply by Volume of day t
    df['volume_adjusted_momentum'] = df['combined_return'] * df['volume']
    
    # Calculate Open-High-Low Volatility Index
    df['high_low_spread'] = df['high'] - df['low']
    df['open_close_spread'] = abs(df['open'] - df['close'])
    df['mean_price'] = (df['open'] + df['close']) / 2
    df['volatility_index'] = (df['high_low_spread'] + df['open_close_spread']) / (2 * df['mean_price'])
    
    # Adjust for Market Conditions
    market_index_df['market_10_day_return'] = (market_index_df['close'].shift(-9) / market_index_df['close']) - 1
    df = df.join(market_index_df['market_10_day_return'], on='date')
    df['market_adjusted_momentum'] = df['volume_adjusted_momentum'] - df['market_10_day_return']
    
    # Final Factor Calculation
    df['final_factor'] = df['market_adjusted_momentum'] - df['volatility_index']
    
    return df['final_factor']

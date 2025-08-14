import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close to Open Ratio
    df['close_to_open_ratio'] = df['close'] / df['open']
    
    # Weighted Difference
    df['weighted_diff'] = (df['intraday_range'] * df['close_to_open_ratio']) * df['volume']
    
    # Multi-Day Momentum
    df['prev_day_return'] = (df['close'].shift(1) - df['close'].shift(2)) / df['close'].shift(2)
    df['two_day_return'] = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    df['three_day_return'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Volatility Component
    daily_returns = df['close'].pct_change()
    df['5d_std'] = daily_returns.rolling(window=5).std()
    df['10d_std'] = daily_returns.rolling(window=10).std()
    df['avg_volatility'] = (df['5d_std'] + df['10d_std']) / 2
    df['vol_adjusted'] = df['weighted_diff'] / df['avg_volatility']
    
    # Sector and Market Cap Adjustments
    # Assuming sector and market cap data are available in the DataFrame
    # For simplicity, using dummy weights for sectors and market cap
    sector_weights = {'sector_A': 0.8, 'sector_B': 0.7, 'sector_C': 0.9}
    market_cap_weights = {'small': 0.5, 'medium': 0.7, 'large': 1.0}
    
    df['sector_weight'] = df['sector'].map(sector_weights)
    df['market_cap_weight'] = df['market_cap'].map(market_cap_weights)
    
    # Calculate Sector Average Return
    sector_avg_return = df.groupby('sector')['prev_day_return'].transform('mean')
    
    # Adjust for Sector Performance
    df['adjusted_for_sector'] = df['vol_adjusted'] * (1 + (df['prev_day_return'] - sector_avg_return))
    
    # Adjust for Market Cap
    df['final_alpha_factor'] = df['adjusted_for_sector'] * df['sector_weight'] * df['market_cap_weight']
    
    return df['final_alpha_factor'].dropna()

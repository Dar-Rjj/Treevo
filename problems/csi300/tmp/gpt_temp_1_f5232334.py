import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Compute Intraday Return
    df['intraday_return'] = df['close'] - df['open']
    
    # Determine Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
    
    # Adjust Volatility by Volume
    df['adjusted_volatility'] = df['intraday_volatility'] * df['volume']
    
    # Calculate Bid-Ask Spread (Assuming we have 'ask' and 'bid' prices in the DataFrame)
    df['bid_ask_spread'] = df['ask'] - df['bid']
    
    # Adjust Intraday Return by Bid-Ask Spread
    df['intraday_return_adj'] = df['intraday_return'] / df['bid_ask_spread']
    
    # Calculate Short-Term Momentum
    df['short_term_momentum'] = df['close'] - df['close'].shift(1)
    
    # Calculate Long-Term Momentum
    df['long_term_momentum'] = df['close'] - df['close'].shift(5)
    
    # Dynamically Weight Momentum Terms
    short_std = df['short_term_momentum'].rolling(window=20).std()
    long_std = df['long_term_momentum'].rolling(window=20).std()
    
    df['short_term_weight'] = 1 / short_std
    df['long_term_weight'] = 1 / long_std
    
    total_weight = df['short_term_weight'] + df['long_term_weight']
    df['short_term_weight_norm'] = df['short_term_weight'] / total_weight
    df['long_term_weight_norm'] = df['long_term_weight'] / total_weight
    
    # Combine Intraday Return, Adjusted Volatility, and Weighted Momentum
    df['combined_value'] = (
        df['intraday_return_adj'] 
        - df['adjusted_volatility'] 
        + df['short_term_momentum'] * df['short_term_weight_norm']
        + df['long_term_momentum'] * df['long_term_weight_norm']
    )
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = df['combined_value'].apply(lambda x: 1 if x > 0 else 0)
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)

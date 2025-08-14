import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Compute Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Determine Intraday Volatility
    high_low_range = df['high'] - df['low']
    close_high_range = np.abs(df['close'] - df['high'])
    close_low_range = np.abs(df['close'] - df['low'])
    df['intraday_volatility'] = np.maximum.reduce([high_low_range, close_high_range, close_low_range])
    
    # Adjust Volatility by Volume
    df['adjusted_volatility'] = df['intraday_volatility'] * df['volume']
    
    # Calculate Intraday Momentum
    # Assign dynamic weights: w1, w2, w3, w4, w5 based on recent volatility
    volatilities = df['intraday_volatility'].rolling(window=5).mean()
    weights = 1 / (volatilities + 0.0001)  # to avoid division by zero
    weights = weights / weights.sum()  # normalize weights
    
    open_prices = df['open'].shift(1).rolling(window=5).apply(lambda x: (x * weights).sum(), raw=False)
    df['intraday_momentum'] = df['close'] - open_prices
    
    # Calculate Short-Term Moving Average
    df['short_term_ma'] = df['close'].rolling(window=5).mean()
    
    # Calculate Long-Term Moving Average
    df['long_term_ma'] = df['close'].rolling(window=20).mean()
    
    # Calculate Relative Strength
    df['relative_strength'] = df['short_term_ma'] / df['long_term_ma']
    
    # Combine Intraday Return, Adjusted Volatility, and Momentum
    df['combined_value'] = df['intraday_return'] + df['intraday_momentum'] - df['adjusted_volatility']
    
    # Incorporate Relative Strength
    df['combined_value_rs'] = df['combined_value'] * df['relative_strength']
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = np.where(df['combined_value_rs'] > 0, 1, 0)
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date')
# alpha_factor_series = heuristics_v2(df)
# print(alpha_factor_series)

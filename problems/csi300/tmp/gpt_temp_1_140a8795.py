import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Price Range
    df['intraday_price_range'] = df['high'] - df['low']
    
    # Compute Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Determine Intraday Volatility
    df['true_range'] = df.apply(
        lambda x: max(x['high'] - x['low'], 
                      abs(x['high'] - df['close'].shift(1).fillna(0)), 
                      abs(df['close'].shift(1).fillna(0) - x['low'])),
        axis=1
    )
    N = 14
    df['average_true_range'] = df['true_range'].rolling(window=N, min_periods=1).mean()
    
    # Adjust Volatility by Volume
    df['adjusted_volatility'] = df['average_true_range'] * df['volume']
    
    # Calculate Short-Term Momentum
    df['short_term_momentum'] = df['close'] - df['close'].shift(1)
    
    # Calculate Medium-Term Momentum
    df['medium_term_momentum'] = df['close'] - df['close'].shift(5)
    
    # Calculate Long-Term Momentum
    df['long_term_momentum'] = df['close'] - df['close'].shift(20)
    
    # Combine Intraday Return, Adjusted Volatility, and Momentum
    df['combined_value'] = (
        df['intraday_return'] 
        - df['adjusted_volatility'] 
        + df['short_term_momentum'] 
        + df['medium_term_momentum'] 
        + df['long_term_momentum']
    )
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = (df['combined_value'] > 0).astype(int)
    
    return df['alpha_factor']

# Example usage:
# alpha_factor = heuristics_v2(df)

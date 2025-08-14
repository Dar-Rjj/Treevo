import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Compute Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Determine Intraday Volatility (Average True Range over 10 days)
    df['true_range'] = df.apply(lambda x: max(x['high'] - x['low'], 
                                              abs(x['high'] - df.loc[x.name - pd.Timedelta(days=1), 'close']), 
                                              abs(x['low'] - df.loc[x.name - pd.Timedelta(days=1), 'close'])), axis=1)
    df['intraday_volatility'] = df['true_range'].rolling(window=10).mean()
    
    # Adjust Volatility by Volume
    df['adjusted_volatility'] = df['intraday_volatility'] * df['volume']
    
    # Calculate Intraday Momentum
    df['intraday_momentum'] = df['close'] - df.shift(1)['open']
    
    # Combine Intraday Return, Adjusted Volatility, and Momentum
    df['combined_value'] = df['intraday_return'] - df['adjusted_volatility'] + df['intraday_momentum']
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = (df['combined_value'] > 0).astype(int)
    
    return df['alpha_factor']

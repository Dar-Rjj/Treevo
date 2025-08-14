import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate Intraday Price Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Compute Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Determine Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
    
    # Adjust Volatility by Volume
    df['adjusted_volatility'] = df['intraday_volatility'] * df['volume']
    
    # Calculate Short-Term Momentum
    df['short_term_momentum'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Long-Term Momentum
    df['long_term_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Incorporate Market Trend
    def calculate_market_trend(close_prices):
        X = np.array(range(len(close_prices))).reshape(-1, 1)
        y = close_prices
        model = LinearRegression().fit(X, y)
        return model.coef_[0]
    
    df['market_trend'] = df['close'].rolling(window=20).apply(calculate_market_trend, raw=False)
    
    # Apply Dynamic Weighting to Short-Term Momentum
    df['weighted_short_term_momentum'] = df['short_term_momentum'] * (1 + df['market_trend'])
    
    # Apply Dynamic Weighting to Long-Term Momentum
    df['weighted_long_term_momentum'] = df['long_term_momentum'] * (1 + df['market_trend'])
    
    # Combine Intraday Return, Adjusted Volatility, and Weighted Momentum
    df['combined_value'] = (df['intraday_return'] 
                            - df['adjusted_volatility'] 
                            + df['weighted_short_term_momentum'] 
                            + df['weighted_long_term_momentum'])
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = df['combined_value'].apply(lambda x: 1 if x > 0 else 0)
    
    return df['alpha_factor']

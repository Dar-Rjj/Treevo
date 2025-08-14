import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Volatility
    df['IV'] = df['high'] - df['low']
    
    # Weight by Volume
    df['W_IV'] = df['IV'] * df['volume']
    
    # Calculate Intraday Momentum
    df['M'] = df['close'] - df['open']
    df['Rolling_M'] = df['M'].rolling(window=5).sum()
    
    # Incorporate More Price Levels
    df['TR'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))), axis=1)
    df['Rolling_TR'] = df['TR'].rolling(window=14).mean()
    
    # Integrate Weighted Volatility, Rolling Momentum, and True Range
    df['Sum'] = df['W_IV'] + df['Rolling_M'] + df['Rolling_TR']
    
    # Apply Dynamic Exponential Smoothing
    alpha = 0.9
    df['ES'] = df['Sum'].ewm(alpha=alpha, adjust=False).mean()
    
    # Ensure Values are Positive
    epsilon = 1e-6
    df['Positive_ES'] = df['ES'] + epsilon
    
    # Apply Logarithmic Transformation
    df['Log_ES'] = np.log(df['Positive_ES'])
    
    # Factor Output
    return df['Log_ES']

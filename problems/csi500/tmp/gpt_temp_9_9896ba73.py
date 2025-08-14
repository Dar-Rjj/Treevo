import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate the log return of close price
    df['log_return'] = df['close'].apply(lambda x: np.log(x)).diff()
    
    # Calculate the volatility factor (standard deviation of log returns over a 30-day window)
    df['volatility'] = df['log_return'].rolling(window=30).std()
    
    # Calculate the liquidity factor (average volume over a 5-day window divided by average amount over a 5-day window)
    df['liquidity'] = df['volume'].rolling(window=5).mean() / df['amount'].rolling(window=5).mean()
    
    # Calculate the momentum factor (slope of the linear regression of close prices over a 60-day period)
    def calculate_momentum(data, window=60):
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(np.arange(window).reshape(-1, 1), data[-window:])
        return reg.coef_[0]
    
    df['momentum'] = df['close'].rolling(window=60).apply(calculate_momentum, raw=False)
    
    # Combine the factors into a composite alpha score
    df['alpha_factor'] = (df['momentum'] * 0.4) - (df['volatility'] * 0.3) + (df['liquidity'] * 0.3)
    
    # Drop rows with NaN values that were created during rolling operations
    df.dropna(inplace=True)
    
    # Return the alpha factor as a series
    return df['alpha_factor']

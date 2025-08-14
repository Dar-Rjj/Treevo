import pandas as pd
import pandas as pd

def heuristics_v2(df, n_days=10):
    # Calculate Intraday Return
    df['Intraday_Return'] = (df['High'] - df['Low']) / df['Close'].shift(1)
    
    # Calculate Volume Impact Score
    volume_sum = df['Volume'].rolling(window=n_days).sum()
    high_avg = df['High'].rolling(window=n_days).mean()
    low_avg = df['Low'].rolling(window=n_days).mean()
    df['Volume_Impact_Score'] = (high_avg - low_avg) / volume_sum
    
    # Calculate Dynamic Volatility Adjusted Intraday Return
    close_rolling_std = df['Close'].rolling(window=n_days).std()
    max_volatility = close_rolling_std.max()
    min_volatility = close_rolling_std.min()
    if max_volatility == min_volatility:
        volatility_weight = 1.0  # avoid division by zero
    else:
        volatility_weight = 1 - (close_rolling_std - min_volatility) / (max_volatility - min_volatility)
    df['Volatility_Adjusted_Intraday_Return'] = df['Intraday_Return'] * volatility_weight
    
    # Multiply Weighted Intraday Return by Volume Impact Score
    df['Factor'] = df['Volatility_Adjusted_Intraday_Return'] * df['Volume_Impact_Score']
    
    return df['Factor']

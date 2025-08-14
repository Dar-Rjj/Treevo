import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate EMAs for Close Price
    df['EMA_5_Close'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20_Close'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate Volatility
    daily_returns = df['close'].pct_change()
    df['Volatility_10'] = daily_returns.rolling(window=10).std()
    df['Volatility_30'] = daily_returns.rolling(window=30).std()
    
    # Dynamic Weights
    df['EMA_Ratio'] = df['EMA_5_Close'] / df['EMA_20_Close']
    
    # Integrate Volume Trends
    df['EMA_5_Volume'] = df['volume'].ewm(span=5, adjust=False).mean()
    df['EMA_20_Volume'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['Volume_Ratio'] = df['EMA_5_Volume'] / df['EMA_20_Volume']
    
    # Sector Performance (Assuming a column 'sector' exists in the DataFrame)
    sector_ema_5 = df.groupby('sector')['close'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    sector_ema_20 = df.groupby('sector')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['Sector_EMA_5_Ratio'] = df['EMA_5_Close'] / sector_ema_5
    df['Sector_EMA_20_Ratio'] = df['EMA_20_Close'] / sector_ema_20
    
    # Balance Factors
    df['Short_Term_Momentum_Adjusted'] = df['EMA_Ratio'] * df['Sector_EMA_5_Ratio']
    df['Adaptive_Volatility_Adjusted'] = (df['Volatility_10'] + df['Volatility_30']) / 2
    df['Dynamic_Weights_Adjusted'] = df['Short_Term_Momentum_Adjusted'] * df['Adaptive_Volatility_Adjusted']
    
    # Liquidity Adjustment
    df['Average_20_Volume'] = df['volume'].rolling(window=20).mean()
    df['Liquidity_Ratio'] = df['volume'] / df['Average_20_Volume']
    
    # Final Alpha Factor
    df['Alpha_Factor'] = df['Dynamic_Weights_Adjusted'] * df['Liquidity_Ratio']
    
    return df['Alpha_Factor']

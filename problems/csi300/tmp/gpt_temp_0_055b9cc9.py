import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Spread
    df['High_Low_Spread'] = df['high'] - df['low']
    
    # Calculate Daily Volume Trend
    df['Volume_MA_10'] = df['volume'].rolling(window=10).mean()
    df['Volume_Trend'] = df['volume'] - df['Volume_MA_10']
    
    # Calculate Short-Term Price Trend (EMA over 10 days)
    df['Close_EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['Short_Term_Price_Trend'] = df['close'] - df['Close_EMA_10']
    
    # Calculate Medium-Term Price Trend (EMA over 30 days)
    df['Close_EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['Medium_Term_Price_Trend'] = df['close'] - df['Close_EMA_30']
    
    # Calculate Long-Term Price Trend (EMA over 50 days)
    df['Close_EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['Long_Term_Price_Trend'] = df['close'] - df['Close_EMA_50']
    
    # Calculate Dynamic Volatility (Standard Deviation over 10 days)
    df['Dynamic_Volatility'] = df['close'].rolling(window=10).std()
    
    # Integrate Momentum and Relative Strength
    df['Close_EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['Close_EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Relative_Strength'] = df['Close_EMA_5'] / df['Close_EMA_20']
    
    # Adjust High-Low Spread by Volume Trend
    df['Adjusted_Spread'] = df['High_Low_Spread'] * df['Volume_Trend'].apply(lambda x: 1.5 if x > 0 else 0.5)
    
    # Incorporate Short-Term Price Trend
    df['Adjusted_Short_Term'] = df['Short_Term_Price_Trend'].apply(lambda x: 1.2 if x > 0 else 0.8)
    
    # Incorporate Medium-Term Price Trend
    df['Adjusted_Medium_Term'] = df['Medium_Term_Price_Trend'].apply(lambda x: 1.1 if x > 0 else 0.9)
    
    # Incorporate Long-Term Price Trend
    df['Adjusted_Long_Term'] = df['Long_Term_Price_Trend'].apply(lambda x: 1.3 if x > 0 else 0.7)
    
    # Incorporate Dynamic Volatility
    df['Adjusted_Volatility'] = df['Dynamic_Volatility'].apply(lambda x: 1.4 if x > df['Dynamic_Volatility'].mean() else 0.6)
    
    # Incorporate Momentum and Relative Strength
    df['Adjusted_Relative_Strength'] = df['Relative_Strength'].apply(lambda x: 1.7 if x > 1 else 0.3)
    
    # Final Alpha Factor
    df['Alpha_Factor'] = (df['Adjusted_Spread'] * 
                          df['Adjusted_Short_Term'] * 
                          df['Adjusted_Medium_Term'] * 
                          df['Adjusted_Long_Term'] * 
                          df['Adjusted_Volatility'] * 
                          df['Adjusted_Relative_Strength'])
    
    return df['Alpha_Factor']

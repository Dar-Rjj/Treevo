import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Range
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Calculate High-Low Range Momentum
    df['HL_Range_Momentum'] = df['High_Low_Range'] - df['High_Low_Range'].shift(1)
    
    # Calculate Daily Price Return
    df['Daily_Price_Return'] = (df['close'] / df['close'].shift(1)) - 1
    
    # Calculate 20-Day Weighted Moving Average of Returns
    df['WMA_Returns'] = df.apply(lambda x: sum(df.loc[x.name - pd.Timedelta(days=i), 'Daily_Price_Return'] * df.loc[x.name - pd.Timedelta(days=i), 'volume'] for i in range(20) if x.name - pd.Timedelta(days=i) in df.index) / sum(df.loc[x.name - pd.Timedelta(days=i), 'volume'] for i in range(20) if x.name - pd.Timedelta(days=i) in df.index), axis=1)
    
    # Adjust for Price Volatility
    df['Avg_Daily_Price_Range'] = df['High_Low_Range'].rolling(window=20).mean()
    df['Adjusted_WMA_Returns'] = df['WMA_Returns'] - df['Avg_Daily_Price_Range']
    
    # Adjust by Average Volume
    df['Avg_Volume'] = df['volume'].rolling(window=20).mean()
    df['Adj_HL_Range_Momentum'] = df['HL_Range_Momentum'] / df['Avg_Volume']
    
    # Calculate Intraday Return Ratio
    df['Intraday_Return_Ratio'] = df['high'] / df['low'] - 1
    
    # Calculate Weighted Open-to-Close Return
    df['Weighted_Open_Close_Return'] = (df['close'] - df['open']) * df['volume']
    
    # Combine Adjusted Momentum and Intraday Factors
    df['Alpha_Factor'] = df['Adj_HL_Range_Momentum'] + df['Adjusted_WMA_Returns'] + df['Intraday_Return_Ratio'] + df['Weighted_Open_Close_Return']
    
    return df['Alpha_Factor']

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Historical Volatility
    df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
    df['Volatility'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)
    
    # Compute High-Low Difference and Open-Close Return
    df['High_Low_Diff'] = df['high'] - df['low']
    df['Open_Close_Return'] = df['close'] - df['open']
    
    # Weighted Average of High-Low Difference and Open-Close Return
    df['Weight_High_Low'] = df['High_Low_Diff'] * (df['Volatility'] / df['Volatility'].mean())
    df['Weight_Open_Close'] = df['Open_Close_Return'] * (df['Volatility'] / df['Volatility'].mean())
    df['Intraday_Momentum'] = (df['Weight_High_Low'] + df['Weight_Open_Close']) / 2
    
    # Apply Volume and Price Shock Filter
    df['Volume_Ratio'] = df['volume'] / df['volume'].shift(1)
    df['Amount_Ratio'] = df['amount'] / df['amount'].shift(1)
    df['Abs_Price_Change'] = abs(df['close'] - df['close'].shift(1))
    df['Price_Range'] = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    df['Price_Shock_Ratio'] = df['Abs_Price_Change'] / df['Price_Range']
    
    # Threshold Filter
    df['Intraday_Momentum_Filtered'] = df['Intraday_Momentum'] * ((df['Volume_Ratio'] > 1.5) & (df['Amount_Ratio'] > 1.5) & (df['Price_Shock_Ratio'] > 0.05))
    
    # Adjust for Volume
    df['Intraday_Momentum_Adjusted'] = df['Intraday_Momentum_Filtered'] / df['volume']
    
    # Measure Volume Synchronization
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Price_Change'] = df['close'] - df['close'].shift(1)
    df['Volume_Sync'] = np.sign(df['Volume_Change'] * df['Price_Change'])
    
    # Calculate EMA Cross Signal
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Momentum'] = np.where(df['EMA_5'] > df['EMA_20'], 1, -1)
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = 0
    df.loc[df['close'] > df['close'].shift(1), 'OBV'] = df['volume']
    df.loc[df['close'] < df['close'].shift(1), 'OBV'] = -df['volume']
    df['OBV'] = df['OBV'].cumsum()
    
    # Calculate MACD
    df['MACD_Line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Signal_Diff'] = abs(df['MACD_Line'] - df['Signal_Line'])
    
    # Weight OBV by |MACD Line - Signal Line|
    df['Weighted_OBV'] = df['OBV'] * df['MACD_Signal_Diff']
    
    # Calculate Price Velocity
    df['Price_Velocity'] = (df['close'] - df['close'].shift(10)) / 10
    
    # Integrate Intraday Metrics
    df['Integrated_Intraday'] = (df['Intraday_Momentum_Adjusted'] * df['Intraday_Return']) / df['volume']
    
    # Synthesize Final Alpha Factor
    df['Alpha_Factor'] = (df['Intraday_Momentum_Adjusted'] + 
                          df['Volume_Sync'] * df['Price_Change'] + 
                          (df['Weighted_OBV'] * df['Price_Velocity']) + 
                          df['Integrated_Intraday'])
    
    return df['Alpha_Factor']

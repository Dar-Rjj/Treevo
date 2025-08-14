import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Weighted Average Price (VWAP)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Daily VWAP Change
    df['VWAP_Change'] = df['VWAP'].diff()
    
    # Separate Positive and Negative Changes
    df['Positive_Change'] = df['VWAP_Change'].apply(lambda x: max(x, 0))
    df['Negative_Change'] = df['VWAP_Change'].apply(lambda x: -min(x, 0))
    
    # 14-Day Averages for RSI
    df['Pos_Avg_14'] = df['Positive_Change'].rolling(window=14).mean()
    df['Neg_Avg_14'] = df['Negative_Change'].rolling(window=14).mean()
    
    # Calculate RSI
    df['RSI'] = 100 - (100 / (1 + (df['Pos_Avg_14'] / df['Neg_Avg_14'])))
    
    # Intraday Range Growth
    df['High_Low_Range'] = df['high'] - df['low']
    df['Prev_High_Low_Range'] = df['High_Low_Range'].shift(1)
    df['Intraday_Range_Growth'] = (df['High_Low_Range'] - df['Prev_High_Low_Range']) / df['Prev_High_Low_Range']
    
    # Volume-Weighted Moving Average (VWMA)
    df['Close_Vol'] = df['close'] * df['volume']
    df['VWMA'] = df['Close_Vol'].rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    
    # Daily Returns
    df['Daily_Return'] = df['close'].pct_change()
    
    # Long-Term VW-Return (Momentum Component)
    df['Long_Term_VW_Return'] = (df['Daily_Return'].rolling(window=100) * df['volume'].rolling(window=100)).sum() / df['volume'].rolling(window=100).sum()
    
    # Short-Term VW-Return (Reversal Component)
    df['Short_Term_VW_Return'] = (df['Daily_Return'].rolling(window=5) * df['volume'].rolling(window=5)).sum() / df['volume'].rolling(window=5).sum()
    
    # Final Alpha Factor
    df['Alpha_Factor'] = df['RSI'] + df['Intraday_Range_Growth'] + df['VWMA'] + df['Long_Term_VW_Return'] - df['Short_Term_VW_Return']
    
    return df['Alpha_Factor']

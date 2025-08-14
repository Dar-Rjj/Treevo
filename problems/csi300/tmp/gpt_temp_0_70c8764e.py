import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['Intraday_Return'] = (df['high'] - df['low']) / df['close']
    
    # Calculate Overnight Return
    df['Overnight_Return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Combine Intraday and Overnight Returns
    df['Combined_Return'] = df['Intraday_Return'] + df['Overnight_Return']
    
    # Compute Volume-Weighted Average Price (VWAP)
    df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
    df['Volume_Weighted_Price'] = df['Typical_Price'] * df['volume']
    df['VWAP'] = df['Volume_Weighted_Price'].rolling(window=len(df), min_periods=1).sum() / df['volume'].rolling(window=len(df), min_periods=1).sum()
    
    # Calculate VWAP Reversal Indicator
    df['Reversal_Indicator'] = np.where(df['VWAP'] > df['close'], 1, -1)
    
    # Integrate Reversal Indicator with Combined Return
    df['Integrated_Reversal_Return'] = df['Combined_Return'] * df['Reversal_Indicator']
    
    # Calculate Volume-Weighted Return
    df['Return'] = (df['close'] - df['open']) / df['open']
    df['Volume_Weighted_Return'] = df['Return'] * df['volume']
    df['Volume_Weighted_Return'] = df['Volume_Weighted_Return'].rolling(window=len(df), min_periods=1).sum() / df['volume'].rolling(window=len(df), min_periods=1).sum()
    
    # Integrate Volume-Weighted Return with Integrated Reversal Return
    df['Initial_Alpha_Factor'] = df['Integrated_Reversal_Return'] + df['Volume_Weighted_Return']
    
    # Calculate Rolling Averages
    df['Rolling_Avg_5'] = df['close'].rolling(window=5).mean()
    df['Rolling_Avg_20'] = df['close'].rolling(window=20).mean()
    
    # Calculate Volatility
    df['Daily_Return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    # Adjust for Volatility
    df['Volatility_Adjusted_Alpha_Factor'] = df['Initial_Alpha_Factor'] / df['Volatility']
    
    # Incorporate Transaction Amount
    df['Amount_Weighted_Return'] = (df['close'] - df['open']) / df['open'] * df['amount']
    df['Amount_Weighted_Return'] = df['Amount_Weighted_Return'].rolling(window=len(df), min_periods=1).sum() / df['amount'].rolling(window=len(df), min_periods=1).sum()
    
    # Integrate Amount-Weighted Return with Volatility-Adjusted Alpha Factor
    df['Final_Alpha_Factor'] = df['Volatility_Adjusted_Alpha_Factor'] + df['Amount_Weighted_Return']
    
    # Incorporate Momentum Signal
    df['Momentum_Signal'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['Updated_Final_Alpha_Factor'] = df['Final_Alpha_Factor'] + df['Momentum_Signal']
    
    # Incorporate Market Sentiment
    df['Market_Sentiment'] = (df['Daily_Return'] + 1).rolling(window=20).apply(np.prod) - 1
    df['Enhanced_Final_Alpha_Factor'] = df['Updated_Final_Alpha_Factor'] + df['Market_Sentiment']
    
    return df['Enhanced_Final_Alpha_Factor']

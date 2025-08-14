import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Compute Volume-Weighted High-Low Price Difference
    df['Volume_Weighted_High_Low'] = (df['high'] - df['low']) * df['volume']
    
    # Calculate Deviation of Current Intraday High-Low Spread
    df['High_Low_Spread'] = df['high'] - df['low']
    historical_avg_spread = df['High_Low_Spread'].rolling(window=5).mean()
    df['Deviation_Intraday_Spread'] = df['High_Low_Spread'] - historical_avg_spread
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = ((df['open'] + df['high'] + df['low'] + df['close']) * df['volume']).cumsum() / df['volume'].cumsum()
    df['Deviation_VWAP_Close'] = df['VWAP'] - df['close']
    
    # Incorporate Volume Impact Factor
    df['Price_Change'] = df['close'].diff().abs()
    df['Volume_Impact'] = df['volume'] * df['Price_Change'].shift(1)
    
    # Integrate Historical High-Low Range and Momentum Contributions
    df['Vol_Weighted_HL_Range'] = df['Volume_Weighted_High_Low'].rolling(window=5).sum()
    df['Momentum_Contribution'] = df['close'].pct_change(10) * df['Vol_Weighted_HL_Range']
    df['Accumulated_Momentum'] = df['Momentum_Contribution'].where(df['close'].diff(5) > 0, 0).cumsum()
    
    # Adjust for Market Sentiment
    df['Price_Volatility'] = (df['high'] - df['low']) / df['close']
    average_volatility = df['Price_Volatility'].rolling(window=5).mean()
    df['Integrated_Value'] = df['Accumulated_Momentum']
    df['Adjusted_Integrated_Value'] = df['Integrated_Value'].apply(lambda x: x * 1.5 if x > average_volatility else x * 0.5)
    
    # Evaluate Overnight Sentiment
    df['Log_Volume'] = np.log(df['volume'])
    df['Overnight_Return'] = np.log(df['open']) - np.log(df['close'].shift(1))
    
    # Integrate Intraday and Overnight Signals
    df['Intraday_Return'] = (np.log(df['high']) - np.log(df['low'])) / 2 + (np.log(df['close']) - np.log(df['open']))
    df['Volume_Adjusted_Indicator'] = df['volume'].rolling(window=5).mean() - df['volume']
    df['Integrated_Signal'] = (df['Intraday_Return'] - df['Overnight_Return']) * df['Volume_Adjusted_Indicator']
    
    # Generate Alpha Factor
    df['Alpha_Factor'] = (df['Deviation_Intraday_Spread'] + df['Deviation_VWAP_Close']) * df['close']
    df['Alpha_Factor'] = df['Alpha_Factor'] + df['Adjusted_Integrated_Value'] + df['Integrated_Signal']
    
    # Synthesize Overall Alpha Factor
    df['Overall_Alpha_Factor'] = df['Alpha_Factor'] + df['Volume_Impact']
    
    return df['Overall_Alpha_Factor']

import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Define lookback periods
    sma_lookback = 20
    atr_lookback = 14
    
    # Calculate Simple Moving Average of Close Price
    df['SMA_Close'] = df['close'].rolling(window=sma_lookback).mean()
    
    # Calculate Price Momentum
    df['Momentum_signal'] = df['close'] - df['SMA_Close']
    
    # Calculate High and Low Price Momentum
    df['Daily_Range'] = df['high'] - df['low']
    df['Close_to_Open_Return'] = (df['close'] - df['open']) / df['open']
    df['High_Low_Momentum'] = df['Daily_Range'] * df['Close_to_Open_Return']
    
    # Combine Momentum Signals
    df['Combined_Momentum'] = df['Momentum_signal'] + df['High_Low_Momentum']
    
    # Calculate Volume Change
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    
    # Weight Combined Momentum by Volume
    df['Volume_Adjusted_Combined_Momentum'] = df['Combined_Momentum'] * df['Volume_Change']
    
    # Calculate True Range
    df['True_Range'] = df[['high' - 'low', 'high' - df['close'].shift(1), df['close'].shift(1) - 'low']].max(axis=1)
    
    # Calculate Average True Range
    df['ATR'] = df['True_Range'].rolling(window=atr_lookback).mean()
    
    # Generate Volatility Adjusted Trend Score
    df['Volatility_Adjusted_Trend_Score'] = df['Volume_Adjusted_Combined_Momentum'] / df['ATR']
    
    return df['Volatility_Adjusted_Trend_Score']

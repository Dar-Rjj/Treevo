import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the True Range for each day
    df['Prev_Close'] = df['close'].shift(1)
    df['True_Range'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['Prev_Close']), abs(x[1] - df['Prev_Close'])), axis=1)
    
    # Calculate the 14-day Simple Moving Average of the True Range
    df['SMA_TR_14'] = df['True_Range'].rolling(window=14).mean()
    
    # Construct the Momentum Factor
    df['Momentum_Factor'] = (df['close'] - df['SMA_TR_14']) / df['SMA_TR_14']
    
    # Calculate 14-Day Volume-Weighted Intraday Return
    df['Intraday_Return'] = df['close'] - df['open']
    df['Volume_Weighted_Intraday_Return'] = df['Intraday_Return'] * df['volume']
    df['Volume_Weighted_Intraday_Return_14'] = df['Volume_Weighted_Intraday_Return'].rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    
    # Identify Consecutive Up/Down Days
    df['Up_Day'] = (df['close'] > df['open']).astype(int)
    df['Down_Day'] = (df['open'] > df['close']).astype(int)
    
    # Adjust for Extreme Movement (High-Low Difference)
    df['High_Low_Diff'] = df['high'] - df['low']
    df['Volume_Weighted_High_Low_Diff'] = df['High_Low_Diff'] * df['volume']
    
    # Incorporate Volume Influence
    df['Avg_Volume_20'] = df['volume'].rolling(window=20).mean()
    df['Volume_Impact'] = df['volume'] / df['Avg_Volume_20']
    df['Weighted_Intraday_Reversal'] = df['Volume_Weighted_Intraday_Return_14'] * df['Volume_Impact']
    
    # Adjust for Volatility
    df['Std_30'] = df['close'].rolling(window=30).std()
    df['Adjusted_Momentum_Factor'] = df['Momentum_Factor'] / df['Std_30']
    
    # Introduce Trend Component
    df['Trend_MA_50'] = df['close'].rolling(window=50).mean()
    df['Trend_Direction'] = (df['close'] > df['Trend_MA_50']).astype(int) * 2 - 1
    
    # Final Alpha Factor
    df['Alpha_Factor'] = (df['Adjusted_Momentum_Factor'] + df['Weighted_Intraday_Reversal']) * df['Trend_Direction']
    
    return df['Alpha_Factor']

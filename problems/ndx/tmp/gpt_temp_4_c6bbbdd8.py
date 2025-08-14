import pandas as pd
import pandas as pd

def heuristics_v2(df):
    N = 5  # Lookback period
    
    # Calculate Simple Moving Average (SMA) of Close price
    df['SMA_Close'] = df['close'].rolling(window=N).mean()
    
    # Calculate Price Difference
    df['Price_Diff'] = df['close'] - df['SMA_Close']
    
    # Compute Momentum Score
    df['Momentum_Score'] = df['Price_Diff'] / df['SMA_Close']
    
    # Calculate Cumulative Volume
    df['Cumulative_Volume'] = df['volume'].rolling(window=N).sum()
    
    # Calculate SMA Volume
    df['SMA_Volume'] = df['volume'].rolling(window=N).mean()
    
    # Adjust Momentum Score by Volume
    df['Volume_Factor'] = df['Cumulative_Volume'] / df['SMA_Volume']
    df['Momentum_Score_Adjusted_Volume'] = df['Momentum_Score'] * df['Volume_Factor']
    
    # Calculate True Range
    df['True_Range'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df.shift(1).loc[x.name, 'close']), abs(x['low'] - df.shift(1).loc[x.name, 'close'])), axis=1)
    
    # Calculate Average True Range (ATR)
    df['ATR'] = df['True_Range'].rolling(window=N).mean()
    
    # Adjust Momentum Score by Volatility
    df['Momentum_Score_Adjusted_Volatility'] = df['Momentum_Score_Adjusted_Volume'] / df['ATR']
    
    return df['Momentum_Score_Adjusted_Volatility'].dropna()

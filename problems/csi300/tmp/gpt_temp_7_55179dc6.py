import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 5-day and 20-day EMA of Close, High, and Low
    df['EMA_5_Close'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_5_High'] = df['high'].ewm(span=5, adjust=False).mean()
    df['EMA_5_Low'] = df['low'].ewm(span=5, adjust=False).mean()
    
    df['EMA_20_Close'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_20_High'] = df['high'].ewm(span=20, adjust=False).mean()
    df['EMA_20_Low'] = df['low'].ewm(span=20, adjust=False).mean()
    
    # Calculate momentum
    df['Momentum_Close'] = df['EMA_5_Close'] - df['EMA_20_Close']
    df['Momentum_High'] = df['EMA_5_High'] - df['EMA_20_High']
    df['Momentum_Low'] = df['EMA_5_Low'] - df['EMA_20_Low']
    
    # Combine Close, High, and Low momentum
    df['Combined_Momentum'] = df['Momentum_Close'] + (df['Momentum_High'] - df['Momentum_Low'])
    
    # Calculate 5-day and 20-day EMA of Volume
    df['EMA_5_Volume'] = df['volume'].ewm(span=5, adjust=False).mean()
    df['EMA_20_Volume'] = df['volume'].ewm(span=20, adjust=False).mean()
    
    # Adjust for Volume Trends
    df['Volume_Trend_Adjustment'] = df.apply(lambda row: 1.5 if row['EMA_5_Volume'] > row['EMA_20_Volume'] else 0.7, axis=1)
    df['Adjusted_Momentum'] = df['Combined_Momentum'] * df['Volume_Trend_Adjustment']
    
    # Calculate True Range
    df['True_Range'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2].shift(1)), abs(x[1] - x[2].shift(1))), axis=1)
    
    # Calculate ATR
    df['ATR'] = df['True_Range'].ewm(span=14, adjust=False).mean()
    
    # Adjust Combined Momentum by ATR
    df['Momentum_ATR_Adjusted'] = df['Adjusted_Momentum'] / df['ATR']
    
    # Calculate Daily Volume Change
    df['Volume_Change'] = df['volume'] / df['volume'].shift(1)
    
    # Adjust by Volume Change and Volume EMA
    df['Volume_EMA_14'] = df['volume'].ewm(span=14, adjust=False).mean()
    df['Momentum_Volume_Adjusted'] = df['Momentum_ATR_Adjusted'] * df['Volume_Change'] * df['Volume_EMA_14']
    
    # Incorporate Enhanced Price Gaps and Reversal Sensitivity
    df['Open_to_Close_Gap'] = df['open'] - df['close'].shift(1)
    df['High_to_Low_Gap'] = df['high'] - df['low'].shift(1)
    df['Total_Gap'] = df['Open_to_Close_Gap'] + df['High_to_Low_Gap']
    
    df['Weighted_High_Low_Spread'] = (df['high'] - df['low']) * df['volume']
    df['Weighted_Open_Close_Spread'] = (df['open'] - df['close']) * df['volume']
    
    df['Weighted_Spreads'] = df['Weighted_High_Low_Spread'] + df['Weighted_Open_Close_Spread']
    
    # Combine with Volume-Adjusted Momentum
    df['Momentum_with_Gaps'] = df['Momentum_Volume_Adjusted'] + df['Total_Gap'] + df['Weighted_Spreads']
    
    # Detect Volume Spikes
    df['Volume_Spike'] = df['volume'] / df['volume'].shift(1)
    
    # Final Alpha Factor
    df['Alpha_Factor'] = df['Momentum_with_Gaps'] + df['Volume_Spike'] - df['Weighted_Spreads']
    
    return df['Alpha_Factor']

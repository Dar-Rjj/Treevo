import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 5-day and 20-day Simple Moving Averages (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # SMA Crossover
    df['SMA_Crossover'] = 0
    df.loc[(df['SMA_5'] > df['SMA_20']) & (df['SMA_5'].shift(1) <= df['SMA_20'].shift(1)), 'SMA_Crossover'] = 1
    df.loc[(df['SMA_5'] < df['SMA_20']) & (df['SMA_5'].shift(1) >= df['SMA_20'].shift(1)), 'SMA_Crossover'] = -1
    
    # Price Rate of Change over 10 days
    df['Price_Rate_of_Change'] = df['close'].pct_change(periods=10)
    
    # Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['amount'] / df['volume']).cumsum() / (df['volume'].cumsum())
    df['VWAP_Factor'] = 0
    df.loc[df['close'] > df['VWAP'], 'VWAP_Factor'] = 1
    df.loc[df['close'] < df['VWAP'], 'VWAP_Factor'] = -1
    
    # Volume Trend
    df['Volume_Trend'] = df['volume'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Relative Strength (RS)
    df['Relative_Strength'] = df['close'] / df['close'].shift(20)
    
    # Daily Price Range
    df['Daily_Price_Range'] = df['high'] - df['low']
    df['Avg_Daily_Price_Range'] = df['Daily_Price_Range'].rolling(window=20).mean()
    df['Range_Factor'] = 0
    df.loc[df['Daily_Price_Range'] > df['Avg_Daily_Price_Range'], 'Range_Factor'] = 1
    df.loc[df['Daily_Price_Range'] < df['Avg_Daily_Price_Range'], 'Range_Factor'] = -1
    
    # True Range
    df['True_Range'] = df[['high' - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))]].max(axis=1)
    df['True_Range_MA'] = df['True_Range'].rolling(window=20).mean()
    df['True_Range_Factor'] = df['True_Range'] / df['True_Range_MA']
    
    # Amount-to-Volume Ratio
    df['Amount_to_Volume_Ratio'] = df['amount'] / df['volume']
    df['Avg_Amount_to_Volume_Ratio'] = df['Amount_to_Volume_Ratio'].rolling(window=20).mean()
    df['Amount_to_Volume_Factor'] = 0
    df.loc[df['Amount_to_Volume_Ratio'] > df['Avg_Amount_to_Volume_Ratio'], 'Amount_to_Volume_Factor'] = 1
    df.loc[df['Amount_to_Volume_Ratio'] < df['Avg_Amount_to_Volume_Ratio'], 'Amount_to_Volume_Factor'] = -1
    
    # Combine factors
    df['Alpha_Factor'] = (
        df['SMA_Crossover'] + 
        df['Price_Rate_of_Change'] + 
        df['VWAP_Factor'] + 
        df['Volume_Trend'] + 
        df['Relative_Strength'] + 
        df['Range_Factor'] + 
        df['True_Range_Factor'] + 
        df['Amount_to_Volume_Factor']
    )
    
    return df['Alpha_Factor']

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics(df, N=20):
    # Calculate Close Price Momentum
    df['Close_Momentum'] = df['close'].diff(periods=N)
    
    # Identify Directional Days
    df['Direction'] = np.where(df['close'] > df['open'], 'Up', 'Down')
    df['Up_Count'] = df['Direction'].rolling(window=N).apply(lambda x: (x == 'Up').sum(), raw=True)
    df['Down_Count'] = df['Direction'].rolling(window=N).apply(lambda x: (x == 'Down').sum(), raw=True)
    df['Net_Directional_Count'] = df['Up_Count'] - df['Down_Count']
    
    # Weight by Volume and Amount
    df['Volume_Weighted_Net_Direction'] = df['Net_Directional_Count'] * (df['volume'] + df['amount'])
    
    # Short-Term Price Momentum
    df['Short_Term_Momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Long-Term Price Momentum
    df['Long_Term_Momentum'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Volume-Weighted Average Prices
    df['Daily_Avg_Price'] = (df['high'] + df['low'] + df['close']) / 3
    df['Volume_Weighted_Avg_Price'] = df['Daily_Avg_Price'] * df['volume']
    df['Sum_Volume_Weighted_Avg_Price'] = df['Volume_Weighted_Avg_Price'].rolling(window=N).sum()
    
    # Volume-Weighted Median Prices
    df['Daily_Median_Price'] = df[['high', 'low', 'close']].median(axis=1)
    df['Volume_Weighted_Median_Price'] = df['Daily_Median_Price'] * df['volume']
    df['Sum_Volume_Weighted_Median_Price'] = df['Volume_Weighted_Median_Price'].rolling(window=N).sum()
    
    # Combined Volume-Weighted Moving Average
    df['Combined_VW_MA'] = (df['Sum_Volume_Weighted_Avg_Price'] + df['Sum_Volume_Weighted_Median_Price']) / 2
    df['Total_Volume'] = df['volume'].rolling(window=N).sum()
    df['Combined_VW_MA'] = df['Combined_VW_MA'] / df['Total_Volume']
    
    # Current Day's Volume-Weighted Price
    df['Current_VW_Price'] = df['Daily_Avg_Price'] * df['volume']
    
    # VWPTI
    df['VWPTI'] = (df['Current_VW_Price'] - df['Combined_VW_MA']) / df['Combined_VW_MA']
    
    # Volume Trend
    df['Volume_Trend'] = df['volume'].rolling(window=5).mean()
    
    # Trading Activity Indicator
    df['Trading_Activity_Indicator'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Volatility Component
    df['Volatility'] = df['close'].rolling(window=20).std()
    
    # Daily Price Return
    df['Daily_Return'] = df['close'] / df['close'].shift(1)
    
    # Volume Shock Factor
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Volume_Long_Term_Avg'] = df['volume'].rolling(window=21).mean()
    df['Volume_Shock_Factor'] = df['Volume_Change'] / df['Volume_Long_Term_Avg']
    
    # Adjust Combined Factors
    df['Adjusted_Combined_Factor'] = (df['Current_VW_Price'] * df['Daily_Return']) * (1 + df['Volume_Shock_Factor'])
    
    # Integrate VWPTI
    df['Integrated_Factor'] = df['Adjusted_Combined_Factor'] * df['VWPTI']
    
    # High-Low Range Momentum
    df['High_Low_Range_Momentum'] = (df['high'] - df['low']) / (df['high'].shift(N) - df['low'].shift(N))
    
    # Include High-Low Range Momentum
    df['Final_Factor'] = df['Integrated_Factor'] + df['High_Low_Range_Momentum']
    
    return df['Final_Factor']

# Example usage:
# df = pd.DataFrame(...)  # Load your data into a DataFrame
# factor = heuristics(df)
# print(factor)

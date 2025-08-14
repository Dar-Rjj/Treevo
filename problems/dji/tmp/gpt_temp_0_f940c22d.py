import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Relative Strength and Intraday Range
    df['Momentum'] = df['Close'] / df['Close'].shift(10)
    df['Intraday_Range'] = df['High'] - df['Low']
    
    # Measure Volume Activity Change
    m = 5
    df['Volume_Change'] = (df['Volume'].rolling(window=m).sum() / m) - df['Volume'].shift(1)
    
    # Combine Relative Strength, Volume Change, and Intraday Range
    df['Combined_Factor'] = df['Momentum'] * df['Volume_Change']
    df['Volume_Anomaly'] = df['Volume_Change']
    df['Adjusted_Intraday_Range'] = df['Intraday_Range'] * (1 + df['Volume_Anomaly'] / df['Volume'])

    # Calculate Daily Price Momentum
    df['Daily_Price_Momentum'] = df['Close'] - df['Close'].shift(1)
    
    # Calculate Short-Term and Long-Term Trends
    df['5d_EMA'] = df['Daily_Price_Momentum'].ewm(span=5, adjust=False).mean()
    df['20d_EMA'] = df['Daily_Price_Momentum'].ewm(span=20, adjust=False).mean()
    df['5d_STD'] = df['Daily_Price_Momentum'].rolling(window=5).std()
    df['20d_STD'] = df['Daily_Price_Momentum'].rolling(window=20).std()
    df['Short_Term_Volatility_Adjusted'] = df['5d_EMA'] / df['5d_STD']
    df['Long_Term_Volatility_Adjusted'] = df['20d_EMA'] / df['20d_STD']
    
    # Generate Volume Synchronized Oscillator
    df['VSO'] = (df['Short_Term_Volatility_Adjusted'] - df['Long_Term_Volatility_Adjusted']) * df['Volume']
    
    # Incorporate Price Movement Intensity
    df['Price_Movement_Intensity'] = (df['High'] - df['Low']) + (df['Close'] - df['Open'])
    
    # Adjust Momentum by Inverse of Volatility
    df['True_Range'] = df['Intraday_Range']
    df['Adjusted_Momentum'] = df['Momentum'] / df['True_Range']
    
    # Identify Volume Spikes
    df['Volume_Change'] = df['Volume'] - df['Volume'].shift(1)
    df['Spike_Threshold'] = df['Volume'].rolling(window=20).median() * 2
    df['Spike_Indicator'] = (df['Volume_Change'] > df['Spike_Threshold']).astype(int)
    
    # Adjust Cumulative Moving Difference by Volume-Weighted Average
    df['Cumulative_High_Low_Diff'] = df['Intraday_Range'].rolling(window=20).sum()
    df['Volume_Weighted_Average'] = (df['Volume'] * df['Close']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    df['Adjusted_Cumulative_Moving_Difference'] = df['Cumulative_High_Low_Diff'] * df['Volume_Weighted_Average'] * df['Spike_Indicator']
    
    # Synthesize Final Alpha Factor
    df['Alpha_Factor'] = (
        df['Combined_Factor'] * 
        df['Adjusted_Cumulative_Moving_Difference'] * 
        df['Adjusted_Intraday_Range'] + 
        (df['Close'] - df['Open']) * 
        df['Adjusted_Momentum'] * 
        df['VSO']
    )
    
    return df['Alpha_Factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)

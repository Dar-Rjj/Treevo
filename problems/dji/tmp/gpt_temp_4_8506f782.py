import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Relative Strength
    n = 5
    df['Relative_Strength'] = df['close'] / df['close'].shift(n)
    
    # Measure Volume Activity Change
    m = 10
    df['Volume_Change'] = df['volume'] - df['volume'].rolling(window=m).mean()
    
    # Combine Relative Strength and Volume Change
    df['Combined_Factor'] = df['Relative_Strength'] * df['Volume_Change']
    
    # Calculate Daily Price Momentum
    df['Daily_Price_Momentum'] = df['close'] - df['close'].shift(1)
    
    # Calculate Short-Term Trend
    df['Short_Term_EMA'] = df['Daily_Price_Momentum'].ewm(span=5, adjust=False).mean()
    df['Short_Term_Volatility'] = df['Daily_Price_Momentum'].rolling(window=5).std()
    df['Short_Term_Trend'] = df['Short_Term_EMA'] / df['Short_Term_Volatility']
    
    # Calculate Long-Term Trend
    df['Long_Term_EMA'] = df['Daily_Price_Momentum'].ewm(span=20, adjust=False).mean()
    df['Long_Term_Volatility'] = df['Daily_Price_Momentum'].rolling(window=20).std()
    df['Long_Term_Trend'] = df['Long_Term_EMA'] / df['Long_Term_Volatility']
    
    # Generate Volume Synchronized Oscillator
    df['Volume_Synchronized_Oscillator'] = (df['Short_Term_Trend'] - df['Long_Term_Trend']) * df['volume']
    
    # Calculate Daily High-Low Difference
    df['High_Low_Diff'] = df['high'] - df['low']
    
    # Cumulate the Moving Difference
    df['Cumulative_High_Low_Diff'] = df['High_Low_Diff'].rolling(window=20).sum()
    
    # Calculate Volume Trend
    df['Volume_Trend'] = df['volume'].rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    
    # Adjust Cumulative Moving Difference by Volume Trend
    df['Adjusted_Cumulative_Diff'] = df['Cumulative_High_Low_Diff'] * df['Volume_Trend']
    
    # Calculate Intraday Range
    df['Intraday_Range'] = df['high'] - df['low']
    
    # Apply Weighted Volume Adjustment
    weighted_avg_volume = df['volume'].rolling(window=21).mean()
    volume_anomaly = df['volume'] - weighted_avg_volume
    df['Adjusted_Intraday_Range'] = df['Intraday_Range'] * (1 + volume_anomaly / weighted_avg_volume)
    
    # Incorporate Price Oscillation
    df['High_Low_Range'] = df['high'] - df['low']
    df['Open_Close_Spread'] = df['close'] - df['open']
    df['Price_Oscillation'] = df['High_Low_Range'] + df['Open_Close_Spread']
    
    # Adjust Momentum by Inverse of Volatility
    df['True_Range'] = df[['high' - 'low', 'high' - df['close'].shift(1), df['close'].shift(1) - 'low']].max(axis=1)
    df['Momentum_Adjusted'] = df['Daily_Price_Momentum'] / df['True_Range']
    
    # Integrate Combined Factors
    df['Integrated_Factor'] = (
        df['Combined_Factor'] * 
        df['Adjusted_Cumulative_Diff'] * 
        df['Adjusted_Intraday_Range'] * 
        df['Price_Oscillation'] * 
        df['Momentum_Adjusted'] * 
        df['Volume_Synchronized_Oscillator']
    )
    
    return df['Integrated_Factor']

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum
    momentum_lookback = 30
    df['Price_Momentum'] = df['close'].diff(momentum_lookback)
    
    # Calculate Volume Acceleration
    volume_lookback = 14
    df['Volume_Acceleration'] = df['volume'].diff(volume_lookback)
    
    # Combine Momentum and Volume
    df['Intermediate_Factor'] = df['Price_Momentum'] + df['Volume_Acceleration']
    
    # Calculate High-to-Low Range
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Adjust for Volume
    df['Adjusted_High_Low_Range'] = df['High_Low_Range'] * np.sqrt(df['volume'])
    
    # Detect Volume Spike
    df['Volume_5D_MA'] = df['volume'].rolling(window=5).mean()
    df['Volume_Spike'] = (df['volume'] > 2 * df['Volume_5D_MA']).astype(int)
    
    # Calculate Volume Trend
    df['Volume_20D_MA'] = df['volume'].rolling(window=20).mean()
    df['Volume_Trend'] = np.where(df['volume'] > df['Volume_20D_MA'], 1, -1)
    
    # Calculate Intraday Return
    df['Intraday_Return'] = df['high'] / df['low']
    
    # Calculate Intraday Volatility
    df['Intraday_Volatility'] = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2)
    
    # Adjust Intraday Return
    df['Intraday_Volatility_Adjusted'] = df['Intraday_Return'] / df['Intraday_Volatility']
    
    # Calculate Volume Displacement
    df['Volume_Displacement'] = df['volume'] - df['volume'].shift(1)
    
    # Combine Intraday Factors
    df['Intraday_Factors'] = df['Intraday_Volatility_Adjusted'] * df['Volume_Displacement']
    
    # Calculate Trading Range Expansion
    trading_range_lookback = 30
    df['True_Range'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['Average_True_Range'] = df['True_Range'].rolling(window=trading_range_lookback).mean()
    
    # Final Combination
    df['Factor'] = (df['Adjusted_High_Low_Range'] * df['Volume_Trend'] + df['Intermediate_Factor']) * (df['Volume_Spike'] * 2 + 1) + df['Intraday_Factors'] + df['Volume_Acceleration'] + df['Average_True_Range']
    
    return df['Factor']

# Example usage:
# df = pd.read_csv('your_stock_data.csv', parse_dates=['date'], index_col='date')
# factor_series = heuristics_v2(df)
# print(factor_series)

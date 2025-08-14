import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Adjusted High-Low Spread
    df['High_Low_Spread'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['close'] / df['open'] - 1
    
    # Apply Conditional Weight to High-Low Spread
    positive_return_weight = 1.5
    negative_return_weight = 0.5
    df['Weighted_High_Low_Spread'] = df['High_Low_Spread'] * np.where(df['Close_to_Open_Return'] > 0, positive_return_weight, negative_return_weight)
    
    # Integrate Volume Weighted Average Price (VWAP)
    df['VWAP'] = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    df['Cumulative_Volume'] = df['volume'].cumsum()
    df['Cumulative_VWAP'] = df['VWAP'].cumsum()
    df['VWAP'] = df['Cumulative_VWAP'] / df['Cumulative_Volume']
    
    # Combine Intraday Indicators with Spread
    df['Intraday_Indicator'] = df['Weighted_High_Low_Spread'] + df['VWAP']
    
    # Compute Exponential Daily Returns
    df['Daily_Return'] = df['close'] / df['close'].shift(1) - 1
    df['Exponential_Daily_Return'] = df['Daily_Return'].ewm(span=5, adjust=False).mean()
    
    # Short-Term and Long-Term Exponential Moving Average of Daily Returns
    df['Short_Term_EMA'] = df['Exponential_Daily_Return'].ewm(span=5, adjust=False).mean()
    df['Long_Term_EMA'] = df['Exponential_Daily_Return'].ewm(span=20, adjust=False).mean()
    
    # Compute Dynamic Difference
    df['Dynamic_Difference'] = df['Short_Term_EMA'] - df['Long_Term_EMA']
    
    # Calculate Intraday Price Movement
    df['Intraday_Price_Movement'] = df['close'] - df['open']
    
    # Determine Volume Increase from Average
    df['5D_MA_Volume'] = df['volume'].rolling(window=5).mean()
    df['Volume_Increase'] = df['volume'] - df['5D_MA_Volume']
    
    # Weight Intraday Price Movement by Volume Spike
    df['Weighted_Intraday_Price_Movement'] = df['Intraday_Price_Movement'] * abs(df['Volume_Increase'])
    
    # Weight Close-to-Open Return by Volume
    df['Weighted_Close_to_Open_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Identify Significant Daily Range
    df['Daily_Range'] = df['high'] - df['low']
    
    # Integrate Price Volatility
    df['5D_MA_Range'] = df['Daily_Range'].rolling(window=5).mean()
    df['Price_Volatility'] = df['Weighted_High_Low_Spread'] / df['5D_MA_Range']
    
    # Calculate Rolling Momentum and Volatility
    df['Rolling_Momentum'] = df['Daily_Return'].rolling(window=10).sum()
    df['Rolling_Volatility'] = np.sqrt(df['Daily_Return'].pow(2).rolling(window=10).sum())
    
    # Incorporate High-Frequency Trading Signals
    df['Triple_Exponential_HLC'] = df[['high', 'low', 'close']].apply(lambda x: x.ewm(span=5, adjust=False).mean().ewm(span=5, adjust=False).mean().ewm(span=5, adjust=False).mean(), axis=1)
    df['Volume_Shock'] = df['volume'] - df['5D_MA_Volume']
    df['Weighted_HLC'] = df['Triple_Exponential_HLC'] * df['Volume_Shock']
    
    # Combine All Components
    df['Factor'] = (
        df['Intraday_Indicator'] +
        df['Weighted_Intraday_Price_Movement'] +
        df['Price_Volatility'] +
        df['Weighted_Close_to_Open_Return'] +
        (df['Rolling_Momentum'] * df['Rolling_Volatility']) +
        df['Dynamic_Difference'] +
        df['Weighted_HLC']
    )
    
    return df['Factor'].dropna()

# Example usage:
# factor_values = heuristics_v2(df)

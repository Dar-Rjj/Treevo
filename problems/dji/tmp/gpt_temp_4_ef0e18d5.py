import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the difference between today's close price and yesterday's close price
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Simple Moving Average of Daily Returns over 5 days
    df['5_day_SMA_Return'] = df['Daily_Return'].rolling(window=5).mean()
    
    # Deviation Factor: Today’s Return minus 5-day SMA
    df['Deviation_Factor'] = df['Daily_Return'] - df['5_day_SMA_Return']
    
    # Standard Deviation of Daily Returns over 20 days
    df['20_day_STD_Return'] = df['Daily_Return'].rolling(window=20).std()
    
    # Modified Momentum-to-Volatility Ratio
    df['Modified_Momentum_Ratio'] = df['Deviation_Factor'] / df['20_day_STD_Return']
    
    # Threshold for Positive and Negative Momentum
    positive_threshold = 1.0
    negative_threshold = -1.0
    df['Momentum_Flag'] = 0
    df.loc[df['Modified_Momentum_Ratio'] > positive_threshold, 'Momentum_Flag'] = 1
    df.loc[df['Modified_Momentum_Ratio'] < negative_threshold, 'Momentum_Flag'] = -1
    
    # Accumulation Distribution Line (ADL)
    df['ADL'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['ADL'] = df['ADL'].cumsum()
    
    # Change in ADL over 10 days
    df['ADL_Change_10_days'] = df['ADL'] - df['ADL'].shift(10)
    
    # Adjusted ADL Change by 10-day ATR
    df['10_day_ATR'] = df['High'].rolling(window=10).max() - df['Low'].rolling(window=10).min()
    df['Adjusted_ADL_Change'] = df['ADL_Change_10_days'] / df['10_day_ATR']
    
    # Binary Indicator for Adjusted ADL Change
    df['ADL_Binary'] = 0
    df.loc[df['Adjusted_ADL_Change'] > 0, 'ADL_Binary'] = 1
    df.loc[df['Adjusted_ADL_Change'] < 0, 'ADL_Binary'] = -1
    
    # On-Balance Volume (OBV)
    df['OBV'] = (df['Close'] > df['Close'].shift(1)).astype(int) * df['Volume']
    df['OBV'] = df['OBV'].cumsum()
    
    # 14-day OBV Slope
    df['14_day_OBV_Slope'] = df['OBV'].rolling(window=14).apply(lambda x: (x[-1] - x[0]) / 14)
    
    # Volume-Adjusted OBV Slope
    df['14_day_AVG_Volume'] = df['Volume'].rolling(window=14).mean()
    df['Volume_Adjusted_OBV_Slope'] = df['14_day_OBV_Slope'] / df['14_day_AVG_Volume']
    
    # Directional Indicator for Volume-Adjusted OBV Slope
    df['OBV_Slope_Directional'] = 0
    df.loc[df['Volume_Adjusted_OBV_Slope'] > 0, 'OBV_Slope_Directional'] = 1
    df.loc[df['Volume_Adjusted_OBV_Slope'] < 0, 'OBV_Slope_Directional'] = -1
    
    # Weighted High-Low Spread by Volume
    df['Weighted_High_Low_Spread'] = (df['High'] - df['Low']) * df['Volume']
    
    # Intraday Volatility Momentum
    df['Intraday_Volatility_Momentum'] = (df['Close'] - df['Open']) / df['Open']
    
    # Combine All Components
    df['Alpha_Factor'] = (
        df['Deviation_Factor'] +
        df['Adjusted_ADL_Change'] +
        df['Volume_Adjusted_OBV_Slope'] +
        df['Weighted_High_Low_Spread'] +
        df['Intraday_Volatility_Momentum']
    )
    
    return df['Alpha_Factor']

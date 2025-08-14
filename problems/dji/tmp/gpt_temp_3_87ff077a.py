import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Daily Return Deviation
    df['Daily_Return'] = df['Close'].diff()
    df['5_day_SMA_Daily_Return'] = df['Daily_Return'].rolling(window=5).mean()
    df['Daily_Return_Deviation'] = df['Daily_Return'] - df['5_day_SMA_Daily_Return']

    # Adjusted Momentum-to-Volatility Ratio
    df['20_day_STD_Daily_Return'] = df['Daily_Return'].rolling(window=20).std()
    df['Adjusted_Momentum_to_Volatility'] = df['Daily_Return_Deviation'] / df['20_day_STD_Daily_Return']

    # Buy/Sell Indicator based on Momentum-to-Volatility
    threshold = 1.0
    df['Momentum_Indicator'] = np.where(df['Adjusted_Momentum_to_Volatility'] > threshold, 1,
                                        np.where(df['Adjusted_Momentum_to_Volatility'] < -threshold, -1, 0))

    # Accumulation Distribution Line (ADL) Analysis
    money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    money_flow_volume = money_flow_multiplier * df['Volume']
    df['ADL'] = money_flow_volume.cumsum()
    df['ADL_Change'] = df['ADL'] - df['ADL'].shift(10)
    df['10_day_ATR'] = df[['High', 'Low']].apply(lambda x: x.diff().abs().max(), axis=1).rolling(window=10).mean()
    df['ADL_Sentiment'] = df['ADL_Change'] / df['10_day_ATR']
    df['ADL_Sentiment_Indicator'] = np.where(df['ADL_Sentiment'] > 0, 1, -1)

    # On-Balance Volume (OBV) Trend
    df['OBV'] = df['Volume'].copy()
    df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['OBV'] + df['Volume'], 
                         np.where(df['Close'] < df['Close'].shift(1), df['OBV'] - df['Volume'], df['OBV']))
    df['14_day_OBV_Slope'] = df['OBV'].diff(14) / 14
    df['14_day_MAV'] = df['Volume'].rolling(window=14).mean()
    df['OBV_Directional_Indicator'] = np.where(df['14_day_OBV_Slope'] / df['14_day_MAV'] > 0, 1, -1)

    # High-Low Spread Weighted by Volume
    df['High_Low_Spread'] = (df['High'] - df['Low']) * df['Volume']
    positive_return_weight = 1.5
    negative_return_weight = 0.5
    df['Weighted_High_Low_Spread'] = np.where(df['Close'] > df['Open'], df['High_Low_Spread'] * positive_return_weight, 
                                              df['High_Low_Spread'] * negative_return_weight)

    # Intraday Price Movement and Volume Spike
    df['Intraday_Price_Movement'] = df['Close'] - df['Open']
    df['Volume_Spike'] = df['Volume'] - df['Volume'].rolling(window=5).mean()
    df['Adjusted_Intraday_Movement'] = abs(df['Volume'] - df['Volume'].rolling(window=5).mean()) * df['Intraday_Price_Movement']

    # Close-to-Open Return
    df['Close_to_Open_Return'] = (df['Close'] / df['Open']) - 1

    # Combine Components
    df['Alpha_Factor'] = (
        df['Momentum_Indicator'] +
        df['ADL_Sentiment_Indicator'] +
        df['OBV_Directional_Indicator'] +
        df['Weighted_High_Low_Spread'] +
        df['Adjusted_Intraday_Movement'] +
        df['Volume_Spike'] +
        df['Close_to_Open_Return']
    )

    return df['Alpha_Factor']

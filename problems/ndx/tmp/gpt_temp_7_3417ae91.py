import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Volume-Weighted High-Low Price Difference
    df['Volume_Weighted_High_Low'] = (df['High'] - df['Low']) * df['Volume']

    # Calculate Deviation of Current Intraday High-Low Spread from Historical Average
    rolling_avg_high_low = df['High'] - df['Low']
    df['Deviation_High_Low_Spread'] = (df['High'] - df['Low']) - rolling_avg_high_low.rolling(window=5).mean()

    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = ((df['Open'] + df['High'] + df['Low'] + df['Close']) * df['Volume']).cumsum() / df['Volume'].cumsum()

    # Calculate Deviation of VWAP from Close
    df['Deviation_VWAP_Close'] = df['VWAP'] - df['Close']

    # Incorporate Volume Impact Factor
    df['Price_Change'] = df['Close'].diff()
    df['Volume_Impact_Factor'] = df['Volume'] * abs(df['Price_Change'])

    # Integrate Historical High-Low Range and Momentum Contributions
    df['Volume_Weighted_High_Low_5D'] = df['Volume_Weighted_High_Low'].rolling(window=5).sum()
    df['Price_Change_10D'] = df['Close'].pct_change(10)
    df['Momentum_Contribution'] = df['Price_Change_10D'] * df['Volume_Weighted_High_Low_5D']
    df['Momentum_Accumulated'] = df['Momentum_Contribution'].where(df['Price_Change_10D'] > 0, 0).rolling(window=5).sum()

    # Evaluate Overnight Sentiment
    df['Log_Volume'] = np.log(df['Volume'])
    df['Overnight_Return'] = np.log(df['Open'] / df['Close'].shift(1))

    # Integrate Intraday and Overnight Signals
    df['Intraday_Return'] = (np.log(df['High'] / df['Low']) + np.log(df['Close'] / df['Open'])) / 2
    df['Intraday_Overnight_Diff'] = df['Intraday_Return'] - df['Overnight_Return']
    df['Volume_Adjusted_Indicator'] = df['Volume'].rolling(window=5).mean() - df['Volume']
    df['Integrated_Intraday_Overnight'] = df['Intraday_Overnight_Diff'] * df['Volume_Adjusted_Indicator']

    # Generate Alpha Factor
    df['Alpha_Factor'] = df['Deviation_High_Low_Spread'] + df['Deviation_VWAP_Close'] + df['Volume_Impact_Factor'] * df['Close']
    df['Alpha_Factor'] = df['Alpha_Factor'] + df['Momentum_Accumulated'] + df['Integrated_Intraday_Overnight']

    # Synthesize Overall Alpha Factor
    df['Overall_Alpha_Factor'] = df['Alpha_Factor'] + (df['Volume'].rolling(window=5).mean() - df['Volume'])

    return df['Overall_Alpha_Factor']

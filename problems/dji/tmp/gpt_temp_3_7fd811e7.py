import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Daily High-Low Difference
    df['High_Low_Diff'] = df['high'] - df['low']

    # Cumulate the Moving Average of High-Low Differences
    window_size = 10
    df['High_Low_MA'] = df['High_Low_Diff'].rolling(window=window_size, min_periods=1).mean()

    # Calculate Volume-Weighted Close Price
    window_size = 20
    df['Volume_Weighted_Close'] = (df['close'] * df['volume']).rolling(window=window_size, min_periods=1).sum() / df['volume'].rolling(window=window_size, min_periods=1).sum()

    # Adjust Cumulative Moving Average by Volume-Weighted Close Price
    df['Adjusted_MA'] = df['High_Low_MA'] * df['Volume_Weighted_Close']

    # Calculate Daily Price Momentum
    df['Price_Momentum'] = df['close'] - df['close'].shift(1)

    # Calculate Short-Term Adjusted Momentum
    short_window = 5
    df['Short_Term_EMA'] = df['Price_Momentum'].ewm(span=short_window, adjust=False).mean()
    df['Short_Term_Std'] = df['Price_Momentum'].rolling(window=short_window, min_periods=1).std()
    df['Short_Term_Adjusted_Momentum'] = df['Short_Term_EMA'] / df['Short_Term_Std']

    # Calculate Long-Term Adjusted Momentum
    long_window = 20
    df['Long_Term_EMA'] = df['Price_Momentum'].ewm(span=long_window, adjust=False).mean()
    df['Long_Term_Std'] = df['Price_Momentum'].rolling(window=long_window, min_periods=1).std()
    df['Long_Term_Adjusted_Momentum'] = df['Long_Term_EMA'] / df['Long_Term_Std']

    # Generate Volume Synchronized Oscillator
    df['Volume_Synchronized_Oscillator'] = (df['Long_Term_Adjusted_Momentum'] - df['Short_Term_Adjusted_Momentum']) * df['volume']

    # Incorporate Price Movement Intensity
    df['High_Low_Range'] = df['high'] - df['low']
    df['Open_Close_Spread'] = df['close'] - df['open']
    df['Price_Movement_Intensity'] = df['High_Low_Range'] + df['Open_Close_Spread']

    # Volume-Weighted Close Price Trend
    trend_window = 30
    df['Volume_Weighted_Close_Trend'] = df['Volume_Weighted_Close'].rolling(window=trend_window, min_periods=1).apply(lambda x: linregress(range(trend_window), x).slope, raw=True)

    # Construct Comprehensive Alpha Factor
    df['Alpha_Factor'] = (df['Adjusted_MA'] * 
                          df['Volume_Weighted_Close_Trend'] * 
                          df['Price_Momentum'] * 
                          df['Volume_Synchronized_Oscillator'] * 
                          df['Price_Movement_Intensity'])

    # Introduce a new component: Relative Strength Indicator (RSI)
    rsi_window = 14
    gain = df['close'].diff().clip(lower=0)
    loss = -df['close'].diff().clip(upper=0)
    avg_gain = gain.rolling(window=rsi_window, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Alpha_Factor'] *= df['RSI']

    return df['Alpha_Factor']

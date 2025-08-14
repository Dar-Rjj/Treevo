import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Reversal
    df['Intraday_High_Low_Spread'] = df['high'] - df['low']
    df['Close_to_Open_Return'] = (df['close'] / df['open']) - 1
    df['Intraday_Reversal'] = 1 - (df['Close_to_Open_Return'] / df['Intraday_High_Low_Spread'])

    # Incorporate Volume Influence
    n_days = 20
    df['Average_Volume'] = df['volume'].rolling(window=n_days).mean()
    df['Intraday_Volume_Impact'] = df['volume'] / df['Average_Volume']
    df['Weighted_Intraday_Reversal'] = df['Intraday_Reversal'] * df['Intraday_Volume_Impact']

    # Calculate Daily Log Returns
    df['Daily_Log_Return'] = np.log(df['close'] / df['close'].shift(1))

    # Compute Daily Volatility
    df['Daily_Volatility'] = df['Daily_Log_Return'].rolling(window=20).std()

    # Calculate Intraday Volatility
    n_days_intraday_vol = 5
    df['Intraday_Volatility'] = df['Intraday_High_Low_Spread'].rolling(window=n_days_intraday_vol).sum()

    # Adjust Close-to-Close Return by Intraday Volatility
    df['Close_to_Close_Return'] = df['close'] / df['close'].shift(1) - 1
    df['Adjusted_Close_to_Close_Return'] = df['Close_to_Close_Return'] / df['Intraday_Volatility']

    # Synthesize Combined Factor
    df['Combined_Factor'] = df['Weighted_Intraday_Reversal'] + df['Adjusted_Close_to_Close_Return']

    # Introduce Momentum Component
    momentum_window = 20
    df['Momentum'] = df['close'] / df['close'].shift(momentum_window) - 1
    df['Combined_Factor'] += df['Momentum']

    # Introduce Trend Analysis
    df['200_day_MA'] = df['close'].rolling(window=200).mean()
    df['50_day_MA'] = df['close'].rolling(window=50).mean()
    df['Trend'] = df['50_day_MA'] - df['200_day_MA']
    df['Combined_Factor'] += df['Trend']

    # Drop rows with NaN values
    df.dropna(inplace=True)

    return df['Combined_FFactor']

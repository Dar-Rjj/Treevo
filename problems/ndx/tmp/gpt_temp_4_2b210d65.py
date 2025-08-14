import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Weighted High-Low Momentum
    df['High_Low_Volume'] = (df['high'] - df['low']) * df['volume']
    df['Price_Change'] = df['close'].diff()
    df['Vol_High_Low_Momentum'] = df['High_Low_Volume'] * df['Price_Change']

    # Integrated Daily Momentum
    df['Daily_Price_Change'] = df['close'].pct_change()
    df['Integrated_Momentum'] = df['volume'] * (df['Daily_Price_Change'] ** 2) * (df['open'] - df['close'].shift(1))

    # Historical Momentum
    df['Momentum_Contribution'] = df['Daily_Price_Change'] * df['volume']
    df['Historical_Momentum'] = df['Momentum_Contribution'].rolling(window=5).sum()
    df['Max_Min_Price_Change'] = df['close'].rolling(window=5).max() - df['close'].rolling(window=5).min()
    df['Historical_Momentum'] = df['Historical_Momentum'] * (df['Max_Min_Price_Change'] > 0).astype(int)

    # Market Sentiment
    df['Volatility_Threshold'] = ((df['high'] - df['low']) / df['close']).rolling(window=5).mean()
    df['Market_Sentiment'] = df['Vol_High_Low_Momentum'] * np.where(df['Volatility_Threshold'] > 0, 1 + np.log1p(df['volume']), 1 - np.log1p(df['volume']))

    # Overnight and Intraday Dynamics
    df['Log_Return'] = np.log(df['open'] / df['close'].shift(1))
    df['Intraday_Ratio_High_Low'] = df['high'] / df['low']
    df['Intraday_Ratio_Close_Open'] = df['close'] / df['open']
    df['Overnight_Sentiment'] = np.log1p(df['volume']) * (df['open'] / df['close'].shift(1)) + (df['high'] - df['low'])

    # Intraday Intensity
    df['Intraday_Trading_Range'] = df['high'] - df['low']
    df['Intraday_Volatility'] = df['Intraday_Trading_Range'] * df['volume']
    df['Intraday_Activity'] = df['Intraday_Volatility'] * df['volume'] * (df['high'] - df['low'])

    # Volume Trend and Reversal
    df['Volume_Direction'] = np.where(df['volume'] > df['volume'].shift(1), 1, -1)
    df['Intraday_High_Low_Diff'] = df['high'] - df['low']
    df['Weighted_Reversal_Potential'] = df['Intraday_High_Low_Diff'] * df['Volume_Direction'] * (df['close'] - df['open'])

    # Synthesize Hybrid Signals
    df['Momentum_Adjusted_High_Low'] = df['Vol_High_Low_Momentum'] + df['Market_Sentiment']
    df['Weighted_Intraday_Return'] = (df['Intraday_Ratio_High_Low'] + df['Intraday_Ratio_Close_Open']) / 2
    df['Overnight_Return'] = df['open'] / df['close'].shift(1)
    df['Hybrid_Signal'] = df['Weighted_Intraday_Return'] - df['Overnight_Return']
    df['Recent_Volume_Spike'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['Enhanced_Signal'] = df['Hybrid_Signal'] + df['Intraday_Activity']
    df['Integrated_Signal'] = df['Adjusted_Momentum'] * (df['Enhanced_Signal'] * (df['high'] - df['low']))

    return df['Integrated_Signal']

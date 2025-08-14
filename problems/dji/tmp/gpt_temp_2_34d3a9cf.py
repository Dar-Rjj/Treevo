import pandas as pd
import numpy as np
def heuristics_v2(df):
    # Calculate Simple Moving Averages (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # Calculate Exponential Moving Averages (EMA)
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Calculate High-Low Range
    df['High_Low_Range'] = df['high'] - df['low']

    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # Calculate Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + RS))

    # Calculate Rate of Change (ROC)
    df['ROC_12'] = df['close'].pct_change(periods=12) * 100

    # Calculate Price Rate of Change (PROC)
    df['PROC_12'] = df['close'].pct_change(periods=12) * 100

    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Calculate Volume-Price Trend (VPT)
    df['VPT'] = (df['volume'] * (df['close'] - df['close'].shift(1))).cumsum()

    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    df['True_Range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Average True Range (ATR)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()

    # Calculate Historical Volatility
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

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

    # Calculate Rate of Change (ROC)
    df['ROC_14'] = df['close'].pct_change(periods=14)

    # Calculate MACD
    df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']

    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Calculate Support and Resistance (Pivot Points)
    df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3

    # Calculate Average Directional Index (ADX)
    def calculate_adx(high, low, close, period=14):
        tr = pd.Series(np.max(np.abs([high - low, high - close.shift(), low - close.shift()]), axis=0), index=close.index)
        atr = tr.rolling(window=period).mean()
        up_move = high - high.shift()
        down_move = low.shift() - low
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=close.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=close.index)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        return adx

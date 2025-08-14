import pandas as pd
import numpy as np
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
    df['ROC_14'] = df['close'].pct_change(periods=14) * 100

    # Calculate MACD
    df['MACD_line'] = df['EMA_12'] - df['EMA_26']
    df['Signal_line'] = df['MACD_line'].ewm(span=9, adjust=False).mean()

    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Calculate Pivot Points
    df['Pivot_High'] = df['high'].rolling(window=20).max()
    df['Pivot_Low'] = df['low'].rolling(window=20).min()

    # Calculate ADX
    def calculate_ADX(high, low, close, n=14):
        tr = pd.Series(np.zeros(len(high)), index=high.index)
        tr[1:] = np.max([high - low, np.abs(high - close.shift()), np.abs(low - close.shift())], axis=0)[1:]
        tr_sma = tr.rolling(window=n).mean()
        up = high - high.shift()
        down = low.shift() - low
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=high.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=high.index)
        plus_di = 100 * (plus_dm.rolling(window=n).mean() / tr_sma)
        minus_di = 100 * (minus_dm.rolling(window=n).mean() / tr_sma)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=n).mean()
        return adx

    df['ADX_14'] = calculate_ADX(df['high'], df['low'], df['close'])

    # Calculate Bollinger Bands
    df['BBands_middle'] = df['close'].rolling(window=20).mean()
    df['BBands_std'] = df['close'].rolling(window=20).std()
    df['BBands_upper'] = df['BBands_middle'] + 2 * df['BBands_std']
    df['BBands_lower'] = df['BBands_middle'] - 2 * df['BBands_std']

    # Calculate True Range
    df['TR'] = pd.Series(np.maximum(
        (df['high'] - df['low']),
        np.maximum(
            np.abs(df['high'] - df['close'].shift()),
            np.abs(df['low'] - df['close'].shift())
        )
    ))

    # Combine all the indicators into a single alpha factor
    df['alpha_factor'] = (
        (df['SMA_5'] - df['SMA_20']) / df['SMA_20'] +
        (df['EMA_5'] - df['EMA_20']) / df['EMA_20'] +
        df['ROC_14'] / 100 +
        (df['MACD_line'] - df['Signal_line']) / df['Signal_line'] +
        (df['OBV'] - df['OBV'].shift(1)) / df['OBV'].shift(1) +
        (df['Pivot_High'] - df['Pivot_Low']) / df['Pivot_Low'] +
        df['ADX_14'] / 100 +
        (df['close'] - df['BBands_middle']) / (df['BBands_upper'] - df['BBands_lower']) +
        df['TR'] / df['close']
    )

    return df['alpha_factor']

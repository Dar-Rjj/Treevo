import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import rankdata

def heuristics_v2(df):
    # Calculate Enhanced Intraday Dynamics
    df['Intraday_Return'] = (df['high'] - df['low']) / df['low']
    df['Close_to_Open_Return'] = (df['close'] - df['open']) / df['open']
    df['Daily_Volume_Change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['Daily_Amount_Change'] = (df['amount'] - df['amount'].shift(1)) / df['amount'].shift(1)
    df['Weighted_Returns'] = (np.abs(df['Intraday_Return']) * df['Intraday_Return']) + (np.abs(df['Close_to_Open_Return']) * df['Close_to_Open_Return'])
    df['Adjusted_Interim_Factor'] = df['Weighted_Returns'] * df['Daily_Volume_Change'] * df['Daily_Amount_Change']

    # 3-day Moving Average of Adjusted Interim Factor
    df['MA_3_Adjusted_Interim_Factor'] = df['Adjusted_Interim_Factor'].rolling(window=3).mean()

    # Enhanced Momentum Indicators
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_Cross'] = np.where(df['SMA_5'] > df['SMA_20'], 1, 0)

    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['EMA_Growth'] = df['EMA_10'] / df['EMA_30'] - 1

    delta = df['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    _gain = up.ewm(com=(14 - 1), min_periods=14).mean()
    _loss = down.abs().ewm(com=(14 - 1), min_periods=14).mean()
    RS = _gain / _loss
    df['RSI_14'] = 100 - (100 / (1 + RS))

    # Enhanced Volatility
    df['True_Range'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))), axis=1)
    df['ATR_20'] = df['True_Range'].rolling(window=20).mean()
    df['Volume_Weighted_Price_Change'] = (np.abs(df['close'] - df['close'].shift(1)) * df['volume'])
    df['Volume_Change_Ratio'] = df['volume'] / df['volume'].shift(1)

    # Final Composite Alpha Factor
    df['Momentum_Score'] = df['SMA_Cross'] + df['EMA_Growth'] + df['RSI_14']
    df['Volatility_Score'] = df['ATR_20'] + df['Volume_Weighted_Price_Change'] + df['Volume_Change_Ratio']
    df['Final_Composite_Alpha_Factor'] = (df['MA_3_Adjusted_Interim_Factor'] + df['Momentum_Score'] + df['Volatility_Score']).apply(lambda x: 1 / (1 + np.exp(-x)))

    # Sigmoid Function and Scaling to [-1, 1]
    df['Final_Composite_Alpha_Factor'] = (df['Final_Composite_Alpha_Factor'] - 0.5) * 2

    return df['Final_Composite_Alpha_Factor'].dropna()

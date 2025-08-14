import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Simple Moving Averages (SMA)
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()

    # Calculate Exponential Moving Averages (EMA)
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # Calculate Rate of Change (ROC) over 10 days
    df['ROC_10'] = df['close'].pct_change(periods=10)

    # Calculate MACD
    df['MACD_line'] = df['EMA_12'] - df['EMA_26']
    df['Signal_line'] = df['MACD_line'].ewm(span=9, adjust=False).mean()

    # Calculate True Range
    df['True_Range'] = df[['high', 'low']].diff(axis=1).abs().max(axis=1)

    # Calculate Bollinger Bands
    df['BB_middle'] = df['SMA_20']
    df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()

    # Calculate On Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Calculate Money Flow Index (MFI) over 14 days
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_money_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_money_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    df['MFI'] = 100 - (100 / (1 + (positive_money_flow.rolling(window=14).sum() / negative_money_flow.rolling(window=14).sum())))

    # Calculate Pivot Points
    df['Pivot_Point'] = (df['high'] + df['low'] + df['close']) / 3
    df['R1'] = 2 * df['Pivot_Point'] - df['low']
    df['S1'] = 2 * df['Pivot_Point'] - df['high']

    # Calculate Gaps
    df['Gap'] = df['open'].diff()

    # Opening Range Breakout
    df['Opening_Range_High'] = df['open'].rolling(window=1).max()
    df['Breakout_Bullish'] = df['high'] > df['Opening_Range_High']

    # Closing Range Analysis
    df['Closing_Range_Proportion'] = (df['close'] - df['open']) / (df['high'] - df['low'])

    # Combine all the factors into a single alpha factor
    df['alpha_factor'] = (df['SMA_10'] - df['SMA_200']) + \
                         (df['EMA_10'] - df['EMA_200']) + \
                         df['ROC_10'] + \
                         (df['MACD_line'] - df['Signal_line']) + \
                         df['True_Range'] + \
                         (df['close'] - df['BB_middle']) + \
                         df['OBV'] + \
                         df['MFI'] + \
                         df['Pivot_Point'] + \
                         df['Gap'] + \
                         df['Breakout_Bullish'].astype(int) + \
                         df['Closing_Range_Proportion']

    return df['alpha_factor'].dropna()

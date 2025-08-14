import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate short-term and long-term EMAs
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # EMA crossover factor
    df['EMA_Crossover'] = 0
    df.loc[df['EMA_5'] > df['EMA_20'], 'EMA_Crossover'] = 1
    df.loc[df['EMA_5'] < df['EMA_20'], 'EMA_Crossover'] = -1

    # Price rate of change over 10 days
    df['Price_Rate_of_Change'] = df['close'].pct_change(periods=10)
    df['Smoothed_Price_Rate_of_Change'] = df['Price_Rate_of_Change'].rolling(window=5).mean()

    # Volume Weighted Average Price (VWAP)
    df['Cum_Volume'] = df['volume'].cumsum()
    df['Cum_Amount'] = (df['close'] * df['volume']).cumsum()
    df['VWAP'] = df['Cum_Amount'] / df['Cum_Volume']
    df['VWAP_Factor'] = 0
    df.loc[df['close'] > df['VWAP'], 'VWAP_Factor'] = 1
    df.loc[df['close'] < df['VWAP'], 'VWAP_Factor'] = -1

    # Volume trend
    df['Volume_Trend'] = np.sign(df['volume'] - df['volume'].shift(1))

    # Amount-to-volume ratio
    df['Amount_to_Volume_Ratio'] = df['amount'] / df['volume']
    df['Amount_to_Volume_Ratio_Factor'] = (df['Amount_to_Volume_Ratio'] - df['Amount_to_Volume_Ratio'].rolling(window=20).mean()) / df['Amount_to_Volume_Ratio'].rolling(window=20).std()

    # Relative strength with dynamic lookback window
    df['Volatility'] = df['close'].rolling(window=20).std()
    df['Lookback_Window'] = (20 * df['Volatility'] / df['Volatility'].rolling(window=20).mean()).astype(int).clip(lower=5, upper=60)
    df['RS'] = df['close'] / df['close'].shift(df['Lookback_Window'])

    # Daily price range
    df['Daily_Range'] = df['high'] - df['low']
    df['Average_Range'] = df['Daily_Range'].rolling(window=20).mean()
    df['Range_Factor'] = (df['Daily_Range'] - df['Average_Range']) / df['Average_Range']

    # True Range
    df['True_Range'] = df[['high', 'low', 'close']].diff(axis=1).abs().max(axis=1)
    df['Average_True_Range'] = df['True_Range'].rolling(window=20).mean()
    df['True_Range_Factor'] = df['True_Range'] / df['Average_True_Range']

    # Combine factors using a simple arithmetic operation
    df['Combined_Factor'] = df['EMA_Crossover'] + df['Smoothed_Price_Rate_of_Change'] + df['VWAP_Factor'] + df['Volume_Trend'] + df['Amount_to_Volume_Ratio_Factor'] + df['RS'] + df['Range_Factor'] + df['True_Range_Factor']

    # Introduce a non-linear combination of factors using a polynomial function
    df['Non_Linear_Combined_Factor'] = df['Combined_Factor']**2 + 0.5 * df['Combined_Factor']**3

    return df['Non_Linear_Combined_Factor']

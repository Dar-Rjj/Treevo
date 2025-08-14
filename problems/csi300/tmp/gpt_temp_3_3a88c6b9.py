import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Enhanced Intraday Dynamics
    df['Intraday_Return'] = (df['high'] - df['low']) / df['low']
    df['Close_to_Open_Return'] = (df['close'] - df['open']) / df['open']
    df['Weighted_Returns'] = (df['Intraday_Return'] ** 2) + (df['Close_to_Open_Return'] ** 2)
    df['Daily_Volume_Change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['Adjusted_Interim_Factor'] = df['Weighted_Returns'] * df['Daily_Volume_Change']

    # Smooth and Combine Interim Factor
    df['5_day_MA_Adjusted_Interim_Factor'] = df['Adjusted_Interim_Factor'].rolling(window=5).mean()

    # Enhanced Momentum Indicators
    df['7_day_SMA'] = df['close'].rolling(window=7).mean()
    df['30_day_SMA'] = df['close'].rolling(window=30).mean()
    df['SMA_Crossover'] = df['7_day_SMA'] > df['30_day_SMA']
    
    df['15_day_EMA'] = df['close'].ewm(span=15, adjust=False).mean()
    df['60_day_EMA'] = df['close'].ewm(span=60, adjust=False).mean()
    df['EMA_Growth'] = df['15_day_EMA'] / df['60_day_EMA'] - 1
    
    gain = df['close'].diff().clip(lower=0)
    loss = -df['close'].diff().clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Enhanced Volatility
    true_range = np.maximum(np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1))), abs(df['low'] - df['close'].shift(1)))
    df['20_day_Average_True_Range'] = true_range.rolling(window=20).mean()
    df['Volume_Weighted_Price_Change'] = (abs(df['close'] - df['close'].shift(1)) * df['volume'])

    # Additional Reversal Indicator
    df['10_day_Price_Reversal'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['20_day_Price_Reversal'] = (df['close'] - df['open']) / (df[['high', 'low']].rolling(window=20).max() - df[['high', 'low']].rolling(window=20).min())

    # Final Composite Alpha Factor
    df['Momentum_Score'] = df['SMA_Crossover'].astype(int) * df['EMA_Growth'] * (df['RSI'] / 100)
    df['Volatility_Score'] = df['20_day_Average_True_Range'] * df['Volume_Weighted_Price_Change']
    df['Reversal_Score'] = df['10_day_Price_Reversal'] * df['20_day_Price_Reversal']
    df['Composite_Alpha_Factor'] = df['5_day_MA_Adjusted_Interim_Factor'] * df['Momentum_Score'] * df['Volatility_Score'] * df['Reversal_Score']
    
    # Sigmoid Function and Scaling to [-1, 1]
    df['Alpha_Factor'] = 2 * (1 / (1 + np.exp(-df['Composite_Alpha_Factor']))) - 1

    return df['Alpha_Factor']

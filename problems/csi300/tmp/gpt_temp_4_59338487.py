import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Dynamics
    df['Intraday_Return'] = (df['high'] - df['low']) / df['low']
    df['Close_to_Open_Return'] = (df['close'] - df['open']) / df['open']
    df['Daily_Volume_Change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['Daily_Amount_Change'] = (df['amount'] - df['amount'].shift(1)) / df['amount'].shift(1)
    
    df['Weighted_Returns'] = (np.abs(df['Intraday_Return']) * df['Intraday_Return']) + (np.abs(df['Close_to_Open_Return']) * df['Close_to_Open_Return'])
    df['Adjusted_Interim_Factor'] = df['Weighted_Returns'] * df['Daily_Volume_Change'] * df['Daily_Amount_Change']

    # Smooth and Combine Factors
    df['Interim_Factor_3d_MA'] = df['Adjusted_Interim_Factor'].rolling(window=3).mean()
    
    # Momentum Indicators
    df['5d_SMA'] = df['close'].rolling(window=5).mean()
    df['20d_SMA'] = df['close'].rolling(window=20).mean()
    df['SMA_Crossover'] = (df['5d_SMA'] > df['20d_SMA']).astype(int)
    
    df['10d_EMA'] = df['close'].ewm(span=10, adjust=False).mean()
    df['30d_EMA'] = df['close'].ewm(span=30, adjust=False).mean()
    df['EMA_Growth'] = df['10d_EMA'] - df['30d_EMA']
    
    def rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI_14d'] = rsi(df['close'], window=14)
    
    # Volatility
    true_range = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], 
                                                           abs(x['high'] - df['close'].shift(1)), 
                                                           abs(x['low'] - df['close'].shift(1))), axis=1)
    df['ATR_20d'] = true_range.rolling(window=20).mean()
    
    df['Volume_Weighted_Price_Change'] = (np.abs(df['close'] - df['close'].shift(1))) * df['volume']
    df['Volume_Change_Ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Final Composite Alpha Factor
    df['Momentum_Score'] = df['SMA_Crossover'] + df['EMA_Growth'] + df['RSI_14d']
    df['Volatility_Score'] = df['ATR_20d'] + df['Volume_Weighted_Price_Change'] + df['Volume_Change_Ratio']
    df['Volume_Score'] = df['Interim_Factor_3d_MA']
    
    df['Composite_Alpha_Factor'] = df['Momentum_Score'] + df['Volatility_Score'] + df['Volume_Score']
    
    # Sigmoid Function and Scaling to [-1, 1]
    df['Final_Alpha_Factor'] = 2 * (1 / (1 + np.exp(-df['Composite_Alpha_Factor']))) - 1
    
    return df['Final_Alpha_Factor']

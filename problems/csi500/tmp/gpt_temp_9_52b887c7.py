import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    df['Intraday_High_Low_Spread'] = df['high'] - df['low']
    
    # Calculate Logarithmic Return
    df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate Close-Open Spread
    df['Close_Open_Spread'] = df['close'] - df['open']
    
    # Calculate Volume Weighted High-Low Spread
    df['Volume_Weighted_High_Low_Spread'] = (df['high'] - df['low']) * df['volume']
    
    # Calculate Average Daily Volume over 20-Day Rolling Window
    df['Average_Daily_Volume'] = df['volume'].rolling(window=20).mean()
    
    # Combine Intraday Volatility, Momentum, and Volume-Weighted Measure
    df['Combined_Measure'] = df['Intraday_High_Low_Spread'] + df['Log_Return'] + df['Volume_Weighted_High_Low_Spread']
    
    # Adjust for Gap Up or Down
    df['Gap_Difference'] = df.apply(lambda row: row['open'] - row['close'].shift(1) if row['open'] > row['close'].shift(1) else row['close'].shift(1) - row['open'], axis=1)
    
    # Incorporate Liquidity Factor
    df['Liquidity_Adjusted_Combined_Measure'] = df['Combined_Measure'] / df['Average_Daily_Volume']
    
    # Final Alpha Factor
    df['Final_Alpha_Factor'] = df['Liquidity_Adjusted_Combined_Measure'] + df['Gap_Difference']
    
    return df['Final_Alpha_Factor']

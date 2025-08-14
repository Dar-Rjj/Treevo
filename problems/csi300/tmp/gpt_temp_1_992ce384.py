import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation

def heuristics_v2(df):
    # Calculate Intraday Price Movement
    df['High_Low_Range'] = df['high'] - df['low']
    df['Close_Open_Diff'] = df['close'] - df['open']
    
    # Incorporate Volume Influence
    df['Volume_Adjusted_Momentum'] = (df['Close_Open_Diff'] + df['High_Low_Range']) * df['volume']
    
    # Adaptive Smoothing via Exponential Moving Average
    recent_volatility = df['Close_Open_Diff'].rolling(window=30).std()
    dynamic_ema_period = 5 + 25 * (1 - 1 / (1 + recent_volatility))
    df['Smoothed_Volume_Adjusted_Momentum'] = df['Volume_Adjusted_Momentum'].ewm(span=dynamic_ema_period, adjust=False).mean()
    
    # Adjust for Market Volatility
    df['Daily_Return'] = df['close'].pct_change()
    df['Absolute_Daily_Return'] = df['Daily_Return'].abs()
    robust_market_volatility = median_abs_deviation(df['Daily_Return'], scale='normal')
    df['Adjusted_Volume_Adjusted_Momentum'] = df['Smoothed_Volume_Adjusted_Momentum'] / robust_market_volatility
    
    # Incorporate Trend Reversal Signal
    df['Short_Term_Momentum'] = df['close'].ewm(span=5, adjust=False).mean()
    df['Long_Term_Momentum'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Momentum_Reversal'] = df['Short_Term_Momentum'] - df['Long_Term_Momentum']
    df['Reversal_Signal'] = df['Momentum_Reversal'].apply(lambda x: 1 if x > 0 else -1)
    
    # Integrate Non-Linear Transformation
    df['Sqrt_Transformed_Momentum'] = np.sqrt(np.abs(df['Adjusted_Volume_Adjusted_Momentum']))
    df['Log_Transformed_Momentum'] = np.log1p(np.abs(df['Adjusted_Volume_Adjusted_Momentum']))
    
    # Enhance Reversal Signal with Adaptive Smoothing
    df['Smoothed_Reversal_Signal'] = df['Reversal_Signal'].ewm(span=dynamic_ema_period, adjust=False).mean()
    df['Combined_Factor'] = df['Sqrt_Transformed_Momentum'] + df['Log_Transformed_Momentum'] + df['Smoothed_Reversal_Signal']
    
    # Final Adaptive Smoothing
    df['Final_Alpha_Factor'] = df['Combined_Factor'].ewm(span=dynamic_ema_period, adjust=False).mean()
    
    return df['Final_Alpha_Factor']

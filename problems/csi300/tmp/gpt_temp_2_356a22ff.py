import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Movement
    df['High_Low_Range'] = df['high'] - df['low']
    df['Close_Open_Diff'] = df['close'] - df['open']
    df['Intraday_Movement'] = (df['High_Low_Range'] + df['Close_Open_Diff']) / 2
    
    # Incorporate Volume Influence
    df['Volume_Adjusted_Momentum'] = df['Intraday_Movement'] * df['volume']
    
    # Adaptive Smoothing via Moving Average
    def dynamic_ema_period(returns, min_periods=5, max_periods=30):
        vol = returns.rolling(window=30).std()
        return (max_periods - min_periods) * (1 - vol / vol.max()) + min_periods
    
    ema_period = dynamic_ema_period(df['close'].pct_change().fillna(0))
    df['Smoothed_Volume_Adjusted_Momentum'] = df['Volume_Adjusted_Momentum'].ewm(span=ema_period, adjust=False).mean()
    
    # Adjust for Market Volatility
    df['Daily_Return'] = df['close'].pct_change()
    df['Abs_Daily_Return'] = df['Daily_Return'].abs()
    df['Robust_Market_Volatility'] = df['Abs_Daily_Return'].rolling(window=30).median() * 1.4826  # MAD scaling factor
    df['Volatility_Adjusted_Momentum'] = df['Smoothed_Volume_Adjusted_Momentum'] / df['Robust_Market_Volatility']
    
    # Incorporate Trend Reversal Signal
    df['Short_Term_Momentum'] = df['close'].ewm(span=5, adjust=False).mean()
    df['Long_Term_Momentum'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Momentum_Reversal'] = df['Short_Term_Momentum'] - df['Long_Term_Momentum']
    df['Reversal_Signal'] = np.where(df['Momentum_Reversal'] > 0, 1, -1)
    
    # Integrate Non-Linear Transformation
    df['Sqrt_Transformed_Momentum'] = np.sqrt(np.abs(df['Volatility_Adjusted_Momentum']))
    df['Log_Transformed_Momentum'] = np.log1p(np.abs(df['Volatility_Adjusted_Momentum']))
    
    # Enhance Reversal Signal with Adaptive Smoothing
    df['Smoothed_Reversal_Signal'] = df['Reversal_Signal'].ewm(span=ema_period, adjust=False).mean()
    df['Combined_Factor'] = df['Sqrt_Transformed_Momentum'] + df['Log_Transformed_Momentum'] + df['Smoothed_Reversal_Signal']
    
    # Refine Final Alpha Factor
    df['Final_Alpha_Factor'] = df['Combined_Factor'].ewm(span=ema_period, adjust=False).mean()
    
    return df['Final_Alpha_Factor']

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
    df['Volume_Adjusted_Momentum'] = (df['High_Low_Range'] + df['Close_Open_Diff']) * df['volume']
    
    # Apply Adaptive Smoothing
    def dynamic_ema_period(volatility):
        return 5 + int(10 * (1 - volatility))
    
    recent_volatility = df['close'].rolling(window=30).std() / df['close'].rolling(window=30).mean()
    ema_periods = [dynamic_ema_period(v) for v in recent_volatility]
    df['Smoothed_Volume_Adjusted_Momentum'] = df['Volume_Adjusted_Momentum'].ewm(span=ema_periods, min_periods=1).mean()

    # Adjust for Market Volatility
    df['Daily_Return'] = df['close'].pct_change()
    df['Abs_Daily_Return'] = df['Daily_Return'].abs()
    df['Robust_Market_Volatility'] = df['Abs_Daily_Return'].rolling(window=30).apply(median_abs_deviation, raw=False)
    df['Adjusted_Volume_Adjusted_Momentum'] = df['Smoothed_Volume_Adjusted_Momentum'] / df['Robust_Market_Volatility']

    # Incorporate Trend Reversal Signal
    df['Short_Term_Momentum'] = df['close'].ewm(span=5, min_periods=1).mean()
    df['Long_Term_Momentum'] = df['close'].ewm(span=20, min_periods=1).mean()
    df['Momentum_Reversal'] = df['Short_Term_Momentum'] - df['Long_Term_Momentum']
    df['Reversal_Signal'] = np.where(df['Momentum_Reversal'] > 0, 1, -1)

    # Integrate Non-Linear Transformation
    df['Sqrt_Transformed_Momentum'] = np.sqrt(np.abs(df['Adjusted_Volume_Adjusted_Momentum']))
    df['Log_Transformed_Momentum'] = np.log(np.abs(df['Adjusted_Volume_Adjusted_Momentum']))

    # Enhance Reversal Signal with Adaptive Smoothing
    df['Smoothed_Reversal_Signal'] = df['Reversal_Signal'].ewm(span=ema_periods, min_periods=1).mean()
    df['Combined_Momentum'] = df['Sqrt_Transformed_Momentum'] + df['Log_Transformed_Momentum'] + df['Smoothed_Reversal_Signal']

    # Final Adaptive Smoothing
    df['Final_Alpha_Factor'] = df['Combined_Momentum'].ewm(span=ema_periods, min_periods=1).mean()

    return df['Final_Alpha_Factor']

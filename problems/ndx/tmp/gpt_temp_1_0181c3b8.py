import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Price (VWP)
    df['VWP'] = (df['High'] * df['Volume'] + df['Low'] * df['Volume']) / (2 * df['Volume'])

    # Generate Intraday Volatility Metric
    avg_vol_14 = df['Volume'].rolling(window=14).mean()
    df['Intraday_Volatility'] = (df['High'] - df['Low']) / avg_vol_14

    # Rolling 14-Day High-to-Low Range Sum
    df['HtoL_Range_Sum_14'] = (df['High'] - df['Low']).rolling(window=14).sum()

    # Difference between Consecutive VWP
    df['VWP_Diff'] = df['VWP'].diff()

    # Integrate High-to-Low Range Momentum
    df['Adjusted_Momentum'] = df['VWP_Diff'] / df['HtoL_Range_Sum_14']

    # Calculate Volume-Adjusted Momentum
    df['VWP_20_EMA'] = df['VWP'].ewm(span=20, adjust=False).mean()
    df['VWP_Momentum'] = df['VWP'] - df['VWP_20_EMA']

    # Incorporate Recent Volatility
    df['Return'] = df['Close'].pct_change()
    df['Std_10'] = df['Return'].rolling(window=10).std()
    df['Vol_Adjusted_Momentum'] = df['VWP_Momentum'] / df['Std_10']

    # Combine Volume-Adjusted Momentum and Composite Volatility
    df['HtoL_MA_20'] = (df['High'] - df['Low']).rolling(window=20).mean()
    df['OtoC_MA_20'] = (df['Open'] - df['Close']).rolling(window=20).mean()
    df['Composite_Volatility'] = (df['HtoL_MA_20'] + df['OtoC_MA_20']) / 2

    # Final Alpha Factor
    df['Alpha_Factor'] = df['Vol_Adjusted_Momentum'] - df['Composite_Volatility']

    return df['Alpha_Factor']

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Movement
    df['High-Low'] = df['high'] - df['low']
    df['Close-Open'] = df['close'] - df['open']

    # Volume-Adjusted Momentum
    df['Volume-Adjusted-Momentum'] = (df['Close-Open'] + df['High-Low']) * df['volume']

    # Apply 10-day EMA to Volume-Adjusted Momentum
    df['EMA_10_Volume_Adjusted_Momentum'] = df['Volume-Adjusted-Momentum'].ewm(span=10).mean()

    # Adjust for Market Volatility
    df['Daily_Return'] = df['close'].pct_change()
    df['Absolute_Daily_Return'] = df['Daily_Return'].abs()
    df['Market_Volatility'] = df['Daily_Return'].rolling(window=30).std()
    
    # Modify Volume-Adjusted Momentum with Market Volatility
    df['Modified_Momentum'] = df['EMA_10_Volume_Adjusted_Momentum'] - df['Market_Volatility']

    # Integrate Trend Reversal Signal
    df['Short_Term_Momentum'] = df['close'].ewm(span=5, adjust=False).mean()
    df['Long_Term_Momentum'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Momentum_Reversal'] = df['Short_Term_Momentum'] - df['Long_Term_Momentum']

    # Label Positive and Negative Reversals
    df['Reversal_Signal'] = np.sign(df['Momentum_Reversal'])

    # Combine Modified Momentum with Reversal Signal
    df['Combined_Factor'] = df['Modified_Momentum'] + df['Reversal_Signal']

    # Enhance Volatility Adjustment
    df['Inverse_Market_Volatility'] = 1 / df['Market_Volatility']
    df['Volatility_Adjusted_Factor'] = df['Combined_Factor'] * df['Inverse_Market_Volatility']

    # Integrate Non-Linear Transformation
    df['Sqrt_Transformed'] = np.sqrt(np.abs(df['Volatility_Adjusted_Factor']))
    df['Log_Transformed'] = np.log1p(np.abs(df['Volatility_Adjusted_Fctor']))

    # Combine Non-Linearly Transformed Alpha Factor with Reversal Signal
    df['Non_Linear_Combined'] = (df['Sqrt_Transformed'] + df['Log_Transformed']) * df['Reversal_Signal']

    # Apply 5-day EMA to Smooth the Final Alpha Factor
    df['Final_Alpha_Factor'] = df['Non_Linear_Combined'].ewm(span=5, adjust=False).mean()

    return df['Final_Alpha_Factor'].dropna()

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday Price Movement
    df['High_Low_Range'] = df['high'] - df['low']
    df['Close_Open_Diff'] = df['close'] - df['open']
    
    # Calculate Volume-Adjusted Momentum
    df['Volume_Adjusted_Momentum'] = (df['Close_Open_Diff'] + df['High_Low_Range']) * df['volume']
    
    # Adaptive Smoothing via Exponential Moving Average
    def dynamic_ema(data, span):
        return data.ewm(span=span, adjust=False).mean()
    
    recent_volatility = df['Close_Open_Diff'].rolling(window=30).std()
    df['Volume_Adjusted_Momentum_Smoothed'] = dynamic_ema(df['Volume_Adjusted_Momentum'], span=recent_volatility)
    
    # Adjust for Market Volatility
    df['Daily_Return'] = df['close'].pct_change()
    df['Absolute_Daily_Return'] = df['Daily_Return'].abs()
    robust_volatility = df['Daily_Return'].rolling(window=30).apply(lambda x: 1.4826 * np.median(np.abs(x - np.median(x))), raw=True)
    df['Volatility_Adjusted_Momentum'] = df['Volume_Adjusted_Momentum_Smoothed'] / robust_volatility
    
    # Incorporate Trend Reversal Signal
    df['Short_Term_Momentum'] = df['close'].ewm(span=5, adjust=False).mean()
    df['Long_Term_Momentum'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Momentum_Reversal'] = df['Short_Term_Momentum'] - df['Long_Term_Momentum']
    df['Reversal_Signal'] = np.where(df['Momentum_Reversal'] > 0, 1, -1)
    
    # Integrate Non-Linear Transformation
    df['Sqrt_Transformed_Momentum'] = np.sqrt(df['Volatility_Adjusted_Momentum'])
    df['Log_Transformed_Momentum'] = np.log(df['Volatility_Adjusted_Momentum'] + 1e-6)  # Add small constant to avoid log(0)
    
    # Enhance Reversal Signal with Adaptive Smoothing
    df['Smoothed_Reversal_Signal'] = dynamic_ema(df['Reversal_Signal'], span=recent_volatility)
    df['Interim_Alpha_Factor'] = df['Sqrt_Transformed_Momentum'] + df['Log_Transformed_Momentum'] + df['Smoothed_Reversal_Signal']
    
    # Final Adaptive Smoothing
    df['Final_Alpha_Factor'] = dynamic_ema(df['Interim_Alpha_Factor'], span=recent_volatility)
    
    return df['Final_Alpha_Factor']

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 102, 101, 103, 104],
#     'high': [105, 107, 106, 108, 109],
#     'low': [99, 100, 101, 102, 103],
#     'close': [103, 105, 104, 106, 107],
#     'amount': [1000, 1200, 1100, 1300, 1400],
#     'volume': [100, 150, 120, 180, 200]
# })
# df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)

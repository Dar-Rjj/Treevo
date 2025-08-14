import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    """
    Generate a novel and interpretable alpha factor that could help predict future stock returns.
    
    Parameters:
    df (pd.DataFrame): A DataFrame with columns (open, high, low, close, volume) and index (date).
    
    Returns:
    pd.Series: A Series with the alpha factor values indexed by (date).
    """
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate the 10-day moving average of the close price
    df['ma_10'] = df['close'].rolling(window=10).mean()
    
    # Calculate the 30-day moving average of the close price
    df['ma_30'] = df['close'].rolling(window=30).mean()
    
    # Calculate the moving average convergence divergence (MACD)
    df['macd_line'] = df['ma_10'] - df['ma_30']
    
    # Calculate the signal line for MACD
    df['signal_line'] = df['macd_line'].rolling(window=9).mean()
    
    # Calculate the MACD histogram
    df['macd_histogram'] = df['macd_line'] - df['signal_line']
    
    # Calculate the 5-day average true range
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x[0], df['close'].shift(1)) - min(x[1], df['close'].shift(1)), axis=1)
    df['atr_5'] = df['true_range'].rolling(window=5).mean()
    
    # Calculate the percentage change in volume
    df['volume_pct_change'] = df['volume'].pct_change()
    
    # Calculate the alpha factor
    df['alpha_factor'] = df['macd_histogram'] * (df['volume_pct_change'] / df['atr_5'])
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df['alpha_factor']

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 101, 99, 102, 103],
#     'high': [102, 103, 100, 104, 105],
#     'low': [98, 99, 97, 100, 101],
#     'close': [101, 100, 102, 103, 104],
#     'volume': [1000, 1500, 1200, 1300, 1600]
# }, index=pd.date_range(start='2023-01-01', periods=5))
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)

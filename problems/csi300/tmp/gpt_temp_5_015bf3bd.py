import pandas as pd
import pandas as pd

def heuristics_v2(df, n=14):
    # Calculate Daily Range
    df['daily_range'] = df['high'] - df['low']
    
    # Calculate Average True Range over N periods
    df['avg_true_range'] = df['daily_range'].rolling(window=n).mean()
    
    # Normalized High
    df['normalized_high'] = df['high'] - df['avg_true_range']
    
    # Normalized Low
    df['normalized_low'] = df['low'] + df['avg_true_range']
    
    # Initialize the momentum signal column
    df['momentum_signal'] = 0
    
    # Calculate Momentum Signal
    for i in range(len(df) - 1):
        if df.iloc[i+1]['open'] > df.iloc[i]['normalized_high']:
            df.at[df.index[i], 'momentum_signal'] = 1
        elif df.iloc[i+1]['open'] < df.iloc[i]['normalized_low']:
            df.at[df.index[i], 'momentum_signal'] = -1
        else:
            df.at[df.index[i], 'momentum_signal'] = 0
    
    return df['momentum_signal']

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 105, 103, 106, 107, 108],
#     'high': [105, 108, 107, 110, 112, 115],
#     'low': [98, 102, 100, 103, 104, 106],
#     'close': [104, 107, 105, 109, 111, 114],
# }, index=pd.date_range(start='2023-01-01', periods=6))
# print(heuristics_v2(df))

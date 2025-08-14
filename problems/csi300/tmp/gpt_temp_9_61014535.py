import pandas as pd
import pandas as pd

def heuristics_v2(df, N=10, positive_multiplier=1.5, negative_multiplier=0.5):
    # Calculate Daily Close-to-Close Return
    df['Close_Return'] = df['close'].pct_change()
    
    # Calculate Volume Trend (VT)
    df['Volume_Change'] = df['volume'].diff()
    df['Volume_Trend'] = 0
    for i in range(1, N+1):
        df['Volume_Trend'] += df['Volume_Change'].shift(i-1).apply(lambda x: x if x > 0 else -x)
    
    # Calculate Volume-Weighted Average of Open and Close Prices
    df['Volume_Weighted_Avg'] = (df['open'] + df['close']) / 2 * df['volume']
    
    # Combine Close-to-Close Return and Volume-Weighted Average
    df['Alpha_Factor'] = df['Close_Return'] * df['Volume_Weighted_Avg']
    
    # Conditionally Adjust Alpha Factor Based on Volume Trend
    df['Final_Alpha'] = df.apply(
        lambda row: row['Alpha_Factor'] * positive_multiplier if row['Volume_Trend'] > 0 else 
                    row['Alpha_Fctor'] * negative_multiplier,
        axis=1
    )
    
    return df['Final_Alpha'].dropna()

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 102, 103, 105, 107],
#     'high': [105, 107, 108, 110, 112],
#     'low': [98, 100, 101, 103, 105],
#     'close': [104, 106, 107, 109, 110],
#     'amount': [1000, 1200, 1300, 1400, 1500],
#     'volume': [100, 120, 130, 140, 150]
# }, index=pd.date_range(start='2023-01-01', periods=5))
# result = heuristics_v2(df)
# print(result)

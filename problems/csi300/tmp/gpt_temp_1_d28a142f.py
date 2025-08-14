import pandas as pd
import pandas as pd

def heuristics_v2(df, n=20):
    # Calculate Daily Price Change
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # Calculate Volume-Weighted Price Change
    df['vol_weighted_price_change'] = df['price_change'] * df['volume']
    
    # Initialize Rolling Sum Variable for Volume
    df['rolling_vol_sum'] = 0
    
    # Update Rolling Sum with Each Day's Volume
    for i in range(n, len(df)):
        df.loc[df.index[i], 'rolling_vol_sum'] = df.loc[df.index[i-n+1:i+1], 'volume'].sum()
    
    # Assign Direction Based on Volume-Weighted Price Change
    df['direction'] = df['vol_weighted_price_change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    # Multiply Direction by Cumulative Volume
    df['direction_weighted_vol'] = df['direction'] * df['rolling_vol_sum']
    
    # Initialize Total Indicator Variable and accumulate
    df['cumulative_indicator'] = df['direction_weighted_vol'].cumsum()
    
    return df['cumulative_indicator']

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 101, 102, 103, 104],
#     'high': [105, 106, 107, 108, 109],
#     'low': [99, 100, 101, 102, 103],
#     'close': [102, 103, 104, 105, 106],
#     'amount': [10000, 10000, 10000, 10000, 10000],
#     'volume': [1000, 1000, 1000, 1000, 1000]
# }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']))

# result = heuristics_v2(df)
# print(result)

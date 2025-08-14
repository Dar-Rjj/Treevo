import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10):
    """
    Calculate the Volume and Amount Adjusted Momentum with Price Gap.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'amount']
                       indexed by date.
    n (int): The lookback period for momentum calculation.
    
    Returns:
    pd.Series: Factor values indexed by date.
    """
    # Calculate Price Gap
    df['price_gap'] = df['open'] - df['close'].shift(1)
    
    # Calculate Momentum
    df['momentum'] = df['close'] - df['close'].shift(n)
    
    # Calculate Volume and Amount-Adjusted Momentum
    df['volume_amount_adjusted_momentum'] = df['momentum'] / (df['volume'] * df['amount'])
    
    # Integrate
    df['factor'] = df['price_gap'] * df['volume_amount_adjusted_momentu m']
    
    return df['factor']

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 102, 101, 99, 103],
#     'high': [105, 107, 106, 104, 108],
#     'low': [98, 99, 100, 97, 102],
#     'close': [103, 105, 104, 102, 106],
#     'volume': [1000, 1200, 1100, 1300, 1400],
#     'amount': [10000, 12000, 11000, 13000, 14000]
# }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']))

# factor_values = heuristics_v2(df)
# print(factor_values)

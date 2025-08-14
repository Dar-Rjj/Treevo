import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df, n=10, m=14):
    # Calculate Daily Return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Volume Change Ratio
    df['volume_change_ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Compute Weighted Momentum
    df['weighted_momentum'] = df['daily_return'] * df['volume_change_ratio']
    df['weighted_momentum_sum'] = df['weighted_momentum'].rolling(window=n).sum()
    
    # Calculate Average True Range
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], 
                                                               abs(x['high'] - df['close'].shift(1)), 
                                                               abs(x['low'] - df['close'].shift(1))), axis=1)
    df['average_true_range'] = df['true_range'].rolling(window=m).mean()
    
    # Adjust for Price Volatility
    df['adjusted_weighted_momentum'] = df['weighted_momentum_sum'] - df['average_true_range']
    
    return df['adjusted_weighted_momentum']

# Example usage:
# df = pd.DataFrame({
#     'date': pd.date_range(start='2023-01-01', periods=100),
#     'open': np.random.uniform(low=50, high=150, size=100),
#     'high': np.random.uniform(low=50, high=150, size=100),
#     'low': np.random.uniform(low=50, high=150, size=100),
#     'close': np.random.uniform(low=50, high=150, size=100),
#     'volume': np.random.randint(low=10000, high=100000, size=100)
# })
# df.set_index('date', inplace=True)
# alpha_factor = heuristics_v2(df)

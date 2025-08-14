import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume-Weighted Average Price (VWAP)
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
    
    # Normalize the values to have a mean of 0 and standard deviation of 1
    df['intraday_range'] = (df['intraday_range'] - df['intraday_range'].mean()) / df['intraday_range'].std()
    df['close_to_open_return'] = (df['close_to_open_return'] - df['close_to_open_return'].mean()) / df['close_to_open_return'].std()
    df['vwap'] = (df['vwap'] - df['vwap'].mean()) / df['vwap'].std()

    # Assign weights based on recency
    recent_weights = {'intraday_range': 0.5, 'close_to_open_return': 0.3, 'vwap': 0.2}
    older_weights = {'intraday_range': 0.4, 'close_to_open_return': 0.4, 'vwap': 0.2}

    # Define a function to calculate the weighted combination
    def weighted_combination(row, recent, days_since_start):
        if days_since_start < 7:
            return (recent['intraday_range'] * row['intraday_range'] +
                    recent['close_to_open_return'] * row['close_to_open_return'] +
                    recent['vwap'] * row['vwap'])
        else:
            return (older_weights['intraday_range'] * row['intraday_range'] +
                    older_weights['close_to_open_return'] * row['close_to_open_return'] +
                    older_weights['vwap'] * row['vwap'])

    # Apply the weighted combination
    df['days_since_start'] = range(len(df))
    df['factor_value'] = df.apply(lambda row: weighted_combination(row, recent_weights, row['days_since_start']), axis=1)

    # Apply dynamic weighting
    df['recency_factor'] = 1 / (1 + np.exp(-(df.index - df.index.min()).days / 30))  # Sigmoid function
    df['momentum_factor'] = df['close'].pct_change().rolling(window=5).mean()
    df['volume_factor'] = df['volume'].pct_change().rolling(window=5).mean()

    # Combine the factors
    df['alpha_factor'] = (df['factor_value'] * df['recency_factor'] * 
                          (1 + df['momentum_factor']) * 
                          (1 + df['volume_factor']))

    # Return the alpha factor
    return df['alpha_factor']

# Example usage:
# df = pd.DataFrame({
#     'date': pd.date_range(start='2023-01-01', periods=60, freq='D'),
#     'open': np.random.uniform(100, 150, 60),
#     'high': np.random.uniform(100, 150, 60),
#     'low': np.random.uniform(100, 150, 60),
#     'close': np.random.uniform(100, 150, 60),
#     'amount': np.random.uniform(100, 150, 60),
#     'volume': np.random.uniform(1000, 1500, 60)
# })
# df.set_index('date', inplace=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)

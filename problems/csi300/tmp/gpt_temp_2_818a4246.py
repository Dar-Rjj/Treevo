import pandas as pd
import pandas as pd

def heuristics_v2(df):
    """
    Calculate a novel alpha factor based on intraday volatility adjusted return.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume'] and index as date.
    
    Returns:
    pd.Series: A series of the calculated alpha factor indexed by date.
    """
    
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df.shift(1)['close']) / df.shift(1)['close']
    
    # Define weights for recent and older data
    recent_weights = {'intraday_range': 0.7, 'close_to_open_return': 0.3}
    older_weights = {'intraday_range': 0.5, 'close_to_open_return': 0.5}
    
    # Assign weights to recent (last 5 days) and older data
    def assign_weights(row):
        if row.name in df.index[-5:]:
            return recent_weights['intraday_range'] * row['intraday_range'] + recent_weights['close_to_open_return'] * row['close_to_open_return']
        else:
            return older_weights['intraday_range'] * row['intraday_range'] + older_weights['close_to_open_return'] * row['close_to_open_return']
    
    # Apply the weighting function to each row
    df['alpha_factor'] = df.apply(assign_weights, axis=1)
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor_values = heuristics_v2(df)
# print(factor_values)

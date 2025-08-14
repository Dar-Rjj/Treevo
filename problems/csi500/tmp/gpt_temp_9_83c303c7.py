import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Weight by Volume
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Add Price Change Momentum
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['volatility'] = df['close'].rolling(window=20).std()
    df['lookback_period'] = (20 * df['volatility']).rolling(window=20).mean().astype(int)
    df['momentum'] = df['price_change'].rolling(window=df['lookback_period']).sum()
    
    # Integrate Volume Trends
    df['volume_trend'] = df['volume'].pct_change().rolling(window=20).sum()
    df['composite_alpha'] = (df['volume_weighted_return'] + df['momentum']) * df['volume_trend']
    
    # Consider Relative Strength
    # Assuming 'benchmark' is the column for benchmark index close prices
    df['relative_strength'] = df['close'] / df['benchmark']
    df['alpha_factor'] = df['composite_alpha'] * df['relative_strength']
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)

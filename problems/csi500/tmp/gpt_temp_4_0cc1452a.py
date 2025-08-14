import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 10-Day Price Variance
    df['close_var_10'] = df['close'].rolling(window=10).var()
    
    # Define the Dynamic Lookback Period
    lookback_period = np.where(df['close_var_10'] > df['close_var_10'].mean(), 5, 20)
    
    # Calculate Dynamic Simple Moving Average of Close Prices
    df['dynamic_sma'] = df.apply(lambda x: df['close'].rolling(window=lookback_period[x.name]).mean(), axis=1)
    
    # Compute Volume-Adjusted Volatility
    df['high_low_diff'] = df['high'] - df['low']
    df['volume_adjusted_volatility'] = df['high_low_diff'] * df['volume']
    
    # Compute Price Momentum
    n = 20  # Number of days to compute the average of last N close prices
    df['avg_close_n'] = df['close'].rolling(window=n).mean()
    df['price_momentum'] = (df['close'] - df['dynamic_sma']) / df['avg_close_n']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['price_momentum'] / df['volume_adjusted_volatility']
    
    # Return the alpha factor as a pandas Series
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)

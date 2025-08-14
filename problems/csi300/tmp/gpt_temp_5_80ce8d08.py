import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Weight by Volume
    df['weighted_volatility'] = df['intraday_volatility'] * df['volume']
    
    # Calculate Intraday Momentum
    df['close_open_diff'] = df['close'] - df['open']
    small_window = 5
    large_window = 20
    df['momentum_small'] = df['close_open_diff'].rolling(window=small_window).sum()
    df['momentum_large'] = df['close_open_diff'].rolling(window=large_window).sum()
    
    # Integrate Momentum and Volatility
    df['integrated_signal'] = df['weighted_volatility'] + (df['momentum_small'] + df['momentum_large']) / 2
    
    # Apply Adaptive Exponential Smoothing
    initial_smoothing_factor = 0.9
    recent_volatility = df['intraday_volatility'].rolling(window=5).std()
    smoothing_factor = initial_smoothing_factor * (1 + recent_volatility)
    df['smoothed_value'] = df['integrated_signal'].ewm(alpha=smoothing_factor, adjust=False).mean()
    
    # Ensure Values are Positive
    small_constant = 1e-6
    df['positive_smoothed_value'] = df['smoothed_value'] + small_constant
    
    # Apply Logarithmic Transformation
    df['log_transformed_value'] = np.log(df['positive_smoothed_value'])
    
    # Factor Output
    return df['log_transformed_value']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor_values = heuristics_v2(df)
# print(factor_values)

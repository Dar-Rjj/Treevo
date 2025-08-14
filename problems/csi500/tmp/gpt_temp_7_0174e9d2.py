import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Define the base lookback period for standard deviation
    base_lookback = 20
    
    # Calculate the rolling standard deviation of close prices
    std_dev = df['close'].rolling(window=base_lookback).std()
    
    # Define the dynamic lookback period for SMA based on market volatility
    def adjust_lookback(std):
        if std > std_dev.mean():
            return int(base_lookback * 0.75)  # Shorter lookback in high volatility
        else:
            return int(base_lookback * 1.25)  # Longer lookback in low volatility
    
    # Apply the function to each row
    df['lookback'] = std_dev.apply(adjust_lookback)
    
    # Calculate the Simple Moving Average (SMA) with the dynamic lookback
    df['sma'] = df['close'].rolling(window=df['lookback']).mean()
    
    # Compute High-Low price difference
    df['high_low_diff'] = df['high'] - df['low']
    
    # Apply volume weighting to the High-Low price difference
    df['vol_weighted_high_low'] = df['volume'] * df['high_low_diff']
    
    # Calculate the rolling average of the volume-weighted high-low differences
    df['vol_adjusted_volatility'] = df['vol_weighted_high_low'].rolling(window=df['lookback']).mean()
    
    # Compute Price Momentum
    df['price_momentum'] = (df['close'] - df['sma']) / df['close'].rolling(window=df['lookback']).sum()
    
    # Final Alpha Factor
    df['alpha_factor'] = df['price_momentum'] / df['vol_adjusted_volatility']
    
    # Return the alpha factor as a pandas Series
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('stock_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)

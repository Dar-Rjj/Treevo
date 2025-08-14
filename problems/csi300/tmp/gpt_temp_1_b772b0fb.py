import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Logarithmic Return
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    
    # Volume-Weighted Daily Return
    df['volume_weighted_return'] = df['log_return'] * df['volume']
    
    # Dynamic Price Volatility
    smoothing_factor = 0.9
    df['abs_log_return'] = np.abs(df['log_return'])
    df['dynamic_volatility'] = df['abs_log_return'].ewm(alpha=1 - smoothing_factor).mean()
    
    # Adjust Volume-Weighted Return for Volatility
    df['adjusted_volume_weighted_return'] = df['volume_weighted_return'] / df['dynamic_volatility']
    
    # Cumulate Adjusted Volume-Weighted Values Over Window (N=5 days)
    df['cumulated_adjusted_return'] = df['adjusted_volume_weighted_return'].rolling(window=5).sum()
    
    # Add Trend Following Component
    M = 20  # Define Moving Average Length
    df['sma'] = df['close'].rolling(window=M).mean()
    df['trend_signal'] = (df['close'] > df['sma']).astype(int)
    
    # Combine Adjusted Volume-Weighted Returns with Trend Signal
    df['trend_adjusted_return'] = df['cumulated_adjusted_return'] * df['trend_signal']
    
    # Calculate Price Momentum Component
    momentum_window = 14
    df['close_moving_avg'] = df['close'].rolling(window=momentum_window).mean()
    df['close_diff'] = df['close'] - df['close_shift_1']  # Close difference
    df['price_momentum'] = df['close_diff'] * df['volume']
    
    # Calculate Price Volatility for Momentum
    price_volatility_window = 5
    df['price_volatility'] = df['close'].rolling(window=price_volatility_window).std()
    df['adjusted_momentum'] = df['price_momentum'] / df['price_volatility']
    
    # Integrate Adjusted Momentum Over Window (N=5 days)
    df['integrated_momentum'] = df['adjusted_momentum'].rolling(window=5).sum()
    
    # Combine All Components
    df['final_indicator'] = df['trend_adjusted_return'] * df['integrated_momentum']
    
    return df['final_indicator']

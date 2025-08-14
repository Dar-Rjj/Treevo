import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily High-Low Difference
    df['high_low_diff'] = df['high'] - df['low']
    
    # Compute Exponential Moving Average (EMA) of High-Low Difference
    ema_span = 14  # Example span
    df['ema_high_low_diff'] = df['high_low_diff'].ewm(span=ema_span, adjust=False).mean()
    
    # Compute High-Low Momentum
    df['high_low_momentum'] = df['high_low_diff'] / df['ema_high_low_diff'].shift(1)
    
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Adjust Intraday Range by Volume
    volume_ema_span = 7  # Example span
    df['volume_ema'] = df['volume'].ewm(span=volume_ema_span, adjust=False).mean()
    df['adjusted_volume'] = df['volume'] / df['volume_ema']
    df['adjusted_intraday_range'] = df['intraday_range'] * df['adjusted_volume']
    
    # Further Adjustment by Close Price Volatility
    df['close_returns'] = df['close'].pct_change()
    close_volatility_window = 10  # Example window
    df['close_volatility'] = df['close_returns'].rolling(window=close_volatility_window).std()
    df['volatility_adjusted_intraday_range'] = df['adjusted_intraday_range'] / df['close_volatility']
    
    # Calculate True Range Using High, Low, and Previous Close Prices
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = df[['high' - df['prev_close'], df['prev_close'] - df['low']]].max(axis=1)
    
    # Adjust Intraday Range by True Range Volatility
    true_range_volatility_window = 10  # Example window
    df['true_range_volatility'] = df['true_range'].rolling(window=true_range_volatility_window).std()
    df['tr_volatility_adjusted_intraday_range'] = df['volatility_adjusted_intraday_range'] / df['true_range_volatility']
    
    # Combine High-Low Momentum and Intraday Range
    df['combined_factor'] = df['high_low_momentum'] * df['tr_volatility_adjusted_intraday_range']
    
    # Calculate Volume Spike
    volume_spike_window = 7  # Example window
    df['median_volume'] = df['volume'].rolling(window=volume_spike_window).median()
    df['volume_spike_ratio'] = df['volume'] / df['median_volume']
    
    # Adjust Combined Factor by Volume
    df['volume_adjusted_combined_factor'] = df['combined_factor'] * df['volume_spike_ratio']
    
    # Further Adjustment by Open Price Volatility
    df['open_returns'] = df['open'].pct_change()
    open_volatility_window = 10  # Example window
    df['open_volatility'] = df['open_returns'].rolling(window=open_volatility_window).std()
    df['open_volatility_adjusted_factor'] = df['volume_adjusted_combined_factor'] / df['open_volatility']
    
    # Incorporate Return Momentum
    df['close_to_close_returns'] = df['close'].pct_change()
    momentum_window = 5  # Example window
    df['momentum'] = df['close_to_close_returns'].rolling(window=momentum_window).sum()
    df['momentum_adjusted_factor'] = df['open_volatility_adjusted_factor'] * df['momentum']
    
    # Further Adjustment by Open-Close Spread
    df['open_close_spread'] = (df['open'] - df['close']).abs()
    df['final_factor'] = df['momentum_adjusted_factor'] / df['open_close_spread']
    
    return df['final_factor'].dropna()

# Example usage:
# factor_values = heuristics_v2(df)

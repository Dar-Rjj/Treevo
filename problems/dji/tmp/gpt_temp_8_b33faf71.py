import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily High-Low Difference
    df['high_low_diff'] = df['high'] - df['low']
    
    # Cumulate the Moving Average of High-Low Differences
    window_size_1 = 10  # Define window size
    df['high_low_ma'] = df['high_low_diff'].rolling(window=window_size_1, min_periods=1).mean()
    df['cum_high_low_ma'] = df['high_low_ma'].expanding().mean()
    
    # Calculate Volume-Weighted Close Price
    window_size_2 = 20  # Define window size
    df['vwap'] = (df['close'] * df['volume']).rolling(window=window_size_2, min_periods=1).sum() / df['volume'].rolling(window=window_size_2, min_periods=1).sum()
    
    # Adjust Cumulative Moving Average by Volume-Weighted Close Price
    df['adjusted_cum_high_low_ma'] = df['cum_high_low_ma'] * df['vwap']
    
    # Calculate Daily Price Momentum
    df['daily_momentum'] = df['close'] - df['close'].shift(1)
    
    # Calculate Short-Term Adjusted Momentum
    short_window = 5
    df['short_term_ema'] = df['daily_momentum'].ewm(span=short_window, adjust=False).mean()
    df['short_term_std'] = df['daily_momentum'].rolling(window=short_window, min_periods=1).std()
    df['short_term_adjusted_momentum'] = df['short_term_ema'] / df['short_term_std']
    
    # Calculate Long-Term Adjusted Momentum
    long_window = 20
    df['long_term_ema'] = df['daily_momentum'].ewm(span=long_window, adjust=False).mean()
    df['long_term_std'] = df['daily_momentum'].rolling(window=long_window, min_periods=1).std()
    df['long_term_adjusted_momentum'] = df['long_term_ema'] / df['long_term_std']
    
    # Generate Volume Synchronized Oscillator
    df['vso'] = (df['long_term_adjusted_momentum'] - df['short_term_adjusted_momentum']) * df['volume']
    
    # Incorporate Price Movement Intensity
    df['high_low_range'] = df['high'] - df['low']
    df['open_close_spread'] = df['close'] - df['open']
    df['price_movement_intensity'] = df['high_low_range'] + df['open_close_spread']
    
    # Volume-Weighted Close Price Trend
    df['vwap_trend'] = df['vwap'].rolling(window=window_size_2, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    
    # Construct Comprehensive Alpha Factor
    df['alpha_factor'] = df['adjusted_cum_high_low_ma'] * df['vwap_trend'] * df['daily_momentum'] * df['vso'] * df['price_movement_intensity']
    
    return df['alpha_factor']

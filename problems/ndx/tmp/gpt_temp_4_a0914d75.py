import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum factor based on price trends
    n = 10  # Number of periods for rate of change
    df['roc'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n)
    
    # Short-term and long-term average returns
    short_term_window = 5
    long_term_window = 20
    
    df['short_term_momentum'] = (df['close'].rolling(window=short_term_window).mean() / df['close'].shift(short_term_window - 1)) - 1
    df['long_term_momentum'] = (df['close'].rolling(window=long_term_window).mean() / df['close'].shift(long_term_window - 1)) - 1
    
    df['relative_momentum'] = df['short_term_momentum'] - df['long_term_momentum']
    
    # Volatility factor using price movement variability
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['volatility'] = df['daily_return'].rolling(window=20).std()
    
    # Analyze volume and price changes
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['up_day'] = df['price_change'] > 0
    df['down_day'] = df['price_change'] < 0
    
    up_days = df[df['up_day']]['volume']
    down_days = df[df['down_day']]['volume']
    
    df['up_volume_avg'] = up_days.rolling(window=20).mean()
    df['down_volume_avg'] = down_days.rolling(window=20).mean()
    
    df['volume_ratio'] = df['up_volume_avg'] / df['down_volume_avg']
    
    # Combine factors into a single alpha factor
    df['alpha_factor'] = df['relative_momentum'] * (1 + df['volatility']) * df['volume_ratio']
    
    return df['alpha_factor'].dropna()

import pandas as pd
import pandas as pd

def heuristics_v2(df, N=10):
    # Calculate Close Price Momentum
    close_momentum = df['close'].diff(periods=N)
    
    # Identify Directional Days
    df['direction'] = 'Up'
    df.loc[df['open'] > df['close'], 'direction'] = 'Down'
    
    # Count Number of Up and Down Days in Last N Days
    up_days = df['direction'].rolling(window=N).apply(lambda x: (x == 'Up').sum(), raw=False)
    down_days = df['direction'].rolling(window=N).apply(lambda x: (x == 'Down').sum(), raw=False)
    
    # Weight by Volume
    volume_weighted_direction = (up_days - down_days) * df['volume']
    
    # Calculate Short-Term Price Momentum
    short_term_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Calculate Long-Term Price Momentum
    long_term_momentum = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Calculate Volume Trend
    volume_trend = df['volume'].rolling(window=5).mean()
    
    # Weighted Combination
    weighted_combination = short_term_momentum * volume_trend + long_term_momentum
    
    # Combine Components
    alpha_factor = close_momentum + volume_weighted_direction + weighted_combination
    
    return alpha_factor

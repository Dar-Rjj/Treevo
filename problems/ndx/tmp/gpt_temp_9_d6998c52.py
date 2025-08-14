import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate High-to-Low Range
    df['daily_range'] = df['High'] - df['Low']
    
    # Calculate Rolling Average of High-to-Low Range
    df['rolling_avg_range'] = df['daily_range'].rolling(window=14).mean()
    
    # Calculate Daily Price Change
    df['price_change'] = df['Close'] - df['Open']
    
    # Compute 5-Day and 10-Day Moving Average of Price Change
    df['ma_5_price_change'] = df['price_change'].rolling(window=5).mean()
    df['ma_10_price_change'] = df['price_change'].rolling(window=10).mean()
    
    # Determine Reversal Signal
    df['reversal_signal'] = 0
    df.loc[df['ma_5_price_change'] > df['ma_10_price_change'], 'reversal_signal'] = 1
    df.loc[df['ma_5_price_change'] < df['ma_10_price_change'], 'reversal_signal'] = -1
    df['reversal_signal'] *= (df['rolling_avg_range'] / df['rolling_avg_range'].mean())
    
    # Filter by Volume
    volume_threshold = df['Volume'].quantile(0.75)
    df['reversal_signal'] = df['reversal_signal'] * (df['Volume'] > volume_threshold)
    
    # Calculate Volume Adjusted Momentum
    df['log_returns'] = (df['Close'] / df['Close'].shift(1)).apply(lambda x: np.log(x))
    df['ema_log_return'] = df['log_returns'].ewm(span=20, adjust=False).mean()
    df['volume_adjusted_momentum'] = df['ema_log_return'] * df['Volume']
    
    # Calculate High-Low Range Volatility
    df['high_low_volatility'] = df['daily_range'].rolling(window=20).std()
    
    # Combine Volume Adjusted Momentum, Reversal Signal, and High-Low Range Volatility
    df['alpha_factor'] = (df['volume_adjusted_momentum'] 
                          + df['reversal_signal'] 
                          - df['high_low_volatility'])
    
    return df['alpha_factor']

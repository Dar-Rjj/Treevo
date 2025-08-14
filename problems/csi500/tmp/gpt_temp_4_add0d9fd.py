import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Volume-Weight the Return
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Compute Dynamic Lookback Period
    # Calculate Volatility using ATR (Average True Range) for simplicity
    df['tr'] = df[['high'-'low', 'high'-'close'].shift(1), 'low'-'close'.shift(1)]).max(axis=1)
    df['atr'] = df['tr'].rolling(window=20).mean()
    df['volatility'] = df['atr'] / df['close']
    
    # Adjust Lookback Period Based on Volatility
    lookback_min, lookback_max = 5, 30  # Define min and max lookback periods
    df['lookback_period'] = np.where(df['volatility'] > df['volatility'].quantile(0.75), lookback_min,
                                     np.where(df['volatility'] < df['volatility'].quantile(0.25), lookback_max, 20))
    
    # Generate Moving Averages
    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_long'] = df['close'].rolling(window=20).mean()
    
    # Differentiate Positive from Negative Returns
    df['positive_return_indicator'] = np.where(df['volume_weighted_return'] > 0, 1, 0)
    df['negative_return_indicator'] = np.where(df['volume_weighted_return'] <= 0, 1, 0)
    
    # Combine components into a single alpha factor
    df['alpha_factor'] = (df['volume_weighted_return'].rolling(window=df['lookback_period']).sum() +
                          (df['ma_short'] - df['ma_long']) +
                          (df['positive_return_indicator'] - df['negative_return_indicator']))
    
    return df['alpha_factor']

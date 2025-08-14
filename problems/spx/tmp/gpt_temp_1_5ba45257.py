import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Indicators
    df['simple_price_change'] = df['close'] - df['close'].shift(5)
    df['log_price_change'] = np.log(df['close'] / df['close'].shift(5))
    
    # Average Simple Price Change
    df['avg_simple_price_change'] = (df['close'].diff().rolling(window=5).sum() / 5)
    
    # Average Logarithmic Price Change
    df['avg_log_price_change'] = (np.log(df['close'] / df['close'].shift(1)).rolling(window=5).sum() / 5)
    
    # Volume-Based Indicators
    df['vwap'] = (df['volume'] * df['close']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    df['volume_to_price_ratio'] = (df['volume'] / df['close']) - (df['volume'].shift(5) / df['close'].shift(5))
    
    # Volatility Indicators
    df['true_range'] = df[['high', 'close']].max(axis=1) - df[['low', 'close'].shift(1)].min(axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (
        df['simple_price_change'] + 
        df['log_price_change'] + 
        df['avg_simple_price_change'] + 
        df['avg_log_price_change'] + 
        df['vwap'] + 
        df['volume_to_price_ratio'] + 
        df['atr']
    )
    
    return df['alpha_factor']

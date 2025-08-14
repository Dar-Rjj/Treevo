import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Exponential Momentum
    df['exponential_momentum'] = (df['close'] / df['close'].shift(30)) - 1
    
    # Calculate 30-Day Average True Range for Stability
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift()), abs(x['low'] - x['close'].shift())), axis=1)
    df['average_true_range_30'] = df['true_range'].rolling(window=30).mean()
    
    # Calculate Short-Term Exponential Moving Average
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate Long-Term Exponential Moving Average
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Adjust Momentum Factor with Inverse of Volatility
    df['adjusted_momentum'] = df['exponential_momentum'] * (1 / df['average_true_range_30'])
    
    # Incorporate Volume into the Momentum Adjustment
    df['volume_percentage_30'] = df['volume'] / df['volume'].rolling(window=30).mean()
    df['volume_adjusted_momentum'] = df['adjusted_momentum'] * df['volume_percentage_30']
    
    # Integrate Price Position Relative to Exponential Moving Averages
    conditions = [
        (df['close'] > df['ema_50']) & (df['close'] > df['ema_200']),
        (df['close'] < df['ema_50']) & (df['close'] < df['ema_200'])
    ]
    choices = [1, -1]
    df['price_position_relative_to_emas'] = np.select(conditions, choices, default=0)
    
    # Final Alpha Factor
    df['alpha_factor'] = df['volume_adjusted_momentum'] * df['price_position_relative_to_emas']
    
    return df['alpha_factor']

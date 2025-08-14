import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['intraday_momentum'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Weighted Sum of Intraday and Close-to-Open Returns
    df['weighted_intraday_return'] = df['intraday_momentum'].abs() * df['intraday_momentum']
    df['weighted_close_to_open_return'] = df['close_to_open_return'].abs() * df['close_to_open_return']
    df['preliminary_factor'] = df['weighted_intraday_return'] + df['weighted_close_to_open_return']
    
    # Calculate True Range (TR)
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = df[['high', 'low', 'prev_close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    
    # Calculate Positive Directional Movement (+DM)
    df['plus_dm'] = np.where((df['high'] > df['high'].shift(1)) & ((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low'])), 
                             df['high'] - df['high'].shift(1), 0)
    
    # Calculate Negative Directional Movement (-DM)
    df['minus_dm'] = np.where((df['low'].shift(1) > df['low']) & ((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1))), 
                              df['low'].shift(1) - df['low'], 0)
    
    # Smooth +DM and -DM
    df['smooth_plus_dm'] = df['plus_dm'].rolling(window=14).sum()
    df['smooth_minus_dm'] = df['minus_dm'].rolling(window=14).sum()
    df['smooth_true_range'] = df['true_range'].rolling(window=14).sum()
    
    # Calculate +DI and -DI
    df['plus_di'] = df['smooth_plus_dm'] / df['smooth_true_range']
    df['minus_di'] = df['smooth_minus_dm'] / df['smooth_true_range']
    
    # Calculate ADMI
    df['admi'] = (df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    
    # Multiply Preliminary Factor by ADMI
    df['intermediate_alpha_factor'] = df['preliminary_factor'] * df['admi']
    
    # Analyze Close Price Trends
    df['close_price_trend'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
    
    # Examine the Relationship Between Close Price and Volume
    df['close_to_avg_volume_ratio'] = df['close'] / df['volume'].rolling(window=20).mean()
    
    # Assess the Volatility
    df['daily_return'] = df['close'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=20).std()
    
    # Combine Intermediate Alpha Factor with New Components
    df['final_alpha_factor'] = df['intermediate_alpha_factor'] * df['close_price_trend'] * df['close_to_avg_volume_ratio'] / df['volatility']
    
    return df['final_alpha_factor']

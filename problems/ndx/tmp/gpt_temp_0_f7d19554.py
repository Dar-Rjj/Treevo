import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=10, m=14, k=14):
    # Calculate Daily Return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Volume Change Ratio
    df['volume_change_ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Compute Weighted Momentum
    df['weighted_momentum'] = (df['daily_return'] * df['volume_change_ratio']).rolling(window=n).sum()
    
    # Adjust for Price Volatility
    df['true_range'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    df['average_true_range'] = df['true_range'].rolling(window=m).mean()
    df['enhanced_atr'] = df['average_true_range'] * (1 + 0.5 * (df['high'] - df['low']) / df['close'].shift(1))
    df['momentum_adjusted_by_volatility'] = df['weighted_momentum'] - df['enhanced_atr']
    
    # Incorporate Trend Strength
    df['positive_dm'] = df['high'].diff().apply(lambda x: max(0, x))
    df['negative_dm'] = -df['low'].diff().apply(lambda x: min(0, x))
    df['smoothed_positive_dm'] = df['positive_dm'].ewm(span=k, adjust=False).mean()
    df['smoothed_negative_dm'] = df['negative_dm'].ewm(span=k, adjust=False).mean()
    df['positive_di'] = 100 * (df['smoothed_positive_dm'] / df['average_true_range'])
    df['negative_di'] = 100 * (df['smoothed_negative_dm'] / df['average_true_range'])
    df['adx'] = 100 * (abs(df['positive_di'] - df['negative_di']) / (df['positive_di'] + df['negative_di']))
    
    # Final Factor
    df['final_factor'] = df['momentum_adjusted_by_volatility'] * (df['adx'] / 100)
    
    return df['final_factor'].dropna()

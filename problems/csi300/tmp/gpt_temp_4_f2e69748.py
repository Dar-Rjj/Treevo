import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Short-term momentum
    n = 10
    df['short_momentum'] = df['close'] - df['close'].shift(n)
    
    # Long-term momentum
    m = 60
    df['long_momentum'] = df['close'] - df['close'].shift(m)
    
    # Relative volume factor
    k = 20
    df['avg_volume_20'] = df['volume'].rolling(window=k).mean()
    df['relative_volume'] = df['volume'] / df['avg_volume_20']
    
    # On-balance volume adjusted for price
    df['obv'] = 0
    df['obv'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume'] - (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    df['obv'] = df['obv'].cumsum()
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['obv_adjusted'] = df['obv'] * (df['price_change'] / df['close'])
    
    # Daily price movement
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    
    # True range
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = df[['high' - 'low', 'high' - 'prev_close', 'prev_close' - 'low']].max(axis=1)
    df['normalized_true_range'] = df['true_range'] / df['close']
    
    # Directional Movement Index (DMI)
    df['high_minus_low'] = df['high'] - df['low'].shift(1)
    df['high_minus_prev_close'] = df['high'] - df['close'].shift(1)
    df['prev_close_minus_low'] = df['close'].shift(1) - df['low']
    df['plus_dm'] = df[['high_minus_low', 'high_minus_prev_close']].max(axis=1)
    df['minus_dm'] = df[['high_minus_low', 'prev_close_minus_low']].max(axis=1)
    df['plus_di'] = df['plus_dm'].rolling(window=14).sum() / df['true_range'].rolling(window=14).sum()
    df['minus_di'] = df['minus_dm'].rolling(window=14).sum() / df['true_range'].rolling(window=14).sum()
    df['dmi'] = abs(df['plus_di'] - df['minus_di'])
    
    # Smoothed DMI
    df['smoothed_dmi'] = df['dmi'].rolling(window=7).mean()
    
    # Combine all factors
    df['alpha_factor'] = (
        df['short_momentum'] + 
        df['long_momentum'] + 
        df['relative_volume'] + 
        df['obv_adjusted'] + 
        df['daily_range'] + 
        df['normalized_true_range'] + 
        df['dmi'] + 
        df['smoothed_dmi']
    )
    
    return df['alpha_factor']

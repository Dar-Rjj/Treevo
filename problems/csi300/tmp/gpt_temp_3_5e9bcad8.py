import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate short-term and long-term EMAs
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # EMA crossover factor
    df['ema_crossover'] = 0
    df.loc[df['ema_5'] > df['ema_20'], 'ema_crossover'] = 1
    df.loc[df['ema_5'] < df['ema_20'], 'ema_crossover'] = -1
    
    # Price rate of change over 10 days
    df['roc_10'] = df['close'].pct_change(periods=10)
    
    # Volume Weighted Average Price (VWAP)
    df['vwap'] = (df['amount'] / df['volume']).cumsum() / (df['volume']).cumsum()
    df['vwap_factor'] = 0
    df.loc[df['close'] > df['vwap'], 'vwap_factor'] = 1
    df.loc[df['close'] < df['vwap'], 'vwap_factor'] = -1
    
    # Volume trend
    df['volume_trend'] = df['volume'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    # Relative Strength with a dynamic lookback window (e.g., 10 days)
    df['relative_strength'] = df['close'] / df['close'].shift(10)
    
    # Daily price range
    df['price_range'] = df['high'] - df['low']
    df['avg_range_10'] = df['price_range'].rolling(window=10).mean()
    df['range_factor'] = df['price_range'] / df['avg_range_10']
    
    # True Range
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1
    )
    df['avg_true_range_10'] = df['true_range'].rolling(window=10).mean()
    df['true_range_factor'] = df['true_range'] / df['avg_true_range_10']
    
    # Combine factors using simple arithmetic (e.g., addition)
    df['alpha_factor'] = (
        df['ema_crossover'] + 
        df['roc_10'] + 
        df['vwap_factor'] + 
        df['volume_trend'] + 
        df['relative_strength'] + 
        df['range_factor'] + 
        df['true_range_factor']
    )
    
    return df['alpha_factor']

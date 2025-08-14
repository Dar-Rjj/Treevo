import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10):
    # Calculate Daily Price Range
    df['daily_range'] = df['high'] - df['low']
    
    # Calculate True Range
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))),
        axis=1
    )
    
    # Calculate Average True Range (ATR) for the last N days
    df['atr'] = df['true_range'].rolling(window=n).mean()
    
    # Calculate Typical Price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate Volume-Weighted Average Price (VWAP) for the last N days
    df['tpv'] = df['typical_price'] * df['volume']
    df['vwap'] = df['tpv'].rolling(window=n).sum() / df['volume'].rolling(window=n).sum()
    
    # Normalize ATR by VWAP
    df['atr_vwap_ratio'] = df['atr'] / df['vwap']
    
    # Alpha Factor
    alpha_factor = df['atr_vwap_ratio']
    
    return alpha_factor

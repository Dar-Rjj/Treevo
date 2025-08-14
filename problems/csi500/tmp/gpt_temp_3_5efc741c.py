import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Price Change Velocity
    df['price_change'] = df['close'].diff()
    
    # Compute Weighted Moving Average of Price Changes
    window_size = 5
    weights = pd.Series(range(1, window_size + 1), index=df.index[-window_size:])
    df['weighted_price_change'] = df['price_change'].rolling(window=window_size).apply(lambda x: (x * weights[:len(x)]).sum() / weights[:len(x)].sum(), raw=False)
    
    # Calculate Adjusted Volume
    df['volume_change_ratio'] = df['volume'] / df['volume'].shift(1)
    df['ema_volume'] = df['volume'].ewm(alpha=0.2, adjust=False).mean()
    df['adjusted_volume'] = df['volume_change_ratio'] * df['ema_volume']
    
    # Combine Price Change Velocity and Adjusted Volume
    df['alpha_factor'] = df['weighted_price_change'] * df['adjusted_volume']
    
    return df['alpha_factor']

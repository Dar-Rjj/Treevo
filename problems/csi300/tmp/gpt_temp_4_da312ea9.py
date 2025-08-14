import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'].diff()
    
    # Calculate 5-Day Price Momentum
    df['5_day_momentum'] = df['daily_price_change'].rolling(window=5).sum()
    
    # Ensure sum is non-zero
    df['5_day_momentum'] = df['5_day_momentum'].where(df['5_day_momentum'] != 0, 1e-6)
    
    # Calculate Volume Confirmation
    df['volume_confirmation'] = (df['volume'] > df['volume'].shift(1)).astype(int)
    df['momentum_with_volume_confirmation'] = df['volume_confirmation'] * df['5_day_momentum']
    
    # Calculate 5-Day Average True Range (ATR)
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2].shift()), abs(x[1] - x[2].shift())), axis=1)
    df['5_day_atr'] = df['true_range'].rolling(window=5).mean()
    
    # Adjust for Volatility
    df['factor'] = df['momentum_with_volume_confirmation'] / df['5_day_atr']
    
    return df['factor'].dropna()

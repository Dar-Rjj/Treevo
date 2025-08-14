import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    true_range = df[['high_low_range', (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()]].max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    
    # Volume-Weighted Intraday Momentum
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    df['volume_weighted_return'] = df['intraday_return'] * df['volume']
    df['sma_5_volume_weighted'] = df['volume_weighted_return'].rolling(window=5).mean()
    
    # Trend Strength Indicator
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['trend_strength'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
    
    # Volume-Weighted Price-Volume Trend
    df['daily_price_change'] = df['close'] - df['close'].shift(1)
    df['price_volume_product'] = df['daily_price_change'] * df['volume']
    df['pvt_30'] = df['price_volume_product'].rolling(window=30).sum()
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['atr_14'] + df['sma_5_volume_weighted'] + df['trend_strength'] + df['pvt_30']) / 4
    
    return df['alpha_factor']

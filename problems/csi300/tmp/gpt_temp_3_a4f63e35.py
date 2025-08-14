import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Simple Moving Averages
    df['5_day_SMA'] = df['close'].rolling(window=5).mean()
    df['20_day_SMA'] = df['close'].rolling(window=20).mean()
    
    # Calculate Exponential Moving Averages
    df['5_day_EMA'] = df['close'].ewm(span=5, adjust=False).mean()
    df['20_day_EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate Momentum Indicators
    df['10_day_momentum'] = df['close'] / df['close'].shift(10) - 1
    df['30_day_momentum'] = df['close'] / df['close'].shift(30) - 1
    
    # Calculate Volatility Indicators
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    df['10_day_volatility'] = df['daily_return'].rolling(window=10).std()
    df['30_day_volatility'] = df['daily_return'].rolling(window=30).std()
    
    # Calculate Volume Moving Averages
    df['5_day_volume_MA'] = df['volume'].rolling(window=5).mean()
    df['20_day_volume_MA'] = df['volume'].rolling(window=20).mean()
    
    # Calculate Amount Moving Averages
    df['5_day_amount_MA'] = df['amount'].rolling(window=5).mean()
    df['20_day_amount_MA'] = df['amount'].rolling(window=20).mean()
    
    # Calculate Daily Range
    df['daily_range'] = df['high'] - df['low']
    
    # Calculate Average True Range
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))), axis=1)
    df['14_day_ATR'] = df['true_range'].rolling(window=14).mean()
    
    # Calculate Bollinger Bands
    df['20_day_SMA_close'] = df['close'].rolling(window=20).mean()
    df['20_day_std_close'] = df['close'].rolling(window=20).std()
    df['upper_bb'] = df['20_day_SMA_close'] + 2 * df['20_day_std_close']
    df['lower_bb'] = df['20_day_SMA_close'] - 2 * df['20_day_std_close']
    
    # Calculate Keltner Channels
    df['20_day_EMA_close'] = df['close'].ewm(span=20, adjust=False).mean()
    df['10_day_ATR'] = df['true_range'].rolling(window=10).mean()
    df['upper_kc'] = df['20_day_EMA_close'] + 2 * df['10_day_ATR']
    df['lower_kc'] = df['20_day_EMA_close'] - 2 * df['10_day_ATR']
    
    # Combine Multiple Indicators
    df['composite_momentum'] = (df['10_day_momentum'] + df['30_day_momentum']) / 2
    df['composite_volatility'] = (df['10_day_volatility'] + df['30_day_volatility']) / 2
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['composite_momentum'] / df['composite_volatility']) * (df['close'] - df['20_day_SMA'])
    
    return df['alpha_factor']

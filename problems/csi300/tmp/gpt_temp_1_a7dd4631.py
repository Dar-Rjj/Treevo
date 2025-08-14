import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open'].shift(1)) / df['open'].shift(1)
    
    # Calculate Daily Volume Change
    df['daily_volume_change'] = df['volume'] / df['volume'].shift(1)
    
    # Calculate Daily Amount Change
    df['daily_amount_change'] = df['amount'] / df['amount'].shift(1)
    
    # Weighted Intraday and Close-to-Open Returns
    df['weighted_intraday_return'] = df['intraday_return'] * abs(df['intraday_return'])
    df['weighted_close_to_open_return'] = df['close_to_open_return'] * abs(df['close_to_open_return'])
    
    # Integrate Volume and Amount Changes
    df['interim_factor'] = (df['weighted_intraday_return'] + df['weighted_close_to_open_return']) * df['daily_volume_change'] * df['daily_amount_change']
    
    # Smooth and Combine Interim Factor
    df['smoothed_interim_factor'] = df['interim_factor'].rolling(window=3).mean()
    
    # Simple Moving Average Cross Over
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_cross_over'] = df['sma_5'] - df['sma_20']
    
    # Exponential Weighted Moving Average Growth
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['ema_growth'] = df['ema_10'] - df['ema_30']
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # True Range Averaging
    tr = df[['high', 'low']].apply(lambda x: max(x[0], df['close'].shift(1)), axis=1) - df[['high', 'low']].apply(lambda x: min(x[1], df['close'].shift(1)), axis=1)
    df['atr_20'] = tr.rolling(window=20).mean()
    
    # Volume Weighted Price Change
    df['abs_close_to_close_change'] = abs(df['close'] - df['close'].shift(1))
    df['volume_weighted_price_change'] = (df['abs_close_to_close_change'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Volume Change Ratio
    df['volume_change_ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Final Composite Alpha Factor
    df['momentum_score'] = df['sma_cross_over'] + df['ema_growth']
    df['volatility_score'] = df['atr_20'] + df['volume_weighted_price_change']
    df['volume_score'] = df['volume_change_ratio']
    
    df['composite_alpha_factor'] = df['smoothed_interim_factor'] + df['momentum_score'] + df['volatility_score'] + df['volume_score']
    
    return df['composite_alpha_factor']

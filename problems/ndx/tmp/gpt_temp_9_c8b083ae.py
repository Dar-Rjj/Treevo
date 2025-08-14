import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Volume-Price Momentum
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['Volume_Price_Momentum'] = (df['VWAP'] - df['VWAP'].shift(1)) / df['VWAP'].shift(1)
    
    # Combined Momentum
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['Combined_Momentum'] = df['SMA_5'] - df['EMA_10']
    
    # True Range Momentum
    df['True_Range'] = df[['high', 'low', 'close']].join(df['close'].shift(1), rsuffix='_prev').max(axis=1) - \
                       df[['high', 'low', 'close']].join(df['close'].shift(1), rsuffix='_prev').min(axis=1)
    df['True_Range_Momentum'] = df['True_Range'] / df['close']
    
    # Volume-Weighted Momentum
    total_volume_5_days = df['volume'].rolling(window=5).sum()
    df['Volume_Weighted_Momentum'] = (df['close'] - df['open']) * (df['volume'] / total_volume_5_days)
    
    # Combined Breakout and Reversal
    df['20_day_high'] = df['high'].rolling(window=20).max().shift(1)
    df['20_day_vol_avg'] = df['volume'].rolling(window=20).mean()
    df['Volume_Spike'] = df['volume'] > 1.5 * df['20_day_vol_avg']
    df['New_High_Volume_Spike'] = (df['high'] > df['20_day_high']) & df['Volume_Spike']
    
    # Volume-Weighted Breakout
    df['20_day_VWAP'] = df['VWAP'].rolling(window=20).mean()
    df['Volume_Weighted_Breakout'] = (df['VWAP'] > df['20_day_VWAP']) & df['Volume_Spike']
    
    # Return the factor values
    return df[['Volume_Price_Momentum', 'Combined_Momentum', 'True_Range_Momentum', 'Volume_Weighted_Momentum', 
               'New_High_Volume_Spike', 'Volume_Weighted_Breakout']]

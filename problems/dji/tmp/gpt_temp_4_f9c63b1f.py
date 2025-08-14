import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Simple Moving Averages
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Calculate Exponential Moving Average with alpha = 2 / (n + 1)
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Average True Range (ATR)
    df['TR'] = df[['high' - 'low', 
                   (df['high'] - df['close'].shift(1)).abs(), 
                   (df['low'] - df['close'].shift(1)).abs()]].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Calculate Volume-Weighted Average Price (VWAP)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate Donchian Channels
    n = 20
    df['Upper_Band'] = df['high'].rolling(window=n).max()
    df['Lower_Band'] = df['low'].rolling(window=n).min()
    df['Middle_Band'] = (df['Upper_Band'] + df['Lower_Band']) / 2
    
    # Calculate Bullish and Bearish Engulfing Patterns
    df['Bullish_Engulfing'] = ((df['close'] > df['open']) & 
                               (df['close'].shift(1) < df['open'].shift(1)) & 
                               (df['close'] > df['open'].shift(1)) & 
                               (df['open'] < df['close'].shift(1)))
    df['Bearish_Engulfing'] = ((df['close'] < df['open']) & 
                               (df['close'].shift(1) > df['open'].shift(1)) & 
                               (df['close'] < df['open'].shift(1)) & 
                               (df['open'] > df['close'].shift(1)))
    
    # Calculate Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_money_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_money_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_money_flow_sum = positive_money_flow.rolling(window=14).sum()
    negative_money_flow_sum = negative_money_flow.rolling(window=14).sum()
    money_ratio = positive_money_flow_sum / negative_money_flow_sum
    df['MFI'] = 100 - (100 / (1 + money_ratio))
    
    # Combine factors into a single alpha factor
    df['alpha_factor'] = (df['SMA_5'] - df['SMA_20']) + \
                         (df['EMA_10'] - df['SMA_20']) + \
                         df['RSI'] - 50 + \
                         df['ATR'] + \
                         df['OBV'] + \
                         df['VWAP'] + \
                         (df['Middle_Band'] - df['close']) + \
                         df['Bullish_Engulfing'] - df['Bearish_Engulfing'] + \
                         (df['MFI'] - 50)
    
    return df['alpha_factor']

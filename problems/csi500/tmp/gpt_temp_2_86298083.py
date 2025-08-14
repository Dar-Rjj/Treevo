import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 50-day and 200-day SMAs
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Calculate the difference between 50-day and 200-day SMAs
    df['SMA_Diff'] = df['SMA_50'] - df['SMA_200']
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = 0
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            df['OBV'].iloc[i] = df['OBV'].iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            df['OBV'].iloc[i] = df['OBV'].iloc[i-1] - df['volume'].iloc[i]
        else:
            df['OBV'].iloc[i] = df['OBV'].iloc[i-1]
    
    # Identify Doji Candlestick Pattern
    df['IsDoji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low'])) < 0.01
    
    # Calculate True Range (TR)
    df['TR'] = df[['high', 'low']].apply(lambda x: max(x['high'], x['low']), axis=1)
    df['TR'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1).iloc[x.name]), abs(x['low'] - df['close'].shift(1).iloc[x.name])), axis=1)
    
    # Calculate Average True Range (ATR) over 14 days
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Calculate Money Flow Index (MFI)
    df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
    df['RawMoneyFlow'] = df['TypicalPrice'] * df['volume']
    
    positive_mf = df[df['TypicalPrice'] > df['TypicalPrice'].shift(1)]['RawMoneyFlow'].rolling(window=14).sum()
    negative_mf = df[df['TypicalPrice'] < df['TypicalPrice'].shift(1)]['RawMoneyFlow'].rolling(window=14).sum()
    
    df['PositiveMF'] = positive_mf
    df['NegativeMF'] = negative_mf
    
    df['MFI'] = 100 - (100 / (1 + (df['PositiveMF'] / df['NegativeMF'])))
    
    # Combine all factors into a single alpha factor
    df['AlphaFactor'] = df['SMA_Diff'] + df['OBV'] + df['IsDoji'].astype(int) + df['ATR'] + df['MFI']
    
    return df['AlphaFactor']

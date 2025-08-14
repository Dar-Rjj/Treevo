import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Momentum Indicators
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['SMA_Diff'] = df['SMA_50'] - df['SMA_200']
    
    # Volume Indicators
    df['OBV'] = 0
    for i in range(1, len(df)):
        if df.loc[df.index[i], 'close'] > df.loc[df.index[i-1], 'close']:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] + df.loc[df.index[i], 'volume']
        elif df.loc[df.index[i], 'close'] < df.loc[df.index[i-1], 'close']:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] - df.loc[df.index[i], 'volume']
        else:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV']
    
    # Price Pattern Indicators
    df['IsDoji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low'])) < 0.01
    
    # Volatility Indicators
    df['TR'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Market Sentiment Indicators
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['PositiveMF'] = 0
    df['NegativeMF'] = 0
    for i in range(1, len(df)):
        if df.loc[df.index[i], 'TP'] > df.loc[df.index[i-1], 'TP']:
            df.loc[df.index[i], 'PositiveMF'] = df.loc[df.index[i], 'TP'] * df.loc[df.index[i], 'volume']
        else:
            df.loc[df.index[i], 'NegativeMF'] = df.loc[df.index[i], 'TP'] * df.loc[df.index[i], 'volume']
    
    positive_rolling_sum = df['PositiveMF'].rolling(window=14).sum()
    negative_rolling_sum = df['NegativeMF'].rolling(window=14).sum()
    df['MFI'] = 100 - (100 / (1 + (positive_rolling_sum / negative_rolling_sum)))
    
    # Combine all indicators to form the alpha factor
    df['AlphaFactor'] = (df['SMA_Diff'] + df['OBV'] + df['IsDoji'].astype(int) + df['ATR'] + df['MFI'])
    
    return df['AlphaFactor']

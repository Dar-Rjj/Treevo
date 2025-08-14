import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['TotalVolume'] = df['volume']
    df['TotalDollarValue'] = df['volume'] * df['close']
    df['VWAP'] = df.groupby(df.index.date)['TotalDollarValue'].cumsum() / df.groupby(df.index.date)['TotalVolume'].cumsum()
    
    # Calculate VWAP Deviation
    df['VWAPDeviation'] = df['close'] - df['VWAP']
    
    # Calculate Exponential Moving Average (EMA) of VWAP
    df['VWAPEMA'] = df['VWAP'].ewm(span=10, adjust=False).mean()
    
    # Calculate VWAP EMA Deviation
    df['VWAPEMADeviation'] = df['VWAP'] - df['VWAPEMA']
    
    # Calculate Cumulative VWAP Deviation
    df['CumulativeVWAPDeviation'] = df['VWAPDeviation'].rolling(window=10).sum()
    
    # Calculate Volume Trend
    df['VolumeTrend'] = df['volume'].pct_change()
    
    # Calculate Multi-Period Momentum
    df['MultiPeriodMomentum'] = df['close'].pct_change().rolling(window=10).sum()
    
    # Calculate True Range
    df['TrueRange'] = df[['high', 'low']].apply(lambda x: max(x['high'], df['close'].shift(1)) - min(x['low'], df['close'].shift(1)), axis=1)
    
    # Calculate Average True Range (ATR)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    
    # Integrate Factors into Final Alpha Signal
    df['AlphaSignal'] = (df['CumulativeVWAPDeviation'] + 
                         df['VWAPEMADeviation'] + 
                         df['VolumeTrend'] + 
                         df['MultiPeriodMomentum'] + 
                         1/df['ATR'])
    
    return df['AlphaSignal']
